"""Free-form answer grader for MMAR difficulty experiments (multi-judge).

Judge prompt assembly is table-driven (``GRADE_PROMPT_RECIPES``). Inspect gold
(with-gt) and nongold (free) prompts, with variable slots left as
``{question}`` / ``{answer}`` / ``{prediction}``::

    python grader.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from audio_flamingo_runtime import resolve_model_dir
from mmar_common import (
    ensure_judge_schema,
    judge_label,
    parse_freeform_output,
    recompute_multi_judge_scores,
    select_grade_question_ids,
    split_last_think_close,
    write_jsonl,
)

DEFAULT_JUDGE_MODEL_IDS = ("Qwen/Qwen3.6-35B-A3B-FP8",)
# Back-compat aliases.
DEFAULT_GRADER_MODEL_ID = DEFAULT_JUDGE_MODEL_IDS[0]
GRADER_LABEL = judge_label(DEFAULT_GRADER_MODEL_ID)

# Default judge sampling fallback (unknown judges).
DEFAULT_JUDGE_SAMPLING: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 4096,
    "seed": 0,
}
DEFAULT_JUDGE_MAX_TOKENS = int(DEFAULT_JUDGE_SAMPLING["max_tokens"])
DEFAULT_JUDGE_BATCH_SIZE = 64

# Short names accepted by judge CLIs (mirrors seed_volume.MODEL_ALIASES).
JUDGE_MODEL_ALIASES: dict[str, str] = {
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3.6-35b-a3b-fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
    "qwen3.6-35b-a3b": "Qwen/Qwen3.6-35B-A3B-FP8",
    "qwen3.6-35b": "Qwen/Qwen3.6-35B-A3B-FP8",
    "qwen3.6": "Qwen/Qwen3.6-35B-A3B-FP8",
}

# Per-judge vLLM engine + SamplingParams (mirrors MODEL_SPECS for test takers).
JUDGE_SPECS: dict[str, dict[str, Any]] = {
    "qwen2.5-3b-instruct": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "engine": {
            "dtype": "bfloat16",
            # Grade prompt + up to max_tokens generation.
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            "enforce_eager": True,
            "enable_prefix_caching": True,
            "trust_remote_code": True,
        },
        # generation_config.json: T=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05
        "sampling": {
            "temperature": 0.0,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "max_tokens": 4096,
            "seed": 0,
        },
        "batch_size": 128,
    },
    "qwen3.6-35b-a3b-fp8": {
        "model_id": "Qwen/Qwen3.6-35B-A3B-FP8",
        "engine": {
            "dtype": "auto",
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            # Only 3B params are active per token, so decode is kernel-launch
            # bound and CUDA graphs dominate: eager measured 332 output tok/s
            # vs 4.8-7k with graphs on one H100 (see tune_judge.py).
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.92,
            # Text-only path: skip vision tower for grading.
            "language_model_only": True,
        },
        # Thinking-mode defaults: T=1.0, top_p=0.95, top_k=20
        # (judging forces temperature=0 for determinism).
        # Grade replies average ~600 tokens; 4096 leaves headroom for the rare
        # runaway (~0.2% of shots) whose truncation would score as a Fail.
        "sampling": {
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 4096,
            "seed": 0,
        },
        # Throughput keeps scaling with concurrency (128→3.5k, 256→4.8k,
        # 512→7.0k output tok/s); KV holds ~1M tokens so 512 seqs fit easily.
        "batch_size": 512,
    },
}

ALL_JUDGE_LABELS = tuple(JUDGE_SPECS.keys())

# Open vLLM models that grade each other in round-robin (not mimo / voxtral /
# AF-Next / Thinking unless they appear in a run and are passed explicitly).
ROUND_ROBIN_SUITE: tuple[str, ...] = (
    "qwen2.5-omni-7b",
    "phi-4-multimodal",
    "gemma-4-e4b",
    "qwen3-omni-instruct",
    "nemotron-3-nano-omni",
)

# Internal names: with_gt (gold shown) vs free (audio judge, no gold).
# Not a CLI choice — include_gold selects which recipe to use.
GRADE_PROMPT_NAMES = ("with_gt", "free")
DEFAULT_GRADE_PROMPT = "with_gt"
DEFAULT_INCLUDE_GOLD = True

# Judge prompt piece text and the gold / free assembly table live in the
# "Judge prompt recipes" section below. Run ``python grader.py`` to print every
# rendered combination with ``{question}`` / ``{answer}`` / ``{prediction}``
# placeholders.


# ---------------------------------------------------------------------------
# Judge resolution (alias → spec → engine / sampling)
# ---------------------------------------------------------------------------


def resolve_judge_model_id(model_id: str) -> str:
    """Expand a judge alias (e.g. ``qwen3.6-35b-a3b-fp8``) to a Hub repo id."""
    key = str(model_id or "").strip()
    if not key:
        return key
    return JUDGE_MODEL_ALIASES.get(key.lower(), key)


def _default_judge_engine(model_id: str) -> dict[str, Any]:
    """Fallback engine kwargs for judges not listed in JUDGE_SPECS."""
    engine: dict[str, Any] = {
        "dtype": "auto" if _looks_fp8(model_id) else "bfloat16",
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "trust_remote_code": True,
    }
    if _needs_language_model_only(model_id):
        engine["language_model_only"] = True
    return engine


def resolve_judge_spec(model_id: str) -> dict[str, Any]:
    """Return the full judge spec dict for a Hub id or alias."""
    resolved_id = resolve_judge_model_id(model_id)
    label = judge_label(resolved_id)
    spec = JUDGE_SPECS.get(label)
    if spec:
        return spec
    return {
        "model_id": resolved_id,
        "engine": _default_judge_engine(resolved_id),
        "sampling": dict(DEFAULT_JUDGE_SAMPLING),
        "batch_size": DEFAULT_JUDGE_BATCH_SIZE,
    }


def resolve_judge_batch_size(
    model_id: str,
    batch_size: int | None = None,
) -> int:
    """Return shots-per-generate batch size for a judge."""
    if batch_size is not None:
        return int(batch_size)
    spec = resolve_judge_spec(model_id)
    return int(spec.get("batch_size", DEFAULT_JUDGE_BATCH_SIZE))


def resolve_judge_sampling(
    model_id: str,
    args: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """Return SamplingParams kwargs for a judge (per-model; no global defaults).

    Optional CLI overrides on ``args`` (``temperature``, ``top_p``,
    ``max_new_tokens``) replace the model values when not ``None``.
    """
    spec = resolve_judge_spec(model_id)
    if "sampling" not in spec:
        raise ValueError(
            f"Judge {model_id!r} ({judge_label(resolve_judge_model_id(model_id))!r}) "
            "has no per-judge sampling config"
        )
    out = dict(spec["sampling"])
    if args is None:
        return out
    if getattr(args, "temperature", None) is not None:
        out["temperature"] = float(args.temperature)
    if getattr(args, "top_p", None) is not None:
        out["top_p"] = float(args.top_p)
    if getattr(args, "max_new_tokens", None) is not None:
        out["max_tokens"] = int(args.max_new_tokens)
    return out


def resolve_judge_engine(
    model_id: str,
    args: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """Return vLLM LLM kwargs for a judge (excluding ``model`` path)."""
    spec = resolve_judge_spec(model_id)
    engine = dict(spec["engine"])
    if args is None:
        return engine
    if getattr(args, "max_num_seqs", None) is not None:
        engine["max_num_seqs"] = int(args.max_num_seqs)
    if getattr(args, "gpu_memory_utilization", None) is not None:
        engine["gpu_memory_utilization"] = float(args.gpu_memory_utilization)
    if getattr(args, "max_model_len", None) is not None:
        engine["max_model_len"] = int(args.max_model_len)
    return engine


def judge_sampling_params(
    model_id: str,
    args: SimpleNamespace | None = None,
    *,
    max_tokens: int | None = None,
):
    """Build a vLLM SamplingParams instance for one judge."""
    from vllm import SamplingParams

    kwargs = resolve_judge_sampling(model_id, args)
    temperature = float(kwargs.get("temperature", 0.0))
    kwargs["temperature"] = temperature if temperature > 0 else 0.0
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    return SamplingParams(**kwargs)


def _looks_fp8(model_id: str) -> bool:
    text = model_id.lower()
    return "fp8" in text or text.endswith("-fp8")


def _needs_language_model_only(model_id: str) -> bool:
    """Qwen3.5/3.6 multimodal checkpoints: skip the vision tower for text judging."""
    text = model_id.lower()
    return any(token in text for token in ("qwen3.5", "qwen3.6", "qwen3_5", "qwen3_6"))


def grade_prompt_name(include_gold: bool = DEFAULT_INCLUDE_GOLD) -> str:
    """``with_gt`` when gold is shown; ``free`` when the audio judge has no gold."""
    return "with_gt" if include_gold else "free"


def normalize_grade_prompt(
    name: str | None,
    *,
    include_gold: bool | None = None,
) -> str:
    """Return ``with_gt`` / ``free``. ``include_gold`` wins when given."""
    if include_gold is not None:
        return grade_prompt_name(include_gold)
    value = str(name or DEFAULT_GRADE_PROMPT).strip().lower()
    if value in GRADE_PROMPT_NAMES:
        return value
    # Legacy permissive/neutral names map onto the gold recipe; nongold
    # callers should pass include_gold so this does not matter.
    if value in {"permissive", "neutral", "all", ""}:
        return DEFAULT_GRADE_PROMPT
    raise ValueError(
        f"Unknown grade prompt {name!r}; expected one of {GRADE_PROMPT_NAMES}"
    )


def parse_grade_prompt_list(
    value: str | None = None,
    *,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> list[str]:
    """Return the single active prompt name. Style lists are ignored."""
    del value
    return [grade_prompt_name(include_gold)]


def gold_tag(include_gold: bool) -> str:
    return "gold" if include_gold else "nongold"


def compose_judge_key(
    model_label: str,
    *,
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> str:
    """Stable key: ``{label}__{with_gt|free}__{gold|nongold}``."""
    label = str(model_label or "").strip()
    if not label:
        raise ValueError("compose_judge_key requires a model_label")
    prompt_name = normalize_grade_prompt(prompt, include_gold=include_gold)
    return f"{label}__{prompt_name}__{gold_tag(include_gold)}"


def resolve_grade_judge_key(
    handle: dict[str, Any],
    *,
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
    judge_key: str | None = None,
) -> str:
    """Composite key ``{label}__{with_gt|free}__{gold|nongold}``.

    Always composed (never the bare label) so new with-gt / free verdicts do
    not collide with older permissive/neutral grades.
    """
    if judge_key:
        return str(judge_key)
    label = str(handle.get("judge_label") or judge_label(handle.get("model_id")) or GRADER_LABEL)
    return compose_judge_key(label, prompt=prompt, include_gold=include_gold)


def parse_shot_indices(first_shot_only: bool) -> tuple[int, ...] | None:
    return (0,) if first_shot_only else None


# ===========================================================================
# Judge prompt recipes
#
# Two combinations (no style axis):
#   include_gold True  → with_gt  (text judge sees the benchmark answer)
#   include_gold False → free     (audio judge hears the clip; no gold)
#
# Adapted from nikhilchandak/answer-matching ``gpqa_judge.py``
# (``get_judge_prompt_with_gt`` / ``get_free_judge_prompt``) for audio:
# the free judge is told it will hear the clip; with_gt is told the
# question is about an audio clip. Output is 0/1 in <answer> tags.
#
# Each recipe is three slots joined with a blank line between slots:
#
#   before_audio     piece names, joined by "\n"
#                    sent first; on nongold the clip is inserted after this
#   field_templates  filled with {question} / {answer} / {prediction}
#                    joined by "\n"
#   after_fields     piece names, joined by "\n"
#                    gold: match-only closer after the fields
#                    free: numeric/vagueness rules + closer after the clip
#                    and fields (audio wraps append this after {fields})
#
# Gold / with_gt:
#   with_gt_role
#   Question / Ground truth / Response
#   with_gt_closer
#
# Nongold / free:
#   free_role
#   [AUDIO]
#   Question / Response
#   free_closer
#
# ``build_grade_prompt`` is the full text (API judges and gold vLLM).
# Audio suite judges take ``instructions`` + ``fields`` from the recipe and
# wrap them with ``NONGOLD_AUDIO_PROMPT_TEMPLATES`` / chat messages.
# Inspect every combo:  python grader.py
# ===========================================================================

# Named pieces. Recipes below refer to these keys, not the raw strings.
JUDGE_PIECES: dict[str, str] = {
    "with_gt_role": (
        "Your task is to judge whether the given response to an audio question "
        "matches a given ground truth answer or not. You are provided with a "
        "question about an audio clip, a ground truth response, and the response "
        "you need to judge.\n"
        'For a response to "match", it must have at least as much information '
        "as the ground-truth. \n"
        "The response can have more information than the ground-truth. It can "
        'be more specific (for example, "Labrador" is more specific than "dog"), '
        "or have additional possible correct answers. But it must cover everything "
        "mentioned in the ground-truth. It is okay if it covers it in different "
        "words, i.e. paraphrased. \n"
        "For numeric answers, the relative error, defined as |response - ground "
        "truth| / mean(response, ground truth), must be less than 1% for the "
        "response to be judged as a correct match. Here, if the ground truth is "
        "a specific numeric quantity but the response is a range, then they don't "
        "match (even if the range contains the ground truth).\n"
        "\n"
        "Possible judgments:\n"
        "\n"
        '"0": The response does not match the ground-truth answer.\n'
        '"1": The response matches the ground-truth.'
    ),
    "with_gt_closer": (
        "Your job is to ONLY check whether the given response matches the ground "
        "truth answer or not in the context of the question. You DO NOT NEED to "
        "assess the correctness of the response. This is part of an automated "
        'evaluation process, therefore you MUST OUTPUT your final answer as "0" '
        'or "1" in <answer> </answer> tags.\n'
        "Think step by step and end your response with <answer>0</answer> OR "
        "<answer>1</answer> TAGS."
    ),
    "free_role": (
        "Your task is to judge whether the given response to an audio question "
        "is correct or not. You are given an audio clip, a question about that "
        "clip, and the response you are judging.\n"
        "Possible judgments:\n"
        '"0": The response is incorrect. \n'
        '"1": The response is correct.'
    ),
    "free_closer": (
        "The response should fully answer the question and must not be vague.\n"
        "For numeric answers, the relative error, defined as |response - ground "
        "truth| / mean(response, ground truth), must be less than 1% for the "
        "response to be judged as a correct match. Here, if the ground truth is "
        "a specific numeric quantity but the response is a range, then they don't "
        "match (even if the range contains the ground truth).\n"
        "\n"
        "To the best of your knowledge: Does the provided response answer the "
        "question correctly, given the audio? This is part of an automated "
        'evaluation process, therefore you MUST OUTPUT your final answer as "0" '
        'or "1" in <answer> </answer> tags.\n'
        "Think step by step and end your response with <answer>0</answer> OR "
        "<answer>1</answer> TAGS."
    ),
}

# Placeholders filled by ``GradePromptRecipe.fields``.
FIELD_QUESTION = 'Question: "{question}"'
FIELD_GOLD = 'Ground truth: "{answer}"'
FIELD_PREDICTION = 'Response: "{prediction}"'

# Dummy values used by ``python grader.py`` so variable slots stay visible.
PROMPT_PLACEHOLDERS: dict[str, str] = {
    "question": "{question}",
    "answer": "{answer}",
    "prediction": "{prediction}",
}


def _fill_placeholders(template: str, **values: str) -> str:
    """Replace ``{name}`` slots; braces inside values are kept literal."""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def _join_pieces(names: tuple[str, ...]) -> str:
    return "\n".join(JUDGE_PIECES[name] for name in names)


@dataclass(frozen=True)
class GradePromptRecipe:
    """One (gold / free) judge prompt, split at the audio insertion point."""

    name: str
    include_gold: bool
    before_audio: tuple[str, ...]
    field_templates: tuple[str, ...]
    after_fields: tuple[str, ...]

    def instructions(self) -> str:
        """Text sent before the clip (nongold) or as the first block (gold)."""
        return _join_pieces(self.before_audio)

    def fields(
        self,
        *,
        question: str,
        answer: str = "",
        prediction: str,
    ) -> str:
        """Question / gold / test-taker answer block (no audio)."""
        return "\n".join(
            _fill_placeholders(
                line,
                question=question,
                answer=answer,
                prediction=prediction,
            )
            for line in self.field_templates
        )

    def closer(self) -> str:
        """Block after the fields (match-only closer, or free-judge rules)."""
        return _join_pieces(self.after_fields)

    def fields_with_closer(
        self,
        *,
        question: str,
        answer: str = "",
        prediction: str,
    ) -> str:
        """Fields plus closer — used after the clip on the nongold audio path."""
        text = self.fields(question=question, answer=answer, prediction=prediction)
        closer = self.closer()
        if closer:
            return f"{text}\n\n{closer}"
        return text

    def as_text(
        self,
        *,
        question: str,
        answer: str = "",
        prediction: str,
    ) -> str:
        """Full text prompt (gold vLLM path and API judges)."""
        parts = [
            self.instructions(),
            self.fields(question=question, answer=answer, prediction=prediction),
        ]
        closer = self.closer()
        if closer:
            parts.append(closer)
        return "\n\n".join(parts)


# Gold (with_gt) and free (nongold). Edit this table — not the builders.
GRADE_PROMPT_RECIPES: dict[bool, GradePromptRecipe] = {
    True: GradePromptRecipe(
        name="with_gt",
        include_gold=True,
        before_audio=("with_gt_role",),
        field_templates=(FIELD_QUESTION, FIELD_GOLD, FIELD_PREDICTION),
        after_fields=("with_gt_closer",),
    ),
    False: GradePromptRecipe(
        name="free",
        include_gold=False,
        before_audio=("free_role",),
        field_templates=(FIELD_QUESTION, FIELD_PREDICTION),
        after_fields=("free_closer",),
    ),
}


def get_grade_prompt_recipe(
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> GradePromptRecipe:
    del prompt
    key = bool(include_gold)
    recipe = GRADE_PROMPT_RECIPES.get(key)
    if recipe is None:
        raise ValueError(f"No grade-prompt recipe for include_gold={include_gold!r}")
    return recipe


def iter_grade_prompt_recipes() -> tuple[GradePromptRecipe, ...]:
    """Gold (with_gt) first, then free (nongold)."""
    return tuple(GRADE_PROMPT_RECIPES[flag] for flag in (True, False))


def build_grade_instructions(
    *,
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> str:
    """Judge preamble sent before the clip (nongold) or as the first block (gold)."""
    return get_grade_prompt_recipe(prompt, include_gold).instructions()


def build_grade_input_fields(
    *,
    question: str,
    answer: str,
    prediction: str,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> str:
    """Question / gold / test-taker answer block (no audio)."""
    return get_grade_prompt_recipe(include_gold=include_gold).fields(
        question=question,
        answer=answer,
        prediction=prediction,
    )


def build_grade_after_audio(
    *,
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> str:
    """Text after the fields. For nongold this is appended after the clip + fields."""
    return get_grade_prompt_recipe(prompt, include_gold).closer()


def build_grade_prompt(
    *,
    question: str,
    answer: str,
    prediction: str,
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> str:
    """Assemble the full text judge prompt for gold or free."""
    return get_grade_prompt_recipe(prompt, include_gold).as_text(
        question=question,
        answer=answer,
        prediction=prediction,
    )


def build_grade_gold_prefix(*, question: str, answer: str) -> str:
    """Cached gold prefix: instructions + question + ground truth (no response)."""
    recipe = get_grade_prompt_recipe(include_gold=True)
    question_line = _fill_placeholders(FIELD_QUESTION, question=question)
    gold_line = _fill_placeholders(FIELD_GOLD, answer=answer)
    return f"{recipe.instructions()}\n\n{question_line}\n{gold_line}\n"


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

# Final-answer tokens: <answer>0</answer> / <answer>1</answer> (preferred),
# plus legacy Pass/Fail and YES/NO from older runs.
# Soft whole-region fallback — keep this narrow so prose like "correct in
# meaning" does not count as a verdict; token labels above still accept
# a bare "correct" / "incorrect" final line.
ANSWER_TAG_RE = re.compile(r"<answer>\s*([01])\s*</answer>", re.IGNORECASE)
PASS_RE = re.compile(r"\b(pass|yes|true)\b", re.IGNORECASE)
FAIL_RE = re.compile(r"\b(fail|no|false)\b", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
PASS_LABELS = frozenset({"PASS", "P", "YES", "Y", "TRUE", "CORRECT", "1"})
FAIL_LABELS = frozenset({"FAIL", "F", "NO", "N", "FALSE", "INCORRECT", "WRONG", "0"})


def _answer_region(text: str) -> str:
    """Prefer text after a closed ``</think>`` block; else the full string."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    match = THINK_BLOCK_RE.search(cleaned)
    if match:
        remainder = (cleaned[: match.start()] + cleaned[match.end() :]).strip()
        if remainder:
            return remainder
    return cleaned


def parse_grade_verdict(text: str) -> bool | None:
    """Parse a 0/1 <answer> tag (or legacy Pass/Fail) reply. None if unparseable."""
    region = _answer_region(text)
    if not region:
        return None
    tag_matches = list(ANSWER_TAG_RE.finditer(region))
    if tag_matches:
        return tag_matches[-1].group(1) == "1"
    lines = [line.strip() for line in region.splitlines() if line.strip()]
    if not lines:
        return None

    def _token_verdict(raw: str) -> bool | None:
        token = raw.strip("`\"' .").upper()
        if ":" in token:
            token = token.split(":")[-1].strip("`\"' .")
        # Keep only the last whitespace-separated word (e.g. "Answer Pass").
        if " " in token:
            token = token.split()[-1].strip("`\"' .")
        if token in PASS_LABELS:
            return True
        if token in FAIL_LABELS:
            return False
        return None

    # Prefer an explicit Pass/Fail (or legacy YES/NO) on the final line.
    last = _token_verdict(lines[-1])
    if last is not None:
        return last

    # Scan earlier lines only for a clean Pass/Fail token (ignore prose).
    for line in reversed(lines[:-1]):
        hit = _token_verdict(line)
        if hit is not None:
            return hit

    # Last resort: whole-region exclusive keyword match (legacy YES/NO dumps).
    has_pass = bool(PASS_RE.search(region))
    has_fail = bool(FAIL_RE.search(region))
    if has_pass and not has_fail:
        return True
    if has_fail and not has_pass:
        return False
    return None


def format_grade_output(verdict: bool | None) -> str | None:
    """Short 0/1 label for schema ``output`` / tips (legacy Pass/Fail still parsed)."""
    if verdict is True:
        return "1"
    if verdict is False:
        return "0"
    return None


def _normalize_grade_answer(text: str) -> str:
    """Lowercase exact-match key for reusing a prior grade."""
    return str(text or "").strip().lower()


def _shot_prediction_text(shot: dict) -> str:
    """Extracted answer shown to the judge; never includes text before last ``</think>``."""
    raw = str(shot.get("model_output") or "")
    extracted = str(shot.get("answer_prediction") or "")
    source = raw if split_last_think_close(raw) is not None else (extracted or raw)
    if split_last_think_close(source) is not None:
        _, answer = parse_freeform_output(source)
        return answer
    return extracted or raw


def _shot_judge_entry(shot: dict, judge_key: str) -> dict | None:
    judges = shot.get("judges")
    if isinstance(judges, dict):
        entry = judges.get(judge_key)
        if entry is not None and entry.get("correct") is not None:
            return entry
    return None


def _grade_reuse_key(
    question: str,
    answer: str,
    prediction: str,
    *,
    include_gold: bool,
) -> tuple[str, str, str]:
    """Identity of a grade prompt: question, gold (if shown), normalized answer."""
    gold = str(answer or "") if include_gold else ""
    return (str(question or ""), gold, _normalize_grade_answer(prediction))


def _shot_needs_grade(shot: dict, judge_key: str) -> bool:
    """True when this judge has no verdict yet for the shot."""
    if _shot_judge_entry(shot, judge_key) is not None:
        return False
    # Legacy flat fields only count for the same judge label/id.
    legacy_id = shot.get("grader")
    if legacy_id and judge_label(legacy_id) == judge_key and shot.get("correct") is not None:
        return False
    return True


def _record_needs_grade(
    record: dict,
    judge_key: str,
    *,
    shot_indices: tuple[int, ...] | list[int] | None = None,
) -> bool:
    shots = record.get("shots") or []
    if shot_indices is not None:
        allowed = {int(i) for i in shot_indices}
        shots = [
            shot for shot in shots if int(shot.get("shot_index", 0)) in allowed
        ]
    if not shots:
        return False
    return any(_shot_needs_grade(shot, judge_key) for shot in shots)


def _suite_label_for(model_id: str) -> str | None:
    """Return a MODEL_SPECS label when ``model_id`` is a suite alias or HF id."""
    from mmar_models import MODEL_SPECS

    key = str(model_id or "").strip()
    if not key:
        return None
    if key in MODEL_SPECS:
        return key
    for label, spec in MODEL_SPECS.items():
        if spec.get("model_id") == key:
            return label
    return None


def judge_is_audio_model(
    model_id: str | None = None,
    handle: dict[str, Any] | None = None,
) -> bool:
    """True when this judge can hear MMAR audio (suite or API audio models)."""
    if handle and handle.get("suite_label"):
        return True
    if handle and handle.get("backend") in {"openai", "gemini"}:
        return True
    keys = [
        (handle or {}).get("suite_label"),
        (handle or {}).get("judge_label"),
        (handle or {}).get("model_id"),
        model_id,
    ]
    if any(_suite_label_for(str(key or "")) is not None for key in keys):
        return True
    from mmar_api import is_api_judge

    return any(is_api_judge(str(key or "")) for key in keys if key)


def require_audio_nongold_judge(
    model_id: str | None = None,
    handle: dict[str, Any] | None = None,
    *,
    include_gold: bool,
) -> None:
    """NO_GOLD grading is audio-only: text judges cannot decide without gold."""
    if include_gold:
        return
    if judge_is_audio_model(model_id, handle):
        return
    shown = (
        (handle or {}).get("suite_label")
        or (handle or {}).get("judge_label")
        or (handle or {}).get("model_id")
        or model_id
        or "<unknown>"
    )
    raise SystemExit(
        "NO_GOLD / --no-include-gold grading requires an audio-capable judge "
        f"that receives the clip (got {shown!r}). Text-only judges cannot "
        "grade without ground truth; use a suite or API audio model or pass "
        "--include-gold."
    )


def _grade_sampling_for_engine(engine: dict[str, Any], sampling: dict[str, Any]) -> dict[str, Any]:
    """Force deterministic grading; cap max_tokens to fit ``max_model_len``."""
    out = dict(sampling)
    out["temperature"] = 0.0
    max_len = int(engine.get("max_model_len") or 8192)
    requested = int(out.get("max_tokens") or DEFAULT_JUDGE_MAX_TOKENS)
    # Leave headroom for the grade prompt; CoT + <answer> tags need room.
    cap = max(256, min(requested, max_len // 2))
    out["max_tokens"] = cap
    out["seed"] = 0
    return out


# ---------------------------------------------------------------------------
# Load a local / suite judge
# ---------------------------------------------------------------------------


def load_grader(
    model_id: str = DEFAULT_GRADER_MODEL_ID,
    args: SimpleNamespace | None = None,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    suite_label = _suite_label_for(model_id)
    if suite_label is not None:
        from mmar_models import MODEL_SPECS, load_model

        spec = MODEL_SPECS[suite_label]
        ns = SimpleNamespace(
            model_id=spec["model_id"],
            tokenizer_id=spec.get("tokenizer_id"),
            local_model_dir=None,
            local_tokenizer_dir=None,
            max_num_seqs=getattr(args, "max_num_seqs", None) if args else None,
            gpu_memory_utilization=(
                getattr(args, "gpu_memory_utilization", None) if args else None
            ),
            max_model_len=getattr(args, "max_model_len", None) if args else None,
            seed=0,
        )
        loaded = load_model(suite_label, ns)
        llm = loaded["llm"]
        tokenizer = llm.get_tokenizer()
        sampling = _grade_sampling_for_engine(spec.get("engine") or {}, spec.get("sampling") or {})
        print(
            f"Freeform grader ready (suite): {suite_label} ({spec['model_id']}) "
            f"sampling={sampling}"
        )
        chat_kwargs = loaded.get("chat_kwargs") or {}
        return {
            "llm": llm,
            "tokenizer": tokenizer,
            "model_id": spec["model_id"],
            "judge_label": suite_label,
            "suite_label": suite_label,
            "backend": loaded.get("backend") or spec.get("backend"),
            "sampling_rate": int(
                loaded.get("sampling_rate") or spec.get("sampling_rate") or 16000
            ),
            "stages": loaded.get("stages"),
            "SamplingParams": SamplingParams,
            "sampling": sampling,
            "lora_request": loaded.get("lora_request"),
            "chat_kwargs": chat_kwargs,
            "chat_template_kwargs": chat_kwargs.get("chat_template_kwargs") or {},
            "batch_size": 32,
        }

    model_id = resolve_judge_model_id(model_id)
    label = judge_label(model_id)
    local_id = resolve_model_dir(model_id, None)
    engine = resolve_judge_engine(model_id, args)
    sampling = resolve_judge_sampling(model_id, args)
    sampling = _grade_sampling_for_engine(engine, sampling)

    llm_kwargs: dict[str, Any] = {"model": local_id, **engine}
    language_model_only = bool(engine.get("language_model_only"))
    try:
        llm = LLM(**llm_kwargs)
    except TypeError as exc:
        # Older vLLM builds may not accept language_model_only.
        if not language_model_only:
            raise
        print(
            f"[grader] language_model_only unsupported ({exc}); "
            "retrying without it"
        )
        llm_kwargs.pop("language_model_only", None)
        llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    dtype = engine.get("dtype", "?")
    print(
        f"Freeform grader ready: {model_id} ({local_id}) "
        f"label={label} dtype={dtype} sampling={sampling}"
        f"{' language_model_only' if language_model_only else ''}"
    )
    return {
        "llm": llm,
        "tokenizer": tokenizer,
        "model_id": model_id,
        "judge_label": label,
        "suite_label": None,
        "backend": None,
        "SamplingParams": SamplingParams,
        "sampling": sampling,
        "lora_request": None,
        "chat_kwargs": {},
        "chat_template_kwargs": {},
    }


def _format_chat(
    tokenizer: Any,
    user_text: str,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = dict(chat_template_kwargs or {})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"User: {user_text}\nAssistant:"


# ---------------------------------------------------------------------------
# Nongold audio wrapping
#
# Same recipe as GRADE_PROMPT_RECIPES[False] (free):
#   {instructions}   = recipe.instructions()   (task + 0/1 judgments)
#   {audio}          = clip or model placeholder
#   {fields}         = Question / Response + free_closer
#
# String templates below are used by vLLM / Omni generate.
# Chat-message wrapping (_nongold_audio_chat_messages) is used by vllm_chat /
# hf_chat backends (gemma-4-e4b, nemotron-3-nano-omni, API-style content).
# ---------------------------------------------------------------------------

NONGOLD_AUDIO_DEFAULT_LABEL = "qwen3-omni"

# {instructions} {fields} {system}  — {system} is filled for models that have one.
_QWEN3_NONGOLD_AUDIO_TEMPLATE = (
    "<|im_start|>user\n"
    "{instructions}\n\n"
    "<|audio_start|><|audio_pad|><|audio_end|>{fields}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

NONGOLD_AUDIO_PROMPT_TEMPLATES: dict[str, str] = {
    "qwen3-omni": _QWEN3_NONGOLD_AUDIO_TEMPLATE,
    "qwen3-omni-instruct": _QWEN3_NONGOLD_AUDIO_TEMPLATE,
    "qwen2.5-omni-7b": (
        "<|im_start|>system\n{system}\n{instructions}<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_bos|><|AUDIO|><|audio_eos|>{fields}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "phi-4-multimodal": (
        "<|user|>{instructions}<|audio_1|>{fields}<|end|><|assistant|>"
    ),
    "af-next-think": (
        "<|im_start|>system\n{system}\n{instructions}<|im_end|>\n"
        "<|im_start|>user\n"
        "<sound>{fields}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "mimo-audio-7b": (
        "<|im_start|>user\n"
        "{instructions}\n\n"
        "<|sosp|><|empty|><|eosp|>{fields}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n"
    ),
    "step-audio-2-mini": (
        "<|im_start|>system\n{instructions}<|im_end|>\n"
        "<|im_start|>user\n<audio_patch>\n"
        "{fields}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}

# Chat-message backends (not a string template). Slots match the recipe fields.
NONGOLD_AUDIO_CHAT_LAYOUT = (
    "user.content:\n"
    "  [0] text  {instructions}\n"
    "  [1] audio {audio}\n"
    "  [2] text  " + FIELD_QUESTION + "\n"
    "  [3] text  " + FIELD_PREDICTION + "\n"
    "  [4] text  {closer}"
)


def _nongold_audio_system_text(label: str) -> str:
    """Model system prompt interpolated into templates that have ``{system}``."""
    if label == "qwen2.5-omni-7b":
        from mmar_models import QWEN25_OMNI_SYSTEM

        return QWEN25_OMNI_SYSTEM
    if label == "af-next-think":
        from mmar_models import AF_NEXT_SYSTEM

        return AF_NEXT_SYSTEM
    return ""


def nongold_audio_prompt_template(label: str) -> str:
    """Return the string wrap for ``label``, falling back to Qwen3-Omni."""
    return (
        NONGOLD_AUDIO_PROMPT_TEMPLATES.get(label)
        or NONGOLD_AUDIO_PROMPT_TEMPLATES[NONGOLD_AUDIO_DEFAULT_LABEL]
    )


def resolve_grade_audio_path(audio_path: str | None) -> Path | None:
    """Resolve a prediction ``audio_path`` against MMAR data (volume or local)."""
    if not audio_path:
        return None
    path = Path(str(audio_path))
    candidates = [path]
    try:
        from modal_cache import DEFAULT_MMAR_DATA_ROOT

        data_root = Path(DEFAULT_MMAR_DATA_ROOT)
        audio_dir = data_root / "audio"
        if not path.is_absolute():
            candidates.append(data_root / path)
        candidates.append(audio_dir / path.name)
        candidates.append(data_root / "audio" / path.name)
    except Exception:
        pass
    repo_root = Path(__file__).resolve().parent
    local_data = repo_root / "data" / "mmar"
    local_audio = local_data / "audio"
    if not path.is_absolute():
        candidates.append(local_data / path)
        candidates.append(repo_root / path)
    candidates.append(local_audio / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _nongold_audio_prompt_string(
    label: str, instructions: str, fields: str, closer: str = ""
) -> str:
    """Wrap NO_GOLD inputs as: instructions, then audio placeholder, then fields + closer."""
    body = f"{fields}\n\n{closer}" if closer else fields
    return _fill_placeholders(
        nongold_audio_prompt_template(label),
        instructions=instructions,
        fields=body,
        system=_nongold_audio_system_text(label),
    )


def _nongold_audio_chat_messages(
    instructions: str,
    question: str,
    prediction: str,
    audio_path: Path,
    *,
    audio_type: str = "audio_url",
    closer: str = "",
) -> list[dict[str, Any]]:
    """User turn: instructions, then audio, question, response, closer."""
    if audio_type == "audio":
        audio_part: dict[str, Any] = {"type": "audio", "audio": str(audio_path)}
    else:
        audio_part = {
            "type": "audio_url",
            "audio_url": {"url": f"file://{audio_path}"},
        }
    content: list[dict[str, Any]] = [
        {"type": "text", "text": instructions},
        audio_part,
        {
            "type": "text",
            "text": _fill_placeholders(FIELD_QUESTION, question=question),
        },
        {
            "type": "text",
            "text": _fill_placeholders(
                FIELD_PREDICTION, prediction=prediction
            ),
        },
    ]
    if closer:
        content.append({"type": "text", "text": closer})
    return [{"role": "user", "content": content}]


def _grade_sampling(handle: dict[str, Any], *, max_tokens: int | None = None):
    from vllm import SamplingParams

    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    if handle.get("suite_label") and handle.get("sampling"):
        kwargs = dict(handle["sampling"])
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        kwargs["temperature"] = 0.0
        return SamplingParams(**kwargs)
    return judge_sampling_params(model_id, max_tokens=max_tokens)


def _completion_text(out: Any) -> str:
    outs = getattr(out, "outputs", None) or []
    if outs:
        return str(getattr(outs[0], "text", "") or "")
    return str(getattr(out, "text", "") or "")


def _grade_results_from_texts(
    jobs: list[dict],
    texts: list[str],
    *,
    model_id: str,
    prompt: str,
    include_gold: bool,
) -> list[dict]:
    results: list[dict] = []
    prompt_name = grade_prompt_name(include_gold)
    del prompt
    for job, text in zip(jobs, texts):
        verdict = parse_grade_verdict(text)
        short = format_grade_output(verdict)
        correct = bool(verdict) if verdict is not None else False
        results.append(
            {
                "correct": correct,
                "verdict": (
                    "pass" if verdict is True else "fail" if verdict is False else None
                ),
                "generation": text,
                "grader_output": short,
                "grader": model_id,
                "grader_verdict_raw": verdict,
                "question": job.get("question"),
                "answer": job.get("answer"),
                "prediction": job.get("prediction"),
                "prompt": prompt_name,
                "include_gold": bool(include_gold),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Batch grading
# ---------------------------------------------------------------------------


def _grade_shot_batch_audio(
    handle: dict[str, Any],
    jobs: list[dict],
    *,
    max_tokens: int | None = None,
    prompt: str | None = None,
) -> list[dict]:
    """NO_GOLD path: audio-capable judges hear the clip, then question + answer."""
    from mmar_models import _load_audio_tuple

    del prompt
    label = str(handle.get("suite_label") or handle.get("judge_label") or "")
    backend = str(handle.get("backend") or "vllm")
    sampling_rate = int(handle.get("sampling_rate") or 16000)
    recipe = get_grade_prompt_recipe(include_gold=False)
    instructions = recipe.instructions()
    closer = recipe.closer()
    sampling = _grade_sampling(handle, max_tokens=max_tokens)
    generate_kwargs: dict[str, Any] = {}
    if handle.get("lora_request") is not None:
        generate_kwargs["lora_request"] = handle["lora_request"]

    resolved: list[tuple[dict, Path]] = []
    for job in jobs:
        audio_path = resolve_grade_audio_path(str(job.get("audio_path") or "") or None)
        if audio_path is None:
            raise SystemExit(
                "NO_GOLD audio grading needs a readable wav for each question; "
                f"missing audio_path for id={job.get('id')!r} "
                f"path={job.get('audio_path')!r}"
            )
        resolved.append((job, audio_path))

    texts: list[str] = []
    if backend == "vllm_chat":
        chat_kwargs = dict(handle.get("chat_kwargs") or {})
        messages = [
            _nongold_audio_chat_messages(
                instructions,
                str(job.get("question") or ""),
                str(job.get("prediction") or ""),
                audio_path,
                closer=closer,
            )
            for job, audio_path in resolved
        ]
        if label in {"gemma-4-e4b", "nemotron-3-nano-omni"}:
            outputs = []
            for msgs in messages:
                outputs.extend(
                    handle["llm"].chat(
                        [msgs], sampling_params=sampling, **chat_kwargs
                    )
                )
        else:
            outputs = handle["llm"].chat(
                messages, sampling_params=sampling, **chat_kwargs
            )
        texts = [_completion_text(out) for out in outputs]
    elif backend in {"hf_chat", "vllm_transformers"}:
        messages = [
            _nongold_audio_chat_messages(
                instructions,
                str(job.get("question") or ""),
                str(job.get("prediction") or ""),
                audio_path,
                audio_type="audio",
                closer=closer,
            )
            for job, audio_path in resolved
        ]
        chat_kwargs = dict(handle.get("chat_kwargs") or {})
        outputs = handle["llm"].chat(
            messages, sampling_params=sampling, **chat_kwargs
        )
        texts = [_completion_text(out) for out in outputs]
    elif backend == "vllm_voxtral":
        from mistral_common.protocol.instruct.chunk import AudioChunk, TextChunk
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.tokens.tokenizers.audio import Audio

        tokenizer = handle["tokenizer"]
        prompts = []
        for job, audio_path in resolved:
            fields = recipe.fields_with_closer(
                question=str(job.get("question") or ""),
                prediction=str(job.get("prediction") or ""),
            )
            audio = Audio.from_file(str(audio_path), strict=False)
            messages = [
                UserMessage(
                    content=[
                        TextChunk(text=instructions),
                        AudioChunk.from_audio(audio),
                        TextChunk(text=fields),
                    ]
                ).to_openai()
            ]
            prompts.append(
                {
                    "prompt_token_ids": tokenizer.apply_chat_template(
                        messages=messages
                    ),
                    "multi_modal_data": {
                        "audio": [(audio.audio_array, audio.sampling_rate)]
                    },
                }
            )
        outputs = handle["llm"].generate(
            prompts, sampling_params=sampling, **generate_kwargs
        )
        texts = [_completion_text(out) for out in outputs]
    elif backend == "vllm_omni":
        from mmar_models import _omni_generate_texts

        prompts = []
        for job, audio_path in resolved:
            fields = recipe.fields(
                question=str(job.get("question") or ""),
                prediction=str(job.get("prediction") or ""),
            )
            audio = _load_audio_tuple(str(audio_path), sampling_rate)
            item = {
                "prompt": _nongold_audio_prompt_string(
                    label, instructions, fields, closer
                ),
                "multi_modal_data": {"audio": audio},
                "modalities": ["text"],
            }
            prompts.append(item)
        texts = _omni_generate_texts(
            handle["llm"],
            prompts,
            [sampling] * len(prompts),
            stages=int(handle.get("stages") or 1),
            n_shots=1,
            debug_label=label,
        )
    else:
        # vllm / vllm_omni: audio placeholder in the prompt string.
        prompts = []
        for job, audio_path in resolved:
            fields = recipe.fields(
                question=str(job.get("question") or ""),
                prediction=str(job.get("prediction") or ""),
            )
            audio = _load_audio_tuple(str(audio_path), sampling_rate)
            item: dict[str, Any] = {
                "prompt": _nongold_audio_prompt_string(
                    label, instructions, fields, closer
                ),
                "multi_modal_data": {"audio": audio},
            }
            if label in {"mimo-audio-7b", "step-audio-2-mini"}:
                item["modalities"] = ["text"]
            prompts.append(item)
        outputs = handle["llm"].generate(
            prompts, sampling_params=sampling, **generate_kwargs
        )
        texts = [_completion_text(out) for out in outputs]

    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    return _grade_results_from_texts(
        jobs,
        texts,
        model_id=model_id,
        prompt=grade_prompt_name(False),
        include_gold=False,
    )


def grade_shot_batch(
    handle: dict[str, Any],
    jobs: list[dict],
    *,
    max_tokens: int | None = None,
    prompt: str = DEFAULT_GRADE_PROMPT,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> list[dict]:
    """Grade a list of ``{question, answer, prediction}`` jobs.

    NO_GOLD jobs must use an audio judge and include ``audio_path``. After the
    grade prompt, inputs are audio, then question, then the response, then
    the 0/1 closer.

    Returns one result dict per job with ``correct``, ``verdict``,
    ``generation`` (full text), ``grader_output`` (short 0/1), and
    ``grader``.
    """
    if not jobs:
        return []
    include_gold = bool(include_gold)
    require_audio_nongold_judge(handle=handle, include_gold=include_gold)
    if not include_gold:
        return _grade_shot_batch_audio(
            handle, jobs, max_tokens=max_tokens, prompt=prompt
        )

    tokenizer = handle["tokenizer"]
    chat_template_kwargs = handle.get("chat_template_kwargs") or {}
    prompts = [
        _format_chat(
            tokenizer,
            build_grade_prompt(
                question=str(job.get("question") or ""),
                answer=str(job.get("answer") or ""),
                prediction=str(job.get("prediction") or ""),
                prompt=prompt,
                include_gold=True,
            ),
            chat_template_kwargs=chat_template_kwargs,
        )
        for job in jobs
    ]
    sampling = _grade_sampling(handle, max_tokens=max_tokens)
    generate_kwargs: dict[str, Any] = {}
    if handle.get("lora_request") is not None:
        generate_kwargs["lora_request"] = handle["lora_request"]
    outputs = handle["llm"].generate(
        prompts, sampling_params=sampling, **generate_kwargs
    )
    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    return _grade_results_from_texts(
        jobs,
        [_completion_text(out) for out in outputs],
        model_id=model_id,
        prompt=prompt,
        include_gold=True,
    )


# ---------------------------------------------------------------------------
# Predictions.jsonl I/O
# ---------------------------------------------------------------------------


def grade_predictions_file(
    predictions_path: Path,
    handle: dict[str, Any],
    *,
    judge_key: str | None = None,
    primary_judge: str | None = None,
    batch_size: int | None = None,
    force: bool = False,
    prompt: str = DEFAULT_GRADE_PROMPT,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
    shot_indices: tuple[int, ...] | list[int] | None = None,
    make_primary: bool = False,
    sidecar_path: Path | None = None,
    n_questions: int | None = None,
) -> dict[str, Any]:
    """Grade shots in a predictions.jsonl file for one judge configuration.

    When ``force`` is True, existing entries for ``judge_key`` are replaced.
    When ``sidecar_path`` is set, verdicts are written there instead of
    rewriting ``predictions.jsonl`` (safe for concurrent round-robin judges).
    ``make_primary`` False keeps each record's existing ``primary_judge``.
    ``n_questions`` grades a prefix of the fixed shuffled id list (see
    ``mmar_common.GRADE_SAMPLE_SEED``); None or < 0 grades every question.
    Duplicate model answers (lowercase, stripped) reuse an existing verdict
    for the same question and gold, including shots already graded for this
    judge when ``force`` is False.
    """
    if not predictions_path.exists():
        return {
            "status": "missing",
            "predictions_path": str(predictions_path),
            "n_records": 0,
            "n_shots_graded": 0,
        }

    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    include_gold = bool(include_gold)
    prompt_name = grade_prompt_name(include_gold)
    del prompt
    require_audio_nongold_judge(handle=handle, include_gold=include_gold)
    key = resolve_grade_judge_key(
        handle, prompt=prompt_name, include_gold=include_gold, judge_key=judge_key
    )
    allowed = {int(i) for i in shot_indices} if shot_indices is not None else None
    if batch_size is not None:
        effective_batch_size = int(batch_size)
    elif handle.get("batch_size"):
        effective_batch_size = int(handle["batch_size"])
    else:
        effective_batch_size = resolve_judge_batch_size(model_id, None)

    records: list[dict] = []
    with open(predictions_path, encoding="utf-8") as handle_in:
        for line in handle_in:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    for record in records:
        ensure_judge_schema(
            record,
            fallback_label=key,
            fallback_model_id=model_id,
        )

    selected_ids = select_grade_question_ids(
        [str(record.get("id") or "") for record in records],
        n_questions,
    )
    allowed_ids = set(selected_ids) if selected_ids is not None else None

    def _in_sample(record: dict) -> bool:
        if allowed_ids is None:
            return True
        return str(record.get("id") or "").strip() in allowed_ids

    reuse_cache: dict[tuple[str, str, str], dict] = {}
    if not force:
        for record in records:
            if not _in_sample(record):
                continue
            question = str(record.get("question") or "")
            answer = str(record.get("answer") or "")
            for shot in record.get("shots") or []:
                shot_index = int(shot.get("shot_index", 0))
                if allowed is not None and shot_index not in allowed:
                    continue
                entry = _shot_judge_entry(shot, key)
                if entry is None:
                    continue
                reuse_cache.setdefault(
                    _grade_reuse_key(
                        question,
                        answer,
                        _shot_prediction_text(shot),
                        include_gold=include_gold,
                    ),
                    dict(entry),
                )

    jobs: list[dict] = []
    owners: list[list[tuple[int, int]]] = []
    job_keys: list[tuple[str, str, str]] = []
    pending_index: dict[tuple[str, str, str], int] = {}
    reuse_owners: list[tuple[int, int, dict]] = []
    for record_index, record in enumerate(records):
        if not _in_sample(record):
            continue
        if not force and not _record_needs_grade(
            record, key, shot_indices=shot_indices
        ):
            continue
        question = str(record.get("question") or "")
        answer = str(record.get("answer") or "")
        for shot in record.get("shots") or []:
            shot_index = int(shot.get("shot_index", 0))
            if allowed is not None and shot_index not in allowed:
                continue
            if not force and not _shot_needs_grade(shot, key):
                continue
            prediction = _shot_prediction_text(shot)
            cache_key = _grade_reuse_key(
                question, answer, prediction, include_gold=include_gold
            )
            cached = reuse_cache.get(cache_key)
            if cached is not None:
                reuse_owners.append((record_index, shot_index, cached))
                continue
            idx = pending_index.get(cache_key)
            if idx is None:
                pending_index[cache_key] = len(jobs)
                jobs.append(
                    {
                        "id": record.get("id"),
                        "question": question,
                        "answer": answer,
                        "prediction": prediction,
                        "audio_path": record.get("audio_path"),
                    }
                )
                owners.append([(record_index, shot_index)])
                job_keys.append(cache_key)
            else:
                owners[idx].append((record_index, shot_index))

    graded = 0
    reused = 0
    partials: list[dict] = []

    def _apply_entry(record_index: int, shot_index: int, entry: dict) -> None:
        nonlocal graded
        record = records[record_index]
        copied = dict(entry)
        if sidecar_path is not None:
            partials.append(
                {
                    "id": record.get("id"),
                    "shot_index": shot_index,
                    "judge_key": key,
                    "entry": copied,
                }
            )
            graded += 1
            return
        for shot in record.get("shots") or []:
            if int(shot.get("shot_index", -1)) != shot_index:
                continue
            shot.setdefault("judges", {})[key] = copied
            graded += 1
            break

    for record_index, shot_index, entry in reuse_owners:
        _apply_entry(record_index, shot_index, entry)
        reused += 1

    for start in range(0, len(jobs), effective_batch_size):
        chunk = jobs[start : start + effective_batch_size]
        chunk_owners = owners[start : start + effective_batch_size]
        chunk_keys = job_keys[start : start + effective_batch_size]
        results = grade_shot_batch(
            handle,
            chunk,
            prompt=prompt_name,
            include_gold=include_gold,
        )
        for cache_key, owner_list, result in zip(chunk_keys, chunk_owners, results):
            entry = {
                "correct": bool(result["correct"]),
                "verdict": result.get("verdict"),
                "output": result.get("grader_output"),
                "generation": result.get("generation") or "",
                "model_id": model_id,
                "prompt": prompt_name,
                "include_gold": include_gold,
            }
            reuse_cache[cache_key] = entry
            for extra_index, (record_index, shot_index) in enumerate(owner_list):
                _apply_entry(record_index, shot_index, entry)
                if extra_index:
                    reused += 1

    if sidecar_path is not None:
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(sidecar_path, partials, mode="w")
        return {
            "status": "ok",
            "predictions_path": str(predictions_path),
            "sidecar_path": str(sidecar_path),
            "n_records": len(records),
            "n_sampled": (
                len(allowed_ids) if allowed_ids is not None else len(records)
            ),
            "n_questions": n_questions,
            "n_shots_graded": graded,
            "n_shots_reused": reused,
            "grader": model_id,
            "judge_label": key,
            "prompt": prompt_name,
            "include_gold": include_gold,
            "replaced": bool(force),
        }

    for record in records:
        if allowed_ids is not None and str(record.get("id") or "").strip() not in allowed_ids:
            continue
        existing_primary = record.get("primary_judge")
        if make_primary:
            use_primary = key
        else:
            use_primary = existing_primary or primary_judge
        existing = [str(x) for x in (record.get("judges") or []) if x]
        ordered: list[str] = []
        if use_primary:
            ordered.append(str(use_primary))
        for label in existing:
            if label not in ordered:
                ordered.append(label)
        if key not in ordered:
            ordered.append(key)
        record["judges"] = ordered
        record["scoring"] = "qwen_freeform_judge"
        recompute_multi_judge_scores(record, use_primary)

    write_jsonl(predictions_path, records, mode="w")
    return {
        "status": "ok",
        "predictions_path": str(predictions_path),
        "n_records": len(records),
        "n_sampled": (
            len(allowed_ids) if allowed_ids is not None else len(records)
        ),
        "n_questions": n_questions,
        "n_shots_graded": graded,
        "n_shots_reused": reused,
        "grader": model_id,
        "judge_label": key,
        "primary_judge": primary_judge,
        "prompt": prompt_name,
        "include_gold": include_gold,
        "replaced": bool(force),
    }


def apply_judge_partials(
    predictions_path: Path,
    partial_paths: list[Path],
    *,
    make_primary: bool = False,
    primary_judge: str | None = None,
) -> dict[str, Any]:
    """Merge sidecar judge verdicts into ``predictions.jsonl``."""
    if not predictions_path.exists():
        return {
            "status": "missing",
            "predictions_path": str(predictions_path),
            "n_applied": 0,
            "judge_keys": [],
        }
    records: list[dict] = []
    by_id: dict[str, dict] = {}
    with open(predictions_path, encoding="utf-8") as handle_in:
        for line in handle_in:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            records.append(record)
            rid = str(record.get("id") or "")
            if rid:
                by_id[rid] = record

    applied = 0
    keys: list[str] = []
    for path in partial_paths:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as handle_in:
            for line in handle_in:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                key = str(row.get("judge_key") or "")
                entry = row.get("entry") or {}
                if not key or not isinstance(entry, dict):
                    continue
                if key not in keys:
                    keys.append(key)
                record = by_id.get(str(row.get("id") or ""))
                if record is None:
                    continue
                shot_index = int(row.get("shot_index", 0))
                ensure_judge_schema(record, fallback_label=key)
                for shot in record.get("shots") or []:
                    if int(shot.get("shot_index", -1)) != shot_index:
                        continue
                    shot.setdefault("judges", {})[key] = entry
                    applied += 1
                    break

    for record in records:
        existing_primary = record.get("primary_judge")
        if make_primary and keys:
            use_primary = keys[0] if primary_judge is None else primary_judge
        else:
            use_primary = existing_primary or primary_judge
        existing = [str(x) for x in (record.get("judges") or []) if x]
        ordered: list[str] = []
        if use_primary:
            ordered.append(str(use_primary))
        for label in existing:
            if label not in ordered:
                ordered.append(label)
        for key in keys:
            if key not in ordered:
                ordered.append(key)
        record["judges"] = ordered
        record["scoring"] = record.get("scoring") or "qwen_freeform_judge"
        recompute_multi_judge_scores(record, use_primary)

    write_jsonl(predictions_path, records, mode="w")
    return {
        "status": "ok",
        "predictions_path": str(predictions_path),
        "n_applied": applied,
        "judge_keys": keys,
    }


# ---------------------------------------------------------------------------
# Inspection: python grader.py
# ---------------------------------------------------------------------------

_BANNER = "=" * 78
_RULE = "-" * 78


def validate_grade_prompt_recipes() -> None:
    """Raise if the recipe table is missing a combo or names an unknown piece."""
    missing = [flag for flag in (True, False) if flag not in GRADE_PROMPT_RECIPES]
    if missing:
        raise RuntimeError(f"Missing grade-prompt recipes for include_gold={missing}")
    for recipe in GRADE_PROMPT_RECIPES.values():
        expected = grade_prompt_name(recipe.include_gold)
        if recipe.name != expected:
            raise RuntimeError(
                f"Recipe name {recipe.name!r} does not match include_gold="
                f"{recipe.include_gold} (expected {expected!r})"
            )
        for name in (*recipe.before_audio, *recipe.after_fields):
            if name not in JUDGE_PIECES:
                raise RuntimeError(
                    f"Unknown piece {name!r} in {recipe.name}/{gold_tag(recipe.include_gold)}"
                )


def _audio_template_groups() -> list[tuple[tuple[str, ...], str]]:
    """Group suite labels that share a nongold string template."""
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for label, template in NONGOLD_AUDIO_PROMPT_TEMPLATES.items():
        if template not in groups:
            groups[template] = []
            order.append(template)
        groups[template].append(label)
    return [(tuple(groups[template]), template) for template in order]


def format_grade_prompt_inspection(
    *,
    include_gold: bool | None = None,
    audio_wraps: bool = True,
) -> str:
    """Render gold (with_gt) and/or free (nongold) prompts with placeholder inputs."""
    validate_grade_prompt_recipes()
    gold_flags = (
        (True, False) if include_gold is None else (bool(include_gold),)
    )
    selected = [get_grade_prompt_recipe(include_gold=gold) for gold in gold_flags]
    chunks: list[str] = []

    chunks.append(_BANNER)
    chunks.append("JUDGE PIECES (named fragments; recipes below only cite these keys)")
    chunks.append(_BANNER)
    for name, text in JUDGE_PIECES.items():
        chunks.append(f"[{name}]")
        chunks.append(text)
        chunks.append("")

    chunks.append(_BANNER)
    chunks.append("FIELD TEMPLATES (variable slots)")
    chunks.append(_BANNER)
    chunks.append(FIELD_QUESTION)
    chunks.append(FIELD_GOLD)
    chunks.append(FIELD_PREDICTION)
    chunks.append("")

    n = len(selected)
    for index, recipe in enumerate(selected, start=1):
        gold = gold_tag(recipe.include_gold)
        chunks.append(_BANNER)
        chunks.append(
            f"{index}/{n}  name={recipe.name}  include_gold={recipe.include_gold}  ({gold})"
        )
        chunks.append(_BANNER)
        chunks.append("assembly (top → bottom):")
        chunks.append(f"  before_audio : {' + '.join(recipe.before_audio) or '(empty)'}")
        if recipe.include_gold:
            chunks.append("  fields       : Question + Ground truth + Response")
            chunks.append(f"  after_fields : {' + '.join(recipe.after_fields) or '(none)'}")
        else:
            chunks.append("  [AUDIO]      : clip inserted here for suite / API audio judges")
            chunks.append("  fields       : Question + Response")
            chunks.append(f"  after_fields : {' + '.join(recipe.after_fields) or '(none)'}")
        chunks.append("")
        chunks.append("rendered text prompt (API judges and gold vLLM):")
        chunks.append(_RULE)
        chunks.append(recipe.as_text(**PROMPT_PLACEHOLDERS))
        chunks.append(_RULE)
        chunks.append("")

    if audio_wraps and any(not recipe.include_gold for recipe in selected):
        nongold = [recipe for recipe in selected if not recipe.include_gold]
        chunks.append(_BANNER)
        chunks.append("NONGOLD AUDIO WRAPS")
        chunks.append("Each wrap is: {instructions} + audio + {fields} + {closer}")
        chunks.append("Unknown suite labels fall back to the qwen3-omni template.")
        chunks.append(_BANNER)
        chunks.append("")
        chunks.append("chat-message backends (vllm_chat / hf_chat / gemma / nemotron):")
        chunks.append(_RULE)
        chunks.append(
            _fill_placeholders(
                NONGOLD_AUDIO_CHAT_LAYOUT,
                instructions="{instructions}",
                audio="{audio}",
                closer="{closer}",
                **PROMPT_PLACEHOLDERS,
            )
        )
        chunks.append(_RULE)
        chunks.append("")

        for labels, template in _audio_template_groups():
            label_list = ", ".join(labels)
            chunks.append(_BANNER)
            chunks.append(f"string wrap: {label_list}")
            chunks.append(_BANNER)
            chunks.append("template:")
            chunks.append(template)
            chunks.append("")
            sample_label = labels[0]
            for recipe in nongold:
                wrapped = _nongold_audio_prompt_string(
                    sample_label,
                    recipe.instructions(),
                    recipe.fields(**PROMPT_PLACEHOLDERS),
                    recipe.closer(),
                )
                chunks.append(
                    f"----- wrapped  name={recipe.name}  nongold -----"
                )
                chunks.append(wrapped)
                chunks.append("")

    return "\n".join(chunks).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Render judge prompts with {question}/{answer}/{prediction} placeholders "
            "so the with-gt (gold) and free (nongold) recipes can be inspected."
        )
    )
    gold = parser.add_mutually_exclusive_group()
    gold.add_argument(
        "--gold",
        dest="include_gold",
        action="store_true",
        default=None,
        help="Only render the with-gt recipe that includes the benchmark answer.",
    )
    gold.add_argument(
        "--nongold",
        dest="include_gold",
        action="store_false",
        help="Only render the free recipe that omits gold (audio judges).",
    )
    parser.add_argument(
        "--no-audio-wraps",
        action="store_true",
        help="Skip model-specific nongold audio wrappers.",
    )
    args = parser.parse_args(argv)
    print(
        format_grade_prompt_inspection(
            include_gold=args.include_gold,
            audio_wraps=not args.no_audio_wraps,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
