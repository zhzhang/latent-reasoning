"""Free-form answer grader for MMAR difficulty experiments (multi-judge).

Judge prompt assembly is table-driven (``JUDGE_FORMATS``). Inspect gold
(with-gt) and nongold (free) prompts, with variable slots left as
``{question}`` / ``{answer}`` / ``{prediction}``::

    python grader.py

Per-judge wraps (the text ``grade_shot_batch`` would send) live in
``render_judge_prompt``; dump them with ``python render_judge_prompts.py``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from audio_flamingo_runtime import resolve_model_dir
from mmar_common import (
    ASSISTANT_THINK_OPEN,
    MUSIC_FLAMINGO_THINK_SUFFIX,
    after_last_think_close,
    ensure_assistant_think_open,
    ensure_judge_schema,
    extract_freeform_answer,
    join_vllm_reasoning,
    judge_label,
    parse_freeform_output,
    recompute_multi_judge_scores,
    select_grade_question_ids,
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
        # Chat template skips CoT only when enable_thinking is false.
        "enable_thinking": True,
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

# Named formats live in ``JUDGE_FORMATS`` below. Default runs still use
# with_gt / free via include_gold; ``--grade-prompt`` selects any table
# key (comma-separated, or ``all``).
DEFAULT_GRADE_PROMPT = "with_gt"
DEFAULT_INCLUDE_GOLD = True
GRADE_PROMPT_ALIASES = {
    "permissive": DEFAULT_GRADE_PROMPT,
    "neutral": "neutral_with_gt",
}
ACCURACY_META_KEYS = frozenset(
    {
        "pack",
        "labels_path",
        "n_label_rows",
        "n_questions",
        "epsilon",
        "modes",
        "by_category",
        "by_modality",
    }
)

# Prompt / closer text lives in JUDGE_FORMATS below. Run ``python grader.py``
# to print every rendered combination with ``{question}`` / ``{answer}`` /
# ``{prediction}`` placeholders.


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
    n: int = 1,
    temperature: float | None = None,
):
    """Build a vLLM SamplingParams instance for one judge.

    ``n > 1`` forks ``n`` completions per prompt so they share prefill
    (same as test-taker ``SamplingParams(n=...)``). Temperature 0 would
    make those copies identical, so n>1 bumps T to 1.0 unless an explicit
    positive ``temperature`` is passed.
    """
    from vllm import SamplingParams

    kwargs = resolve_judge_sampling(model_id, args)
    if temperature is not None:
        t = float(temperature)
    else:
        t = float(kwargs.get("temperature", 0.0))
    kwargs["temperature"] = t if t > 0 else 0.0
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    n = max(1, int(n))
    if n > 1:
        kwargs["n"] = n
        if kwargs["temperature"] <= 0:
            kwargs["temperature"] = 1.0
    return SamplingParams(**kwargs)


def _looks_fp8(model_id: str) -> bool:
    text = model_id.lower()
    return "fp8" in text or text.endswith("-fp8")


def _needs_language_model_only(model_id: str) -> bool:
    """Qwen3.5/3.6 multimodal checkpoints: skip the vision tower for text judging."""
    text = model_id.lower()
    return any(token in text for token in ("qwen3.5", "qwen3.6", "qwen3_5", "qwen3_6"))


def grade_prompt_names() -> tuple[str, ...]:
    """Insertion-order keys of ``JUDGE_FORMATS``."""
    return tuple(JUDGE_FORMATS)


def grade_mode_title(name: str) -> str:
    """Human label derived from a format's gold / audio flags."""
    fmt = JUDGE_FORMATS.get(name)
    if fmt is None:
        return str(name or "")
    gold = "sees gold" if fmt.include_gold else "no gold"
    audio = "hears audio" if fmt.audio_included else "text"
    return f"{name} ({audio}, {gold})"


def grade_mode_titles() -> dict[str, str]:
    return {name: grade_mode_title(name) for name in JUDGE_FORMATS}


def grade_prompt_name(include_gold: bool = DEFAULT_INCLUDE_GOLD) -> str:
    """Default boolean gold flag → ``JUDGE_FORMATS`` key (``with_gt`` / ``free``)."""
    preferred = "with_gt" if include_gold else "free"
    if preferred in JUDGE_FORMATS:
        return preferred
    matches = [
        name
        for name, fmt in JUDGE_FORMATS.items()
        if fmt.include_gold is bool(include_gold)
    ]
    if matches:
        return matches[0]
    raise ValueError(
        f"No JUDGE_FORMATS entry with include_gold={bool(include_gold)}"
    )


def normalize_grade_prompt(
    name: str | None,
    *,
    include_gold: bool | None = None,
) -> str:
    """Return a ``JUDGE_FORMATS`` key. Explicit names win over ``include_gold``."""
    value = str(name or "").strip().lower()
    value = GRADE_PROMPT_ALIASES.get(value, value)
    if value in JUDGE_FORMATS:
        return value
    if value:
        raise ValueError(
            f"Unknown grade prompt {name!r}; expected one of {grade_prompt_names()}"
        )
    if include_gold is not None:
        return grade_prompt_name(include_gold)
    return DEFAULT_GRADE_PROMPT


def gold_mode_flags(include_gold: bool | None) -> list[bool]:
    """``None`` means both with-GT then no-GT; otherwise a single mode."""
    if include_gold is None:
        return [True, False]
    return [bool(include_gold)]


def parse_grade_prompt_list(
    value: str | None = None,
    *,
    include_gold: bool | None = DEFAULT_INCLUDE_GOLD,
) -> list[str]:
    """Return prompt names. An explicit list wins over ``include_gold``.

    ``all`` / ``*`` expands to every key in ``JUDGE_FORMATS``.
    """
    if value and str(value).strip():
        names: list[str] = []
        for raw in str(value).split(","):
            item = raw.strip()
            if not item:
                continue
            if item.lower() in {"all", "*"}:
                return list(grade_prompt_names())
            names.append(normalize_grade_prompt(item))
        if names:
            return names
    return [grade_prompt_name(flag) for flag in gold_mode_flags(include_gold)]


def iter_grade_modes(
    prompt: str | None = None,
    include_gold: bool | None = None,
) -> list[tuple[str, bool]]:
    """``(prompt_name, include_gold)`` pairs for one judging run."""
    names = parse_grade_prompt_list(prompt, include_gold=include_gold)
    return [(name, JUDGE_FORMATS[name].include_gold) for name in names]


def resolve_grade_allowed_ids(
    record_ids: list[str] | tuple[str, ...],
    *,
    question_ids: list[str] | tuple[str, ...] | None = None,
    n_questions: int | None = None,
) -> set[str] | None:
    """Question ids to grade, or ``None`` to grade every record.

    ``question_ids`` restricts to that set. ``n_questions`` then takes a prefix
    of the fixed shuffle of the remaining ids.
    """
    wanted: set[str] | None = None
    if question_ids is not None:
        wanted = {str(qid).strip() for qid in question_ids if str(qid).strip()}
    pool = [
        str(qid).strip()
        for qid in record_ids
        if str(qid).strip() and (wanted is None or str(qid).strip() in wanted)
    ]
    selected = select_grade_question_ids(pool, n_questions)
    if selected is not None:
        return set(selected)
    return wanted


def gold_tag(include_gold: bool) -> str:
    return "gold" if include_gold else "nongold"


def compose_judge_key(
    model_label: str,
    *,
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> str:
    """Stable key: ``{label}__{prompt}__{gold|nongold}``."""
    label = str(model_label or "").strip()
    if not label:
        raise ValueError("compose_judge_key requires a model_label")
    prompt_name = normalize_grade_prompt(prompt, include_gold=include_gold)
    fmt = JUDGE_FORMATS[prompt_name]
    return f"{label}__{prompt_name}__{gold_tag(fmt.include_gold)}"


def resolve_grade_judge_key(
    handle: dict[str, Any],
    *,
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
    judge_key: str | None = None,
) -> str:
    """Composite key ``{label}__{prompt}__{gold|nongold}``.

    Always composed (never the bare label) so new with-gt / free / named
    verdicts do not collide with older permissive/neutral grades.
    """
    if judge_key:
        return str(judge_key)
    label = str(handle.get("judge_label") or judge_label(handle.get("model_id")) or GRADER_LABEL)
    return compose_judge_key(label, prompt=prompt, include_gold=include_gold)


def parse_judge_key(key: str) -> dict[str, str]:
    """Split ``{label}__{prompt}__{gold|nongold}`` (prompt may contain ``_``)."""
    parts = [p for p in str(key or "").split("__") if p]
    if len(parts) >= 3:
        gold_tag = parts[-1]
        prompt = parts[-2]
        model = "__".join(parts[:-2])
        return {
            "label": str(key),
            "model": model,
            "prompt": prompt,
            "gold_tag": gold_tag,
        }
    return {"label": str(key or ""), "model": str(key or ""), "prompt": "", "gold_tag": ""}


def judge_mode_bucket(judge_key: str, entry: dict | None = None) -> str | None:
    """Map a verdict to a ``JUDGE_FORMATS`` key.

    Prefer the stored ``prompt`` / key slot so named gold recipes are not
    folded into ``with_gt`` just because gold is shown.
    """
    prompt = ""
    if isinstance(entry, dict):
        prompt = str(entry.get("prompt") or "").strip().lower()
        prompt = GRADE_PROMPT_ALIASES.get(prompt, prompt)
        if prompt in JUDGE_FORMATS:
            return prompt
    parsed = parse_judge_key(judge_key)
    slot = GRADE_PROMPT_ALIASES.get(parsed["prompt"].lower(), parsed["prompt"].lower())
    if slot in JUDGE_FORMATS:
        return slot
    if isinstance(entry, dict):
        if entry.get("include_gold") is True:
            return "with_gt"
        if entry.get("include_gold") is False:
            return "free"
    gold_tag = parsed["gold_tag"].lower()
    if gold_tag == "nongold":
        return "free"
    if gold_tag == "gold":
        return "with_gt"
    key = str(judge_key or "")
    if key.endswith("__nongold"):
        return "free"
    if key.endswith("__gold"):
        return "with_gt"
    return None


def accuracy_mode_names(payload: dict | None = None) -> list[str]:
    """Ordered accuracy buckets: ``JUDGE_FORMATS`` keys, then any extra tables."""
    names = list(grade_prompt_names())
    if not isinstance(payload, dict):
        return names
    for key, value in payload.items():
        if key in ACCURACY_META_KEYS or key in names:
            continue
        if isinstance(value, dict):
            names.append(str(key))
    return names


def parse_shot_indices(first_shot_only: bool) -> tuple[int, ...] | None:
    return (0,) if first_shot_only else None


# ===========================================================================
# Judge prompt formats
#
# Named formats live in ``JUDGE_FORMATS`` below. Adding a key there is
# enough for CLIs, grading, accuracy, and the viewer. Gold is inferred
# from ``FIELD_GOLD`` in ``field_templates``; audio vs text from
# ``audio_included``.
#
# Adapted from nikhilchandak/answer-matching ``gpqa_judge.py``
# (``get_judge_prompt_with_gt`` / ``get_free_judge_prompt``) for audio:
# the free judge is told it will hear the clip; with_gt is told the
# question is about an audio clip. Output is 0/1 in <answer> tags.
#
# Each format is two segments plus filled fields:
#
#   prompt           preamble sent first; on free the clip is inserted after
#   field_templates  filled with {question} / {answer} / {prediction}
#   closer           after the fields (match-only on with_gt; rules + 0/1
#                    tags on free). Audio wraps append this after {fields}.
#
# Gold / with_gt:
#   prompt
#   Question / Ground truth / Response
#   closer
#
# Nongold / free:
#   prompt
#   [AUDIO]
#   Question / Response
#   closer
#
# ``build_grade_prompt`` is the full text (API judges and gold vLLM).
# Audio suite judges take ``prompt`` + ``fields`` from the format and
# wrap them with ``NONGOLD_AUDIO_PROMPT_TEMPLATES`` / chat messages.
# Inspect every combo:  python grader.py
# ===========================================================================

# Placeholders filled by ``JudgeFormat.fields``.
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


@dataclass(frozen=True)
class JudgeFormat:
    """One named judge format: prompt and optional closer segments."""

    prompt: str
    closer: str = ""
    audio_included: bool = False
    field_templates: tuple[str, ...] = ()

    @property
    def include_gold(self) -> bool:
        """True when the ground-truth answer is in the field block."""
        return FIELD_GOLD in self.field_templates

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

    def fields_with_closer(
        self,
        *,
        question: str,
        answer: str = "",
        prediction: str,
    ) -> str:
        """Fields plus closer — used after the clip on the nongold audio path."""
        text = self.fields(question=question, answer=answer, prediction=prediction)
        if self.closer:
            return f"{text}\n\n{self.closer}"
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
            self.prompt,
            self.fields(question=question, answer=answer, prediction=prediction),
        ]
        if self.closer:
            parts.append(self.closer)
        return "\n\n".join(parts)


# Named formats. Edit this table — not the builders.
JUDGE_FORMATS: dict[str, JudgeFormat] = {
    "with_gt": JudgeFormat(
        prompt=(
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
        closer=(
            "Your job is to ONLY check whether the given response matches the ground "
            "truth answer or not in the context of the question. You DO NOT NEED to "
            "assess the correctness of the response. This is part of an automated "
            'evaluation process, therefore you MUST OUTPUT your final answer as "0" '
            'or "1" in <answer> </answer> tags.\n'
            "Think step by step and end your response with <answer>0</answer> OR "
            "<answer>1</answer> TAGS."
        ),
        audio_included=False,
        field_templates=(FIELD_QUESTION, FIELD_GOLD, FIELD_PREDICTION),
    ),
    "free": JudgeFormat(
        prompt=(
            "Your task is to judge whether the given response to an audio question "
            "is correct or not. You are given an audio clip, a question about that "
            "clip, and the response you are judging.\n"
            "Possible judgments:\n"
            '"0": The response is incorrect. \n'
            '"1": The response is correct.'
        ),
        closer=(
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
        audio_included=True,
        field_templates=(FIELD_QUESTION, FIELD_PREDICTION),
    ),
    "neutral_with_gt": JudgeFormat(
        prompt=(
            "Your task is to judge whether the given response to an audio question "
            "is correct or not. You are given an audio clip, a question about that "
            "clip, the ground truth answer, and the response you are judging.\n"
            "Reason briefly, then give your judgement of the response in a single "
            "final line with one word: \"Correct\" or \"Incorrect\""
        ),
        audio_included=True,
        field_templates=(FIELD_QUESTION, FIELD_GOLD, FIELD_PREDICTION),
    ),
    "neutral_with_gt_no_audio": JudgeFormat(
        prompt=(
            "Your task is to judge whether the given response to an audio question "
            "is correct or not. You are given an audio clip, a question about that "
            "clip, the ground truth answer, and the response you are judging.\n"
            "Reason briefly, then give your judgement of the response in a single "
            "final line with one word: \"Correct\" or \"Incorrect\""
        ),
        field_templates=(FIELD_QUESTION, FIELD_GOLD, FIELD_PREDICTION),
    ),
    "neutral_no_gt": JudgeFormat(
        prompt=(
            "Your task is to judge whether the given response to an audio question "
            "is correct or not. You are given an audio clip, a question about that "
            "clip, and the response you are judging.\n"
            "Reason briefly, then give a concise, answer in a single final line in "
            "exactly this format:\n"
            "Answer: <Correct or Incorrect>"
        ),
        audio_included=True,
        field_templates=(FIELD_QUESTION, FIELD_PREDICTION),
    ),
}


def get_judge_format(
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> JudgeFormat:
    """Look up a format by name; ``include_gold`` selects with_gt vs free."""
    name = normalize_grade_prompt(prompt, include_gold=include_gold)
    fmt = JUDGE_FORMATS.get(name)
    if fmt is None:
        raise ValueError(f"Unknown judge format {name!r}")
    return fmt


def iter_judge_formats() -> tuple[tuple[str, JudgeFormat], ...]:
    """Every named format in ``JUDGE_FORMATS`` (insertion order)."""
    return tuple(JUDGE_FORMATS.items())


def build_grade_instructions(
    *,
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> str:
    """Judge preamble sent before the clip (nongold) or as the first block (gold)."""
    return get_judge_format(prompt, include_gold).prompt


def build_grade_input_fields(
    *,
    question: str,
    answer: str,
    prediction: str,
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> str:
    """Question / gold / test-taker answer block (no audio)."""
    return get_judge_format(prompt, include_gold).fields(
        question=question,
        answer=answer,
        prediction=prediction,
    )


def build_grade_after_audio(
    *,
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> str:
    """Closer after the fields. For nongold this is appended after the clip + fields."""
    return get_judge_format(prompt, include_gold).closer


def build_grade_prompt(
    *,
    question: str,
    answer: str,
    prediction: str,
    prompt: str | None = None,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
) -> str:
    """Assemble the full text judge prompt for gold or free."""
    return get_judge_format(prompt, include_gold).as_text(
        question=question,
        answer=answer,
        prediction=prediction,
    )


def build_grade_gold_prefix(
    *,
    question: str,
    answer: str,
    prompt: str | None = None,
    include_gold: bool = True,
) -> str:
    """Cached gold prefix: prompt plus every field before the response."""
    fmt = get_judge_format(prompt, include_gold=include_gold)
    lines = [
        _fill_placeholders(template, question=question, answer=answer, prediction="")
        for template in fmt.field_templates
        if template != FIELD_PREDICTION
    ]
    body = "\n".join(lines)
    if body:
        return f"{fmt.prompt}\n\n{body}\n"
    return f"{fmt.prompt}\n\n"


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

# Extraction order matches test-taker parsers (``extract_freeform_answer``):
# last ``Answer:`` line, then ``<answer>`` tags, then last-line / boxed /
# keyword fallbacks. Soft whole-region fallback stays narrow so prose like
# "correct in meaning" does not count as a verdict.
ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(0|1|correct|incorrect|pass|fail|yes|no|true|false|wrong)\s*</answer>",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)
PASS_RE = re.compile(r"\b(pass|yes|true)\b", re.IGNORECASE)
FAIL_RE = re.compile(r"\b(fail|no|false)\b", re.IGNORECASE)
# incorrect before correct so the longer token wins as a last-line word.
VERDICT_LINE_RE = re.compile(
    r"""
    ^
    [\s*`"'*_(\[]*
    (?:(?:final\s+)?(?:answer|verdict|judgement|judgment|label|decision)\s*[:=]\s*)?
    (?P<label>incorrect|correct|pass|fail|yes|no|true|false|wrong|[01])
    [\s.`"'*_!?,;:)\]\\]*
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)
LAST_WORD_VERDICT_RE = re.compile(
    r"\b(?P<label>incorrect|correct|pass|fail|wrong)\b\s*[.!?]?\s*$",
    re.IGNORECASE,
)
PASS_LABELS = frozenset({"PASS", "P", "YES", "Y", "TRUE", "CORRECT", "1"})
FAIL_LABELS = frozenset({"FAIL", "F", "NO", "N", "FALSE", "INCORRECT", "WRONG", "0"})


def _label_verdict(raw: str) -> bool | None:
    token = str(raw or "").strip().upper()
    if token in PASS_LABELS:
        return True
    if token in FAIL_LABELS:
        return False
    return None


def _clean_verdict_token(raw: str) -> str:
    token = str(raw or "").strip()
    boxed = BOXED_RE.search(token)
    if boxed:
        token = boxed.group(1).strip()
    token = token.strip("`\"'*_.!? ,;:()[]")
    if ":" in token:
        token = token.split(":")[-1].strip("`\"'*_.!? ,;:()[]")
    if " " in token:
        token = token.split()[-1].strip("`\"'*_.!? ,;:()[]")
    return token.upper()


def _verdict_from_text(snippet: str) -> bool | None:
    """Map a snippet (extracted answer or post-think region) to a verdict."""
    snippet = (snippet or "").strip()
    if not snippet:
        return None
    if "\n" not in snippet:
        verdict = _label_verdict(_clean_verdict_token(snippet))
        if verdict is not None:
            return verdict
    tag_matches = list(ANSWER_TAG_RE.finditer(snippet))
    if tag_matches:
        return _label_verdict(tag_matches[-1].group(1))
    boxed_matches = list(BOXED_RE.finditer(snippet))
    if boxed_matches:
        boxed = _label_verdict(_clean_verdict_token(boxed_matches[-1].group(1)))
        if boxed is not None:
            return boxed
    lines = [line.strip() for line in snippet.splitlines() if line.strip()]
    if not lines:
        return None

    def _line_verdict(raw: str) -> bool | None:
        match = VERDICT_LINE_RE.match(raw.strip())
        if match:
            return _label_verdict(match.group("label"))
        match = LAST_WORD_VERDICT_RE.search(raw.strip())
        if match:
            return _label_verdict(match.group("label"))
        return _label_verdict(_clean_verdict_token(raw))

    last = _line_verdict(lines[-1])
    if last is not None:
        return last
    for line in reversed(lines[:-1]):
        hit = _line_verdict(line)
        if hit is not None:
            return hit

    # Last resort: whole-region exclusive keyword match (legacy YES/NO dumps).
    # Do not search for correct/incorrect here — those words appear in prose.
    has_pass = bool(PASS_RE.search(snippet))
    has_fail = bool(FAIL_RE.search(snippet))
    if has_pass and not has_fail:
        return True
    if has_fail and not has_pass:
        return False
    return None


def majority_grade_verdict(verdicts: list[bool | None]) -> bool | None:
    """Strict majority over ``len(verdicts)`` slots. Unparsed shots do not vote."""
    n = len(verdicts)
    if n == 0:
        return None
    need = n // 2 + 1
    n_true = sum(1 for value in verdicts if value is True)
    n_false = sum(1 for value in verdicts if value is False)
    if n_true >= need:
        return True
    if n_false >= need:
        return False
    return None


def parse_grade_verdict(text: str) -> bool | None:
    """Parse a 0/1 tag, Correct/Incorrect, or legacy Pass/Fail reply.

    Uses the same answer extractor as test-takers (``Answer:`` line, then
    ``<answer>`` tags, then freeform fallbacks). Returns None if unparseable.
    ``incorrect`` is matched before ``correct`` so a last-line ``Incorrect``
    is never read as ``Correct``.
    """
    text = (text or "").strip()
    if not text:
        return None
    extracted = extract_freeform_answer(text)
    region = after_last_think_close(text) or text
    for snippet in (extracted, region):
        verdict = _verdict_from_text(snippet)
        if verdict is not None:
            return verdict
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


def _shot_prediction_text(shot: dict, *, model_label: str | None = None) -> str:
    """Extracted answer shown to the judge; never includes text before last ``</think>``."""
    raw = str(shot.get("model_output") or "")
    extracted = str(shot.get("answer_prediction") or "")
    source = (raw or extracted).strip()
    if not source:
        return ""
    return extract_freeform_answer(source) or extracted or raw


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
    prompt: str | None = None,
) -> tuple[str, ...]:
    """Identity of a grade prompt: optional format, question, gold, answer.

    ``prompt`` is prepended when given so ``with_gt`` and ``neutral_with_gt``
    (both show gold) do not share verdicts. Callers that grade one format at
    a time may omit it; the 3-tuple shape stays stable for those caches.
    """
    gold = str(answer or "") if include_gold else ""
    body = (str(question or ""), gold, _normalize_grade_answer(prediction))
    if prompt is None:
        return body
    name = normalize_grade_prompt(prompt, include_gold=include_gold)
    return (name, *body)


def _job_grade_spec(
    job: dict,
    *,
    prompt: str,
    include_gold: bool,
) -> tuple[str, bool, JudgeFormat]:
    """Per-job format: ``job['prompt']`` / ``job['include_gold']`` override the batch."""
    job_prompt = job.get("prompt", prompt)
    if job.get("include_gold") is not None:
        job_gold = bool(job["include_gold"])
    else:
        job_gold = include_gold
    name = normalize_grade_prompt(job_prompt, include_gold=job_gold)
    fmt = get_judge_format(name, include_gold=job_gold)
    return name, fmt.include_gold, fmt


def _shots_in_index_order(
    record: dict,
    shot_indices: tuple[int, ...] | list[int] | None,
) -> list[tuple[int, dict]]:
    """Return ``(shot_index, shot)`` in ``shot_indices`` order (or index order)."""
    by_idx: dict[int, dict] = {}
    for shot in record.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        try:
            idx = int(shot.get("shot_index", 0))
        except (TypeError, ValueError):
            idx = 0
        by_idx.setdefault(idx, shot)
    if shot_indices is None:
        return [(idx, by_idx[idx]) for idx in sorted(by_idx)]
    out: list[tuple[int, dict]] = []
    seen: set[int] = set()
    for raw in shot_indices:
        idx = int(raw)
        if idx in seen:
            continue
        seen.add(idx)
        shot = by_idx.get(idx)
        if shot is not None:
            out.append((idx, shot))
    return out


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


def _sidecar_row_key(row: dict) -> tuple[str, int]:
    qid = str(row.get("id") or "").strip()
    try:
        shot_index = int(row.get("shot_index", 0))
    except (TypeError, ValueError):
        shot_index = 0
    return (qid, shot_index)


def _load_sidecar_rows(path: Path | None) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    if path is None or not Path(path).is_file():
        return rows
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            key = _sidecar_row_key(row)
            if key[0]:
                rows[key] = row
    return rows


def _overlay_sidecar_rows(
    records: list[dict],
    rows: dict[tuple[str, int], dict],
    *,
    judge_key: str,
) -> None:
    """Copy sidecar verdicts onto matching shots that lack this judge."""
    if not rows:
        return
    by_id = {
        str(record.get("id") or "").strip(): record for record in records
    }
    by_id.pop("", None)
    for (qid, shot_index), row in rows.items():
        record = by_id.get(qid)
        if record is None:
            continue
        key = str(row.get("judge_key") or judge_key or "")
        entry = row.get("entry")
        if key != judge_key or not isinstance(entry, dict):
            continue
        for shot in record.get("shots") or []:
            try:
                idx = int(shot.get("shot_index", 0))
            except (TypeError, ValueError):
                idx = 0
            if idx != shot_index:
                continue
            judges = shot.setdefault("judges", {})
            if not isinstance(judges, dict):
                judges = {}
                shot["judges"] = judges
            judges.setdefault(key, entry)
            break


def pending_grade_ids(
    predictions_path: Path,
    judge_key: str,
    *,
    question_ids: list[str] | None = None,
    shot_indices: tuple[int, ...] | list[int] | None = None,
    sidecar_path: Path | None = None,
) -> list[str]:
    """Question ids whose allowed shots still lack a verdict for ``judge_key``.

    Sidecar rows count as graded. Preserves ``question_ids`` order (or file
    order when that is omitted). Ids with no prediction record are omitted.
    """
    path = Path(predictions_path)
    if not path.is_file():
        return []
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    for record in records:
        ensure_judge_schema(record, fallback_label=judge_key)
    _overlay_sidecar_rows(
        records, _load_sidecar_rows(sidecar_path), judge_key=judge_key
    )
    by_id: dict[str, dict] = {}
    file_order: list[str] = []
    for record in records:
        qid = str(record.get("id") or "").strip()
        if not qid or qid in by_id:
            continue
        by_id[qid] = record
        file_order.append(qid)
    wanted = [
        str(qid).strip()
        for qid in (question_ids if question_ids is not None else file_order)
        if str(qid).strip()
    ]
    pending: list[str] = []
    seen: set[str] = set()
    for qid in wanted:
        if qid in seen:
            continue
        seen.add(qid)
        record = by_id.get(qid)
        if record is None:
            continue
        if _record_needs_grade(record, judge_key, shot_indices=shot_indices):
            pending.append(qid)
    return pending


def remaining_grade_work(
    pack_dir: Path,
    model_labels: list[str],
    judge_key: str,
    *,
    question_ids: list[str] | None = None,
    shot_indices: tuple[int, ...] | list[int] | None = None,
    sidecar: bool = False,
) -> dict[str, list[str]]:
    """Map of test-taker label -> ungraded ids. Fully graded models are omitted."""
    pack = Path(pack_dir)
    remaining: dict[str, list[str]] = {}
    for label in model_labels:
        pred = pack / "models" / label / "predictions.jsonl"
        sidecar_path = (
            pack / "models" / label / "judge_partials" / f"{judge_key}.jsonl"
            if sidecar
            else None
        )
        pending = pending_grade_ids(
            pred,
            judge_key,
            question_ids=question_ids,
            shot_indices=shot_indices,
            sidecar_path=sidecar_path,
        )
        if pending:
            remaining[label] = pending
    return remaining


def _suite_label_for(model_id: str) -> str | None:
    """Return a MODEL_SPECS label when ``model_id`` is a spec key or HF id."""
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
    """True when this judge can hear MMAR audio (MODEL_SPECS or API)."""
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
    audio_required: bool | None = None,
) -> None:
    """Audio formats (no-gold, or gold+audio) need an audio-capable judge."""
    needed = bool(audio_required) if audio_required is not None else not include_gold
    if not needed:
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
        "Audio grading requires an audio-capable judge that receives the "
        f"clip (got {shown!r}). Text-only judges cannot grade this format; "
        "use a MODEL_SPECS or API audio model, or a text-only with-GT recipe."
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
        llm = loaded.get("llm")
        tokenizer = loaded.get("tokenizer")
        getter = getattr(llm, "get_tokenizer", None) if llm is not None else None
        if tokenizer is None and callable(getter):
            tokenizer = getter()
        local_id = resolve_model_dir(
            spec["model_id"], getattr(ns, "local_model_dir", None)
        )
        if tokenizer is not None:
            _ensure_tokenizer_chat_template(
                tokenizer,
                model_dir=local_id,
                fallback=_CHATML_JINJA if "qwen" in suite_label else None,
            )
        sampling = _grade_sampling_for_engine(
            spec.get("engine") or {}, spec.get("sampling") or {}
        )
        backend = loaded.get("backend") or spec.get("backend")
        print(
            f"Freeform grader ready (suite): {suite_label} ({spec['model_id']}) "
            f"backend={backend} sampling={sampling}"
        )
        chat_kwargs = loaded.get("chat_kwargs") or {}
        try:
            from vllm import SamplingParams
        except ImportError:
            SamplingParams = None  # type: ignore[misc, assignment]
        # HF .chat() is sequential; small chunks keep progress logs moving.
        hf_chat = llm is None and loaded.get("model") is not None
        return {
            "llm": llm,
            "model": loaded.get("model"),
            "tokenizer": tokenizer,
            "model_id": spec["model_id"],
            "judge_label": suite_label,
            "suite_label": suite_label,
            "backend": backend,
            "sampling_rate": int(
                loaded.get("sampling_rate") or spec.get("sampling_rate") or 16000
            ),
            "stages": loaded.get("stages"),
            "SamplingParams": SamplingParams,
            "sampling": sampling,
            "lora_request": loaded.get("lora_request"),
            "chat_kwargs": chat_kwargs,
            "chat_template_kwargs": chat_kwargs.get("chat_template_kwargs") or {},
            "batch_size": (
                8
                if hf_chat
                else int((spec.get("engine") or {}).get("max_num_seqs") or 32)
            ),
        }

    from modal_cache import configure_compile_cache

    model_id = resolve_judge_model_id(model_id)
    label = judge_label(model_id)
    configure_compile_cache(label)

    from vllm import LLM, SamplingParams

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
    _ensure_tokenizer_chat_template(tokenizer, model_dir=local_id)
    spec = resolve_judge_spec(model_id)
    chat_template_kwargs: dict[str, Any] = {}
    if "enable_thinking" in spec:
        chat_template_kwargs["enable_thinking"] = bool(spec["enable_thinking"])
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
        "chat_kwargs": (
            {"chat_template_kwargs": chat_template_kwargs}
            if chat_template_kwargs
            else {}
        ),
        "chat_template_kwargs": chat_template_kwargs,
    }


# Qwen3-Omni stores this in chat_template.json, not tokenizer_config.json.
# vLLM's get_tokenizer() therefore has chat_template unset.
_CHATML_JINJA = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
)


def _tokenizer_objects(tokenizer: Any) -> list[Any]:
    seen: list[Any] = []
    for obj in (
        tokenizer,
        getattr(tokenizer, "tokenizer", None),
        getattr(tokenizer, "_tokenizer", None),
    ):
        if obj is not None and obj not in seen:
            seen.append(obj)
    return seen


def _chat_template_of(tokenizer: Any) -> str | None:
    for obj in _tokenizer_objects(tokenizer):
        template = getattr(obj, "chat_template", None)
        if isinstance(template, str) and template.strip():
            return template
        if isinstance(template, dict):
            for key in ("default", "chat_template"):
                value = template.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            value = next(iter(template.values()), None)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _read_model_chat_template(model_dir: str | Path | None) -> tuple[str | None, str | None]:
    if not model_dir:
        return None, None
    root = Path(model_dir)
    jinja = root / "chat_template.jinja"
    if jinja.is_file():
        text = jinja.read_text(encoding="utf-8").strip()
        if text:
            return text, str(jinja)
    json_path = root / "chat_template.json"
    if json_path.is_file():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, str) and data.strip():
            return data, str(json_path)
        if isinstance(data, dict):
            for key in ("chat_template", "default", "template"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value, str(json_path)
    return None, None


def _ensure_tokenizer_chat_template(
    tokenizer: Any,
    *,
    model_dir: str | Path | None = None,
    fallback: str | None = None,
) -> str | None:
    """Attach a chat template when vLLM loaded the tokenizer without one."""
    existing = _chat_template_of(tokenizer)
    if existing:
        return existing
    dirs: list[str | Path] = []
    if model_dir:
        dirs.append(model_dir)
    for obj in _tokenizer_objects(tokenizer):
        name = getattr(obj, "name_or_path", None)
        if name:
            dirs.append(name)
    template = None
    source = None
    for path in dirs:
        template, source = _read_model_chat_template(path)
        if template:
            break
    if not template:
        template = fallback
        source = "ChatML fallback" if template else None
    if not template:
        return None
    for obj in _tokenizer_objects(tokenizer):
        if hasattr(obj, "chat_template"):
            try:
                obj.chat_template = template
            except (AttributeError, TypeError):
                pass
    print(f"[grader] tokenizer.chat_template was unset; loaded from {source}")
    return template


def _format_chatml(user_text: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _format_chat(
    tokenizer: Any,
    user_text: str,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    messages = [{"role": "user", "content": user_text}]
    if not hasattr(tokenizer, "apply_chat_template"):
        return f"User: {user_text}\nAssistant:"
    kwargs = dict(chat_template_kwargs or {})
    template = kwargs.get("chat_template") or _chat_template_of(tokenizer)
    if template and "chat_template" not in kwargs:
        kwargs["chat_template"] = template
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            pass
    except ValueError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=template or _CHATML_JINJA,
            )
        except (TypeError, ValueError):
            pass
    return _format_chatml(user_text)


# ---------------------------------------------------------------------------
# Nongold audio wrapping
#
# Same format as JUDGE_FORMATS["free"]:
#   {instructions}   = format.prompt   (task + 0/1 judgments)
#   {audio}          = clip or model placeholder
#   {fields}         = Question / Response + format.closer
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
        "<|im_start|>assistant\n" + ASSISTANT_THINK_OPEN
    ),
    "music-flamingo": (
        "<|im_start|>system\n{system}\n{instructions}<|im_end|>\n"
        "<|im_start|>user\n"
        "<sound>{fields}<|im_end|>\n"
        "<|im_start|>assistant\n" + ASSISTANT_THINK_OPEN
    ),
}

# Chat-message backends (not a string template). Slots match the format fields.
NONGOLD_AUDIO_CHAT_LAYOUT = (
    "user.content:\n"
    "  [0] text  {instructions}\n"
    "  [1] audio {audio}\n"
    "  [2] text  " + FIELD_QUESTION + "\n"
    "  [3] text  " + FIELD_PREDICTION + "\n"
    "  [4] text  {closer}"
)


_ANSWER_FORMAT_TAIL_RE = re.compile(
    r",?\s*then give a concise, answer in a single final line in exactly this format:\n"
    r"Answer:.*$",
    re.IGNORECASE | re.DOTALL,
)


def _adapt_judge_instructions(label: str, instructions: str) -> str:
    """Match generation output format: ``Answer:`` line, except Music Flamingo."""
    text = instructions
    if label == "af-next-think" and "with timestamps" not in text:
        if "Reason step by step before answering" in text:
            text = text.replace(
                "Reason step by step before answering",
                "Reason step by step with timestamps before answering",
                1,
            )
        elif "Reason briefly," in text:
            text = text.replace("Reason briefly,", "Reason briefly with timestamps,", 1)
        else:
            text = text.replace("Reason briefly ", "Reason briefly with timestamps ", 1)
    if label == "music-flamingo":
        stripped = _ANSWER_FORMAT_TAIL_RE.sub(".", text).rstrip()
        text = f"{stripped}\n\n{MUSIC_FLAMINGO_THINK_SUFFIX}"
    return text


def _nongold_audio_system_text(label: str) -> str:
    """Model system prompt interpolated into templates that have ``{system}``."""
    if label == "qwen2.5-omni-7b":
        from mmar_models import QWEN25_OMNI_SYSTEM

        return QWEN25_OMNI_SYSTEM
    if label == "af-next-think":
        from mmar_models import AF_NEXT_SYSTEM

        return AF_NEXT_SYSTEM
    if label == "music-flamingo":
        from mmar_models import MUSIC_FLAMINGO_SYSTEM

        return MUSIC_FLAMINGO_SYSTEM
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
    text = _fill_placeholders(
        nongold_audio_prompt_template(label),
        instructions=instructions,
        fields=body,
        system=_nongold_audio_system_text(label),
    )
    return ensure_assistant_think_open(label, text)


def _nongold_audio_chat_messages(
    instructions: str,
    question: str,
    prediction: str,
    audio_path: Path,
    *,
    audio_type: str = "audio_url",
    closer: str = "",
    answer: str = "",
    field_templates: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """User turn: instructions, then audio, then format fields, then closer."""
    if audio_type == "audio":
        audio_part: dict[str, Any] = {"type": "audio", "audio": str(audio_path)}
    else:
        audio_part = {
            "type": "audio_url",
            "audio_url": {"url": f"file://{audio_path}"},
        }
    templates = field_templates or (FIELD_QUESTION, FIELD_PREDICTION)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": instructions},
        audio_part,
    ]
    for template in templates:
        content.append(
            {
                "type": "text",
                "text": _fill_placeholders(
                    template,
                    question=question,
                    answer=answer,
                    prediction=prediction,
                ),
            }
        )
    if closer:
        content.append({"type": "text", "text": closer})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Prompt rendering (no tokenizer / GPU)
# ---------------------------------------------------------------------------

_CHAT_JUDGE_BACKENDS = frozenset({"vllm_chat", "hf_chat", "vllm_transformers"})
_HF_AUDIO_TYPE_BACKENDS = frozenset({"hf_chat", "vllm_transformers"})
_DEFAULT_RENDER_AUDIO = "/cache/data/mmar/audio/example.wav"


def judge_render_labels() -> tuple[str, ...]:
    """Labels ``render_judge_prompt`` can dump (MODEL_SPECS, dedicated, API)."""
    from mmar_api import API_SPECS
    from mmar_models import ALL_MODEL_LABELS

    labels: list[str] = []
    seen: set[str] = set()

    def _add(items) -> None:
        for item in items:
            key = str(item or "").strip()
            if key and key not in seen:
                labels.append(key)
                seen.add(key)

    _add(ALL_MODEL_LABELS)
    _add(JUDGE_SPECS)
    _add(NONGOLD_AUDIO_PROMPT_TEMPLATES)
    _add(API_SPECS)
    return tuple(labels)


def parse_judge_list(value: str) -> list[str]:
    """Comma-separated judge labels, or ``all``."""
    raw = [item.strip() for item in value.split(",") if item.strip()]
    known = judge_render_labels()
    if not raw or any(item.lower() == "all" for item in raw):
        return list(known)
    labels: list[str] = []
    unknown: list[str] = []
    for item in raw:
        resolved = _resolve_render_judge_label(item)
        if resolved is None:
            unknown.append(item)
        elif resolved not in labels:
            labels.append(resolved)
    if unknown:
        raise ValueError(
            f"Unknown judge label(s): {unknown}. "
            f"Choose from {list(known)} or 'all'."
        )
    return labels


def _resolve_render_judge_label(raw: str) -> str | None:
    from mmar_api import resolve_api_judge_label
    from mmar_models import MODEL_SPECS

    key = str(raw or "").strip()
    if not key:
        return None
    if key in judge_render_labels():
        return key
    api = resolve_api_judge_label(key)
    if api:
        return api
    if key in MODEL_SPECS:
        return key
    alias = JUDGE_MODEL_ALIASES.get(key.lower())
    if alias:
        slug = judge_label(alias)
        if slug in JUDGE_SPECS:
            return slug
    if key in JUDGE_SPECS:
        return key
    return _suite_label_for(key)


def _judge_backend(label: str) -> str:
    from mmar_api import API_SPECS, resolve_api_judge_label
    from mmar_models import MODEL_SPECS

    if label in MODEL_SPECS:
        return str(MODEL_SPECS[label].get("backend") or "vllm")
    api = resolve_api_judge_label(label)
    if api:
        return str((API_SPECS.get(api) or {}).get("backend") or "")
    if label in JUDGE_SPECS:
        return "vllm"
    return "vllm"


def judge_can_hear_audio(label: str) -> bool:
    """True when this label can take an ``audio_included`` format."""
    from mmar_api import is_api_judge, is_batch_api_judge
    from mmar_models import MODEL_SPECS

    if is_batch_api_judge(label):
        return False
    if is_api_judge(label):
        return True
    if label in NONGOLD_AUDIO_PROMPT_TEMPLATES:
        return True
    return label in MODEL_SPECS


def render_judge_prompt(
    label: str,
    job: dict,
    *,
    prompt: str | None = None,
    include_gold: bool | None = None,
) -> str:
    """Return the text ``grade_shot_batch`` would send for ``label``.

    Audio is represented by the same placeholders used at grade time
    (``<sound>``, ``<|audio_pad|>``, ``file://`` URLs, etc.). Chat backends
    dump the messages ``LLM.chat`` receives; the model's Jinja chat template
    is applied later by vLLM / HF. API judges send ``build_grade_prompt``
    text and attach audio out of band when the format includes it.
    """
    from mmar_api import is_api_judge, is_batch_api_judge
    from mmar_models import _format_chat_messages

    resolved = _resolve_render_judge_label(label) or label
    gold_flag = DEFAULT_INCLUDE_GOLD if include_gold is None else include_gold
    fmt = get_judge_format(prompt, include_gold=gold_flag)
    question = str(job.get("question") or "")
    answer = str(job.get("answer") or "")
    prediction = str(job.get("prediction") or "")
    audio_raw = str(job.get("audio_path") or "") or _DEFAULT_RENDER_AUDIO
    audio_path = Path(audio_raw)

    if not fmt.audio_included:
        text = fmt.as_text(
            question=question, answer=answer, prediction=prediction
        )
        return text if text.endswith("\n") else f"{text}\n"

    if is_batch_api_judge(resolved):
        raise ValueError(
            f"{resolved!r} is a Batch API judge and only supports text formats"
        )
    if not judge_can_hear_audio(resolved):
        raise ValueError(
            f"{resolved!r} cannot hear audio; use a suite or live API judge"
        )

    if is_api_judge(resolved):
        text = fmt.as_text(
            question=question, answer=answer, prediction=prediction
        )
        body = text if text.endswith("\n") else f"{text}\n"
        return f"[audio attached]\n{body}"

    backend = _judge_backend(resolved)
    instructions = _adapt_judge_instructions(resolved, fmt.prompt)
    if backend in _CHAT_JUDGE_BACKENDS:
        audio_type = "audio" if backend in _HF_AUDIO_TYPE_BACKENDS else "audio_url"
        messages = _nongold_audio_chat_messages(
            instructions,
            question,
            prediction,
            audio_path,
            audio_type=audio_type,
            closer=fmt.closer,
            answer=answer,
            field_templates=fmt.field_templates,
        )
        return _format_chat_messages(messages)
    if backend == "vllm_voxtral":
        fields = fmt.fields_with_closer(
            question=question, answer=answer, prediction=prediction
        )
        return (
            "role=user\n"
            f"{instructions}\n"
            f"<audio>{audio_path}</audio>\n"
            f"{fields}\n"
        )
    fields = fmt.fields(
        question=question, answer=answer, prediction=prediction
    )
    return _nongold_audio_prompt_string(
        resolved, instructions, fields, fmt.closer
    )


def _grade_sampling(
    handle: dict[str, Any],
    *,
    max_tokens: int | None = None,
    n: int = 1,
    temperature: float | None = None,
    seed: int | None = None,
):
    from vllm import SamplingParams

    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    n = max(1, int(n))
    if handle.get("suite_label") and handle.get("sampling"):
        kwargs = dict(handle["sampling"])
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        if temperature is not None:
            t = float(temperature)
            kwargs["temperature"] = t if t > 0 else 0.0
        else:
            kwargs["temperature"] = 0.0
        if seed is not None:
            kwargs["seed"] = int(seed)
        if n > 1:
            kwargs["n"] = n
            if float(kwargs.get("temperature") or 0) <= 0:
                kwargs["temperature"] = 1.0
        return SamplingParams(**kwargs)
    return judge_sampling_params(
        model_id, max_tokens=max_tokens, n=n, temperature=temperature
    )


def _completion_texts(out: Any, n: int = 1) -> list[str]:
    """All ``n`` completion strings from one vLLM request (padded if short).

    Concatenates vLLM's separated ``reasoning`` / ``reasoning_content`` onto
    ``.text`` so LALM CoT is not dropped when a reasoning parser is active.
    """
    n = max(1, int(n))
    request_output = getattr(out, "request_output", None)
    if request_output is not None:
        out = (
            request_output[0]
            if isinstance(request_output, (list, tuple)) and request_output
            else request_output
        )
    outs = getattr(out, "outputs", None) or []
    if outs:
        texts = [join_vllm_reasoning(item) for item in outs]
    else:
        texts = [join_vllm_reasoning(out)]
    if len(texts) < n:
        texts = texts + [""] * (n - len(texts))
    return texts[:n]


def _completion_text(out: Any) -> str:
    return _completion_texts(out, n=1)[0]


def _verdict_fields(text: str) -> dict[str, Any]:
    verdict = parse_grade_verdict(text)
    reasoning, _answer = parse_freeform_output(text)
    return {
        "correct": bool(verdict) if verdict is not None else False,
        "verdict": (
            "pass" if verdict is True else "fail" if verdict is False else None
        ),
        "generation": text,
        "reasoning": reasoning or "",
        "grader_output": format_grade_output(verdict),
        "grader_verdict_raw": verdict,
    }


def _grade_results_from_texts(
    jobs: list[dict],
    texts: list[str],
    *,
    model_id: str,
    prompt: str,
    include_gold: bool,
) -> list[dict]:
    results: list[dict] = []
    for job, text in zip(jobs, texts):
        prompt_name, gold, _ = _job_grade_spec(
            job, prompt=prompt, include_gold=include_gold
        )
        fields = _verdict_fields(text)
        results.append(
            {
                **fields,
                "grader": model_id,
                "question": job.get("question"),
                "answer": job.get("answer"),
                "prediction": job.get("prediction"),
                "prompt": prompt_name,
                "include_gold": gold,
            }
        )
    return results


def _grade_results_from_sample_groups(
    jobs: list[dict],
    sample_groups: list[list[str]],
    *,
    model_id: str,
    prompt: str,
    include_gold: bool,
) -> list[dict]:
    """One result per job: majority vote over ``n`` sampled judge replies."""
    results: list[dict] = []
    fallback_gold = include_gold
    for job, texts in zip(jobs, sample_groups):
        prompt_name, gold, _ = _job_grade_spec(
            job, prompt=prompt, include_gold=fallback_gold
        )
        sample_fields = [_verdict_fields(text) for text in texts]
        raw = [item["grader_verdict_raw"] for item in sample_fields]
        majority = majority_grade_verdict(raw)
        generation = ""
        reasoning = ""
        for item, verdict in zip(sample_fields, raw):
            if majority is not None and verdict is majority:
                generation = item["generation"]
                reasoning = item.get("reasoning") or ""
                break
        if not generation and sample_fields:
            generation = sample_fields[0]["generation"]
            reasoning = sample_fields[0].get("reasoning") or ""
        results.append(
            {
                "correct": bool(majority) if majority is not None else False,
                "verdict": (
                    "pass"
                    if majority is True
                    else "fail"
                    if majority is False
                    else None
                ),
                "generation": generation,
                "reasoning": reasoning,
                "grader_output": format_grade_output(majority),
                "grader": model_id,
                "grader_verdict_raw": majority,
                "samples": [
                    {
                        "correct": item["correct"],
                        "verdict": item["verdict"],
                        "generation": item["generation"],
                        "reasoning": item.get("reasoning") or "",
                        "output": item["grader_output"],
                    }
                    for item in sample_fields
                ],
                "n_samples": len(sample_fields),
                "question": job.get("question"),
                "answer": job.get("answer"),
                "prediction": job.get("prediction"),
                "prompt": prompt_name,
                "include_gold": gold,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Batch grading
# ---------------------------------------------------------------------------


def _is_hf_chat_handle(handle: dict[str, Any]) -> bool:
    """True when the suite loader fell back to Transformers ``model.chat()``."""
    model = handle.get("model")
    return handle.get("llm") is None and model is not None and callable(
        getattr(model, "chat", None)
    )


def _hf_generation_config(
    handle: dict[str, Any],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    n: int = 1,
) -> dict[str, Any]:
    """Map judge sampling onto an HF ``generation_config``."""
    sampling = dict(handle.get("sampling") or {})
    if max_tokens is not None:
        sampling["max_tokens"] = int(max_tokens)
    t = 0.0 if temperature is None else float(temperature)
    if max(1, int(n)) > 1 and t <= 0:
        t = 1.0
    cfg: dict[str, Any] = {
        "max_new_tokens": int(sampling.get("max_tokens") or DEFAULT_JUDGE_MAX_TOKENS),
        "do_sample": t > 0,
        "top_p": float(sampling.get("top_p") or 1.0),
        "repetition_penalty": float(sampling.get("repetition_penalty") or 1.0),
    }
    if t > 0:
        cfg["temperature"] = t
    if seed is not None:
        cfg["seed"] = int(seed)
    elif sampling.get("seed") is not None:
        cfg["seed"] = int(sampling["seed"])
    return cfg


def _run_hf_chat(
    handle: dict[str, Any],
    messages_list: list[list[dict[str, Any]]],
    *,
    generation_config: dict[str, Any],
) -> list[str]:
    """Call ``model.chat(tokenizer, generation_config, messages)`` one job at a time."""
    import torch

    from audio_flamingo_runtime import seed_everything

    seed = generation_config.get("seed")
    if seed is not None:
        seed_everything(int(seed))
    cfg = {key: value for key, value in generation_config.items() if key != "seed"}
    model = handle["model"]
    tokenizer = handle["tokenizer"]
    texts: list[str] = []
    with torch.inference_mode():
        for messages in messages_list:
            response = model.chat(tokenizer, cfg, messages)
            if isinstance(response, tuple):
                response = response[0]
            texts.append(str(response or ""))
    return texts


def _grade_hf_chat_messages(
    handle: dict[str, Any],
    jobs: list[dict],
    messages_list: list[list[dict[str, Any]]],
    *,
    max_tokens: int | None,
    prompt_name: str,
    include_gold: bool,
    n_samples: int,
    temperature: float | None,
) -> list[dict]:
    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    n_samples = max(1, int(n_samples))
    if n_samples <= 1:
        cfg = _hf_generation_config(
            handle, max_tokens=max_tokens, temperature=temperature
        )
        texts = _run_hf_chat(handle, messages_list, generation_config=cfg)
        return _grade_results_from_texts(
            jobs,
            texts,
            model_id=model_id,
            prompt=prompt_name,
            include_gold=include_gold,
        )
    sample_groups: list[list[str]] = []
    for messages in messages_list:
        group: list[str] = []
        for shot_i in range(n_samples):
            cfg = _hf_generation_config(
                handle,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=shot_i,
                n=n_samples,
            )
            group.extend(
                _run_hf_chat(handle, [messages], generation_config=cfg)
            )
        sample_groups.append(group)
    return _grade_results_from_sample_groups(
        jobs,
        sample_groups,
        model_id=model_id,
        prompt=prompt_name,
        include_gold=include_gold,
    )


def _grade_results_from_audio_outputs(
    jobs: list[dict],
    *,
    n_samples: int,
    model_id: str,
    prompt: str,
    include_gold: bool,
    outputs: list[Any] | None = None,
    texts: list[str] | None = None,
    sample_groups: list[list[str]] | None = None,
) -> list[dict]:
    n_samples = max(1, int(n_samples))
    if n_samples > 1:
        groups = sample_groups
        if groups is None:
            groups = [
                _completion_texts(out, n=n_samples) for out in (outputs or [])
            ]
        return _grade_results_from_sample_groups(
            jobs,
            groups,
            model_id=model_id,
            prompt=prompt,
            include_gold=include_gold,
        )
    if texts is None:
        texts = [_completion_text(out) for out in (outputs or [])]
    return _grade_results_from_texts(
        jobs,
        texts,
        model_id=model_id,
        prompt=prompt,
        include_gold=include_gold,
    )


def _text_grade_user_messages(
    job: dict, prompt_name: str, include_gold: bool
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": build_grade_prompt(
                        question=str(job.get("question") or ""),
                        answer=str(job.get("answer") or ""),
                        prediction=str(job.get("prediction") or ""),
                        prompt=prompt_name,
                        include_gold=include_gold,
                    ),
                }
            ],
        }
    ]


def _require_job_audio_path(job: dict) -> Path:
    audio_path = resolve_grade_audio_path(str(job.get("audio_path") or "") or None)
    if audio_path is None:
        raise SystemExit(
            "Audio grading needs a readable wav for each question; "
            f"missing audio_path for id={job.get('id')!r} "
            f"path={job.get('audio_path')!r}"
        )
    return audio_path


def _chat_messages_for_grade_job(
    handle: dict[str, Any],
    job: dict,
    spec: tuple[str, bool, JudgeFormat],
    *,
    audio_type: str,
) -> list[dict[str, Any]]:
    prompt_name, include_gold, fmt = spec
    if not fmt.audio_included:
        return _text_grade_user_messages(job, prompt_name, include_gold)
    label = str(handle.get("suite_label") or handle.get("judge_label") or "")
    instructions = _adapt_judge_instructions(label, fmt.prompt)
    return _nongold_audio_chat_messages(
        instructions,
        str(job.get("question") or ""),
        str(job.get("prediction") or ""),
        _require_job_audio_path(job),
        audio_type=audio_type,
        closer=fmt.closer,
        answer=str(job.get("answer") or ""),
        field_templates=fmt.field_templates,
    )


def _vllm_text_grade_prompt(
    handle: dict[str, Any],
    job: dict,
    prompt_name: str,
    include_gold: bool,
) -> str:
    tokenizer = handle.get("tokenizer")
    if tokenizer is None:
        getter = getattr(handle.get("llm"), "get_tokenizer", None)
        tokenizer = getter() if callable(getter) else None
    if tokenizer is None:
        raise SystemExit("Text grade jobs need a tokenizer on the judge handle")
    return _format_chat(
        tokenizer,
        build_grade_prompt(
            question=str(job.get("question") or ""),
            answer=str(job.get("answer") or ""),
            prediction=str(job.get("prediction") or ""),
            prompt=prompt_name,
            include_gold=include_gold,
        ),
        chat_template_kwargs=handle.get("chat_template_kwargs") or {},
    )


def _vllm_generate_item_for_grade_job(
    handle: dict[str, Any],
    job: dict,
    spec: tuple[str, bool, JudgeFormat],
    *,
    label: str,
    sampling_rate: int,
) -> str | dict[str, Any]:
    from mmar_models import _load_audio_tuple

    prompt_name, include_gold, fmt = spec
    if not fmt.audio_included:
        return _vllm_text_grade_prompt(handle, job, prompt_name, include_gold)
    audio_path = _require_job_audio_path(job)
    instructions = _adapt_judge_instructions(label, fmt.prompt)
    fields = fmt.fields(
        question=str(job.get("question") or ""),
        answer=str(job.get("answer") or ""),
        prediction=str(job.get("prediction") or ""),
    )
    audio = _load_audio_tuple(str(audio_path), sampling_rate)
    return {
        "prompt": _nongold_audio_prompt_string(
            label, instructions, fields, fmt.closer
        ),
        "multi_modal_data": {"audio": audio},
    }


def _voxtral_item_for_grade_job(
    handle: dict[str, Any],
    job: dict,
    spec: tuple[str, bool, JudgeFormat],
    *,
    label: str,
) -> dict[str, Any]:
    from mistral_common.protocol.instruct.chunk import AudioChunk, TextChunk
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.tokens.tokenizers.audio import Audio

    prompt_name, include_gold, fmt = spec
    tokenizer = handle["tokenizer"]
    if not fmt.audio_included:
        text = build_grade_prompt(
            question=str(job.get("question") or ""),
            answer=str(job.get("answer") or ""),
            prediction=str(job.get("prediction") or ""),
            prompt=prompt_name,
            include_gold=include_gold,
        )
        messages = [UserMessage(content=[TextChunk(text=text)]).to_openai()]
        return {
            "prompt_token_ids": tokenizer.apply_chat_template(messages=messages)
        }
    instructions = _adapt_judge_instructions(label, fmt.prompt)
    audio_path = _require_job_audio_path(job)
    fields = fmt.fields_with_closer(
        question=str(job.get("question") or ""),
        answer=str(job.get("answer") or ""),
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
    return {
        "prompt_token_ids": tokenizer.apply_chat_template(messages=messages),
        "multi_modal_data": {
            "audio": [(audio.audio_array, audio.sampling_rate)]
        },
    }


def _grade_shot_batch_audio(
    handle: dict[str, Any],
    jobs: list[dict],
    *,
    max_tokens: int | None = None,
    prompt: str | None = None,
    include_gold: bool = False,
    n_samples: int = 1,
    temperature: float | None = None,
) -> list[dict]:
    """One ``chat`` / ``generate`` for mixed audio and text grade jobs.

    Jobs that include a clip keep the audio-then-fields layout. Text-only
    jobs (with-GT, gold-no-audio, …) use the text grade prompt so a run can
    send every format in one vLLM call. ``n_samples > 1`` majority-votes
    that many completions per job. Plain vLLM backends use
    ``SamplingParams(n=...)``. HF duplicates the prompt (those backends
    cannot fork ``n`` on one prefill).
    """
    from mmar_models import backend_duplicates_shots

    label = str(handle.get("suite_label") or handle.get("judge_label") or "")
    backend = str(handle.get("backend") or "vllm")
    sampling_rate = int(handle.get("sampling_rate") or 16000)
    prompt_name = normalize_grade_prompt(prompt, include_gold=include_gold)
    specs = [
        _job_grade_spec(job, prompt=prompt_name, include_gold=include_gold)
        for job in jobs
    ]
    n_samples = max(1, int(n_samples))
    duplicate_prompts = n_samples > 1 and backend_duplicates_shots(backend)
    sampling = _grade_sampling(
        handle,
        max_tokens=max_tokens,
        n=1 if duplicate_prompts else n_samples,
        temperature=temperature,
    )
    generate_kwargs: dict[str, Any] = {}
    if handle.get("lora_request") is not None:
        generate_kwargs["lora_request"] = handle["lora_request"]

    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    result_kwargs = dict(
        n_samples=n_samples,
        model_id=model_id,
        prompt=prompt_name,
        include_gold=include_gold,
    )

    if backend == "vllm_chat":
        chat_kwargs = dict(handle.get("chat_kwargs") or {})
        messages = [
            _chat_messages_for_grade_job(
                handle, job, spec, audio_type="audio_url"
            )
            for job, spec in zip(jobs, specs)
        ]
        outputs = handle["llm"].chat(
            messages, sampling_params=sampling, **chat_kwargs
        )
        return _grade_results_from_audio_outputs(
            jobs, outputs=outputs, **result_kwargs
        )
    if _is_hf_chat_handle(handle) or backend in {"hf_chat", "vllm_transformers"}:
        messages = [
            _chat_messages_for_grade_job(
                handle, job, spec, audio_type="audio"
            )
            for job, spec in zip(jobs, specs)
        ]
        if _is_hf_chat_handle(handle):
            return _grade_hf_chat_messages(
                handle,
                jobs,
                messages,
                max_tokens=max_tokens,
                prompt_name=prompt_name,
                include_gold=include_gold,
                n_samples=n_samples,
                temperature=temperature,
            )
        chat_kwargs = dict(handle.get("chat_kwargs") or {})
        if duplicate_prompts:
            sample_groups: list[list[str]] = []
            for msgs in messages:
                group: list[str] = []
                for shot_i in range(n_samples):
                    sp = _grade_sampling(
                        handle,
                        max_tokens=max_tokens,
                        n=1,
                        temperature=temperature,
                        seed=shot_i,
                    )
                    outs = handle["llm"].chat(
                        [msgs], sampling_params=sp, **chat_kwargs
                    )
                    group.append(
                        _completion_text(outs[0]) if outs else ""
                    )
                sample_groups.append(group)
            return _grade_results_from_audio_outputs(
                jobs, sample_groups=sample_groups, **result_kwargs
            )
        outputs = handle["llm"].chat(
            messages, sampling_params=sampling, **chat_kwargs
        )
        return _grade_results_from_audio_outputs(
            jobs, outputs=outputs, **result_kwargs
        )
    if backend == "vllm_voxtral":
        prompts = [
            _voxtral_item_for_grade_job(handle, job, spec, label=label)
            for job, spec in zip(jobs, specs)
        ]
        outputs = handle["llm"].generate(
            prompts, sampling_params=sampling, **generate_kwargs
        )
        return _grade_results_from_audio_outputs(
            jobs, outputs=outputs, **result_kwargs
        )

    prompts = [
        _vllm_generate_item_for_grade_job(
            handle, job, spec, label=label, sampling_rate=sampling_rate
        )
        for job, spec in zip(jobs, specs)
    ]
    outputs = handle["llm"].generate(
        prompts, sampling_params=sampling, **generate_kwargs
    )
    return _grade_results_from_audio_outputs(
        jobs, outputs=outputs, **result_kwargs
    )


def grade_shot_batch(
    handle: dict[str, Any],
    jobs: list[dict],
    *,
    max_tokens: int | None = None,
    prompt: str = DEFAULT_GRADE_PROMPT,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
    n_samples: int = 1,
    temperature: float | None = None,
) -> list[dict]:
    """Grade a list of ``{question, answer, prediction}`` jobs.

    Each job may carry its own ``prompt`` / ``include_gold``. Audio jobs
    must use an audio judge and include ``audio_path``. After the grade
    prompt, those inputs are audio, then question, then the response, then
    the 0/1 closer. Mixed audio and text jobs go to one vLLM ``generate``
    or ``chat`` call.

    ``n_samples > 1`` uses ``SamplingParams(n=...)`` so the copies share
    prefill (plain vLLM), then majority-votes the parsed verdicts. HF
    backends duplicate the prompt instead.

    Returns one result dict per job with ``correct``, ``verdict``,
    ``generation`` (full text, including any vLLM-separated reasoning),
    ``reasoning`` (CoT span), ``grader_output`` (short 0/1), and
    ``grader``. Majority runs also include ``samples`` / ``n_samples``.
    """
    if not jobs:
        return []
    fallback_fmt = get_judge_format(prompt, include_gold=include_gold)
    include_gold = fallback_fmt.include_gold
    prompt_name = normalize_grade_prompt(prompt, include_gold=include_gold)
    specs = [
        _job_grade_spec(job, prompt=prompt_name, include_gold=include_gold)
        for job in jobs
    ]
    n_samples = max(1, int(n_samples))
    for _, gold, fmt in specs:
        require_audio_nongold_judge(
            handle=handle,
            include_gold=gold,
            audio_required=fmt.audio_included,
        )
    if any(fmt.audio_included for _, _, fmt in specs):
        return _grade_shot_batch_audio(
            handle,
            jobs,
            max_tokens=max_tokens,
            prompt=prompt_name,
            include_gold=include_gold,
            n_samples=n_samples,
            temperature=temperature,
        )

    if _is_hf_chat_handle(handle):
        messages = [
            _text_grade_user_messages(job, name, gold)
            for job, (name, gold, _) in zip(jobs, specs)
        ]
        return _grade_hf_chat_messages(
            handle,
            jobs,
            messages,
            max_tokens=max_tokens,
            prompt_name=prompt_name,
            include_gold=include_gold,
            n_samples=n_samples,
            temperature=temperature,
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
                prompt=name,
                include_gold=gold,
            ),
            chat_template_kwargs=chat_template_kwargs,
        )
        for job, (name, gold, _) in zip(jobs, specs)
    ]
    sampling = _grade_sampling(
        handle, max_tokens=max_tokens, n=n_samples, temperature=temperature
    )
    generate_kwargs: dict[str, Any] = {}
    if handle.get("lora_request") is not None:
        generate_kwargs["lora_request"] = handle["lora_request"]
    outputs = handle["llm"].generate(
        prompts, sampling_params=sampling, **generate_kwargs
    )
    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    if n_samples <= 1:
        return _grade_results_from_texts(
            jobs,
            [_completion_text(out) for out in outputs],
            model_id=model_id,
            prompt=prompt_name,
            include_gold=include_gold,
        )
    return _grade_results_from_sample_groups(
        jobs,
        [_completion_texts(out, n=n_samples) for out in outputs],
        model_id=model_id,
        prompt=prompt_name,
        include_gold=include_gold,
    )


def _log_prompt_progress(
    label: str,
    done: int,
    total: int,
    *,
    elapsed_s: float,
    extra: str = "",
) -> None:
    """Print ``done/total prompts`` for a judge worker."""
    if total > 0:
        body = (
            f"{done}/{total} prompts "
            f"({100.0 * done / total:.0f}%, {elapsed_s:.0f}s)"
        )
    else:
        body = f"{done}/{total} prompts ({elapsed_s:.0f}s)"
    suffix = f" {extra}" if extra else ""
    print(f"[{label}] {body}{suffix}", flush=True)


def _progress_judge_label(handle: dict[str, Any], model_id: str) -> str:
    return str(
        handle.get("judge_label")
        or handle.get("suite_label")
        or model_id
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
    question_ids: list[str] | None = None,
    n_samples: int = 1,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Grade shots in a predictions.jsonl file for one judge configuration.

    When ``force`` is True, existing entries for ``judge_key`` are replaced.
    When ``sidecar_path`` is set, verdicts are written there instead of
    rewriting ``predictions.jsonl`` (safe for concurrent round-robin judges).
    Existing sidecar rows are treated as already graded unless ``force``
    (resume does not redo them, and the sidecar is not overwritten with
    only the new rows).
    ``make_primary`` False keeps each record's existing ``primary_judge``.
    ``question_ids`` limits grading to those ids. ``n_questions`` then grades
    a prefix of the fixed shuffled remaining id list (see
    ``mmar_common.GRADE_SAMPLE_SEED``); None or < 0 keeps the full set.
    Duplicate model answers (lowercase, stripped) reuse an existing verdict
    for the same question and gold, including shots already graded for this
    judge when ``force`` is False.
    ``n_samples > 1`` majority-votes that many judge completions per answer
    (shared-prefill ``SamplingParams(n=...)`` on plain vLLM; prompt copies
    on HF audio backends).
    ``batch_size`` is ignored: remaining jobs go to one vLLM ``generate`` /
    ``chat`` call (continuous batching). Already-graded shots are skipped
    before that call.
    """
    del batch_size
    if not predictions_path.exists():
        return {
            "status": "missing",
            "predictions_path": str(predictions_path),
            "n_records": 0,
            "n_shots_graded": 0,
        }

    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    gradee_label = predictions_path.parent.name
    fmt = get_judge_format(prompt, include_gold=include_gold)
    include_gold = fmt.include_gold
    prompt_name = normalize_grade_prompt(prompt, include_gold=include_gold)
    require_audio_nongold_judge(
        handle=handle,
        include_gold=include_gold,
        audio_required=fmt.audio_included,
    )
    key = resolve_grade_judge_key(
        handle, prompt=prompt_name, include_gold=include_gold, judge_key=judge_key
    )
    n_samples = max(1, int(n_samples))
    allowed = {int(i) for i in shot_indices} if shot_indices is not None else None

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

    existing_sidecar: dict[tuple[str, int], dict] = {}
    if sidecar_path is not None:
        existing_sidecar = _load_sidecar_rows(sidecar_path)
        if not force:
            _overlay_sidecar_rows(records, existing_sidecar, judge_key=key)

    allowed_ids = resolve_grade_allowed_ids(
        [str(record.get("id") or "") for record in records],
        question_ids=question_ids,
        n_questions=n_questions,
    )

    def _in_sample(record: dict) -> bool:
        if allowed_ids is None:
            return True
        return str(record.get("id") or "").strip() in allowed_ids

    reuse_cache: dict[tuple[str, ...], dict] = {}
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
                        _shot_prediction_text(shot, model_label=gradee_label),
                        include_gold=include_gold,
                        prompt=prompt_name,
                    ),
                    dict(entry),
                )

    jobs: list[dict] = []
    owners: list[list[tuple[int, int]]] = []
    job_keys: list[tuple[str, ...]] = []
    pending_index: dict[tuple[str, ...], int] = {}
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
            prediction = _shot_prediction_text(shot, model_label=gradee_label)
            cache_key = _grade_reuse_key(
                question,
                answer,
                prediction,
                include_gold=include_gold,
                prompt=prompt_name,
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
                        "prompt": prompt_name,
                        "include_gold": include_gold,
                    }
                )
                owners.append([(record_index, shot_index)])
                job_keys.append(cache_key)
            else:
                owners[idx].append((record_index, shot_index))

    graded = 0
    reused = 0
    partials: list[dict] = []
    progress_label = _progress_judge_label(handle, model_id)
    progress_extra = f"gradee={gradee_label} prompt={prompt_name}"
    n_prompts = len(jobs)
    started = time.perf_counter()
    if n_prompts == 0:
        print(
            f"[{progress_label}] 0 prompts to generate "
            f"(reused={len(reuse_owners)}) {progress_extra}",
            flush=True,
        )
    else:
        _log_prompt_progress(
            progress_label,
            0,
            n_prompts,
            elapsed_s=0.0,
            extra=progress_extra,
        )

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

    if jobs:
        results = grade_shot_batch(
            handle,
            jobs,
            prompt=prompt_name,
            include_gold=include_gold,
            n_samples=n_samples,
            temperature=temperature,
        )
        for cache_key, owner_list, result in zip(job_keys, owners, results):
            entry = {
                "correct": bool(result["correct"]),
                "verdict": result.get("verdict"),
                "output": result.get("grader_output"),
                "generation": result.get("generation") or "",
                "reasoning": result.get("reasoning") or "",
                "model_id": model_id,
                "prompt": prompt_name,
                "include_gold": include_gold,
            }
            if result.get("samples"):
                entry["samples"] = result["samples"]
                entry["n_samples"] = result.get("n_samples") or n_samples
            reuse_cache[cache_key] = entry
            for extra_index, (record_index, shot_index) in enumerate(owner_list):
                _apply_entry(record_index, shot_index, entry)
                if extra_index:
                    reused += 1
        _log_prompt_progress(
            progress_label,
            n_prompts,
            n_prompts,
            elapsed_s=time.perf_counter() - started,
            extra=progress_extra,
        )

    if sidecar_path is not None:
        if force or partials:
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            combined = {} if force else dict(existing_sidecar)
            for row in partials:
                row_key = _sidecar_row_key(row)
                if row_key[0]:
                    combined[row_key] = row
            write_jsonl(sidecar_path, list(combined.values()), mode="w")
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
            "n_samples": n_samples,
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
        "n_samples": n_samples,
    }


def grade_predictions_pack(
    pack_dir: Path,
    handle: dict[str, Any],
    model_labels: list[str],
    *,
    judge_key: str | None = None,
    batch_size: int | None = None,
    force: bool = False,
    prompt: str = DEFAULT_GRADE_PROMPT,
    include_gold: bool = DEFAULT_INCLUDE_GOLD,
    modes: list[tuple[str, bool]] | None = None,
    shot_indices: tuple[int, ...] | list[int] | None = None,
    question_ids: list[str] | None = None,
    n_questions: int | None = None,
    n_samples: int = 1,
    temperature: float | None = None,
    sidecar: bool = True,
    make_primary: bool = False,
    primary_judge: str | None = None,
    on_batch: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Grade every remaining format × test-taker × shot in one generate.

    Jobs are emitted as ``for question: for mode: for test-taker: for shot``
    so the judge's question/audio prefix stays in cache across gradees, and
    every prompt in the run is one vLLM ``generate`` / ``chat`` call.
    Sidecars are written under each gradee's ``judge_partials/`` unless
    ``sidecar`` is False (then each ``predictions.jsonl`` is rewritten at
    the end). ``on_batch`` runs after the sidecar flush (e.g. volume
    commit). ``batch_size`` is ignored.
    """
    del batch_size
    pack = Path(pack_dir)
    labels = [str(label) for label in model_labels if str(label).strip()]
    if not labels:
        return {
            "status": "ok",
            "order": "question,mode,model,shot",
            "by_model": {},
            "n_shots_graded": 0,
            "n_shots_reused": 0,
        }

    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    audio_ok = judge_is_audio_model(handle=handle)
    if modes is None:
        mode_pairs = [
            (
                normalize_grade_prompt(prompt, include_gold=include_gold),
                get_judge_format(prompt, include_gold=include_gold).include_gold,
            )
        ]
    else:
        mode_pairs = list(modes)
    mode_specs: list[tuple[str, bool, str]] = []
    progress_label = _progress_judge_label(handle, model_id)
    for prompt_name, gold_flag in mode_pairs:
        fmt = get_judge_format(prompt_name, include_gold=gold_flag)
        if fmt.audio_included and not audio_ok:
            print(
                f"[{progress_label}] skipping audio format {prompt_name} "
                f"for text judge",
                flush=True,
            )
            continue
        name = normalize_grade_prompt(prompt_name, include_gold=fmt.include_gold)
        require_audio_nongold_judge(
            handle=handle,
            include_gold=fmt.include_gold,
            audio_required=fmt.audio_included,
        )
        key = resolve_grade_judge_key(
            handle,
            prompt=name,
            include_gold=fmt.include_gold,
            judge_key=judge_key if len(mode_pairs) == 1 else None,
        )
        mode_specs.append((name, fmt.include_gold, key))
    if not mode_specs:
        return {
            "status": "ok",
            "order": "question,mode,model,shot",
            "by_model": {},
            "n_shots_graded": 0,
            "n_shots_reused": 0,
            "modes": [],
        }
    n_samples = max(1, int(n_samples))
    fallback_prompt, fallback_gold, fallback_key = mode_specs[0]
    keys = [key for _, _, key in mode_specs]

    files: dict[str, list[dict]] = {}
    by_id: dict[str, dict[str, dict]] = {}
    sidecar_paths: dict[tuple[str, str], Path] = {}
    existing_sidecar: dict[tuple[str, str], dict[tuple[str, int], dict]] = {}
    all_ids: list[str] = []
    for gradee in labels:
        pred = pack / "models" / gradee / "predictions.jsonl"
        records: list[dict] = []
        if pred.is_file():
            with pred.open(encoding="utf-8") as handle_in:
                for line in handle_in:
                    stripped = line.strip()
                    if stripped:
                        records.append(json.loads(stripped))
        for record in records:
            ensure_judge_schema(
                record,
                fallback_label=fallback_key,
                fallback_model_id=model_id,
            )
        files[gradee] = records
        index: dict[str, dict] = {}
        for record in records:
            qid = str(record.get("id") or "").strip()
            if not qid or qid in index:
                continue
            index[qid] = record
            all_ids.append(qid)
        by_id[gradee] = index
        if sidecar:
            for _name, _gold, key in mode_specs:
                path = pred.parent / "judge_partials" / f"{key}.jsonl"
                sidecar_paths[(gradee, key)] = path
                existing_sidecar[(gradee, key)] = _load_sidecar_rows(path)
                if not force:
                    _overlay_sidecar_rows(
                        records,
                        existing_sidecar[(gradee, key)],
                        judge_key=key,
                    )

    allowed_ids = resolve_grade_allowed_ids(
        all_ids, question_ids=question_ids, n_questions=n_questions
    )
    if question_ids is not None:
        qid_order = [str(qid).strip() for qid in question_ids if str(qid).strip()]
    else:
        qid_order = list(dict.fromkeys(all_ids))
    if allowed_ids is not None:
        qid_order = [qid for qid in qid_order if qid in allowed_ids]

    reuse_cache: dict[tuple[str, ...], dict] = {}
    if not force:
        for qid in qid_order:
            for name, gold, key in mode_specs:
                for gradee in labels:
                    record = by_id.get(gradee, {}).get(qid)
                    if record is None:
                        continue
                    question = str(record.get("question") or "")
                    answer = str(record.get("answer") or "")
                    for shot_index, shot in _shots_in_index_order(
                        record, shot_indices
                    ):
                        entry = _shot_judge_entry(shot, key)
                        if entry is None:
                            continue
                        reuse_cache.setdefault(
                            _grade_reuse_key(
                                question,
                                answer,
                                _shot_prediction_text(shot, model_label=gradee),
                                include_gold=gold,
                                prompt=name,
                            ),
                            dict(entry),
                        )

    jobs: list[dict] = []
    owners: list[list[tuple[str, str, int, str]]] = []
    job_keys: list[tuple[str, ...]] = []
    pending_index: dict[tuple[str, ...], int] = {}
    reuse_owners: list[tuple[str, str, int, str, dict]] = []
    for qid in qid_order:
        for name, gold, key in mode_specs:
            for gradee in labels:
                record = by_id.get(gradee, {}).get(qid)
                if record is None:
                    continue
                question = str(record.get("question") or "")
                answer = str(record.get("answer") or "")
                for shot_index, shot in _shots_in_index_order(record, shot_indices):
                    if not force and not _shot_needs_grade(shot, key):
                        continue
                    prediction = _shot_prediction_text(shot, model_label=gradee)
                    cache_key = _grade_reuse_key(
                        question,
                        answer,
                        prediction,
                        include_gold=gold,
                        prompt=name,
                    )
                    cached = reuse_cache.get(cache_key)
                    if cached is not None:
                        reuse_owners.append(
                            (gradee, qid, shot_index, key, cached)
                        )
                        continue
                    idx = pending_index.get(cache_key)
                    if idx is None:
                        pending_index[cache_key] = len(jobs)
                        jobs.append(
                            {
                                "id": qid,
                                "question": question,
                                "answer": answer,
                                "prediction": prediction,
                                "audio_path": record.get("audio_path"),
                                "prompt": name,
                                "include_gold": gold,
                                "judge_key": key,
                            }
                        )
                        owners.append([(gradee, qid, shot_index, key)])
                        job_keys.append(cache_key)
                    else:
                        owners[idx].append((gradee, qid, shot_index, key))

    print(
        f"[grade] order=question,mode,test-taker,shot "
        f"jobs={len(jobs)} questions={len(qid_order)} "
        f"modes={len(mode_specs)} gradees={len(labels)} n_samples={n_samples}"
    )

    graded = 0
    reused = 0
    graded_by = {(gradee, key): 0 for gradee in labels for key in keys}
    reused_by = {(gradee, key): 0 for gradee in labels for key in keys}
    progress_extra = (
        f"modes={len(mode_specs)} gradees={len(labels)}"
    )
    n_prompts = len(jobs)
    started = time.perf_counter()
    if n_prompts == 0:
        print(
            f"[{progress_label}] 0 prompts to generate "
            f"(reused={len(reuse_owners)}) {progress_extra}",
            flush=True,
        )
    else:
        _log_prompt_progress(
            progress_label,
            0,
            n_prompts,
            elapsed_s=0.0,
            extra=progress_extra,
        )
    partials: dict[tuple[str, str], list[dict]] = {
        (gradee, key): [] for gradee in labels for key in keys
    }

    def _find_shot(gradee: str, qid: str, shot_index: int) -> dict | None:
        record = by_id.get(gradee, {}).get(qid)
        if record is None:
            return None
        for idx, shot in _shots_in_index_order(record, None):
            if idx == shot_index:
                return shot
        return None

    def _apply(
        gradee: str, qid: str, shot_index: int, key: str, entry: dict
    ) -> None:
        nonlocal graded
        copied = dict(entry)
        if sidecar:
            partials[(gradee, key)].append(
                {
                    "id": qid,
                    "shot_index": shot_index,
                    "judge_key": key,
                    "entry": copied,
                }
            )
            graded += 1
            return
        shot = _find_shot(gradee, qid, shot_index)
        if shot is None:
            return
        shot.setdefault("judges", {})[key] = copied
        graded += 1

    def _flush_sidecars() -> None:
        if not sidecar:
            return
        wrote = False
        for (gradee, key), rows in partials.items():
            if not rows:
                continue
            path = sidecar_paths[(gradee, key)]
            path.parent.mkdir(parents=True, exist_ok=True)
            combined = dict(existing_sidecar[(gradee, key)])
            for row in rows:
                row_key = _sidecar_row_key(row)
                if row_key[0]:
                    combined[row_key] = row
            existing_sidecar[(gradee, key)] = combined
            write_jsonl(path, list(combined.values()), mode="w")
            rows.clear()
            wrote = True
        if wrote and on_batch is not None:
            on_batch()

    for gradee, qid, shot_index, key, entry in reuse_owners:
        _apply(gradee, qid, shot_index, key, entry)
        reused += 1
        reused_by[(gradee, key)] += 1
        graded_by[(gradee, key)] += 1
    _flush_sidecars()

    if jobs:
        results = grade_shot_batch(
            handle,
            jobs,
            prompt=fallback_prompt,
            include_gold=fallback_gold,
            n_samples=n_samples,
            temperature=temperature,
        )
        for cache_key, owner_list, result in zip(job_keys, owners, results):
            result_prompt = str(result.get("prompt") or fallback_prompt)
            result_gold = (
                bool(result["include_gold"])
                if result.get("include_gold") is not None
                else fallback_gold
            )
            entry = {
                "correct": bool(result["correct"]),
                "verdict": result.get("verdict"),
                "output": result.get("grader_output"),
                "generation": result.get("generation") or "",
                "reasoning": result.get("reasoning") or "",
                "model_id": model_id,
                "prompt": result_prompt,
                "include_gold": result_gold,
            }
            if result.get("samples"):
                entry["samples"] = result["samples"]
                entry["n_samples"] = result.get("n_samples") or n_samples
            reuse_cache[cache_key] = entry
            for extra_index, (gradee, qid, shot_index, key) in enumerate(
                owner_list
            ):
                _apply(gradee, qid, shot_index, key, entry)
                graded_by[(gradee, key)] += 1
                if extra_index:
                    reused += 1
                    reused_by[(gradee, key)] += 1
        _log_prompt_progress(
            progress_label,
            n_prompts,
            n_prompts,
            elapsed_s=time.perf_counter() - started,
            extra=progress_extra,
        )
        _flush_sidecars()

    if not sidecar:
        last_key = keys[-1] if keys else None
        for gradee in labels:
            records = files[gradee]
            pred = pack / "models" / gradee / "predictions.jsonl"
            for record in records:
                qid = str(record.get("id") or "").strip()
                if allowed_ids is not None and qid not in allowed_ids:
                    continue
                existing_primary = record.get("primary_judge")
                if make_primary and last_key:
                    use_primary = last_key
                else:
                    use_primary = existing_primary or primary_judge
                existing = [str(x) for x in (record.get("judges") or []) if x]
                ordered: list[str] = []
                if use_primary:
                    ordered.append(str(use_primary))
                for item in existing:
                    if item not in ordered:
                        ordered.append(item)
                for key in keys:
                    if key not in ordered:
                        ordered.append(key)
                record["judges"] = ordered
                record["scoring"] = "qwen_freeform_judge"
                recompute_multi_judge_scores(record, use_primary)
            if pred.parent.is_dir() or records:
                pred.parent.mkdir(parents=True, exist_ok=True)
                write_jsonl(pred, records, mode="w")

    by_model: dict[str, dict[str, Any]] = {}
    for gradee in labels:
        for name, gold, key in mode_specs:
            sidecar_path = sidecar_paths.get((gradee, key))
            by_model[f"{gradee}/{key}"] = {
                "status": "ok" if files[gradee] else "missing",
                "predictions_path": str(
                    pack / "models" / gradee / "predictions.jsonl"
                ),
                "sidecar_path": str(sidecar_path) if sidecar_path else None,
                "n_records": len(files[gradee]),
                "n_sampled": (
                    sum(1 for qid in qid_order if qid in by_id.get(gradee, {}))
                ),
                "n_questions": n_questions,
                "n_shots_graded": graded_by[(gradee, key)],
                "n_shots_reused": reused_by[(gradee, key)],
                "grader": model_id,
                "judge_label": key,
                "prompt": name,
                "include_gold": gold,
                "replaced": bool(force),
                "n_samples": n_samples,
            }
    return {
        "status": "ok",
        "order": "question,mode,model,shot",
        "pack_dir": str(pack),
        "by_model": by_model,
        "n_shots_graded": graded,
        "n_shots_reused": reused,
        "grader": model_id,
        "judge_label": fallback_key,
        "prompt": fallback_prompt,
        "include_gold": fallback_gold,
        "modes": [
            {"prompt": name, "include_gold": gold, "judge_key": key}
            for name, gold, key in mode_specs
        ],
        "replaced": bool(force),
        "n_samples": n_samples,
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


def validate_judge_formats() -> None:
    """Raise if a ``JUDGE_FORMATS`` entry is missing its prompt."""
    if not JUDGE_FORMATS:
        raise RuntimeError("JUDGE_FORMATS is empty")
    for name, fmt in JUDGE_FORMATS.items():
        if not str(name or "").strip():
            raise RuntimeError("JUDGE_FORMATS has a blank key")
        if not fmt.prompt.strip():
            raise RuntimeError(f"{name} is missing a prompt segment")


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
    prompt: str | None = None,
    audio_wraps: bool = True,
) -> str:
    """Render named ``JUDGE_FORMATS`` with placeholder inputs."""
    validate_judge_formats()
    if prompt and str(prompt).strip():
        names = parse_grade_prompt_list(prompt, include_gold=include_gold)
        selected = [(name, JUDGE_FORMATS[name]) for name in names]
    elif include_gold is None:
        selected = list(iter_judge_formats())
    else:
        flag = bool(include_gold)
        selected = [
            (name, fmt)
            for name, fmt in iter_judge_formats()
            if fmt.include_gold is flag
        ]
    if not selected:
        raise SystemExit("No judge formats matched the requested filter")
    chunks: list[str] = []

    chunks.append(_BANNER)
    chunks.append("JUDGE FORMATS (prompt + closer segments)")
    chunks.append(_BANNER)
    for name, fmt in selected:
        chunks.append(
            f"[{name}]  audio_included={fmt.audio_included}  "
            f"include_gold={fmt.include_gold}"
        )
        chunks.append("prompt:")
        chunks.append(fmt.prompt)
        chunks.append("")
        chunks.append("closer:")
        chunks.append(fmt.closer or "(none)")
        chunks.append("")

    chunks.append(_BANNER)
    chunks.append("FIELD TEMPLATES (variable slots)")
    chunks.append(_BANNER)
    chunks.append(FIELD_QUESTION)
    chunks.append(FIELD_GOLD)
    chunks.append(FIELD_PREDICTION)
    chunks.append("")

    n = len(selected)
    for index, (name, fmt) in enumerate(selected, start=1):
        gold = gold_tag(fmt.include_gold)
        chunks.append(_BANNER)
        chunks.append(
            f"{index}/{n}  name={name}  audio_included={fmt.audio_included}  "
            f"include_gold={fmt.include_gold}  ({gold})"
        )
        chunks.append(_BANNER)
        chunks.append("assembly (top → bottom):")
        chunks.append("  prompt       : preamble")
        if fmt.audio_included:
            chunks.append("  [AUDIO]      : clip inserted here for suite / API audio judges")
            if fmt.include_gold:
                chunks.append("  fields       : Question + Ground truth + Response")
            else:
                chunks.append("  fields       : Question + Response")
        else:
            chunks.append("  fields       : Question + Ground truth + Response")
        chunks.append("  closer       : closer" if fmt.closer else "  closer       : (none)")
        chunks.append("")
        chunks.append("rendered text prompt (API judges and gold vLLM):")
        chunks.append(_RULE)
        chunks.append(fmt.as_text(**PROMPT_PLACEHOLDERS))
        chunks.append(_RULE)
        chunks.append("")

    if audio_wraps and any(fmt.audio_included for _, fmt in selected):
        nongold = [(name, fmt) for name, fmt in selected if fmt.audio_included]
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
            for name, fmt in nongold:
                wrapped = _nongold_audio_prompt_string(
                    sample_label,
                    _adapt_judge_instructions(sample_label, fmt.prompt),
                    fmt.fields(**PROMPT_PLACEHOLDERS),
                    fmt.closer,
                )
                chunks.append(f"----- wrapped  name={name}  nongold -----")
                chunks.append(wrapped)
                chunks.append("")

    return "\n".join(chunks).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Render judge prompts with {question}/{answer}/{prediction} placeholders "
            "for every format in JUDGE_FORMATS."
        )
    )
    gold = parser.add_mutually_exclusive_group()
    gold.add_argument(
        "--gold",
        dest="include_gold",
        action="store_true",
        default=None,
        help="Only render formats that include the benchmark answer.",
    )
    gold.add_argument(
        "--nongold",
        dest="include_gold",
        action="store_false",
        help="Only render formats that omit gold (audio judges).",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Comma-separated JUDGE_FORMATS keys, or 'all'. Default: every format.",
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
            prompt=(args.prompt or "").strip() or None,
            audio_wraps=not args.no_audio_wraps,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
