"""Free-form answer grader for MMAR difficulty experiments (multi-judge)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from audio_flamingo_runtime import resolve_model_dir
from mmar_common import (
    ensure_judge_schema,
    judge_label,
    recompute_multi_judge_scores,
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

# Final-answer tokens (preferred) plus legacy YES/NO from older runs.
# Soft whole-region fallback — keep this narrow so prose like "correct in
# meaning" does not count as a verdict; token labels above still accept
# a bare "correct" / "incorrect" final line.
PASS_RE = re.compile(r"\b(pass|yes|true)\b", re.IGNORECASE)
FAIL_RE = re.compile(r"\b(fail|no|false)\b", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
PASS_LABELS = frozenset({"PASS", "P", "YES", "Y", "TRUE", "CORRECT"})
FAIL_LABELS = frozenset({"FAIL", "F", "NO", "N", "FALSE", "INCORRECT", "WRONG"})


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


def build_grade_prompt(*, question: str, answer: str, prediction: str) -> str:
    return (
        "You are grading a free-form answer to an audio understanding question.\n"
        "Decide whether the model answer is semantically equivalent to the "
        "correct answer. Accept answers where it is clear that the test taker understands "
        "what the correct answer is. If the model answer is empty, nonsense, or contradicts the "
        "correct answer, mark it incorrect.\n\n"
        f"Question: {question}\n"
        f"Correct answer: {answer}\n"
        f"Model answer: {prediction}\n\n"
        "Reason briefly if needed, then end your reply with a single final line "
        'containing only "Pass" or "Fail".'
    )


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
    """Parse a Pass/Fail (or legacy YES/NO) judge reply. Returns None if unparseable."""
    region = _answer_region(text)
    if not region:
        return None
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
    """Short Pass/Fail label for schema ``output`` / tips."""
    if verdict is True:
        return "Pass"
    if verdict is False:
        return "Fail"
    return None


def _shot_needs_grade(shot: dict, judge_key: str) -> bool:
    """True when this judge has no verdict yet for the shot."""
    judges = shot.get("judges")
    if isinstance(judges, dict) and judge_key in judges:
        entry = judges[judge_key]
        if entry is not None and entry.get("correct") is not None:
            return False
    # Legacy flat fields only count for the same judge label/id.
    legacy_id = shot.get("grader")
    if legacy_id and judge_label(legacy_id) == judge_key and shot.get("correct") is not None:
        return False
    return True


def _record_needs_grade(record: dict, judge_key: str) -> bool:
    shots = record.get("shots") or []
    if not shots:
        return False
    return any(_shot_needs_grade(shot, judge_key) for shot in shots)


def load_grader(
    model_id: str = DEFAULT_GRADER_MODEL_ID,
    args: SimpleNamespace | None = None,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    model_id = resolve_judge_model_id(model_id)
    label = judge_label(model_id)
    local_id = resolve_model_dir(model_id, None)
    engine = resolve_judge_engine(model_id, args)
    sampling = resolve_judge_sampling(model_id, args)

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
        "SamplingParams": SamplingParams,
        "sampling": sampling,
    }


def _format_chat(tokenizer: Any, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {user_text}\nAssistant:"


def grade_shot_batch(
    handle: dict[str, Any],
    jobs: list[dict],
    *,
    max_tokens: int | None = None,
) -> list[dict]:
    """Grade a list of ``{question, answer, prediction}`` jobs.

    Returns one result dict per job with ``correct``, ``verdict``,
    ``generation`` (full text), ``grader_output`` (short Pass/Fail), and
    ``grader``.
    """
    if not jobs:
        return []
    tokenizer = handle["tokenizer"]
    prompts = [
        _format_chat(
            tokenizer,
            build_grade_prompt(
                question=str(job.get("question") or ""),
                answer=str(job.get("answer") or ""),
                prediction=str(job.get("prediction") or ""),
            ),
        )
        for job in jobs
    ]
    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    sampling = judge_sampling_params(model_id, max_tokens=max_tokens)
    outputs = handle["llm"].generate(prompts, sampling_params=sampling)
    results: list[dict] = []
    for job, out in zip(jobs, outputs):
        text = ""
        outs = getattr(out, "outputs", None) or []
        if outs:
            text = str(getattr(outs[0], "text", "") or "")
        verdict = parse_grade_verdict(text)
        short = format_grade_output(verdict)
        # Empty / unparseable answers default to incorrect.
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
            }
        )
    return results


def grade_predictions_file(
    predictions_path: Path,
    handle: dict[str, Any],
    *,
    judge_key: str | None = None,
    primary_judge: str | None = None,
    batch_size: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Grade (or re-grade) one judge for all shots in a predictions.jsonl file.

    When ``force`` is True, existing entries for ``judge_key`` are replaced.
    """
    if not predictions_path.exists():
        return {
            "status": "missing",
            "predictions_path": str(predictions_path),
            "n_records": 0,
            "n_shots_graded": 0,
        }

    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    key = judge_key or judge_label(model_id) or GRADER_LABEL
    primary = primary_judge or key
    effective_batch_size = resolve_judge_batch_size(model_id, batch_size)

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

    jobs: list[dict] = []
    owners: list[tuple[int, int]] = []  # (record_index, shot_index)
    for record_index, record in enumerate(records):
        if not force and not _record_needs_grade(record, key):
            continue
        question = str(record.get("question") or "")
        answer = str(record.get("answer") or "")
        for shot in record.get("shots") or []:
            shot_index = int(shot.get("shot_index", 0))
            if not force and not _shot_needs_grade(shot, key):
                continue
            prediction = (
                shot.get("answer_prediction")
                or shot.get("model_output")
                or ""
            )
            jobs.append(
                {
                    "question": question,
                    "answer": answer,
                    "prediction": prediction,
                }
            )
            owners.append((record_index, shot_index))

    graded = 0
    for start in range(0, len(jobs), effective_batch_size):
        chunk = jobs[start : start + effective_batch_size]
        chunk_owners = owners[start : start + effective_batch_size]
        results = grade_shot_batch(handle, chunk)
        for (record_index, shot_index), result in zip(chunk_owners, results):
            record = records[record_index]
            for shot in record.get("shots") or []:
                if int(shot.get("shot_index", -1)) != shot_index:
                    continue
                judges = shot.setdefault("judges", {})
                # Replace any previous entry for this judge key.
                judges[key] = {
                    "correct": bool(result["correct"]),
                    "verdict": result.get("verdict"),
                    "output": result.get("grader_output"),
                    "generation": result.get("generation") or "",
                    "model_id": model_id,
                }
                graded += 1
                break

    for record in records:
        # Keep judge order: primary first, then any previously seen, then this key.
        existing = [str(x) for x in (record.get("judges") or []) if x]
        ordered: list[str] = []
        if primary:
            ordered.append(primary)
        for label in existing:
            if label not in ordered:
                ordered.append(label)
        if key not in ordered:
            ordered.append(key)
        record["judges"] = ordered
        record["primary_judge"] = primary
        record["scoring"] = "qwen_freeform_judge"
        recompute_multi_judge_scores(record, primary)

    write_jsonl(predictions_path, records, mode="w")
    return {
        "status": "ok",
        "predictions_path": str(predictions_path),
        "n_records": len(records),
        "n_shots_graded": graded,
        "grader": model_id,
        "judge_label": key,
        "primary_judge": primary,
        "replaced": bool(force),
    }
