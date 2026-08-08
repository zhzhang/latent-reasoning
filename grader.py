"""Free-form answer grader for MMAR difficulty experiments (multi-judge)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from audio_flamingo_runtime import resolve_model_dir
from mmar_common import (
    ensure_judge_schema,
    judge_label,
    recompute_multi_judge_scores,
    write_jsonl,
)

DEFAULT_JUDGE_MODEL_IDS = ("Qwen/Qwen2.5-3B-Instruct",)
# Back-compat aliases.
DEFAULT_GRADER_MODEL_ID = DEFAULT_JUDGE_MODEL_IDS[0]
GRADER_LABEL = judge_label(DEFAULT_GRADER_MODEL_ID)

YES_RE = re.compile(r"\b(yes|true|correct|match(?:es)?)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|false|incorrect|wrong|mismatch(?:es)?)\b", re.IGNORECASE)


def build_grade_prompt(*, question: str, answer: str, prediction: str) -> str:
    return (
        "You are grading a free-form answer to an audio understanding question.\n"
        "Decide whether the model answer is semantically equivalent to the "
        "correct answer. Minor wording differences are fine; the meaning must "
        "match. If the model answer is empty, nonsense, or contradicts the "
        "correct answer, mark it incorrect.\n\n"
        f"Question: {question}\n"
        f"Correct answer: {answer}\n"
        f"Model answer: {prediction}\n\n"
        "Reply with only YES or NO."
    )


def parse_grade_verdict(text: str) -> bool | None:
    """Parse a YES/NO grader reply. Returns None if unparseable."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    # Prefer the last non-empty line (models sometimes preamble).
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    candidate = lines[-1] if lines else cleaned
    # Strip common wrappers.
    candidate = candidate.strip("`\"' ").upper()
    if candidate in {"Y", "YES", "TRUE", "CORRECT"}:
        return True
    if candidate in {"N", "NO", "FALSE", "INCORRECT", "WRONG"}:
        return False
    has_yes = bool(YES_RE.search(candidate))
    has_no = bool(NO_RE.search(candidate))
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    # Fall back to full text.
    has_yes = bool(YES_RE.search(cleaned))
    has_no = bool(NO_RE.search(cleaned))
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
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


def load_grader(model_id: str = DEFAULT_GRADER_MODEL_ID) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    local_id = resolve_model_dir(model_id, None)
    llm = LLM(
        model=local_id,
        dtype="bfloat16",
        # Cap judge context (not a KV/throughput lever).
        max_model_len=4096,
        max_num_batched_tokens=8192,
        enforce_eager=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    print(f"Freeform grader ready: {model_id} ({local_id})")
    return {
        "llm": llm,
        "tokenizer": tokenizer,
        "model_id": model_id,
        "SamplingParams": SamplingParams,
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
    max_tokens: int = 8,
) -> list[dict]:
    """Grade a list of ``{question, answer, prediction}`` jobs.

    Returns one result dict per job with ``correct``, ``grader_output``,
    and ``grader``.
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
    sampling = handle["SamplingParams"](
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(max_tokens),
        seed=0,
    )
    outputs = handle["llm"].generate(prompts, sampling_params=sampling)
    results: list[dict] = []
    model_id = handle.get("model_id") or DEFAULT_GRADER_MODEL_ID
    for job, out in zip(jobs, outputs):
        text = ""
        outs = getattr(out, "outputs", None) or []
        if outs:
            text = str(getattr(outs[0], "text", "") or "")
        verdict = parse_grade_verdict(text)
        # Empty / unparseable answers default to incorrect.
        correct = bool(verdict) if verdict is not None else False
        results.append(
            {
                "correct": correct,
                "grader_output": text,
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
    batch_size: int = 64,
    force: bool = False,
) -> dict[str, Any]:
    """Grade (or re-grade) one judge for all shots in a predictions.jsonl file."""
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
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        chunk_owners = owners[start : start + batch_size]
        results = grade_shot_batch(handle, chunk)
        for (record_index, shot_index), result in zip(chunk_owners, results):
            record = records[record_index]
            for shot in record.get("shots") or []:
                if int(shot.get("shot_index", -1)) != shot_index:
                    continue
                judges = shot.setdefault("judges", {})
                judges[key] = {
                    "correct": bool(result["correct"]),
                    "output": result["grader_output"],
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
    }
