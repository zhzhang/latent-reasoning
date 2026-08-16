"""Shared helpers for MMAR-Rubrics evaluation over existing MC runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation_rubrics import InputItem, string_match
from mmar_common import (
    ensure_rubric_meta,
    load_jsonl,
    summarize_evaluated,
    write_json,
    write_jsonl,
)

SOURCE_EXPERIMENT = "exp-mmar-question-difficulty"
RUBRICS_EXPERIMENT = "exp-mmar-rubrics"
RUBRICS_API_EXPERIMENT = "exp-mmar-rubrics-api"
RUBRIC_META_KEYS = ("thinking", "cue", "rubric")
DEFAULT_MODEL_LABEL = "qwen3-omni"
AF_NEXT_MODEL_LABEL = "af-next-think"
DEFAULT_LIMIT = 100
# Viewer / API composite key: "<source_run_id>::<model_label>"
EVAL_RUN_SEP = "::"


def load_question_ids(run_dir: Path) -> list[str]:
    path = run_dir / "question_ids.json"
    if not path.is_file():
        raise FileNotFoundError(f"question_ids.json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict):
        ids = payload.get("ids")
        if isinstance(ids, list):
            return [str(x) for x in ids]
    raise ValueError(f"Unrecognized question_ids.json format at {path}")


def source_run_dir(results_root: Path, source_run_id: str) -> Path:
    return Path(results_root).expanduser().resolve() / SOURCE_EXPERIMENT / source_run_id


def rubrics_run_dir(results_root: Path, source_run_id: str, *, api: bool = False) -> Path:
    experiment = RUBRICS_API_EXPERIMENT if api else RUBRICS_EXPERIMENT
    return Path(results_root).expanduser().resolve() / experiment / source_run_id


def judge_model_dir(run_dir: Path, judge_label: str, model_label: str) -> Path:
    return run_dir / "judges" / judge_label / "models" / model_label


def discover_test_taker_labels(run_dir: Path, manifest: dict | None = None) -> list[str]:
    """Test-taker model labels present in a rubrics output directory."""
    labels: list[str] = []
    seen: set[str] = set()

    def _add(label: str | None) -> None:
        text = str(label or "").strip()
        if text and text not in seen:
            seen.add(text)
            labels.append(text)

    judges_root = Path(run_dir) / "judges"
    if judges_root.is_dir():
        for path in sorted(judges_root.glob("*/models/*")):
            if path.is_dir():
                _add(path.name)

    manifest = manifest or {}
    for entry in manifest.get("models") or []:
        if isinstance(entry, str):
            _add(entry)
        elif isinstance(entry, dict):
            _add(entry.get("label"))
    _add(manifest.get("model_label"))
    return labels


def eval_run_key(source_run_id: str, model_label: str, kind: str | None = None) -> str:
    """Stable id for one (source run, test-taker[, kind]) eval in the viewer."""
    base = f"{source_run_id}{EVAL_RUN_SEP}{model_label}"
    extra = str(kind or "").strip()
    if extra and extra != "rubrics":
        return f"{base}{EVAL_RUN_SEP}{extra}"
    return base


def parse_eval_run_key(key: str, *, default_model: str = DEFAULT_MODEL_LABEL) -> tuple[str, str]:
    """Split a viewer run id into ``(source_run_id, model_label)``.

    Bare source-run ids (legacy bookmarks) keep the default test-taker.
    A trailing kind (e.g. ``::groundedness``) is ignored here; use
    ``parse_eval_run_parts`` when the kind is needed.
    """
    source_run_id, model_label, _kind = parse_eval_run_parts(
        key, default_model=default_model
    )
    return source_run_id, model_label


def parse_eval_run_parts(
    key: str, *, default_model: str = DEFAULT_MODEL_LABEL
) -> tuple[str, str, str]:
    """Split a viewer run id into ``(source_run_id, model_label, kind)``."""
    text = str(key or "").strip()
    parts = [part.strip() for part in text.split(EVAL_RUN_SEP)]
    parts = [part for part in parts if part]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], "rubrics"
    return text, default_model, "rubrics"


def load_predictions_by_id(predictions_path: Path) -> dict[str, dict]:
    if not predictions_path.is_file():
        raise FileNotFoundError(f"predictions not found: {predictions_path}")
    out: dict[str, dict] = {}
    for record in load_jsonl(predictions_path):
        record_id = record.get("id")
        if record_id:
            out[str(record_id)] = record
    return out


def load_meta_by_id(meta_path: Path) -> dict[str, dict]:
    meta_path = ensure_rubric_meta(meta_path)
    return {str(item["id"]): item for item in load_jsonl(meta_path) if item.get("id")}


def first_shot_fields(record: dict) -> tuple[str, str, int]:
    """Return ``(thinking, answer, shot_index)`` for the first generated shot.

    MC aggregation promotes the first *correct* shot onto the record top-level
    ``thinking_prediction`` / ``answer_prediction`` fields. Rubrics eval scores
    shot 0 so we judge the model's first reasoning trace, not a later retry.
    """
    shots = record.get("shots") or []
    if shots:
        def _shot_index(shot: dict) -> int:
            raw = shot.get("shot_index")
            if raw is None:
                return 10**9
            try:
                return int(raw)
            except (TypeError, ValueError):
                return 10**9

        first = min(shots, key=_shot_index)
        idx = first.get("shot_index")
        try:
            shot_index = int(idx) if idx is not None else 0
        except (TypeError, ValueError):
            shot_index = 0
        return (
            str(first.get("thinking_prediction") or ""),
            str(first.get("answer_prediction") or ""),
            shot_index,
        )
    return (
        str(record.get("thinking_prediction") or ""),
        str(record.get("answer_prediction") or ""),
        0,
    )


def primary_shot_fields(record: dict) -> tuple[str, str]:
    """Back-compat wrapper: first-shot thinking/answer only."""
    thinking, answer, _shot_index = first_shot_fields(record)
    return thinking, answer


def merge_rubric_fields(record: dict, meta: dict | None) -> dict:
    """Attach thinking/cue/rubric from meta onto a prediction record."""
    out = dict(record)
    if not meta:
        return out
    for key, value in meta.items():
        if key in RUBRIC_META_KEYS or key not in out:
            out[key] = value
    return out


def build_input_item(record: dict, *, source_model: str | None = None) -> dict[str, Any]:
    thinking_prediction, answer_prediction, shot_index = first_shot_fields(record)
    item = {
        "id": record["id"],
        "question": record.get("question") or "",
        "answer": record.get("answer") or "",
        "thinking": record.get("thinking") or "",
        "cue": record.get("cue") or [],
        "choices": record.get("choices") or [],
        "rubric": record.get("rubric") or [],
        "thinking_prediction": thinking_prediction,
        "answer_prediction": answer_prediction,
        "shot_index": shot_index,
        "modality": record.get("modality") or "unknown",
        "category": record.get("category") or "unknown",
    }
    if source_model:
        item["source_model"] = source_model
    for optional in (
        "sub-category",
        "audio_path",
        "language",
        "source",
        "timestamp",
        "url",
        "model_output",
        "n_shots",
        "shots",
    ):
        if optional in record:
            item[optional] = record[optional]
    return item


def build_rubric_input_items(
    source_dir: Path,
    *,
    model_label: str = DEFAULT_MODEL_LABEL,
    meta_path: Path,
    limit: int = DEFAULT_LIMIT,
    question_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build InputItem-shaped records for the first ``limit`` question ids."""
    all_ids = question_ids if question_ids is not None else load_question_ids(source_dir)
    selected_ids = list(all_ids[: max(0, int(limit))])
    predictions_path = source_dir / "models" / model_label / "predictions.jsonl"
    preds = load_predictions_by_id(predictions_path)
    meta_by_id = load_meta_by_id(meta_path)

    items: list[dict[str, Any]] = []
    missing: list[str] = []
    for record_id in selected_ids:
        pred = preds.get(record_id)
        if pred is None:
            missing.append(record_id)
            continue
        merged = merge_rubric_fields(pred, meta_by_id.get(record_id))
        item = build_input_item(merged, source_model=model_label)
        required = InputItem.__required_keys__
        if not required <= set(item.keys()) or not item.get("rubric"):
            missing.append(record_id)
            continue
        items.append(item)

    if missing and len(items) == 0:
        raise SystemExit(
            f"No usable predictions for model={model_label} under {predictions_path}. "
            f"Missing/invalid ids sample: {missing[:5]}"
        )
    if missing:
        print(
            f"[mmar_rubrics] skipping {len(missing)} ids without usable "
            f"prediction/rubric meta (sample={missing[:3]})"
        )
    return items, selected_ids


def is_fully_graded(item: dict) -> bool:
    """True when a rubric judge actually ran (has a raw LLM response)."""
    if not item.get("id") or "score" not in item:
        return False
    raw = item.get("raw_responses") or []
    return any(str(r or "").strip() for r in raw)


def load_completed_ids(evaluated_path: Path) -> set[str]:
    if not evaluated_path.is_file() or evaluated_path.stat().st_size == 0:
        return set()
    return {
        str(item["id"])
        for item in load_jsonl(evaluated_path)
        if is_fully_graded(item)
    }


def prune_incomplete_evaluations(evaluated_path: Path) -> int:
    """Drop short-circuited / incomplete rows so they can be rejudged.

    Returns how many incomplete rows were removed.
    """
    if not evaluated_path.is_file() or evaluated_path.stat().st_size == 0:
        return 0
    records = load_jsonl(evaluated_path)
    keep = [item for item in records if is_fully_graded(item)]
    removed = len(records) - len(keep)
    if removed <= 0:
        return 0
    # Dedupe by id, last complete wins.
    by_id: dict[str, dict] = {}
    for item in keep:
        by_id[str(item["id"])] = item
    write_jsonl(evaluated_path, list(by_id.values()), mode="w")
    return removed


def evaluated_record_from_result(item: dict, result) -> dict[str, Any]:
    return {
        **item,
        "new": True,
        "score": result.score,
        "correct": result.correct,
        "raw_responses": result.raw_responses,
        "rubric_results": result.rubric_results,
    }


def write_rubrics_manifest(
    run_dir: Path,
    *,
    source_run_id: str,
    model_label: str,
    question_ids: list[str],
    judge_label: str,
    judge_model_id: str,
    num_raters: int,
    limit: int,
    backend: str,
    experiment: str | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    judges = {
        str(entry.get("label")): dict(entry)
        for entry in (existing.get("judges") or [])
        if isinstance(entry, dict) and entry.get("label")
    }
    judges[judge_label] = {
        "label": judge_label,
        "model_id": judge_model_id,
        "backend": backend,
        "num_raters": num_raters,
    }

    models: dict[str, dict[str, Any]] = {}
    for entry in existing.get("models") or []:
        if isinstance(entry, str) and entry.strip():
            models[entry.strip()] = {"label": entry.strip()}
        elif isinstance(entry, dict) and entry.get("label"):
            models[str(entry["label"])] = dict(entry)
    legacy_label = existing.get("model_label")
    if legacy_label and str(legacy_label) not in models:
        models[str(legacy_label)] = {"label": str(legacy_label)}
    models[model_label] = {
        **(models.get(model_label) or {}),
        "label": model_label,
        "shot": "first",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    inferred_experiment = experiment
    if inferred_experiment is None:
        parent_name = run_dir.parent.name
        if parent_name.startswith("exp-mmar-rubrics"):
            inferred_experiment = parent_name
        else:
            inferred_experiment = existing.get("experiment") or RUBRICS_EXPERIMENT

    payload = {
        **existing,
        "experiment": inferred_experiment,
        "source_run_id": source_run_id,
        "source_experiment": SOURCE_EXPERIMENT,
        "model_label": model_label,
        "models": list(models.values()),
        "shot": "first",
        "limit": limit,
        "n_questions": len(question_ids),
        "num_raters": num_raters,
        "scoring": "mmar_rubrics_single_rater" if num_raters == 1 else "mmar_rubrics",
        "judges": list(judges.values()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if "created_at" not in payload:
        payload["created_at"] = payload["updated_at"]
    write_json(path, payload)
    write_json(
        run_dir / "question_ids.json",
        {
            "source_run_id": source_run_id,
            "limit": limit,
            "n": len(question_ids),
            "ids": question_ids,
        },
    )
    return path


def write_judge_scores(
    model_dir: Path,
    evaluated_path: Path,
    *,
    judge_label: str,
    judge_model_id: str,
) -> dict:
    summary = summarize_evaluated(evaluated_path)
    summary.update(
        {
            "judge_label": judge_label,
            "judge_model_id": judge_model_id,
            "scoring": "mmar_rubrics_single_rater",
            "evaluated_path": str(evaluated_path),
        }
    )
    write_json(model_dir / "scores.json", summary)
    return summary


def partition_by_string_match(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split items into (string_match_pass, string_match_failed).

    Both groups are judged by the rubric LLM; this is only for metrics /
    logging.
    """
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for item in items:
        if string_match(item["answer"], item["answer_prediction"], item["choices"]):
            passed.append(item)
        else:
            failed.append(item)
    return passed, failed


def append_evaluated(evaluated_path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    write_jsonl(evaluated_path, records, mode="a")
