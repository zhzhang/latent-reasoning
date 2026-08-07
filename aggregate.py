"""Aggregate per-model MMAR n-shot predictions into a difficulty ranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_LABELS = (
    "af-next-think",
    "mimo-audio-7b",
    "interactive-omni-8b",
    "qwen3-omni",
    "voxtral-small-24b",
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, indent=2, fp=handle, ensure_ascii=False)
        handle.write("\n")
    return path


def _index_by_id(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for record in records:
        record_id = record.get("id")
        if record_id:
            out[str(record_id)] = record
    return out


def load_model_predictions(
    run_dir: Path,
    model_labels: tuple[str, ...] | list[str] = MODEL_LABELS,
) -> dict[str, dict[str, dict]]:
    """Return ``{model_label: {question_id: prediction_record}}``."""
    models_root = run_dir / "models"
    by_model: dict[str, dict[str, dict]] = {}
    for label in model_labels:
        path = models_root / label / "predictions.jsonl"
        by_model[label] = _index_by_id(load_jsonl(path))
    return by_model


def aggregate_difficulty(
    run_dir: Path | str,
    *,
    question_ids: list[str] | None = None,
    model_labels: tuple[str, ...] | list[str] = MODEL_LABELS,
) -> dict[str, Any]:
    """Write ``difficulty.jsonl`` (hardest first) and ``scores.json``.

    Per-question ``avg_success_rate`` is the mean of each model's
    ``shot_success_rate``. Questions missing a model contribution are
    still included using the mean over available models.
    """
    run_path = Path(run_dir)
    by_model = load_model_predictions(run_path, model_labels)

    ids_path = run_path / "question_ids.json"
    if question_ids is None:
        if ids_path.exists():
            payload = json.loads(ids_path.read_text(encoding="utf-8"))
            question_ids = [str(x) for x in payload.get("ids", payload)]
        else:
            # Fall back to union of prediction ids.
            union: set[str] = set()
            for preds in by_model.values():
                union.update(preds)
            question_ids = sorted(union)

    # Prefer meta fields from the first available model record.
    difficulty_rows: list[dict] = []
    for qid in question_ids:
        per_model: dict[str, Any] = {}
        rates: list[float] = []
        base: dict | None = None
        for label in model_labels:
            record = by_model.get(label, {}).get(qid)
            if record is None:
                per_model[label] = {
                    "shot_success_rate": None,
                    "n_shot_correct": None,
                    "n_shots": None,
                    "correct": None,
                    "missing": True,
                }
                continue
            if base is None:
                base = record
            rate = float(record.get("shot_success_rate") or 0.0)
            rates.append(rate)
            per_model[label] = {
                "shot_success_rate": rate,
                "n_shot_correct": record.get("n_shot_correct"),
                "n_shots": record.get("n_shots"),
                "correct": record.get("correct"),
                "answer_prediction": record.get("answer_prediction"),
                "missing": False,
            }

        avg = (sum(rates) / len(rates)) if rates else 0.0
        row = {
            "id": qid,
            "avg_success_rate": avg,
            "n_models_scored": len(rates),
            "per_model": per_model,
            "question": (base or {}).get("question"),
            "choices": (base or {}).get("choices"),
            "answer": (base or {}).get("answer"),
            "audio_path": (base or {}).get("audio_path"),
            "modality": (base or {}).get("modality"),
            "category": (base or {}).get("category"),
            "sub-category": (base or {}).get("sub-category"),
            "language": (base or {}).get("language"),
        }
        difficulty_rows.append(row)

    difficulty_rows.sort(key=lambda r: (r["avg_success_rate"], str(r["id"])))

    # Per-model summary over the fixed question set.
    model_summaries: dict[str, Any] = {}
    for label in model_labels:
        preds = by_model.get(label, {})
        rates = []
        n_correct = 0
        for qid in question_ids:
            record = preds.get(qid)
            if record is None:
                continue
            rates.append(float(record.get("shot_success_rate") or 0.0))
            n_correct += int(bool(record.get("correct")))
        n = len(rates)
        model_summaries[label] = {
            "n": n,
            "accuracy": (n_correct / n) if n else None,
            "avg_shot_success_rate": (sum(rates) / n) if n else None,
            "missing": len(question_ids) - n,
        }

    overall_rates = [r["avg_success_rate"] for r in difficulty_rows if r["n_models_scored"]]
    # Prefer per-record scoring stamp (freeform judge) when present.
    scoring = "mean_shot_success_rate_string_match"
    for label in model_labels:
        for record in by_model.get(label, {}).values():
            if record.get("scoring"):
                scoring = f"mean_shot_success_rate_{record['scoring']}"
                break
        else:
            continue
        break
    scores = {
        "n_questions": len(difficulty_rows),
        "n_models": len(model_labels),
        "model_labels": list(model_labels),
        "scoring": scoring,
        "sort": "avg_success_rate_asc_hardest_first",
        "avg_success_rate": (
            sum(overall_rates) / len(overall_rates) if overall_rates else None
        ),
        "by_model": model_summaries,
    }

    write_jsonl(run_path / "difficulty.jsonl", difficulty_rows)
    write_json(run_path / "scores.json", scores)
    return {
        "n_questions": len(difficulty_rows),
        "difficulty_path": str(run_path / "difficulty.jsonl"),
        "scores_path": str(run_path / "scores.json"),
        "scores": scores,
    }
