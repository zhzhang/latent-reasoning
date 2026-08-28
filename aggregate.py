"""Aggregate per-model MMAR n-shot predictions into a difficulty ranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default experiment model set (used by run_experiment aggregation).
MODEL_LABELS = (
    "af-next-think",
    "mimo-audio-7b",
    "interactive-omni-8b",
    "qwen3-omni",
    "voxtral-small-24b",
)
# Preferred display / discovery order; extras append after known labels.
MODEL_LABEL_ORDER = MODEL_LABELS + (
    "music-flamingo",
    "qwen2.5-omni-7b",
    "phi-4-multimodal",
    "gemma-4-e4b",
    "gemma-4-12b",
    "qwen3-omni-instruct",
    "nemotron-3-nano-omni",
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


def order_model_labels(labels: list[str] | tuple[str, ...]) -> list[str]:
    found = list(dict.fromkeys(str(x) for x in labels if x))
    known = [label for label in MODEL_LABEL_ORDER if label in found]
    rest = [label for label in found if label not in MODEL_LABEL_ORDER]
    return known + rest


def discover_model_labels(
    run_dir: Path,
    *,
    manifest: dict | None = None,
    fallback: tuple[str, ...] | list[str] = MODEL_LABELS,
) -> list[str]:
    """Models that have ``predictions.jsonl`` on disk (else manifest / fallback)."""
    labels: list[str] = []
    models_root = Path(run_dir) / "models"
    if models_root.is_dir():
        for child in sorted(models_root.iterdir()):
            if child.is_dir() and (child / "predictions.jsonl").is_file():
                labels.append(child.name)
    if labels:
        return order_model_labels(labels)
    from_manifest = [str(x) for x in (manifest or {}).get("models") or []]
    if from_manifest:
        return order_model_labels(from_manifest)
    return order_model_labels(list(fallback))


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


def _record_rate(record: dict) -> float | None:
    """Return shot success rate, or None when the record is still unscored."""
    rate = record.get("shot_success_rate")
    if rate is not None:
        return float(rate)
    if record.get("pending_grade"):
        return None
    n_shots = record.get("n_shots")
    n_correct = record.get("n_shot_correct")
    if n_shots and n_correct is not None:
        return float(n_correct) / float(n_shots)
    shots = record.get("shots") or []
    if shots and all(shot.get("correct") is not None for shot in shots):
        return sum(1 for shot in shots if shot.get("correct")) / len(shots)
    return None


def _collect_judge_meta(
    manifest: dict,
    by_model: dict[str, dict[str, dict]],
) -> tuple[list[dict], str | None]:
    """Return ``(judges entries, primary_judge label)`` for scores.json."""
    entries: list[dict] = []
    seen: set[str] = set()
    primary = manifest.get("primary_judge")

    for raw in manifest.get("judges") or []:
        if isinstance(raw, dict):
            label = str(raw.get("label") or "")
            if not label or label in seen:
                continue
            seen.add(label)
            entries.append(
                {
                    "label": label,
                    "model_id": raw.get("model_id"),
                    "primary": bool(raw.get("primary")) or label == primary,
                }
            )
        elif raw:
            label = str(raw)
            if label in seen:
                continue
            seen.add(label)
            entries.append(
                {
                    "label": label,
                    "model_id": None,
                    "primary": label == primary,
                }
            )

    # Fall back to scanning prediction records.
    if not entries:
        for preds in by_model.values():
            for record in preds.values():
                for label in record.get("judges") or []:
                    if label and label not in seen:
                        seen.add(str(label))
                        entries.append(
                            {
                                "label": str(label),
                                "model_id": None,
                                "primary": False,
                            }
                        )
                for shot in record.get("shots") or []:
                    for label in (shot.get("judges") or {}):
                        if label and label not in seen:
                            seen.add(str(label))
                            entries.append(
                                {
                                    "label": str(label),
                                    "model_id": ((shot.get("judges") or {}).get(label) or {}).get(
                                        "model_id"
                                    ),
                                    "primary": False,
                                }
                            )

    if not primary and entries:
        primary = entries[0]["label"]
    for entry in entries:
        entry["primary"] = entry["label"] == primary
    # Ensure primary is first.
    if primary:
        entries.sort(key=lambda e: (0 if e["label"] == primary else 1, e["label"]))
    return entries, primary


def aggregate_difficulty(
    run_dir: Path | str,
    *,
    question_ids: list[str] | None = None,
    model_labels: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Write ``difficulty.jsonl`` (hardest first) and ``scores.json``.

    Per-question ``avg_success_rate`` is the mean of each model's
    ``shot_success_rate``. Questions missing a model contribution are
    still included using the mean over available scored models. Ungraded
    freeform records (``pending_grade`` / null rate) are excluded from
    averages rather than treated as 0%. Ranking uses the primary judge's
    canonical rates; per-judge breakdowns are stored alongside.
    """
    run_path = Path(run_dir)
    manifest: dict = {}
    manifest_path = run_path / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    labels = (
        order_model_labels(list(model_labels))
        if model_labels is not None
        else discover_model_labels(run_path, manifest=manifest)
    )
    by_model = load_model_predictions(run_path, labels)
    judge_entries, primary_judge = _collect_judge_meta(manifest, by_model)

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
        for label in labels:
            record = by_model.get(label, {}).get(qid)
            if record is None:
                per_model[label] = {
                    "shot_success_rate": None,
                    "n_shot_correct": None,
                    "n_shots": None,
                    "correct": None,
                    "missing": True,
                    "pending_grade": False,
                    "per_judge": {},
                }
                continue
            if base is None:
                base = record
            rate = _record_rate(record)
            pending = bool(record.get("pending_grade")) or rate is None
            per_model[label] = {
                "shot_success_rate": rate,
                "n_shot_correct": record.get("n_shot_correct"),
                "n_shots": record.get("n_shots"),
                "correct": record.get("correct"),
                "answer_prediction": record.get("answer_prediction"),
                "missing": False,
                "pending_grade": pending and rate is None,
                "per_judge": dict(record.get("per_judge") or {}),
                "primary_judge": record.get("primary_judge") or primary_judge,
            }
            if rate is not None:
                rates.append(rate)

        avg = (sum(rates) / len(rates)) if rates else None
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

    # Hardest first among scored questions; unscored (all pending) last.
    difficulty_rows.sort(
        key=lambda r: (
            0 if r["n_models_scored"] else 1,
            float(r["avg_success_rate"]) if r["avg_success_rate"] is not None else 1.0,
            str(r["id"]),
        )
    )

    # Per-model summary over the fixed question set.
    model_summaries: dict[str, Any] = {}
    for label in labels:
        preds = by_model.get(label, {})
        rates: list[float] = []
        n_correct = 0
        n_pending = 0
        per_judge_rates: dict[str, list[float]] = {
            e["label"]: [] for e in judge_entries
        }
        per_judge_correct: dict[str, int] = {e["label"]: 0 for e in judge_entries}
        for qid in question_ids:
            record = preds.get(qid)
            if record is None:
                continue
            rate = _record_rate(record)
            if rate is None:
                n_pending += 1
                continue
            rates.append(rate)
            n_correct += int(bool(record.get("correct")))
            for jlabel, stats in (record.get("per_judge") or {}).items():
                jrate = stats.get("shot_success_rate")
                if jrate is None:
                    continue
                per_judge_rates.setdefault(jlabel, []).append(float(jrate))
                per_judge_correct[jlabel] = per_judge_correct.get(jlabel, 0) + int(
                    bool(stats.get("correct"))
                )
        n = len(rates)
        by_judge: dict[str, Any] = {}
        for jlabel, jrates in per_judge_rates.items():
            jn = len(jrates)
            by_judge[jlabel] = {
                "n": jn,
                "accuracy": (per_judge_correct.get(jlabel, 0) / jn) if jn else None,
                "avg_shot_success_rate": (sum(jrates) / jn) if jn else None,
            }
        model_summaries[label] = {
            "n": n,
            "n_pending": n_pending,
            "accuracy": (n_correct / n) if n else None,
            "avg_shot_success_rate": (sum(rates) / n) if n else None,
            "missing": len(question_ids) - n - n_pending,
            "per_judge": by_judge,
        }

    overall_rates = [
        r["avg_success_rate"]
        for r in difficulty_rows
        if r["n_models_scored"] and r["avg_success_rate"] is not None
    ]
    # Prefer per-record scoring stamp (freeform judge) when present.
    scoring = "mean_shot_success_rate_string_match"
    for label in labels:
        for record in by_model.get(label, {}).values():
            if record.get("scoring"):
                scoring = f"mean_shot_success_rate_{record['scoring']}"
                break
        else:
            continue
        break
    if manifest.get("scoring"):
        scoring = str(manifest["scoring"])

    scores = {
        "n_questions": len(difficulty_rows),
        "n_models": len(labels),
        "model_labels": list(labels),
        "scoring": scoring,
        "sort": "avg_success_rate_asc_hardest_first",
        "avg_success_rate": (
            sum(overall_rates) / len(overall_rates) if overall_rates else None
        ),
        "by_model": model_summaries,
        "n_questions_scored": len(overall_rates),
        "n_questions_pending": len(difficulty_rows) - len(overall_rates),
    }
    if manifest.get("mode"):
        scores["mode"] = manifest["mode"]
    if manifest.get("grader_model_id"):
        scores["grader_model_id"] = manifest["grader_model_id"]
    if judge_entries:
        scores["judges"] = judge_entries
    if primary_judge:
        scores["primary_judge"] = primary_judge
    if manifest.get("source_run_id"):
        scores["source_run_id"] = manifest["source_run_id"]
    if manifest.get("n_shots") is not None:
        scores["n_shots"] = manifest["n_shots"]

    write_jsonl(run_path / "difficulty.jsonl", difficulty_rows)
    write_json(run_path / "scores.json", scores)
    return {
        "n_questions": len(difficulty_rows),
        "difficulty_path": str(run_path / "difficulty.jsonl"),
        "scores_path": str(run_path / "scores.json"),
        "scores": scores,
    }
