"""Migrate existing experiment outputs to the multi-judge schema.

Schema-only (local, no GPU): lifts legacy flat ``grader`` / ``grader_output``
fields into ``shot["judges"]``, synthesizes a ``string-match`` judge for MC
runs, recomputes ``per_judge`` aggregates, updates ``manifest.json``, and
re-aggregates ``difficulty.jsonl`` / ``scores.json``.

Usage:

    uv run python retrofit_judges.py
    uv run python retrofit_judges.py --run 20260807T145000Z --dry-run
    uv run python retrofit_judges.py --set-primary qwen2.5-3b-instruct --backup
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from aggregate import aggregate_difficulty, discover_model_labels
from mmar_common import (
    STRING_MATCH_JUDGE_LABEL,
    ensure_judge_schema,
    judge_label,
    recompute_multi_judge_scores,
    write_json,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs"
EXPERIMENT_SUBDIR = "exp-mmar-question-difficulty"


def resolve_run_roots(results_dir: Path) -> list[Path]:
    results_dir = results_dir.expanduser().resolve()
    roots: list[Path] = []
    exp = results_dir / EXPERIMENT_SUBDIR
    if exp.is_dir():
        roots.append(exp)
    if results_dir.name == EXPERIMENT_SUBDIR and results_dir.is_dir():
        if results_dir not in roots:
            roots.append(results_dir)
    elif results_dir.is_dir() and any(
        (child / "manifest.json").is_file() or (child / "models").is_dir()
        for child in results_dir.iterdir()
        if child.is_dir()
    ):
        if results_dir not in roots:
            roots.append(results_dir)
    return roots


def discover_run_dirs(results_dir: Path, run_ids: list[str] | None) -> list[Path]:
    if run_ids:
        found: list[Path] = []
        for run_id in run_ids:
            matched = None
            for root in resolve_run_roots(results_dir):
                candidate = root / run_id
                if candidate.is_dir():
                    matched = candidate
                    break
            if matched is None:
                raise SystemExit(f"Run not found: {run_id} under {results_dir}")
            found.append(matched)
        return found

    runs: list[Path] = []
    for root in resolve_run_roots(results_dir):
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "manifest.json").is_file() or (child / "models").is_dir():
                runs.append(child)
    return runs


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    items: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                items.append(json.loads(stripped))
    return items


def _infer_mode(manifest: dict, records: list[dict]) -> str:
    mode = str(manifest.get("mode") or "").strip().lower()
    if mode in {"freeform", "free_form", "free-form", "open"}:
        return "freeform"
    if mode in {"mc", "multiple_choice", "multiple-choice", "mcq", "choice"}:
        return "mc"
    for record in records:
        scoring = str(record.get("scoring") or "").lower()
        if "freeform" in scoring or record.get("grader"):
            return "freeform"
        for shot in record.get("shots") or []:
            if shot.get("grader") or shot.get("grader_output") or shot.get("pending_grade"):
                return "freeform"
            judges = shot.get("judges") or {}
            if any(label != STRING_MATCH_JUDGE_LABEL for label in judges):
                return "freeform"
    return "mc"


def _collect_judge_entries(
    records: list[dict],
    *,
    manifest: dict,
    primary: str | None,
) -> list[dict]:
    ordered: list[str] = []
    model_ids: dict[str, str | None] = {}

    for raw in manifest.get("judges") or []:
        if isinstance(raw, dict) and raw.get("label"):
            label = str(raw["label"])
            if label not in ordered:
                ordered.append(label)
            model_ids[label] = raw.get("model_id")
        elif raw:
            label = str(raw)
            if label not in ordered:
                ordered.append(label)

    for record in records:
        for label in record.get("judges") or []:
            if label not in ordered:
                ordered.append(str(label))
        for shot in record.get("shots") or []:
            for label, entry in (shot.get("judges") or {}).items():
                if label not in ordered:
                    ordered.append(str(label))
                if isinstance(entry, dict) and entry.get("model_id"):
                    model_ids.setdefault(label, entry.get("model_id"))
        if record.get("grader"):
            label = judge_label(record.get("grader"))
            if label and label not in ordered:
                ordered.append(label)
            if label:
                model_ids.setdefault(label, record.get("grader"))

    if not ordered and manifest.get("grader_model_id"):
        label = judge_label(manifest["grader_model_id"])
        ordered.append(label)
        model_ids[label] = manifest["grader_model_id"]

    if primary and primary not in ordered:
        ordered.insert(0, primary)
    if primary:
        ordered = [primary] + [x for x in ordered if x != primary]

    return [
        {
            "label": label,
            "model_id": model_ids.get(label),
            "primary": label == primary,
        }
        for label in ordered
    ]


def migrate_predictions_file(
    path: Path,
    *,
    primary: str | None,
    fallback_label: str | None,
    fallback_model_id: str | None,
    dry_run: bool,
    backup: bool,
) -> dict[str, Any]:
    records = load_jsonl(path)
    if not records:
        return {"path": str(path), "n_records": 0, "changed": False}

    before = json.dumps(records, sort_keys=True, ensure_ascii=False)
    for record in records:
        ensure_judge_schema(
            record,
            fallback_label=fallback_label,
            fallback_model_id=fallback_model_id,
        )
        # Prefer explicit primary; else keep existing / first discovered.
        record_primary = primary or record.get("primary_judge")
        if primary:
            ordered = [primary] + [
                x for x in (record.get("judges") or []) if x != primary
            ]
            # Also include any shot-level judges not yet listed.
            for shot in record.get("shots") or []:
                for label in (shot.get("judges") or {}):
                    if label not in ordered:
                        ordered.append(label)
            record["judges"] = ordered
            record["primary_judge"] = primary
        recompute_multi_judge_scores(record, record_primary)

    after = json.dumps(records, sort_keys=True, ensure_ascii=False)
    changed = before != after
    if changed and not dry_run:
        if backup:
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(path, bak)
        write_jsonl(path, records, mode="w")
    return {
        "path": str(path),
        "n_records": len(records),
        "changed": changed,
        "primary_judge": primary or (records[0].get("primary_judge") if records else None),
        "judges": (records[0].get("judges") if records else []),
    }


def retrofit_run(
    run_dir: Path,
    *,
    set_primary: str | None,
    dry_run: bool,
    backup: bool,
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path)
    model_labels = discover_model_labels(run_dir, manifest=manifest)

    sample_records: list[dict] = []
    for label in model_labels:
        preds = load_jsonl(run_dir / "models" / label / "predictions.jsonl")
        sample_records.extend(preds[:2])
        if len(sample_records) >= 4:
            break
    mode = _infer_mode(manifest, sample_records)

    fallback_model_id = manifest.get("grader_model_id")
    fallback_label = (
        judge_label(fallback_model_id)
        if fallback_model_id
        else (STRING_MATCH_JUDGE_LABEL if mode == "mc" else "qwen2.5-3b-instruct")
    )
    primary = set_primary or manifest.get("primary_judge") or fallback_label

    per_model: dict[str, Any] = {}
    all_records: list[dict] = []
    for label in model_labels:
        path = run_dir / "models" / label / "predictions.jsonl"
        result = migrate_predictions_file(
            path,
            primary=primary if mode == "freeform" or set_primary else (
                STRING_MATCH_JUDGE_LABEL if mode == "mc" else primary
            ),
            fallback_label=fallback_label,
            fallback_model_id=fallback_model_id,
            dry_run=dry_run,
            backup=backup,
        )
        per_model[label] = result
        all_records.extend(load_jsonl(path) if not dry_run else [])

    # For dry-run, re-read original and simulate on a copy for manifest metadata.
    if dry_run:
        for label in model_labels:
            path = run_dir / "models" / label / "predictions.jsonl"
            records = load_jsonl(path)
            for record in records:
                ensure_judge_schema(
                    record,
                    fallback_label=fallback_label,
                    fallback_model_id=fallback_model_id,
                )
                recompute_multi_judge_scores(
                    record,
                    primary if mode == "freeform" or set_primary else (
                        STRING_MATCH_JUDGE_LABEL if mode == "mc" else primary
                    ),
                )
            all_records.extend(records)

    effective_primary = (
        primary
        if mode == "freeform" or set_primary
        else (STRING_MATCH_JUDGE_LABEL if mode == "mc" else primary)
    )
    judge_entries = _collect_judge_entries(
        all_records, manifest=manifest, primary=effective_primary
    )
    if not judge_entries and mode == "mc":
        judge_entries = [
            {
                "label": STRING_MATCH_JUDGE_LABEL,
                "model_id": None,
                "primary": True,
            }
        ]
        effective_primary = STRING_MATCH_JUDGE_LABEL

    primary_entry = next(
        (e for e in judge_entries if e.get("label") == effective_primary),
        judge_entries[0] if judge_entries else None,
    )
    updated_manifest = dict(manifest)
    updated_manifest["judges"] = judge_entries
    updated_manifest["primary_judge"] = effective_primary
    if primary_entry and primary_entry.get("model_id"):
        updated_manifest["grader_model_id"] = primary_entry["model_id"]
    elif mode == "mc":
        updated_manifest.setdefault("grader_model_id", None)

    manifest_changed = updated_manifest != manifest
    if manifest_changed and not dry_run:
        if backup and manifest_path.is_file():
            bak = manifest_path.with_suffix(manifest_path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(manifest_path, bak)
        write_json(manifest_path, updated_manifest)

    agg = None
    if not dry_run:
        agg = aggregate_difficulty(run_dir, model_labels=model_labels)
        # Stamp judge metadata onto scores.json (aggregate already does this
        # from the updated manifest).
        scores_path = run_dir / "scores.json"
        scores = load_json(scores_path)
        scores["judges"] = judge_entries
        scores["primary_judge"] = effective_primary
        if updated_manifest.get("grader_model_id"):
            scores["grader_model_id"] = updated_manifest["grader_model_id"]
        write_json(scores_path, scores)

    n_changed = sum(1 for info in per_model.values() if info.get("changed"))
    return {
        "run_id": run_dir.name,
        "mode": mode,
        "n_models": len(model_labels),
        "n_models_changed": n_changed,
        "manifest_changed": manifest_changed,
        "primary_judge": effective_primary,
        "judges": [e["label"] for e in judge_entries],
        "per_model": per_model,
        "aggregate": {
            "n_questions": (agg or {}).get("n_questions"),
            "scores_path": (agg or {}).get("scores_path"),
        }
        if agg
        else None,
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Local outputs directory (discovers exp-mmar-question-difficulty/)",
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        default=None,
        help="Run id to migrate (repeatable). Default: all runs.",
    )
    parser.add_argument(
        "--set-primary",
        default=None,
        help="Force this judge label as primary (e.g. qwen2.5-3b-instruct).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write .bak alongside predictions.jsonl / manifest.json before overwrite.",
    )
    args = parser.parse_args()

    run_dirs = discover_run_dirs(args.results_dir, args.runs)
    if not run_dirs:
        raise SystemExit(f"No runs found under {args.results_dir}")

    print(
        f"Retrofitting {len(run_dirs)} run(s) under {args.results_dir}"
        f"{' [dry-run]' if args.dry_run else ''}"
    )
    for run_dir in run_dirs:
        result = retrofit_run(
            run_dir,
            set_primary=args.set_primary,
            dry_run=args.dry_run,
            backup=args.backup,
        )
        print(
            f"  {result['run_id']}: mode={result['mode']} "
            f"judges={result['judges']} primary={result['primary_judge']} "
            f"models_changed={result['n_models_changed']}/{result['n_models']} "
            f"manifest_changed={result['manifest_changed']}"
        )


if __name__ == "__main__":
    main()
