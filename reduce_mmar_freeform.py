"""Reduce the collated MMAR freeform pack: drop Nemotron and gpt-audio-mini.

Rewrites ``outputs/mmar-freeform`` in place:

  * delete ``models/nemotron-3-nano-omni`` and ``models/gpt-audio-mini``
  * drop those models' judge keys from remaining predictions, sidecars, and
    the manifest (``{label}__…``)
  * keep the first ``--n-shots`` attempts per question (by ``shot_index``)
  * drop those models' rows from ``labels.csv``
  * re-aggregate ``difficulty.jsonl`` / ``scores.json``

Usage::

    uv run python reduce_mmar_freeform.py --dry-run
    uv run python reduce_mmar_freeform.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aggregate import aggregate_difficulty, write_jsonl
from mmar_common import load_jsonl, recompute_multi_judge_scores, write_json

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PACK = REPO_ROOT / "outputs" / "mmar-freeform"
DEFAULT_DROP_MODELS = ("nemotron-3-nano-omni", "gpt-audio-mini")
DEFAULT_N_SHOTS = 3


def _shot_index(shot: dict[str, Any]) -> int:
    raw = shot.get("shot_index")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def is_dropped_judge(key: str, drop_models: set[str]) -> bool:
    """True when ``key`` is a dropped model label or ``{label}__…`` judge slug."""
    text = str(key or "")
    if not text:
        return False
    for label in drop_models:
        if text == label or text.startswith(f"{label}__"):
            return True
    return False


def downsample_record(
    record: dict[str, Any],
    *,
    n_shots: int,
    drop_models: set[str],
) -> dict[str, Any]:
    shots = list(record.get("shots") or [])
    shots.sort(key=_shot_index)
    shots = shots[:n_shots]
    for shot in shots:
        judges = shot.get("judges")
        if isinstance(judges, dict):
            for key in list(judges):
                if is_dropped_judge(key, drop_models):
                    judges.pop(key, None)
    record["shots"] = shots
    record["n_shots"] = len(shots)
    primary = record.get("primary_judge")
    if primary and is_dropped_judge(str(primary), drop_models):
        record["primary_judge"] = None
    if isinstance(record.get("judges"), list):
        record["judges"] = [
            label
            for label in record["judges"]
            if not is_dropped_judge(str(label), drop_models)
        ]
    per_judge = record.get("per_judge")
    if isinstance(per_judge, dict):
        record["per_judge"] = {
            key: value
            for key, value in per_judge.items()
            if not is_dropped_judge(str(key), drop_models)
        }
    recompute_multi_judge_scores(record)
    return record


def filter_partials(
    path: Path,
    *,
    keep_indexes: set[int],
    drop_models: set[str],
) -> tuple[int, int]:
    """Rewrite a judge sidecar, dropping Nemotron rows and extra shots.

    Returns ``(kept, dropped)``.
    """
    rows = load_jsonl(path)
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        key = str(row.get("judge_key") or path.stem)
        if is_dropped_judge(key, drop_models):
            dropped += 1
            continue
        try:
            shot_index = int(row.get("shot_index", 0))
        except (TypeError, ValueError):
            shot_index = 0
        if shot_index not in keep_indexes:
            dropped += 1
            continue
        kept.append(row)
    write_jsonl(path, kept)
    return len(kept), dropped


def update_manifest(
    manifest: dict[str, Any],
    *,
    n_shots: int,
    drop_models: set[str],
    remaining: list[str],
) -> dict[str, Any]:
    manifest["n_shots"] = n_shots
    manifest["name"] = "mmar-freeform"
    manifest["models"] = remaining
    progress = dict(manifest.get("progress") or {})
    for label in list(progress):
        if label in drop_models:
            progress.pop(label, None)
            continue
        row = dict(progress[label] or {})
        row["n_shots"] = n_shots
        progress[label] = row
    manifest["progress"] = progress
    sources = []
    for source in manifest.get("sources") or []:
        if not isinstance(source, dict):
            sources.append(source)
            continue
        models = [
            label
            for label in (source.get("models") or [])
            if str(label) not in drop_models
        ]
        source = dict(source)
        source["models"] = models
        sources.append(source)
    if "sources" in manifest:
        manifest["sources"] = sources
    if "judges" in manifest:
        judges = []
        for entry in manifest.get("judges") or []:
            if isinstance(entry, dict):
                label = str(entry.get("label") or "")
            else:
                label = str(entry or "")
            if is_dropped_judge(label, drop_models):
                continue
            judges.append(entry)
        manifest["judges"] = judges
    primary = str(manifest.get("primary_judge") or "")
    if not primary or is_dropped_judge(primary, drop_models):
        replacement = None
        for entry in manifest.get("judges") or []:
            if isinstance(entry, dict):
                model_id = str(entry.get("model_id") or "")
                label = str(entry.get("label") or "")
            else:
                model_id = ""
                label = str(entry or "")
            if model_id and not is_dropped_judge(model_id, drop_models):
                replacement = model_id
                break
            if label and not is_dropped_judge(label, drop_models):
                replacement = label
                break
        manifest["primary_judge"] = replacement
    now = datetime.now(timezone.utc).isoformat()
    manifest["updated_at"] = now
    return manifest


def filter_accuracy_json(path: Path, drop_models: set[str]) -> None:
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    payload["pack"] = "mmar-freeform"
    payload["labels_path"] = str(path.parent / "labels.csv")
    for mode in ("with_gt", "free"):
        bucket = payload.get(mode)
        if isinstance(bucket, dict):
            payload[mode] = {
                key: value
                for key, value in bucket.items()
                if not is_dropped_judge(str(key), drop_models)
            }
    write_json(path, payload)


def filter_labels_csv(path: Path, drop_models: set[str]) -> tuple[int, int]:
    """Drop labeled rows whose ``model_label`` is a removed test-taker.

    Returns ``(kept, dropped)``.
    """
    if not path.is_file():
        return 0, 0

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        return 0, 0
    kept = [
        row
        for row in rows
        if str(row.get("model_label") or "") not in drop_models
    ]
    dropped = len(rows) - len(kept)
    if dropped:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
    return len(kept), dropped


def reduce_pack(
    pack_dir: Path,
    *,
    n_shots: int,
    drop_models: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    models_root = pack_dir / "models"
    if not models_root.is_dir():
        raise SystemExit(f"models/ not found under {pack_dir}")

    present = sorted(p.name for p in models_root.iterdir() if p.is_dir())
    to_drop = [label for label in present if label in drop_models]
    remaining = [label for label in present if label not in drop_models]
    keep_indexes = set(range(n_shots))

    stats: dict[str, Any] = {
        "dropped_models": to_drop,
        "remaining_models": remaining,
        "n_shots": n_shots,
        "records": 0,
        "shots_before": 0,
        "shots_after": 0,
        "partial_files_removed": 0,
        "partial_rows_dropped": 0,
        "label_rows_kept": 0,
        "label_rows_dropped": 0,
    }

    if dry_run:
        for label in remaining:
            pred_path = models_root / label / "predictions.jsonl"
            if not pred_path.is_file():
                continue
            for record in load_jsonl(pred_path):
                stats["records"] += 1
                n_have = len(record.get("shots") or [])
                stats["shots_before"] += n_have
                stats["shots_after"] += min(n_have, n_shots)
            partials = models_root / label / "judge_partials"
            if partials.is_dir():
                for path in sorted(partials.glob("*.jsonl")):
                    if is_dropped_judge(path.stem, drop_models):
                        stats["partial_files_removed"] += 1
                        continue
                    for row in load_jsonl(path):
                        try:
                            shot_index = int(row.get("shot_index", 0))
                        except (TypeError, ValueError):
                            shot_index = 0
                        if shot_index not in keep_indexes:
                            stats["partial_rows_dropped"] += 1
        return stats

    for label in to_drop:
        shutil.rmtree(models_root / label)

    for label in remaining:
        pred_path = models_root / label / "predictions.jsonl"
        if not pred_path.is_file():
            continue
        records = load_jsonl(pred_path)
        rewritten: list[dict[str, Any]] = []
        for record in records:
            stats["records"] += 1
            stats["shots_before"] += len(record.get("shots") or [])
            downsample_record(record, n_shots=n_shots, drop_models=drop_models)
            stats["shots_after"] += len(record.get("shots") or [])
            rewritten.append(record)
        write_jsonl(pred_path, rewritten)

        partials = models_root / label / "judge_partials"
        if not partials.is_dir():
            continue
        for path in sorted(partials.glob("*.jsonl")):
            if is_dropped_judge(path.stem, drop_models):
                path.unlink()
                stats["partial_files_removed"] += 1
                continue
            _, dropped = filter_partials(
                path, keep_indexes=keep_indexes, drop_models=drop_models
            )
            stats["partial_rows_dropped"] += dropped

    manifest_path = pack_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            manifest = payload
    if remaining:
        # Preserve collate / manifest order, then any extras on disk.
        known = [label for label in (manifest.get("models") or []) if label in remaining]
        rest = [label for label in remaining if label not in known]
        remaining = known + rest
    manifest = update_manifest(
        manifest,
        n_shots=n_shots,
        drop_models=drop_models,
        remaining=remaining,
    )
    write_json(manifest_path, manifest)

    ids_path = pack_dir / "question_ids.json"
    if ids_path.is_file():
        try:
            ids_payload = json.loads(ids_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            ids_payload = None
        if isinstance(ids_payload, dict):
            ids_payload["n_shots"] = n_shots
            write_json(ids_path, ids_payload)

    kept_labels, dropped_labels = filter_labels_csv(pack_dir / "labels.csv", drop_models)
    stats["label_rows_kept"] = kept_labels
    stats["label_rows_dropped"] = dropped_labels
    filter_accuracy_json(pack_dir / "judge_accuracy.json", drop_models)
    aggregate_difficulty(pack_dir)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop Nemotron and gpt-audio-mini; keep 3 shots per question."
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=DEFAULT_PACK,
        help="Freeform pack directory (default: outputs/mmar-freeform).",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=DEFAULT_N_SHOTS,
        help="Shots to keep per question (default: 3).",
    )
    parser.add_argument(
        "--drop-model",
        action="append",
        default=None,
        help="Model label to delete (repeatable). Default: nemotron-3-nano-omni, gpt-audio-mini.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pack_dir = Path(args.pack).expanduser().resolve()
    if not pack_dir.is_dir():
        raise SystemExit(f"pack not found: {pack_dir}")
    n_shots = max(1, int(args.n_shots))
    drop_models = set(args.drop_model or DEFAULT_DROP_MODELS)
    stats = reduce_pack(
        pack_dir,
        n_shots=n_shots,
        drop_models=drop_models,
        dry_run=args.dry_run,
    )
    prefix = "dry-run: " if args.dry_run else ""
    dropped = ", ".join(stats["dropped_models"]) or "(none)"
    remaining = ", ".join(stats["remaining_models"]) or "(none)"
    print(f"{prefix}{pack_dir}")
    print(f"drop models: {dropped}")
    print(f"keep models: {remaining}")
    print(
        f"records={stats['records']} shots {stats['shots_before']} -> {stats['shots_after']} "
        f"(n_shots={n_shots})"
    )
    print(
        f"judge_partials: removed {stats['partial_files_removed']} files, "
        f"dropped {stats['partial_rows_dropped']} extra-shot rows"
    )
    print(
        f"labels.csv: kept {stats.get('label_rows_kept', 0)}, "
        f"dropped {stats.get('label_rows_dropped', 0)}"
    )
    if args.dry_run:
        print("dry-run: no files written")


if __name__ == "__main__":
    main()
