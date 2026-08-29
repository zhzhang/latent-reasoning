"""Drop named models as judges (not as gradees) from the judging pack.

Removes ``shots[].judges[<key>]`` where ``key`` is ``{label}`` or
``{label}__…``, plus matching ``judge_partials/*.jsonl`` sidecars, manifest
``judges`` entries, and ``judge_accuracy.json`` tables. Keeps
``models/<label>/`` so a later ``run_judges.py`` can retry those judges.
Does not change ``primary_judge`` unless it is one of the dropped keys.

Default labels: ``qwen3-omni-instruct``, ``music-flamingo``. Does not match
``qwen3-omni`` (the thinking checkpoint).

Usage::

    uv run python scrub_judge_models.py --dry-run
    uv run python scrub_judge_models.py
    uv run modal run scrub_judge_models.py --dry-run
    uv run modal run scrub_judge_models.py
    uv run modal run scrub_judge_models.py --skip-local
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from mmar_common import load_jsonl, recompute_multi_judge_scores, write_json, write_jsonl
from modal_cache import JUDGING_MOUNT, JUDGING_VOLUME_NAME, judging_volume

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PACK = REPO_ROOT / "outputs" / "mmar-judging"
DEFAULT_JUDGES = ("qwen3-omni-instruct", "music-flamingo")
_ACCURACY_META_KEYS = frozenset(
    {
        "pack",
        "labels_path",
        "n_label_rows",
        "n_questions",
        "skipped_models",
        "epsilon",
        "modes",
        "by_category",
        "by_modality",
    }
)

app = modal.App("scrub-judge-models")
cpu_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source(
    "mmar_common",
    "modal_cache",
)


def _csv_parts(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def is_dropped_judge(key: str, drop_models: set[str]) -> bool:
    """True when ``key`` is a dropped model label or ``{label}__…`` judge slug."""
    text = str(key or "")
    if not text:
        return False
    for label in drop_models:
        if text == label or text.startswith(f"{label}__"):
            return True
    return False


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write_jsonl(tmp, rows, mode="w")
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write_json(tmp, payload)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp.replace(path)


def _clear_stale_legacy(record: dict) -> None:
    """If the primary verdict is gone, drop mirrored shot ``correct`` / ``grader``.

    Leave those flags set and ``_shot_needs_grade`` can skip a retry via the
    legacy flat fields.
    """
    primary = record.get("primary_judge")
    for shot in record.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        judges = shot.get("judges") if isinstance(shot.get("judges"), dict) else {}
        primary_entry = judges.get(primary) if primary else None
        if isinstance(primary_entry, dict) and primary_entry.get("correct") is not None:
            continue
        shot["correct"] = None
        shot.pop("grader", None)
        shot.pop("grader_output", None)
        shot["pending_grade"] = True


def scrub_record(
    record: dict,
    *,
    drop_models: set[str],
) -> tuple[int, Counter[str]]:
    """Drop matching judge entries in place. Returns ``(n_dropped, by_key)``."""
    n_dropped = 0
    dropped_by_key: Counter[str] = Counter()
    for shot in record.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        judges = shot.get("judges")
        if not isinstance(judges, dict):
            continue
        for key in list(judges):
            if not is_dropped_judge(str(key), drop_models):
                continue
            del judges[key]
            n_dropped += 1
            dropped_by_key[str(key)] += 1
    if not n_dropped:
        return 0, dropped_by_key
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
    primary = str(record.get("primary_judge") or "")
    dropped_primary = bool(primary and is_dropped_judge(primary, drop_models))
    if dropped_primary:
        record["primary_judge"] = None
    recompute_multi_judge_scores(record)
    if dropped_primary:
        _clear_stale_legacy(record)
    return n_dropped, dropped_by_key


def scrub_sidecars(
    pack_dir: Path,
    labels: list[str],
    *,
    drop_models: set[str],
    dry_run: bool,
) -> tuple[int, int, int]:
    """Drop matching sidecar files/rows. Returns ``(files_seen, files_removed, rows_dropped)``."""
    n_files = 0
    n_removed = 0
    n_rows_dropped = 0
    for label in labels:
        partials_dir = pack_dir / "models" / label / "judge_partials"
        if not partials_dir.is_dir():
            continue
        for path in sorted(partials_dir.glob("*.jsonl")):
            n_files += 1
            if is_dropped_judge(path.stem, drop_models):
                n_removed += 1
                if not dry_run:
                    path.unlink()
                continue
            rows = [row for row in load_jsonl(path) if isinstance(row, dict)]
            kept: list[dict] = []
            file_dropped = 0
            for row in rows:
                key = str(row.get("judge_key") or path.stem)
                if is_dropped_judge(key, drop_models):
                    file_dropped += 1
                    continue
                kept.append(row)
            n_rows_dropped += file_dropped
            if dry_run or not file_dropped:
                continue
            if kept:
                _atomic_write_jsonl(path, kept)
            else:
                path.unlink()
                n_removed += 1
    return n_files, n_removed, n_rows_dropped


def scrub_manifest(
    pack_dir: Path,
    *,
    drop_models: set[str],
    dry_run: bool,
) -> tuple[int, str | None]:
    """Drop matching ``judges`` entries. Returns ``(n_dropped, primary_judge)``."""
    path = pack_dir / "manifest.json"
    if not path.is_file():
        return 0, None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0, None
    if not isinstance(manifest, dict):
        return 0, None
    judges = []
    n_dropped = 0
    for entry in manifest.get("judges") or []:
        if isinstance(entry, dict):
            label = str(entry.get("label") or "")
        else:
            label = str(entry or "")
        if is_dropped_judge(label, drop_models):
            n_dropped += 1
            continue
        judges.append(entry)
    if n_dropped:
        manifest["judges"] = judges
    primary = str(manifest.get("primary_judge") or "")
    if primary and is_dropped_judge(primary, drop_models):
        replacement = None
        for entry in judges:
            if isinstance(entry, dict):
                label = str(entry.get("label") or "")
            else:
                label = str(entry or "")
            if label and not is_dropped_judge(label, drop_models):
                replacement = label
                break
        manifest["primary_judge"] = replacement
        primary_entry = next(
            (
                e
                for e in judges
                if isinstance(e, dict) and e.get("label") == replacement
            ),
            None,
        )
        if isinstance(primary_entry, dict) and primary_entry.get("model_id"):
            manifest["grader_model_id"] = primary_entry["model_id"]
        for entry in judges:
            if isinstance(entry, dict):
                entry["primary"] = entry.get("label") == replacement
    if n_dropped and not dry_run:
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_write_json(path, manifest)
    return n_dropped, manifest.get("primary_judge")


def scrub_accuracy_json(
    pack_dir: Path,
    *,
    drop_models: set[str],
    dry_run: bool,
) -> int:
    """Drop matching judge keys from Alt-Test tables. Returns keys removed."""
    path = pack_dir / "judge_accuracy.json"
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0

    n_dropped = 0

    def _filter_table(bucket: dict) -> dict:
        nonlocal n_dropped
        out: dict = {}
        for key, value in bucket.items():
            if is_dropped_judge(str(key), drop_models):
                n_dropped += 1
                continue
            out[key] = value
        return out

    def _filter_mode_tables(tables: dict) -> dict:
        out: dict = {}
        for mode, bucket in tables.items():
            if isinstance(bucket, dict) and mode not in _ACCURACY_META_KEYS:
                out[mode] = _filter_table(bucket)
            else:
                out[mode] = bucket
        return out

    for mode, bucket in list(payload.items()):
        if mode in _ACCURACY_META_KEYS:
            continue
        if isinstance(bucket, dict):
            payload[mode] = _filter_table(bucket)
    for slice_key in ("by_category", "by_modality"):
        slices = payload.get(slice_key)
        if not isinstance(slices, dict):
            continue
        payload[slice_key] = {
            name: _filter_mode_tables(tables) if isinstance(tables, dict) else tables
            for name, tables in slices.items()
        }
    if n_dropped and not dry_run:
        _atomic_write_json(path, payload)
    return n_dropped


def _discover_labels(pack_dir: Path) -> list[str]:
    models_root = pack_dir / "models"
    labels = sorted(
        path.parent.name
        for path in models_root.glob("*/predictions.jsonl")
        if path.is_file()
    )
    if not labels:
        raise SystemExit(f"no models with predictions.jsonl under {models_root}")
    return labels


def _print_stats(stats: dict[str, Any]) -> None:
    where = stats.get("where") or stats.get("pack")
    print(
        f"[scrub-judges] {where} judges={stats.get('judges')} "
        f"dry_run={stats.get('dry_run')}"
    )
    print(f"models: {stats.get('models')}")
    print(
        f"predictions: dropped {stats.get('n_verdicts_dropped')} verdicts "
        f"across {stats.get('n_records')} records"
    )
    dropped_by_key = stats.get("dropped_by_key") or {}
    if dropped_by_key:
        print("dropped by judge key:")
        for key, count in sorted(dropped_by_key.items()):
            print(f"  {key}: {count}")
    print(
        f"sidecars: {stats.get('sidecar_files_seen')} files, "
        f"removed {stats.get('sidecar_files_removed')}, "
        f"dropped {stats.get('sidecar_rows_dropped')} extra rows"
    )
    print(f"manifest judges dropped: {stats.get('manifest_judges_dropped')}")
    print(f"accuracy keys dropped: {stats.get('accuracy_keys_dropped')}")
    print(f"primary_judge: {stats.get('primary_judge')}")
    if stats.get("dry_run"):
        print("dry-run: no files written")


def scrub_pack(
    pack_dir: Path,
    *,
    drop_models: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    labels = _discover_labels(pack_dir)
    dropped_by_key: Counter[str] = Counter()
    n_records = 0
    n_verdicts = 0
    for label in labels:
        pred_path = pack_dir / "models" / label / "predictions.jsonl"
        records = load_jsonl(pred_path)
        changed = False
        for record in records:
            if not isinstance(record, dict):
                continue
            n_records += 1
            dropped, by_key = scrub_record(record, drop_models=drop_models)
            n_verdicts += dropped
            dropped_by_key.update(by_key)
            if dropped:
                changed = True
        if changed and not dry_run:
            _atomic_write_jsonl(pred_path, records)
            print(f"wrote {pred_path}")

    n_sc_files, n_sc_removed, n_sc_rows = scrub_sidecars(
        pack_dir, labels, drop_models=drop_models, dry_run=dry_run
    )
    n_manifest, primary = scrub_manifest(
        pack_dir, drop_models=drop_models, dry_run=dry_run
    )
    n_accuracy = scrub_accuracy_json(
        pack_dir, drop_models=drop_models, dry_run=dry_run
    )
    stats = {
        "pack": str(pack_dir),
        "dry_run": dry_run,
        "judges": sorted(drop_models),
        "models": labels,
        "n_records": n_records,
        "n_verdicts_dropped": n_verdicts,
        "dropped_by_key": dict(dropped_by_key),
        "sidecar_files_seen": n_sc_files,
        "sidecar_files_removed": n_sc_removed,
        "sidecar_rows_dropped": n_sc_rows,
        "manifest_judges_dropped": n_manifest,
        "accuracy_keys_dropped": n_accuracy,
        "primary_judge": primary,
    }
    return stats


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={JUDGING_MOUNT: judging_volume},
)
def scrub_volume(
    judges: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scrub the ``mmar-judging`` Volume pack in place."""
    judging_volume.reload()
    drop_models = set(judges or DEFAULT_JUDGES)
    stats = scrub_pack(Path(JUDGING_MOUNT), drop_models=drop_models, dry_run=dry_run)
    stats["where"] = f"volume:{JUDGING_VOLUME_NAME}"
    if not dry_run:
        judging_volume.commit()
    _print_stats(stats)
    return stats


@app.local_entrypoint()
def modal_main(
    dry_run: bool = False,
    skip_local: bool = False,
    skip_modal: bool = False,
    judges: str = ",".join(DEFAULT_JUDGES),
    pack: str = str(DEFAULT_PACK),
):
    """Scrub the local pack, then the Modal Volume (unless skipped)."""
    drop_models = set(_csv_parts(judges) or DEFAULT_JUDGES)
    if not skip_local:
        local_dir = Path(pack).expanduser().resolve()
        stats = scrub_pack(local_dir, drop_models=drop_models, dry_run=dry_run)
        stats["where"] = f"local:{local_dir}"
        _print_stats(stats)
    if not skip_modal:
        scrub_volume.remote(judges=sorted(drop_models), dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove qwen3-omni-instruct and music-flamingo judge verdicts "
            "so a later run_judges.py will re-grade them."
        )
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=DEFAULT_PACK,
        help="Judging pack directory (default: outputs/mmar-judging).",
    )
    parser.add_argument(
        "--judges",
        default=",".join(DEFAULT_JUDGES),
        help="Comma-separated judge model labels to drop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing.",
    )
    args = parser.parse_args()
    drop_models = set(_csv_parts(args.judges) or DEFAULT_JUDGES)
    stats = scrub_pack(
        Path(args.pack).expanduser().resolve(),
        drop_models=drop_models,
        dry_run=args.dry_run,
    )
    stats["where"] = f"local:{stats['pack']}"
    _print_stats(stats)


if __name__ == "__main__":
    main()
