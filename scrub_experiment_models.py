"""Drop named models as gradees from a ``run_experiment.py`` pack.

Deletes ``models/<label>/``, then rewrites ``manifest.json`` so
``prepare_run`` will not merge the leftover into ``merged_models``
(``resolve_sampling`` currently dies on ``music-flamingo``, which is no
longer in ``MODEL_SPECS``). Re-aggregates ``difficulty.jsonl`` /
``scores.json``. Does not touch ``primary_judge`` or remaining models'
prediction files.

Default label: ``music-flamingo``. Does not match ``af-next-think``.

Usage::

    uv run python scrub_experiment_models.py --dry-run
    uv run python scrub_experiment_models.py
    uv run modal run scrub_experiment_models.py --dry-run
    uv run modal run scrub_experiment_models.py
    uv run modal run scrub_experiment_models.py --skip-local
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from aggregate import aggregate_difficulty
from modal_cache import (
    MMAR_FREEFORM_THINKING_MOUNT,
    MMAR_FREEFORM_THINKING_VOLUME_NAME,
    mmar_freeform_thinking_volume,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PACK = REPO_ROOT / "outputs" / "mmar-freeform-thinking"
DEFAULT_DROP = ("music-flamingo",)
_MANIFEST_MODEL_MAPS = ("workload", "model_sampling", "model_specs", "progress")

app = modal.App("scrub-experiment-models")
cpu_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source(
    "aggregate",
    "modal_cache",
)


def _csv_parts(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp.replace(path)


def _present_labels(pack_dir: Path) -> list[str]:
    models_root = pack_dir / "models"
    if not models_root.is_dir():
        return []
    return sorted(
        child.name
        for child in models_root.iterdir()
        if child.is_dir()
    )


def _dir_file_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.rglob("*") if _.is_file())


def scrub_manifest(
    pack_dir: Path,
    *,
    drop_models: set[str],
    remaining: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    """Drop gradees from pack-level maps. Returns counts of keys removed."""
    path = pack_dir / "manifest.json"
    stats: dict[str, Any] = {
        "present": path.is_file(),
        "models_dropped": 0,
        "map_keys_dropped": 0,
    }
    if not path.is_file():
        return stats
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        stats["error"] = "manifest.json is not valid JSON"
        return stats
    if not isinstance(manifest, dict):
        stats["error"] = "manifest.json is not an object"
        return stats

    prior = [str(x) for x in (manifest.get("models") or [])]
    kept = [label for label in prior if label not in drop_models]
    extras = [label for label in remaining if label not in kept]
    manifest["models"] = kept + extras
    stats["models_dropped"] = len(prior) - len(kept)

    for key in _MANIFEST_MODEL_MAPS:
        bucket = manifest.get(key)
        if not isinstance(bucket, dict):
            continue
        dropped = [label for label in list(bucket) if label in drop_models]
        if not dropped:
            continue
        for label in dropped:
            bucket.pop(label, None)
        stats["map_keys_dropped"] += len(dropped)

    sources = []
    n_source_dropped = 0
    for source in manifest.get("sources") or []:
        if not isinstance(source, dict):
            sources.append(source)
            continue
        models = [
            label
            for label in (source.get("models") or [])
            if str(label) not in drop_models
        ]
        n_source_dropped += len(source.get("models") or []) - len(models)
        source = dict(source)
        source["models"] = models
        sources.append(source)
    if "sources" in manifest:
        manifest["sources"] = sources
        stats["map_keys_dropped"] += n_source_dropped

    if not dry_run and (stats["models_dropped"] or stats["map_keys_dropped"]):
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_write_json(path, manifest)
    return stats


def scrub_pack(
    pack_dir: Path,
    *,
    drop_models: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    if not pack_dir.is_dir():
        return {
            "pack": str(pack_dir),
            "dry_run": dry_run,
            "missing": True,
            "drop": sorted(drop_models),
        }

    present = _present_labels(pack_dir)
    to_drop = [label for label in present if label in drop_models]
    remaining = [label for label in present if label not in drop_models]
    dropped_files = {label: _dir_file_count(pack_dir / "models" / label) for label in to_drop}

    if not dry_run:
        for label in to_drop:
            shutil.rmtree(pack_dir / "models" / label)

    manifest_stats = scrub_manifest(
        pack_dir,
        drop_models=drop_models,
        remaining=remaining,
        dry_run=dry_run,
    )

    aggregated: dict[str, Any] | None = None
    if not dry_run and remaining:
        aggregated = aggregate_difficulty(pack_dir)
        scores = (aggregated or {}).get("scores") or {}
        aggregated = {
            "n_questions": aggregated.get("n_questions"),
            "n_models": scores.get("n_models"),
            "model_labels": scores.get("model_labels"),
        }

    return {
        "pack": str(pack_dir),
        "dry_run": dry_run,
        "missing": False,
        "drop": sorted(drop_models),
        "dropped_dirs": to_drop,
        "dropped_files": dropped_files,
        "remaining": remaining,
        "manifest": manifest_stats,
        "aggregated": aggregated,
    }


def _print_stats(stats: dict[str, Any]) -> None:
    where = stats.get("where") or stats.get("pack")
    print(f"[scrub-experiment] {where} drop={stats.get('drop')} dry_run={stats.get('dry_run')}")
    if stats.get("missing"):
        print("pack not found; skipped")
        return
    dropped = stats.get("dropped_dirs") or []
    if dropped:
        print("deleted models/:")
        for label in dropped:
            n_files = (stats.get("dropped_files") or {}).get(label, 0)
            print(f"  {label}/ ({n_files} files)")
    else:
        print("deleted models/: (none on disk)")
    remaining = stats.get("remaining") or []
    print(f"remaining models: {', '.join(remaining) or '(none)'}")
    manifest = stats.get("manifest") or {}
    if not manifest.get("present"):
        print("manifest.json: missing")
    elif manifest.get("error"):
        print(f"manifest.json: {manifest['error']}")
    else:
        print(
            f"manifest.json: dropped {manifest.get('models_dropped', 0)} models, "
            f"{manifest.get('map_keys_dropped', 0)} map keys"
        )
    aggregated = stats.get("aggregated")
    if aggregated:
        print(
            f"aggregated: n_questions={aggregated.get('n_questions')} "
            f"n_models={aggregated.get('n_models')} "
            f"labels={aggregated.get('model_labels')}"
        )
    elif stats.get("dry_run"):
        print("dry-run: no files written")
    elif not remaining:
        print("aggregated: skipped (no remaining models)")


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={MMAR_FREEFORM_THINKING_MOUNT: mmar_freeform_thinking_volume},
)
def scrub_volume(
    models: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scrub the ``mmar-freeform-thinking`` Volume pack in place."""
    mmar_freeform_thinking_volume.reload()
    drop_models = set(models or DEFAULT_DROP)
    stats = scrub_pack(
        Path(MMAR_FREEFORM_THINKING_MOUNT),
        drop_models=drop_models,
        dry_run=dry_run,
    )
    stats["where"] = f"volume:{MMAR_FREEFORM_THINKING_VOLUME_NAME}"
    if not dry_run:
        mmar_freeform_thinking_volume.commit()
    _print_stats(stats)
    return stats


@app.local_entrypoint()
def modal_main(
    dry_run: bool = False,
    skip_local: bool = False,
    skip_modal: bool = False,
    models: str = ",".join(DEFAULT_DROP),
    pack: str = str(DEFAULT_PACK),
):
    """Scrub the local pack, then the Modal Volume (unless skipped)."""
    drop_models = set(_csv_parts(models) or DEFAULT_DROP)
    if not skip_local:
        local_dir = Path(pack).expanduser().resolve()
        stats = scrub_pack(local_dir, drop_models=drop_models, dry_run=dry_run)
        stats["where"] = f"local:{local_dir}"
        _print_stats(stats)
    if not skip_modal:
        scrub_volume.remote(models=sorted(drop_models), dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove music-flamingo (or other gradees) from a run_experiment.py "
            "pack so prepare_run no longer merges them."
        )
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=DEFAULT_PACK,
        help="Experiment pack directory (default: outputs/mmar-freeform-thinking).",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_DROP),
        help="Comma-separated gradee labels to delete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing.",
    )
    args = parser.parse_args()
    drop_models = set(_csv_parts(args.models) or DEFAULT_DROP)
    stats = scrub_pack(
        Path(args.pack).expanduser().resolve(),
        drop_models=drop_models,
        dry_run=args.dry_run,
    )
    stats["where"] = f"local:{stats['pack']}"
    _print_stats(stats)


if __name__ == "__main__":
    main()
