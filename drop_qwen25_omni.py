"""Drop Qwen2.5-Omni-7B from local outputs and Modal volumes.

Never touches ``exports/``. Removes the model as a gradee (``models/<label>/``
plus pack manifests) and as a judge (verdicts, sidecars, accuracy tables).
Also deletes Hub weights and the vLLM compile-cache tree on the seed volume.

Usage::

    uv run python drop_qwen25_omni.py --dry-run
    uv run python drop_qwen25_omni.py
    uv run modal run drop_qwen25_omni.py --dry-run
    uv run modal run drop_qwen25_omni.py
    uv run modal run drop_qwen25_omni.py --skip-local
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import modal

from aggregate import DROPPED_MODEL_LABELS
from modal_cache import (
    FREEFORM_THINKING_MOUNT,
    FREEFORM_THINKING_VOLUME_NAME,
    JUDGING_MOUNT,
    JUDGING_VOLUME_NAME,
    MMAR_DESCRIPTIONS_MOUNT,
    MMAR_DESCRIPTIONS_VOLUME_NAME,
    MMAR_FREEFORM_THINKING_MOUNT,
    MMAR_FREEFORM_THINKING_VOLUME_NAME,
    RESULTS_MOUNT,
    RESULTS_VOLUME_NAME,
    VOLUME_MOUNT,
    VOLUME_NAME,
    compile_cache_dir,
    freeform_thinking_volume,
    judging_volume,
    mmar_descriptions_volume,
    mmar_freeform_thinking_volume,
    results_volume,
    volume,
)
from scrub_experiment_models import (
    _present_labels,
    scrub_manifest,
    scrub_pack as scrub_experiment_pack,
)
from scrub_judge_models import discover_packs, scrub_tree

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DROP = tuple(sorted(DROPPED_MODEL_LABELS))
EXPORTS_DIR = REPO_ROOT / "exports"
OUTPUTS_DIR = REPO_ROOT / "outputs"
_SKIP_WALK_NAMES = frozenset({"exports"})
WEIGHT_REPOS = {
    "qwen2.5-omni-7b": "Qwen/Qwen2.5-Omni-7B",
}

app = modal.App("drop-qwen25-omni")
cpu_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source(
    "aggregate",
    "mmar_common",
    "modal_cache",
    "scrub_experiment_models",
    "scrub_judge_models",
)


def _csv_parts(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _is_exports(path: Path) -> bool:
    try:
        path.resolve().relative_to(EXPORTS_DIR.resolve())
        return True
    except ValueError:
        return "exports" in path.parts


def _dir_file_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.rglob("*") if _.is_file())


def _rmtree(path: Path, *, dry_run: bool) -> dict[str, Any]:
    existed = path.exists()
    n_files = _dir_file_count(path) if path.is_dir() else (1 if path.is_file() else 0)
    if existed and not dry_run:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return {
        "path": str(path),
        "existed": existed,
        "n_files": n_files,
        "deleted": bool(existed and not dry_run),
    }


def find_model_dirs(root: Path, labels: set[str]) -> list[Path]:
    """``models/<label>/`` trees under ``root``, skipping ``exports/``."""
    found: list[Path] = []
    if not root.is_dir() or _is_exports(root):
        return found
    for dirpath, dirnames, _filenames in os.walk(root, topdown=True):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_WALK_NAMES]
        base = Path(dirpath)
        if base.name != "models":
            continue
        for label in labels:
            child = base / label
            if child.is_dir():
                found.append(child)
        dirnames[:] = [name for name in dirnames if name not in labels]
    return found


def drop_gradee_dir(pack_dir: Path, *, drop_models: set[str], dry_run: bool) -> dict[str, Any]:
    """Delete ``models/<label>/`` and drop those keys from ``manifest.json``."""
    present = _present_labels(pack_dir)
    to_drop = [label for label in present if label in drop_models]
    remaining = [label for label in present if label not in drop_models]
    deleted = []
    for label in to_drop:
        deleted.append(
            _rmtree(pack_dir / "models" / label, dry_run=dry_run)
        )
    manifest_stats = scrub_manifest(
        pack_dir,
        drop_models=drop_models,
        remaining=remaining,
        dry_run=dry_run,
    )
    return {
        "pack": str(pack_dir),
        "dropped_dirs": to_drop,
        "deleted": deleted,
        "remaining": remaining,
        "manifest": manifest_stats,
    }


def drop_experiment_pack(
    pack_dir: Path, *, drop_models: set[str], dry_run: bool
) -> dict[str, Any]:
    stats = scrub_experiment_pack(
        pack_dir, drop_models=drop_models, dry_run=dry_run
    )
    stats["kind"] = "experiment"
    return stats


def drop_judging_tree(
    root: Path, *, drop_models: set[str], dry_run: bool, where_prefix: str
) -> dict[str, Any]:
    if not root.is_dir():
        return {
            "root": str(root),
            "kind": "judging",
            "missing": True,
            "dry_run": dry_run,
        }
    packs = discover_packs(root)
    if not packs:
        leftover = [
            drop_gradee_dir(root, drop_models=drop_models, dry_run=dry_run)
        ]
        return {
            "root": str(root),
            "kind": "judging",
            "missing": False,
            "n_packs": 0,
            "gradees": leftover,
            "dry_run": dry_run,
        }
    summary = scrub_tree(
        root,
        drop_models=drop_models,
        dry_run=dry_run,
        where_prefix=where_prefix,
    )
    gradees = []
    for pack_dir in packs:
        gradees.append(
            drop_gradee_dir(pack_dir, drop_models=drop_models, dry_run=dry_run)
        )
    summary["kind"] = "judging"
    summary["gradees"] = gradees
    return summary


def drop_leftover_model_dirs(
    root: Path, *, drop_models: set[str], dry_run: bool
) -> list[dict[str, Any]]:
    return [
        _rmtree(path, dry_run=dry_run)
        for path in find_model_dirs(root, drop_models)
        if not _is_exports(path)
    ]


def drop_seed_trees(
    cache_root: Path, *, drop_models: set[str], dry_run: bool
) -> dict[str, Any]:
    """Hub weights under ``/cache/models`` and compile caches under ``/cache/vllm``."""
    weights = []
    caches = []
    for label in sorted(drop_models):
        repo = WEIGHT_REPOS.get(label)
        if repo:
            weights.append(_rmtree(cache_root / "models" / repo, dry_run=dry_run))
        caches.append(_rmtree(compile_cache_dir(label), dry_run=dry_run))
    return {"weights": weights, "compile_caches": caches}


def _print_block(title: str, payload: Any) -> None:
    print(f"[drop-qwen25] {title}")
    if isinstance(payload, dict) and payload.get("missing"):
        print("  skipped (missing)")
        return
    print(f"  {payload}")


def drop_local(*, drop_models: set[str], dry_run: bool) -> dict[str, Any]:
    """Scrub every local pack under ``outputs/``. Leaves ``exports/`` alone."""
    results: dict[str, Any] = {
        "dry_run": dry_run,
        "drop": sorted(drop_models),
        "exports": str(EXPORTS_DIR),
    }
    results["experiment"] = drop_experiment_pack(
        OUTPUTS_DIR / "mmar-freeform-thinking",
        drop_models=drop_models,
        dry_run=dry_run,
    )
    _print_block("local experiment", results["experiment"])
    results["descriptions"] = drop_experiment_pack(
        OUTPUTS_DIR / "mmar-descriptions",
        drop_models=drop_models,
        dry_run=dry_run,
    )
    _print_block("local descriptions", results["descriptions"])
    results["judging"] = drop_judging_tree(
        OUTPUTS_DIR / "mmar-judging",
        drop_models=drop_models,
        dry_run=dry_run,
        where_prefix="local",
    )
    _print_block("local judging", results["judging"])
    results["leftovers"] = drop_leftover_model_dirs(
        OUTPUTS_DIR, drop_models=drop_models, dry_run=dry_run
    )
    if results["leftovers"]:
        _print_block("local leftovers", results["leftovers"])
    else:
        print("[drop-qwen25] local leftovers: none")
    return results


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={MMAR_FREEFORM_THINKING_MOUNT: mmar_freeform_thinking_volume},
)
def drop_experiment_volume(
    models: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    mmar_freeform_thinking_volume.reload()
    drop_models = set(models or DEFAULT_DROP)
    stats = drop_experiment_pack(
        Path(MMAR_FREEFORM_THINKING_MOUNT),
        drop_models=drop_models,
        dry_run=dry_run,
    )
    stats["where"] = f"volume:{MMAR_FREEFORM_THINKING_VOLUME_NAME}"
    if not dry_run:
        mmar_freeform_thinking_volume.commit()
    _print_block(stats["where"], stats)
    return stats


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={MMAR_DESCRIPTIONS_MOUNT: mmar_descriptions_volume},
)
def drop_descriptions_volume(
    models: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    mmar_descriptions_volume.reload()
    drop_models = set(models or DEFAULT_DROP)
    stats = drop_experiment_pack(
        Path(MMAR_DESCRIPTIONS_MOUNT),
        drop_models=drop_models,
        dry_run=dry_run,
    )
    stats["where"] = f"volume:{MMAR_DESCRIPTIONS_VOLUME_NAME}"
    if not dry_run:
        mmar_descriptions_volume.commit()
    _print_block(stats["where"], stats)
    return stats


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={JUDGING_MOUNT: judging_volume},
)
def drop_judging_volume(
    models: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    judging_volume.reload()
    drop_models = set(models or DEFAULT_DROP)
    stats = drop_judging_tree(
        Path(JUDGING_MOUNT),
        drop_models=drop_models,
        dry_run=dry_run,
        where_prefix=f"volume:{JUDGING_VOLUME_NAME}",
    )
    stats["where"] = f"volume:{JUDGING_VOLUME_NAME}"
    if not dry_run:
        judging_volume.commit()
    _print_block(stats["where"], stats)
    return stats


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={FREEFORM_THINKING_MOUNT: freeform_thinking_volume},
)
def drop_legacy_freeform_volume(
    models: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    freeform_thinking_volume.reload()
    drop_models = set(models or DEFAULT_DROP)
    root = Path(FREEFORM_THINKING_MOUNT)
    stats = drop_experiment_pack(root, drop_models=drop_models, dry_run=dry_run)
    leftovers = drop_leftover_model_dirs(
        root, drop_models=drop_models, dry_run=dry_run
    )
    stats["leftovers"] = leftovers
    stats["where"] = f"volume:{FREEFORM_THINKING_VOLUME_NAME}"
    if not dry_run:
        freeform_thinking_volume.commit()
    _print_block(stats["where"], stats)
    return stats


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={RESULTS_MOUNT: results_volume},
)
def drop_results_volume(
    models: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    results_volume.reload()
    drop_models = set(models or DEFAULT_DROP)
    leftovers = drop_leftover_model_dirs(
        Path(RESULTS_MOUNT), drop_models=drop_models, dry_run=dry_run
    )
    stats = {
        "where": f"volume:{RESULTS_VOLUME_NAME}",
        "kind": "results",
        "dry_run": dry_run,
        "leftovers": leftovers,
    }
    if not dry_run:
        results_volume.commit()
    _print_block(stats["where"], stats)
    return stats


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume},
)
def drop_seed_volume(
    models: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    volume.reload()
    drop_models = set(models or DEFAULT_DROP)
    stats = drop_seed_trees(
        Path(VOLUME_MOUNT), drop_models=drop_models, dry_run=dry_run
    )
    stats["where"] = f"volume:{VOLUME_NAME}"
    stats["kind"] = "seed"
    stats["dry_run"] = dry_run
    if not dry_run:
        volume.commit()
    _print_block(stats["where"], stats)
    return stats


@app.local_entrypoint()
def modal_main(
    dry_run: bool = False,
    skip_local: bool = False,
    skip_modal: bool = False,
    models: str = ",".join(DEFAULT_DROP),
):
    """Drop Qwen2.5-Omni locally, then on every Modal volume (unless skipped)."""
    drop_models = set(_csv_parts(models) or DEFAULT_DROP)
    if EXPORTS_DIR.is_dir():
        print(f"[drop-qwen25] leaving {EXPORTS_DIR} untouched")
    if not skip_local:
        drop_local(drop_models=drop_models, dry_run=dry_run)
    if skip_modal:
        return
    drop_experiment_volume.remote(models=sorted(drop_models), dry_run=dry_run)
    drop_descriptions_volume.remote(models=sorted(drop_models), dry_run=dry_run)
    drop_judging_volume.remote(models=sorted(drop_models), dry_run=dry_run)
    drop_legacy_freeform_volume.remote(models=sorted(drop_models), dry_run=dry_run)
    drop_results_volume.remote(models=sorted(drop_models), dry_run=dry_run)
    drop_seed_volume.remote(models=sorted(drop_models), dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove qwen2.5-omni-7b from local outputs/ packs. "
            "Does not touch exports/. Use modal run for Modal volumes."
        )
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_DROP),
        help="Comma-separated labels to drop (default: qwen2.5-omni-7b).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing.",
    )
    args = parser.parse_args()
    drop_models = set(_csv_parts(args.models) or DEFAULT_DROP)
    if EXPORTS_DIR.is_dir():
        print(f"[drop-qwen25] leaving {EXPORTS_DIR} untouched")
    drop_local(drop_models=drop_models, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
