"""Delete tiny test / partial MMAR difficulty runs from Modal results.

Inspects ``latent-reasoning-results/exp-mmar-question-difficulty/<run_id>/``
and removes runs whose total *file* bytes fall below a size cutoff.

Cutoff rationale (from the live volume as of 2026-07-29)
--------------------------------------------------------
Run sizes are strongly bimodal:

* **≤ ~187 KiB** — smoke tests (``num_samples`` 2 or 8), empty aborted
  startups (manifest + question_ids only), and named debug dirs
  (``step-audio-debug-*``, ``timing-2sample-compare``).
* **≥ ~1.5 MiB** — scaled / full experiments (50–200 questions), including
  incomplete full runs that still have substantial prediction payloads.

There is nearly an order-of-magnitude gap between those clusters, so the
default ``--min-bytes`` is **1 MiB** (1_048_576). Anything smaller is
clearly junk; anything larger is kept even if incomplete (e.g. a crashed
200-question run with one model's predictions).

Override with ``--min-bytes`` if you want a stricter keep set
(e.g. ``6000000`` keeps only the multi-MB full / near-full runs).

Usage::

    # Dry-run (default): print keep / delete table
    uv run modal run cleanup_runs.py

    # Actually delete
    uv run modal run cleanup_runs.py --execute

    # Stricter cutoff
    uv run modal run cleanup_runs.py \\
      --min-bytes 6000000 --execute
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import modal
from modal.volume import FileEntryType

from modal_cache import RESULTS_VOLUME_NAME, results_volume

EXPERIMENT_PREFIX = "exp-mmar-question-difficulty"
# Elbow between smoke/aborted (~≤187 KiB) and real payloads (~≥1.5 MiB).
DEFAULT_MIN_BYTES = 1_048_576

app = modal.App("exp-mmar-question-difficulty-cleanup")


@dataclass
class RunInfo:
    run_id: str
    file_bytes: int = 0
    n_files: int = 0
    models: set[str] = field(default_factory=set)
    has_manifest: bool = False
    has_scores: bool = False
    has_difficulty: bool = False
    num_samples: int | None = None
    n_shots: int | None = None
    mode: str | None = None

    @property
    def remote_path(self) -> str:
        return f"{EXPERIMENT_PREFIX}/{self.run_id}"


def _read_json(path: str) -> dict | None:
    try:
        data = b"".join(results_volume.read_file(path))
    except Exception:
        return None
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def list_runs(*, read_manifests: bool = True) -> list[RunInfo]:
    """Aggregate per-run file sizes under the experiment prefix."""
    entries = results_volume.listdir(EXPERIMENT_PREFIX, recursive=True)
    by_run: dict[str, RunInfo] = {}

    for entry in entries:
        parts = entry.path.strip("/").split("/")
        if len(parts) < 2:
            continue
        run_id = parts[1]
        info = by_run.setdefault(run_id, RunInfo(run_id=run_id))

        if entry.type == FileEntryType.DIRECTORY:
            if len(parts) == 4 and parts[2] == "models":
                info.models.add(parts[3])
            continue

        size = int(entry.size or 0)
        info.file_bytes += size
        info.n_files += 1
        rel = "/".join(parts[2:])
        if rel == "manifest.json":
            info.has_manifest = True
        elif rel == "scores.json":
            info.has_scores = True
        elif rel == "difficulty.jsonl":
            info.has_difficulty = True
        elif (
            len(parts) >= 5
            and parts[2] == "models"
            and parts[4] == "predictions.jsonl"
        ):
            info.models.add(parts[3])

    if read_manifests:
        for info in by_run.values():
            if not info.has_manifest:
                continue
            manifest = _read_json(f"{info.remote_path}/manifest.json")
            if not manifest:
                continue
            if "num_samples" in manifest:
                try:
                    info.num_samples = int(manifest["num_samples"])
                except (TypeError, ValueError):
                    pass
            if "n_shots" in manifest:
                try:
                    info.n_shots = int(manifest["n_shots"])
                except (TypeError, ValueError):
                    pass
            mode = manifest.get("mode")
            if isinstance(mode, str):
                info.mode = mode

    return sorted(by_run.values(), key=lambda r: (r.file_bytes, r.run_id))


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KiB"
    if n < 1024**3:
        return f"{n / 1024**2:.2f} MiB"
    return f"{n / 1024**3:.2f} GiB"


def _fmt_manifest(info: RunInfo) -> str:
    bits: list[str] = []
    if info.num_samples is not None:
        bits.append(f"n={info.num_samples}")
    if info.n_shots is not None:
        bits.append(f"shots={info.n_shots}")
    if info.mode:
        bits.append(f"mode={info.mode}")
    models = ",".join(sorted(info.models)) or "-"
    bits.append(f"models={models}")
    flags = []
    if info.has_scores:
        flags.append("scores")
    if info.has_difficulty:
        flags.append("difficulty")
    if flags:
        bits.append("+".join(flags))
    return " ".join(bits)


def plan_cleanup(
    runs: list[RunInfo], min_bytes: int
) -> tuple[list[RunInfo], list[RunInfo]]:
    delete = [r for r in runs if r.file_bytes < min_bytes]
    keep = [r for r in runs if r.file_bytes >= min_bytes]
    return delete, keep


def delete_runs(runs: list[RunInfo]) -> None:
    for info in runs:
        path = info.remote_path
        print(f"  rm -r {path}  ({_fmt_bytes(info.file_bytes)})")
        results_volume.remove_file(path, recursive=True)


@app.local_entrypoint()
def main(
    min_bytes: int = DEFAULT_MIN_BYTES,
    execute: bool = False,
    skip_manifests: bool = False,
):
    """List (and optionally delete) undersized experiment runs.

    Args:
        min_bytes: Delete runs whose total file bytes are strictly below this
            (default 1 MiB — see module docstring for the size-gap rationale).
        execute: Actually delete. Without this flag the script only prints a
            dry-run plan.
        skip_manifests: Skip reading manifests (faster; table loses n/shots).
    """
    print(f"Volume: {RESULTS_VOLUME_NAME}/{EXPERIMENT_PREFIX}")
    print(f"Cutoff: delete if file_bytes < {min_bytes} ({_fmt_bytes(min_bytes)})")
    print(
        "Rationale: live sizes jump from ≤~187 KiB (smoke/aborted) to "
        "≥~1.5 MiB (real payloads); default cutoff sits in that gap."
    )
    print()

    runs = list_runs(read_manifests=not skip_manifests)
    if not runs:
        print("(no runs found)")
        return

    delete, keep = plan_cleanup(runs, min_bytes=min_bytes)

    print(f"{'action':<8} {'size':>10}  {'files':>5}  run_id  detail")
    print("-" * 100)
    for info in runs:
        action = "DELETE" if info.file_bytes < min_bytes else "keep"
        print(
            f"{action:<8} {_fmt_bytes(info.file_bytes):>10}  {info.n_files:5d}  "
            f"{info.run_id}  {_fmt_manifest(info)}"
        )

    delete_bytes = sum(r.file_bytes for r in delete)
    keep_bytes = sum(r.file_bytes for r in keep)
    print()
    print(
        f"Plan: delete {len(delete)} runs ({_fmt_bytes(delete_bytes)}), "
        f"keep {len(keep)} runs ({_fmt_bytes(keep_bytes)})"
    )

    if not delete:
        print("Nothing to delete.")
        return

    if not execute:
        print("Dry-run only. Re-run with --execute to remove DELETE rows.")
        return

    print("Deleting…")
    delete_runs(delete)
    print(f"Removed {len(delete)} runs.")
