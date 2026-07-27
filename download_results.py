"""Download eval artifacts from the ``latent-reasoning-results`` Modal Volume.

Volume layout (mirrors locally under ``<repo>/outputs/``):

    mmar/af3/<run_id>/
      predictions.jsonl              # generations + CoT + answers
      predictions.evaluated.jsonl    # OpenAI rubric grades
      scores.json
      manifest.json
    mmar/af-next-think/<run_id>/     # Audio Flamingo Next Think runs
    exp-mmar-question-difficulty/<run_id>/
      difficulty.jsonl
      scores.json
      manifest.json
      models/<label>/predictions.jsonl

``modal volume get`` places the last path component under the local
destination, so nested downloads must target the parent directory to
preserve the volume tree. This script computes that destination
automatically.

Usage:

    uv run modal run download_results.py
    uv run modal run download_results.py --list-only
    uv run modal run download_results.py --remote-path mmar/af3
    uv run modal run download_results.py --remote-path mmar/af-next-think
    uv run modal run download_results.py --remote-path mmar/af3/20260712T185300Z
    uv run modal run download_results.py --remote-path exp-mmar-question-difficulty
    uv run modal run download_results.py \\
      --remote-path exp-mmar-question-difficulty/20260727T154400Z
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

RESULTS_VOLUME_NAME = "latent-reasoning-results"
DEFAULT_REMOTE_PATH = "/"
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "outputs"

app = modal.App("download-results")
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _normalize_remote(path: str) -> str:
    cleaned = path.strip() or "/"
    if cleaned != "/":
        cleaned = cleaned.strip("/")
    return cleaned


def resolve_local_dest(remote_path: str, local_dir: str | Path) -> Path:
    """Map a Volume subpath to the local parent dir ``modal volume get`` expects.

    ``modal volume get VOLUME a/b/c DEST`` writes to ``DEST/c/``. To mirror
    ``outputs/a/b/c/`` we therefore pass ``DEST=outputs/a/b``.
    """
    remote = _normalize_remote(remote_path)
    base = Path(local_dir).expanduser().resolve()
    if remote == "/":
        return base
    parts = remote.split("/")
    if len(parts) == 1:
        return base
    return base.joinpath(*parts[:-1])


def list_results(remote_path: str = "/") -> list[str]:
    """Return recursive paths under ``remote_path`` on the results Volume."""
    remote = _normalize_remote(remote_path)
    entries = results_volume.listdir(remote, recursive=True)
    paths: list[str] = []
    for entry in entries:
        path = getattr(entry, "path", None) or str(entry)
        paths.append(path)
        print(path)
    if not paths:
        print(f"(empty) volume:{RESULTS_VOLUME_NAME}/{remote}")
    return paths


def download_results(
    remote_path: str = DEFAULT_REMOTE_PATH,
    local_dir: str | Path = DEFAULT_LOCAL_DIR,
    force: bool = True,
) -> Path:
    """Download ``remote_path`` from the results Volume into ``local_dir``.

    Volume root (``/``) maps flatly onto ``local_dir`` (default ``./outputs``).
    """
    remote = _normalize_remote(remote_path)
    dest = resolve_local_dest(remote_path, local_dir)
    dest.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "modal",
        "volume",
        "get",
        RESULTS_VOLUME_NAME,
        remote if remote != "/" else "/",
        str(dest),
    ]
    if force:
        cmd.append("--force")

    print(f"Downloading volume:{RESULTS_VOLUME_NAME}/{remote} -> {dest}")
    subprocess.run(cmd, check=True)
    saved = dest / remote.split("/")[-1] if remote != "/" else dest
    print(f"Saved to {saved}")
    return saved


@app.local_entrypoint()
def main(
    remote_path: str = DEFAULT_REMOTE_PATH,
    local_dir: str = str(DEFAULT_LOCAL_DIR),
    list_only: bool = False,
    force: bool = True,
):
    """List or download files from ``latent-reasoning-results``.

    Args:
        remote_path: Path inside the Volume (default: ``/``, the full tree).
        local_dir: Local destination directory (default: ``<repo>/outputs``).
        list_only: Only print remote paths; do not download.
        force: Overwrite existing local files (passed to ``modal volume get``).
    """
    if list_only:
        list_results(remote_path)
        return

    try:
        download_results(remote_path=remote_path, local_dir=local_dir, force=force)
    except subprocess.CalledProcessError as exc:
        print(
            f"Download failed (exit {exc.returncode}). "
            "List what is on the volume with:\n"
            "  uv run modal run download_results.py --list-only --remote-path /"
        )
        raise SystemExit(exc.returncode) from exc
