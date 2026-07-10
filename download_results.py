"""Download eval artifacts from the ``latent-reasoning-results`` Modal Volume.

Layout produced by eval runners (e.g. ``run_audio_flamingo3_mmar.py``):

    mmar/audio_flamingo3_mmar/<run_id>/
      predictions.jsonl              # generations + CoT + answers
      predictions.evaluated.jsonl    # OpenAI rubric grades
      scores.json
      manifest.json
    mmar/audio_flamingo3_mmar/latest/   # copy of the most recent run

Usage:

    uv run modal run download_results.py
    uv run modal run download_results.py --list-only
    uv run modal run download_results.py --remote-path mmar/audio_flamingo3_mmar/latest
    uv run modal run download_results.py --remote-path mmar --local-dir ./output/results
    uv run modal run download_results.py --remote-path / --local-dir ./output/results --force
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

RESULTS_VOLUME_NAME = "latent-reasoning-results"
DEFAULT_REMOTE_PATH = "mmar/audio_flamingo3_mmar/latest"
DEFAULT_LOCAL_DIR = Path("output/results")

app = modal.App("download-results")
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _normalize_remote(path: str) -> str:
    cleaned = path.strip() or "/"
    if cleaned != "/":
        cleaned = cleaned.strip("/")
    return cleaned


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
    """Download ``remote_path`` from the results Volume into ``local_dir``."""
    remote = _normalize_remote(remote_path)
    dest = Path(local_dir).expanduser().resolve()
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
    print(f"Saved to {dest}")
    return dest


@app.local_entrypoint()
def main(
    remote_path: str = DEFAULT_REMOTE_PATH,
    local_dir: str = str(DEFAULT_LOCAL_DIR),
    list_only: bool = False,
    force: bool = True,
):
    """List or download files from ``latent-reasoning-results``.

    Args:
        remote_path: Path inside the Volume (default: latest MMAR AF3 run).
        local_dir: Local destination directory.
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
