"""Download the judging pack from the ``mmar-judging`` Modal Volume.

Fetches what ``run_judges.py --accuracy-only`` reads: ``labels.csv``,
``manifest.json``, and each ``models/<label>/predictions.jsonl``. Sidecars
under ``judge_partials/`` are not downloaded.

Usage::

    uv run modal run download_judges.py
    uv run modal run download_judges.py --list-only
    uv run python view_judges.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

from modal_cache import JUDGING_VOLUME_NAME, judging_volume

DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "outputs" / "mmar-judging"

app = modal.App("download-judges")


def list_judges(remote_path: str = "/") -> list[str]:
    """Return recursive paths on the ``mmar-judging`` Volume."""
    remote = remote_path.strip() or "/"
    if remote != "/":
        remote = remote.strip("/")
    entries = judging_volume.listdir(remote, recursive=True)
    paths: list[str] = []
    for entry in entries:
        path = getattr(entry, "path", None) or str(entry)
        paths.append(path)
        print(path)
    if not paths:
        print(f"(empty) volume:{JUDGING_VOLUME_NAME}/{remote}")
    return paths


def _volume_get(remote: str, dest: Path, force: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "modal",
        "volume",
        "get",
        JUDGING_VOLUME_NAME,
        remote,
        str(dest),
    ]
    if force:
        cmd.append("--force")
    print(f"Downloading volume:{JUDGING_VOLUME_NAME}/{remote} -> {dest}")
    subprocess.run(cmd, check=True)


def download_judges(
    local_dir: str | Path = DEFAULT_LOCAL_DIR,
    force: bool = True,
) -> Path:
    """Download ``labels.csv``, ``manifest.json``, and merged predictions."""
    dest = Path(local_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    _volume_get("labels.csv", dest, force)
    _volume_get("manifest.json", dest, force)

    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    labels = [str(x) for x in (manifest.get("models") or []) if x]
    if not labels:
        raise SystemExit(f"No models listed in {dest / 'manifest.json'}")

    for label in labels:
        _volume_get(
            f"models/{label}/predictions.jsonl",
            dest / "models" / label,
            force,
        )

    print(f"Saved to {dest}")
    return dest


@app.local_entrypoint()
def main(
    local_dir: str = str(DEFAULT_LOCAL_DIR),
    list_only: bool = False,
    force: bool = True,
):
    """List or download the judging pack from ``mmar-judging``.

    Args:
        local_dir: Local pack directory (default: ``<repo>/outputs/mmar-judging``).
        list_only: Only print remote paths; do not download.
        force: Overwrite existing local files (passed to ``modal volume get``).
    """
    if list_only:
        list_judges()
        return

    try:
        saved = download_judges(local_dir=local_dir, force=force)
    except subprocess.CalledProcessError as exc:
        print(
            f"Download failed (exit {exc.returncode}). "
            "List what is on the volume with:\n"
            "  uv run modal run download_judges.py --list-only"
        )
        raise SystemExit(exc.returncode) from exc

    print("View with:\n  uv run python view_judges.py")
    print(f"(saved under {saved})")


if __name__ == "__main__":
    print(
        "Run via Modal:\n"
        "  uv run modal run download_judges.py\n"
        "  uv run modal run download_judges.py --list-only",
        file=sys.stderr,
    )
    raise SystemExit(2)
