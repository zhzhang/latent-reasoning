"""Download eval artifacts from Modal Volumes.

Default: the ``mmar-freeform-thinking`` volume (``run_experiment.py`` pack
root) into ``outputs/mmar-freeform-thinking/``.

Also supports:

    latent-reasoning-results          # legacy run_id folders
      exp-mmar-question-difficulty/<run_id>/
      mmar/af3/<run_id>/
      mmar/af-next-think/<run_id>/
    mmar-freeform-5-shot-thinking     # API collated pack (volume root)

Judge packs live on the separate ``mmar-judging`` volume; use
``download_judges.py`` / ``judge-quality/download_judge_quality.py``.

``modal volume get`` places the last path component under the local
destination, so nested downloads must target the parent directory to
preserve the volume tree. This script computes that destination
automatically. Volume-root downloads (``/``) write into ``local_dir``.

Usage:

    # Default: mmar-freeform-thinking volume root
    uv run modal run download_results.py
    uv run modal run download_results.py --list-only
    # Legacy question-difficulty runs:
    uv run modal run download_results.py \\
      --volume-name latent-reasoning-results
    uv run modal run download_results.py \\
      --volume-name latent-reasoning-results \\
      --remote-path exp-mmar-question-difficulty/20260807T031152Z
    uv run modal run download_results.py \\
      --volume-name mmar-freeform-5-shot-thinking
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

from modal_cache import (
    FREEFORM_THINKING_VOLUME_NAME,
    MMAR_DESCRIPTIONS_VOLUME_NAME,
    MMAR_FREEFORM_THINKING_VOLUME_NAME,
    RESULTS_VOLUME_NAME,
    freeform_thinking_volume,
    mmar_descriptions_volume,
    mmar_freeform_thinking_volume,
    results_volume,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VOLUME_NAME = MMAR_FREEFORM_THINKING_VOLUME_NAME

VOLUME_SPECS: dict[str, dict] = {
    MMAR_FREEFORM_THINKING_VOLUME_NAME: {
        "handle": mmar_freeform_thinking_volume,
        "remote": "/",
        "local": REPO_ROOT / "outputs" / "mmar-freeform-thinking",
    },
    RESULTS_VOLUME_NAME: {
        "handle": results_volume,
        "remote": "exp-mmar-question-difficulty",
        "local": REPO_ROOT / "outputs",
    },
    FREEFORM_THINKING_VOLUME_NAME: {
        "handle": freeform_thinking_volume,
        "remote": "/",
        "local": REPO_ROOT / "outputs" / "mmar-freeform-5-shot-thinking",
    },
    MMAR_DESCRIPTIONS_VOLUME_NAME: {
        "handle": mmar_descriptions_volume,
        "remote": "/",
        "local": REPO_ROOT / "outputs" / "mmar-descriptions",
    },
}

app = modal.App("download-results")


def _volume_spec(volume_name: str) -> dict:
    name = (volume_name or DEFAULT_VOLUME_NAME).strip() or DEFAULT_VOLUME_NAME
    spec = VOLUME_SPECS.get(name)
    if spec is None:
        known = ", ".join(VOLUME_SPECS)
        raise SystemExit(f"Unknown volume {name!r}. Choose from: {known}")
    return spec


def _normalize_remote(path: str) -> str:
    cleaned = path.strip() or "/"
    if cleaned != "/":
        cleaned = cleaned.strip("/")
    return cleaned


def resolve_local_dest(remote_path: str, local_dir: str | Path) -> Path:
    """Map a Volume subpath to the local parent dir ``modal volume get`` expects.

    ``modal volume get VOLUME a/b/c DEST`` writes to ``DEST/c/``. To mirror
    ``outputs/a/b/c/`` we therefore pass ``DEST=outputs/a/b``. Volume root
    (``/``) writes directly into ``local_dir``.
    """
    remote = _normalize_remote(remote_path)
    base = Path(local_dir).expanduser().resolve()
    if remote == "/":
        return base
    parts = remote.split("/")
    if len(parts) == 1:
        return base
    return base.joinpath(*parts[:-1])


def list_results(
    remote_path: str = "/",
    volume_name: str = DEFAULT_VOLUME_NAME,
) -> list[str]:
    """Return recursive paths under ``remote_path`` on the selected Volume."""
    spec = _volume_spec(volume_name)
    remote = _normalize_remote(remote_path) if remote_path else spec["remote"]
    remote = _normalize_remote(str(remote))
    handle = spec["handle"]
    entries = handle.listdir(remote, recursive=True)
    paths: list[str] = []
    for entry in entries:
        path = getattr(entry, "path", None) or str(entry)
        paths.append(path)
        print(path)
    if not paths:
        print(f"(empty) volume:{volume_name}/{remote}")
    return paths


def download_results(
    remote_path: str | None = None,
    local_dir: str | Path | None = None,
    force: bool = True,
    volume_name: str = DEFAULT_VOLUME_NAME,
) -> Path:
    """Download ``remote_path`` from ``volume_name`` into ``local_dir``."""
    spec = _volume_spec(volume_name)
    remote = _normalize_remote(
        remote_path if remote_path not in (None, "") else str(spec["remote"])
    )
    dest_root = Path(local_dir).expanduser().resolve() if local_dir else spec["local"]
    dest = resolve_local_dest(remote, dest_root)
    dest.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "modal",
        "volume",
        "get",
        volume_name,
        remote if remote != "/" else "/",
        str(dest),
    ]
    if force:
        cmd.append("--force")

    print(f"Downloading volume:{volume_name}/{remote} -> {dest}")
    subprocess.run(cmd, check=True)
    saved = dest / remote.split("/")[-1] if remote != "/" else dest
    print(f"Saved to {saved}")
    return saved


@app.local_entrypoint()
def main(
    volume_name: str = DEFAULT_VOLUME_NAME,
    remote_path: str = "",
    local_dir: str = "",
    list_only: bool = False,
    force: bool = True,
):
    """List or download files from a results Volume.

    Args:
        volume_name: Modal Volume (default: ``mmar-freeform-thinking``).
            Also ``latent-reasoning-results``, ``mmar-freeform-5-shot-thinking``.
        remote_path: Path inside the Volume. Empty uses the volume default
            (``/`` for pack volumes, ``exp-mmar-question-difficulty`` for
            the legacy results volume). Pass ``/`` for the full tree.
        local_dir: Local destination. Empty uses the volume default
            (``outputs/mmar-freeform-thinking`` for the experiment pack).
        list_only: Only print remote paths; do not download.
        force: Overwrite existing local files (passed to ``modal volume get``).
    """
    spec = _volume_spec(volume_name)
    remote = remote_path if remote_path.strip() else str(spec["remote"])
    dest = local_dir if local_dir.strip() else str(spec["local"])

    if list_only:
        list_results(remote, volume_name=volume_name)
        return

    try:
        saved = download_results(
            remote_path=remote,
            local_dir=dest,
            force=force,
            volume_name=volume_name,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"Download failed (exit {exc.returncode}). "
            "List what is on the volume with:\n"
            "  uv run modal run download_results.py --list-only"
        )
        raise SystemExit(exc.returncode) from exc

    print("View with:\n  uv run python view_difficulty.py")
    print(f"(saved under {saved})")
