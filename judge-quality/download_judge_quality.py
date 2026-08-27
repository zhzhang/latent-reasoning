"""Download judge-quality packs from the ``mmar-judging`` Modal Volume.

Pulls the first-shot LLM-GT and LALM-no-GT packs written by
``run_llm_judge_gt.py`` and ``run_lalm_judge_no_gt.py`` into
``outputs/judge-quality/``.

Test-taker generations for those scripts come from the
``mmar-freeform-thinking`` volume; download that pack with
``uv run modal run download_results.py`` into
``outputs/mmar-freeform-thinking/``.

Usage::

    uv run modal run judge-quality/download_judge_quality.py
    uv run modal run judge-quality/download_judge_quality.py --list-only
    uv run modal run judge-quality/download_judge_quality.py --pack llm-judge-gt
    uv run python judge-quality/score_lalm_vs_gt.py
    uv run python judge-quality/view_judge_quality.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import modal

from modal_cache import JUDGING_VOLUME_NAME, judging_volume

GT_PACK_NAME = "llm-judge-gt"
LALM_PACK_NAME = "lalm-judge-no-gt"
DEFAULT_PACKS = (GT_PACK_NAME, LALM_PACK_NAME)
DEFAULT_LOCAL_DIR = _REPO_ROOT / "outputs" / "judge-quality"

app = modal.App("download-judge-quality")


def _normalize_pack(raw: str) -> list[str]:
    text = str(raw or "all").strip().lower()
    if not text or text == "all":
        return list(DEFAULT_PACKS)
    wanted = [part.strip() for part in text.split(",") if part.strip()]
    known = {name.lower(): name for name in DEFAULT_PACKS}
    out: list[str] = []
    for item in wanted:
        name = known.get(item) or item
        if name not in DEFAULT_PACKS:
            raise SystemExit(
                f"Unknown pack {item!r}. Expected one of {list(DEFAULT_PACKS)} or all."
            )
        if name not in out:
            out.append(name)
    return out


def list_judge_quality(remote_path: str = "/") -> list[str]:
    """Print recursive paths on ``mmar-judging``, highlighting quality packs."""
    remote = (remote_path or "/").strip() or "/"
    if remote != "/":
        remote = remote.strip("/")
    entries = judging_volume.listdir(remote, recursive=True)
    paths: list[str] = []
    for entry in entries:
        path = getattr(entry, "path", None) or str(entry)
        paths.append(path)
        if any(path == name or path.startswith(f"{name}/") for name in DEFAULT_PACKS):
            print(path)
        elif remote != "/" and path.startswith(remote):
            print(path)
    shown = [
        path
        for path in paths
        if any(path == name or path.startswith(f"{name}/") for name in DEFAULT_PACKS)
    ]
    if not shown:
        print(f"(no {', '.join(DEFAULT_PACKS)} under volume:{JUDGING_VOLUME_NAME}/{remote})")
        for path in paths[:40]:
            print(f"  {path}")
        if len(paths) > 40:
            print(f"  … {len(paths) - 40} more")
    return paths


def download_pack(pack: str, local_dir: str | Path, *, force: bool = True) -> Path:
    dest = Path(local_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "modal",
        "volume",
        "get",
        JUDGING_VOLUME_NAME,
        pack,
        str(dest),
    ]
    if force:
        cmd.append("--force")
    print(f"Downloading volume:{JUDGING_VOLUME_NAME}/{pack} -> {dest / pack}")
    subprocess.run(cmd, check=True)
    saved = dest / pack
    print(f"Saved to {saved}")
    return saved


@app.local_entrypoint()
def main(
    pack: str = "all",
    local_dir: str = str(DEFAULT_LOCAL_DIR),
    list_only: bool = False,
    force: bool = True,
) -> None:
    """List or download judge-quality packs from ``mmar-judging``.

    Args:
        pack: ``all``, ``llm-judge-gt``, ``lalm-judge-no-gt``, or a comma list.
        local_dir: Local parent directory (default: ``outputs/judge-quality``).
        list_only: Only print remote paths; do not download.
        force: Overwrite existing local files.
    """
    if list_only:
        list_judge_quality()
        return

    packs = _normalize_pack(pack)
    saved: list[Path] = []
    try:
        for name in packs:
            saved.append(download_pack(name, local_dir=local_dir, force=force))
    except subprocess.CalledProcessError as exc:
        print(
            f"Download failed (exit {exc.returncode}). "
            "List what is on the volume with:\n"
            "  uv run modal run judge-quality/download_judge_quality.py --list-only"
        )
        raise SystemExit(exc.returncode) from exc

    print("Score LALM judges vs LLM-GT:\n  uv run python judge-quality/score_lalm_vs_gt.py")
    print("Browse per question:\n  uv run python judge-quality/view_judge_quality.py")
    for path in saved:
        print(f"(saved under {path})")


if __name__ == "__main__":
    print(
        "Run via Modal:\n"
        "  uv run modal run judge-quality/download_judge_quality.py\n"
        "  uv run modal run judge-quality/download_judge_quality.py --list-only",
        file=sys.stderr,
    )
    raise SystemExit(2)
