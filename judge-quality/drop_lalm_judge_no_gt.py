"""Drop the ``lalm-judge-no-gt`` pack (``neutral_no_gt`` LALM-as-judge).

Deletes the whole pack from the ``mmar-judging`` Modal volume and from
local downloads. That pack is only the default LALM format
(``neutral_no_gt``, no gold). Does not touch ``llm-judge-gt`` or
``run_judges.py`` outputs under ``models/``.

Resume on ``run_lalm_judge_no_gt.py`` would otherwise keep the mixed-source
generations and sidecar verdicts. After this drop, a fresh prepare copies
only ``mmar-freeform-thinking``.

Usage::

    uv run modal run judge-quality/drop_lalm_judge_no_gt.py --dry-run
    uv run modal run judge-quality/drop_lalm_judge_no_gt.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import modal

from modal_cache import JUDGING_VOLUME_NAME, judging_volume

PACK_NAME = "lalm-judge-no-gt"
GRADE_PROMPT = "neutral_no_gt"

LOCAL_PACK_DIRS = (
    _REPO_ROOT / "outputs" / "judge-quality" / PACK_NAME,
    _REPO_ROOT / "outputs" / "mmar-judging" / PACK_NAME,
)

app = modal.App("drop-lalm-judge-no-gt")


def _entry_path(entry: object) -> str:
    return str(getattr(entry, "path", None) or entry)


def _remote_paths(pack: str = PACK_NAME) -> list[str]:
    prefix = f"{pack}/"
    try:
        return [_entry_path(entry) for entry in judging_volume.listdir(pack, recursive=True)]
    except Exception:
        pass
    try:
        entries = judging_volume.listdir("/", recursive=False)
    except Exception as exc:
        print(f"[drop-lalm-judge-no-gt] volume:{JUDGING_VOLUME_NAME}/ missing ({exc})")
        return []
    names = {_entry_path(entry).rstrip("/") for entry in entries}
    if pack not in names and not any(name.startswith(prefix) for name in names):
        return []
    try:
        entries = judging_volume.listdir("/", recursive=True)
    except Exception as exc:
        print(f"[drop-lalm-judge-no-gt] volume:{JUDGING_VOLUME_NAME}/ missing ({exc})")
        return []
    return [
        path
        for entry in entries
        if (path := _entry_path(entry)) == pack or path.startswith(prefix)
    ]


def _drop_remote(*, dry_run: bool) -> dict[str, object]:
    paths = _remote_paths()
    print(
        f"[drop-lalm-judge-no-gt] volume:{JUDGING_VOLUME_NAME}/{PACK_NAME} "
        f"({len(paths)} entries, format={GRADE_PROMPT})"
    )
    for path in paths[:20]:
        print(f"  {path}")
    if len(paths) > 20:
        print(f"  … {len(paths) - 20} more")
    if dry_run:
        print("[drop-lalm-judge-no-gt] dry-run: skip volume delete")
        return {"status": "dry-run", "n_entries": len(paths), "deleted": False}
    if not paths:
        print("[drop-lalm-judge-no-gt] nothing to delete on volume")
        return {"status": "absent", "n_entries": 0, "deleted": False}
    judging_volume.remove_file(PACK_NAME, recursive=True)
    leftover = _remote_paths()
    if leftover:
        raise SystemExit(
            f"Volume pack still present after delete: {len(leftover)} entries"
        )
    print(f"[drop-lalm-judge-no-gt] deleted volume:{JUDGING_VOLUME_NAME}/{PACK_NAME}")
    return {"status": "ok", "n_entries": len(paths), "deleted": True}


def _drop_local(*, dry_run: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for path in LOCAL_PACK_DIRS:
        exists = path.is_dir()
        n_files = sum(1 for _ in path.rglob("*") if _.is_file()) if exists else 0
        print(
            f"[drop-lalm-judge-no-gt] local {path} "
            f"({'present' if exists else 'absent'}, {n_files} files)"
        )
        if dry_run:
            results.append(
                {
                    "path": str(path),
                    "status": "dry-run" if exists else "absent",
                    "n_files": n_files,
                    "deleted": False,
                }
            )
            continue
        if not exists:
            results.append(
                {
                    "path": str(path),
                    "status": "absent",
                    "n_files": 0,
                    "deleted": False,
                }
            )
            continue
        shutil.rmtree(path)
        if path.exists():
            raise SystemExit(f"Local pack still present after delete: {path}")
        print(f"[drop-lalm-judge-no-gt] deleted {path}")
        results.append(
            {
                "path": str(path),
                "status": "ok",
                "n_files": n_files,
                "deleted": True,
            }
        )
    return results


@app.local_entrypoint()
def main(dry_run: bool = False) -> dict:
    """Delete ``lalm-judge-no-gt`` from Modal and local downloads.

    Args:
        dry_run: Print what would be removed; do not delete.
    """
    print(
        f"[drop-lalm-judge-no-gt] pack={PACK_NAME} format={GRADE_PROMPT} "
        f"dry_run={dry_run}"
    )
    remote = _drop_remote(dry_run=dry_run)
    local = _drop_local(dry_run=dry_run)
    result = {"pack": PACK_NAME, "format": GRADE_PROMPT, "remote": remote, "local": local}
    print("[drop-lalm-judge-no-gt] done:", result)
    return result


if __name__ == "__main__":
    print(
        "Run via Modal:\n"
        "  uv run modal run judge-quality/drop_lalm_judge_no_gt.py --dry-run\n"
        "  uv run modal run judge-quality/drop_lalm_judge_no_gt.py",
        file=sys.stderr,
    )
    raise SystemExit(2)
