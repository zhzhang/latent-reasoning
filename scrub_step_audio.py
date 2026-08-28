"""Scrub ``step-audio-r1.1`` from result manifests, Modal volumes, and local packs.

Deletes model dirs / smoke traces / compile-cache / seeded weights, then
rewrites leftover ``manifest.json``, scores, CSVs, and prediction sidecars
so the label (and ``{label}__…`` judge slugs) are gone.

Targets:

  Volumes
    mmar-freeform-thinking
    mmar-freeform-5-shot-thinking
    mmar-descriptions
    mmar-judging
    latent-reasoning-results
    latent-reasoning          # weights + ``vllm/step-audio-r1.1`` (skip with --skip-cache)

  Local
    outputs/
    exports/

Usage::

    uv run modal run scrub_step_audio.py --dry-run
    uv run modal run scrub_step_audio.py
    uv run python scrub_step_audio.py --local-only --dry-run
    uv run python scrub_step_audio.py --local-only
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import modal

from modal_cache import (
    FREEFORM_THINKING_VOLUME_NAME,
    JUDGING_VOLUME_NAME,
    MMAR_DESCRIPTIONS_VOLUME_NAME,
    MMAR_FREEFORM_THINKING_VOLUME_NAME,
    RESULTS_VOLUME_NAME,
    VOLUME_NAME,
    freeform_thinking_volume,
    judging_volume,
    mmar_descriptions_volume,
    mmar_freeform_thinking_volume,
    results_volume,
    volume,
)
from mmar_common import recompute_multi_judge_scores

REPO_ROOT = Path(__file__).resolve().parent
DROP_LABEL = "step-audio-r1.1"
DROP_REPO_ID = "stepfun-ai/Step-Audio-R1.1"
DROP_REPO_DIR = "Step-Audio-R1.1"
DROP_TOKENS = frozenset(
    {
        DROP_LABEL,
        DROP_LABEL.lower(),
        DROP_REPO_ID,
        DROP_REPO_ID.lower(),
        DROP_REPO_DIR,
        DROP_REPO_DIR.lower(),
    }
)
ROW_ID_KEYS = (
    "model_label",
    "model",
    "label",
    "judge_key",
    "judge",
    "grader_model_id",
    "primary_judge",
)
REWRITE_SUFFIXES = {".json", ".jsonl", ".csv"}
LOCAL_ROOTS = (
    REPO_ROOT / "outputs",
    REPO_ROOT / "exports",
)
CACHE_PREFIXES = (
    f"models/{DROP_REPO_ID}",
    f"vllm/{DROP_LABEL}",
)

app = modal.App("scrub-step-audio")

RESULTS_VOLUMES: tuple[tuple[str, modal.Volume], ...] = (
    (MMAR_FREEFORM_THINKING_VOLUME_NAME, mmar_freeform_thinking_volume),
    (FREEFORM_THINKING_VOLUME_NAME, freeform_thinking_volume),
    (MMAR_DESCRIPTIONS_VOLUME_NAME, mmar_descriptions_volume),
    (JUDGING_VOLUME_NAME, judging_volume),
    (RESULTS_VOLUME_NAME, results_volume),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_drop_token(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lower = text.lower()
    if lower in DROP_TOKENS or text in DROP_TOKENS:
        return True
    return lower == DROP_LABEL or lower.startswith(f"{DROP_LABEL}__")


def _normalize_remote(path: str) -> str:
    return str(path or "").strip().strip("/")


def _parts(path: str) -> list[str]:
    return [part for part in _normalize_remote(path).split("/") if part]


def is_owned_path(path: str) -> bool:
    """True when this file/dir belongs to Step-Audio (delete the tree)."""
    parts = _parts(path)
    if not parts:
        return False
    for index, part in enumerate(parts):
        stem = part.rsplit(".", 1)[0] if "." in part else part
        if is_drop_token(part) or is_drop_token(stem):
            return True
        if part.lower() == "stepfun-ai" and index + 1 < len(parts):
            if is_drop_token(parts[index + 1]):
                return True
    return False


def owned_delete_root(path: str) -> str:
    """Shortest prefix that identifies the Step-Audio artifact."""
    parts = _parts(path)
    for index, part in enumerate(parts):
        stem = part.rsplit(".", 1)[0] if "." in part else part
        if is_drop_token(part) or is_drop_token(stem):
            return "/".join(parts[: index + 1])
        if part.lower() == "stepfun-ai" and index + 1 < len(parts):
            if is_drop_token(parts[index + 1]):
                return "/".join(parts[: index + 2])
    return _normalize_remote(path)


def collapse_delete_roots(paths: Iterable[str]) -> list[str]:
    roots: list[str] = []
    for path in sorted({owned_delete_root(p) for p in paths}, key=lambda p: (p.count("/"), p)):
        if any(path == root or path.startswith(f"{root}/") for root in roots):
            continue
        roots.append(path)
    return roots


def _under_deleted(path: str, roots: Iterable[str]) -> bool:
    text = _normalize_remote(path)
    for root in roots:
        if text == root or text.startswith(f"{root}/"):
            return True
    return False


def scrub_json(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        changed = False
        for key, child in value.items():
            if is_drop_token(key):
                changed = True
                continue
            cleaned, child_changed = scrub_json(child)
            changed = changed or child_changed
            out[str(key)] = cleaned
        for field in ("model_id", "grader_model_id", "primary_judge"):
            if field in out and is_drop_token(out[field]):
                out[field] = None
                changed = True
        return out, changed
    if isinstance(value, list):
        kept: list[Any] = []
        changed = False
        for item in value:
            if isinstance(item, str) and is_drop_token(item):
                changed = True
                continue
            if isinstance(item, dict):
                identity = (
                    item.get("label")
                    or item.get("model_label")
                    or item.get("model")
                    or item.get("model_id")
                    or item.get("judge_key")
                )
                if identity is not None and is_drop_token(identity):
                    changed = True
                    continue
            cleaned, child_changed = scrub_json(item)
            changed = changed or child_changed
            kept.append(cleaned)
        return kept, changed
    return value, False


def _drop_record(record: dict[str, Any]) -> bool:
    for key in ROW_ID_KEYS:
        if key in record and is_drop_token(record.get(key)):
            return True
    return False


def _has_judge_payload(record: dict[str, Any]) -> bool:
    if record.get("judges") or record.get("per_judge"):
        return True
    return any(
        isinstance(shot, dict) and shot.get("judges")
        for shot in (record.get("shots") or [])
    )


def scrub_prediction_record(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    record, changed = scrub_json(record)
    if not isinstance(record, dict):
        return record, changed
    if changed and _has_judge_payload(record):
        recompute_multi_judge_scores(record)
    return record, changed


def scrub_json_text(text: str) -> tuple[str, bool]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text, False
    cleaned, changed = scrub_json(payload)
    if not changed:
        return text, False
    if isinstance(cleaned, dict) and any(
        key in cleaned for key in ("models", "progress", "workload", "updated_at")
    ):
        cleaned["updated_at"] = _now()
    new_text = json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n"
    return new_text, True


def scrub_jsonl_text(text: str, *, predictions: bool) -> tuple[str, bool]:
    out_lines: list[str] = []
    changed = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out_lines.append(raw)
            continue
        if not isinstance(row, dict):
            out_lines.append(raw)
            continue
        if _drop_record(row):
            changed = True
            continue
        if predictions:
            cleaned, row_changed = scrub_prediction_record(row)
        else:
            cleaned, row_changed = scrub_json(row)
        changed = changed or row_changed
        out_lines.append(json.dumps(cleaned, ensure_ascii=False) if row_changed else line)
    if not changed:
        return text, False
    return "\n".join(out_lines) + ("\n" if out_lines else ""), True


def scrub_csv_text(text: str) -> tuple[str, bool]:
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        return text, False
    kept: list[dict[str, str]] = []
    dropped = 0
    for row in reader:
        if any(is_drop_token(row.get(key)) for key in ROW_ID_KEYS if key in row):
            dropped += 1
            continue
        kept.append(row)
    if not dropped:
        return text, False
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept)
    return buf.getvalue(), True


def scrub_text(path: str, text: str) -> tuple[str, bool]:
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return scrub_csv_text(text)
    if suffix == ".jsonl":
        return scrub_jsonl_text(text, predictions=name == "predictions.jsonl")
    if suffix == ".json":
        return scrub_json_text(text)
    return text, False


def mentions_drop(text: str) -> bool:
    lower = text.lower()
    return DROP_LABEL in lower or "step-audio-r1" in lower or "stepfun-ai/step-audio" in lower


def _entry_path(entry: object) -> str:
    return _normalize_remote(getattr(entry, "path", None) or entry)


def _list_volume(handle: modal.Volume, prefix: str = "/") -> list[str]:
    remote = prefix.strip() or "/"
    try:
        entries = handle.listdir(remote, recursive=True)
    except Exception as exc:  # noqa: BLE001 — missing prefix is fine
        if remote not in {"", "/"}:
            return []
        print(f"  listdir failed: {exc}")
        return []
    return [_entry_path(entry) for entry in entries]


def _read_volume_text(handle: modal.Volume, path: str) -> str:
    chunks: list[bytes] = []
    for chunk in handle.read_file(path):
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8")


def _put_volume_text(handle: modal.Volume, path: str, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=Path(path).suffix, delete=False) as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        with handle.batch_upload(force=True) as batch:
            batch.put_file(tmp_path, f"/{_normalize_remote(path)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _aggregate_local_pack(pack_dir: Path) -> None:
    if not (pack_dir / "models").is_dir():
        return
    if not any((pack_dir / "models").glob("*/predictions.jsonl")):
        return
    try:
        from aggregate import aggregate_difficulty
    except Exception as exc:  # noqa: BLE001
        print(f"  skip aggregate {pack_dir}: {exc}")
        return
    try:
        aggregate_difficulty(pack_dir)
        print(f"  re-aggregated {pack_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"  aggregate failed {pack_dir}: {exc}")


def scrub_local_tree(root: Path, *, dry_run: bool) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "root": str(root),
        "deleted": [],
        "rewritten": [],
        "absent": not root.exists(),
    }
    if not root.exists():
        print(f"[local] {root} absent")
        return stats

    owned: list[str] = []
    rewrite_candidates: list[Path] = []
    for path in root.rglob("*"):
        rel = str(path.relative_to(root))
        if is_owned_path(rel):
            owned.append(rel)
            continue
        if path.is_file() and path.suffix.lower() in REWRITE_SUFFIXES:
            rewrite_candidates.append(path)

    delete_roots = collapse_delete_roots(owned)
    print(f"[local] {root} delete {len(delete_roots)} trees, scan {len(rewrite_candidates)} files")
    for rel in delete_roots:
        target = root / rel
        print(f"  delete {target}")
        stats["deleted"].append(str(target))
        if dry_run or not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    rewritten_packs: set[Path] = set()
    for path in rewrite_candidates:
        rel = str(path.relative_to(root))
        if _under_deleted(rel, delete_roots):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not mentions_drop(text):
            continue
        new_text, changed = scrub_text(rel, text)
        if not changed:
            continue
        print(f"  rewrite {path}")
        stats["rewritten"].append(str(path))
        if dry_run:
            continue
        path.write_text(new_text, encoding="utf-8")
        if path.name == "manifest.json":
            rewritten_packs.add(path.parent)

    if not dry_run:
        for pack_dir in sorted(rewritten_packs):
            _aggregate_local_pack(pack_dir)
    return stats


def scrub_volume(
    name: str,
    handle: modal.Volume,
    *,
    dry_run: bool,
    prefixes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    print(f"[volume] {name}")
    paths: list[str] = []
    if prefixes:
        for prefix in prefixes:
            found = _list_volume(handle, prefix)
            print(f"  {prefix}: {len(found)} entries")
            paths.extend(found)
    else:
        paths = _list_volume(handle, "/")
        print(f"  /: {len(paths)} entries")

    owned = [path for path in paths if is_owned_path(path)]
    delete_roots = collapse_delete_roots(owned)
    rewrite_files = [
        path
        for path in paths
        if Path(path).suffix.lower() in REWRITE_SUFFIXES
        and not _under_deleted(path, delete_roots)
    ]
    stats: dict[str, Any] = {
        "volume": name,
        "n_entries": len(paths),
        "deleted": list(delete_roots),
        "rewritten": [],
    }
    for root in delete_roots:
        print(f"  delete volume:{name}/{root}")
        if dry_run:
            continue
        handle.remove_file(root, recursive=True)

    for path in rewrite_files:
        suffix = Path(path).suffix.lower()
        if suffix not in REWRITE_SUFFIXES:
            continue
        try:
            text = _read_volume_text(handle, path)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip read {path}: {exc}")
            continue
        if not mentions_drop(text):
            continue
        new_text, changed = scrub_text(path, text)
        if not changed:
            continue
        print(f"  rewrite volume:{name}/{path}")
        stats["rewritten"].append(path)
        if dry_run:
            continue
        _put_volume_text(handle, path, new_text)
    return stats


def run_scrub(*, dry_run: bool, skip_cache: bool, local_only: bool) -> dict[str, Any]:
    print(
        f"[scrub-step-audio] label={DROP_LABEL} repo={DROP_REPO_ID} "
        f"dry_run={dry_run} skip_cache={skip_cache} local_only={local_only}"
    )
    result: dict[str, Any] = {
        "label": DROP_LABEL,
        "dry_run": dry_run,
        "local": [],
        "volumes": [],
    }
    for root in LOCAL_ROOTS:
        result["local"].append(scrub_local_tree(root, dry_run=dry_run))
    if local_only:
        return result
    for name, handle in RESULTS_VOLUMES:
        result["volumes"].append(scrub_volume(name, handle, dry_run=dry_run))
    if not skip_cache:
        result["volumes"].append(
            scrub_volume(
                VOLUME_NAME,
                volume,
                dry_run=dry_run,
                prefixes=CACHE_PREFIXES,
            )
        )
    return result


@app.local_entrypoint()
def main(
    dry_run: bool = False,
    skip_cache: bool = False,
    local_only: bool = False,
) -> dict:
    """Remove ``step-audio-r1.1`` from result packs.

    Args:
        dry_run: Print the plan; do not delete or rewrite.
        skip_cache: Leave ``latent-reasoning`` weights / compile cache.
        local_only: Only touch ``outputs/`` and ``exports/``.
    """
    result = run_scrub(dry_run=dry_run, skip_cache=skip_cache, local_only=local_only)
    print("[scrub-step-audio] done")
    return result


def _parse_argv(argv: list[str]) -> dict[str, bool]:
    flags = {"dry_run": False, "skip_cache": False, "local_only": False}
    for arg in argv:
        if arg in {"--dry-run", "--dry_run"}:
            flags["dry_run"] = True
        elif arg in {"--skip-cache", "--skip_cache"}:
            flags["skip_cache"] = True
        elif arg in {"--local-only", "--local_only"}:
            flags["local_only"] = True
        elif arg in {"-h", "--help"}:
            print(__doc__)
            raise SystemExit(0)
        else:
            raise SystemExit(f"unknown arg: {arg}\n{__doc__}")
    return flags


if __name__ == "__main__":
    flags = _parse_argv(sys.argv[1:])
    if not flags["local_only"]:
        print(
            "Run via Modal to touch volumes:\n"
            "  uv run modal run scrub_step_audio.py --dry-run\n"
            "  uv run modal run scrub_step_audio.py\n"
            "Or pass --local-only to rewrite outputs/ and exports/ only.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    run_scrub(dry_run=flags["dry_run"], skip_cache=True, local_only=True)
