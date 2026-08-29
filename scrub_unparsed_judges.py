"""Drop judge verdicts that the answer extractor cannot parse.

``run_judges.py`` stores an unparsed reply as ``correct=False`` / ``verdict=None``.
Resume then treats that shot as already graded, so the fail is sticky. This
script removes those entries from ``predictions.jsonl`` and matching
``judge_partials/*.jsonl`` sidecars so a later ``run_judges.py`` will retry them.

Does not change ``manifest.json`` / ``primary_judge``.

Usage::

    uv run python scrub_unparsed_judges.py --dry-run
    uv run python scrub_unparsed_judges.py
    uv run python scrub_unparsed_judges.py --judge qwen3-omni-instruct
    uv run python scrub_unparsed_judges.py --model gemma-4-e4b --examples 5
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

from grader import majority_grade_verdict, parse_grade_verdict
from mmar_common import (
    STRING_MATCH_JUDGE_LABEL,
    load_jsonl,
    recompute_multi_judge_scores,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PACK = REPO_ROOT / "outputs" / "mmar-judging"


def _csv_parts(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _judge_wanted(key: str, prefixes: list[str]) -> bool:
    if not prefixes:
        return True
    return any(key == prefix or key.startswith(prefix) for prefix in prefixes)


def parsed_judge_verdict(entry: dict) -> bool | None:
    """Re-run the grade extractor on stored judge text. None = unparsed."""
    samples = entry.get("samples")
    if isinstance(samples, list) and samples:
        verdicts = [
            parse_grade_verdict(str(sample.get("generation") or ""))
            for sample in samples
            if isinstance(sample, dict)
        ]
        if verdicts:
            return majority_grade_verdict(verdicts)
    generation = str(entry.get("generation") or "")
    if generation.strip():
        return parse_grade_verdict(generation)
    stored = entry.get("verdict")
    if stored == "pass":
        return True
    if stored == "fail":
        return False
    output = entry.get("output")
    if output == "1":
        return True
    if output == "0":
        return False
    return None


def is_unparsed_judge_entry(label: str, entry: object) -> bool:
    if label == STRING_MATCH_JUDGE_LABEL or not isinstance(entry, dict):
        return False
    return parsed_judge_verdict(entry) is None


def _clear_stale_legacy(record: dict) -> None:
    """Unparsed fails were mirrored onto shot ``correct`` / ``grader``.

    Leave those flags set and ``_shot_needs_grade`` will skip the retry via
    the legacy flat fields even after the judges map entry is gone.
    """
    primary = record.get("primary_judge")
    for shot in record.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        judges = shot.get("judges") if isinstance(shot.get("judges"), dict) else {}
        primary_entry = judges.get(primary) if primary else None
        if isinstance(primary_entry, dict) and primary_entry.get("correct") is not None:
            continue
        shot["correct"] = None
        shot.pop("grader", None)
        shot.pop("grader_output", None)
        shot["pending_grade"] = True


def scrub_record(
    record: dict,
    *,
    prefixes: list[str],
) -> tuple[int, int, Counter[str], list[tuple[str, str, int, str]]]:
    """Drop unparsed judge entries in place.

    Returns ``(n_entries, n_dropped, dropped_by_key, examples)``.
    """
    n_entries = 0
    n_dropped = 0
    dropped_by_key: Counter[str] = Counter()
    examples: list[tuple[str, str, int, str]] = []
    qid = str(record.get("id") or "")
    for shot in record.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        judges = shot.get("judges")
        if not isinstance(judges, dict):
            continue
        shot_index = int(shot.get("shot_index", 0))
        for key in list(judges):
            if not _judge_wanted(str(key), prefixes):
                continue
            entry = judges.get(key)
            n_entries += 1
            if not is_unparsed_judge_entry(str(key), entry):
                continue
            generation = ""
            if isinstance(entry, dict):
                generation = str(entry.get("generation") or "")
            examples.append((str(key), qid, shot_index, generation))
            del judges[key]
            n_dropped += 1
            dropped_by_key[str(key)] += 1
    if n_dropped:
        recompute_multi_judge_scores(record)
        _clear_stale_legacy(record)
    return n_entries, n_dropped, dropped_by_key, examples


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write_jsonl(tmp, rows, mode="w")
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp.replace(path)


def _discover_labels(pack_dir: Path, model: str) -> list[str]:
    models_root = pack_dir / "models"
    if str(model).lower() == "all":
        labels = sorted(
            path.parent.name
            for path in models_root.glob("*/predictions.jsonl")
            if path.is_file()
        )
    else:
        labels = [model]
    if not labels:
        raise SystemExit(f"no models with predictions.jsonl under {models_root}")
    return labels


def scrub_sidecars(
    pack_dir: Path,
    labels: list[str],
    *,
    prefixes: list[str],
    dry_run: bool,
) -> tuple[int, int, int]:
    """Drop unparsed sidecar rows. Returns ``(files, rows, dropped)``."""
    n_files = 0
    n_rows = 0
    n_dropped = 0
    for label in labels:
        partials_dir = pack_dir / "models" / label / "judge_partials"
        if not partials_dir.is_dir():
            continue
        for path in sorted(partials_dir.glob("*.jsonl")):
            n_files += 1
            kept: list[dict] = []
            file_dropped = 0
            for row in load_jsonl(path):
                if not isinstance(row, dict):
                    continue
                n_rows += 1
                key = str(row.get("judge_key") or path.stem)
                if not _judge_wanted(key, prefixes):
                    kept.append(row)
                    continue
                entry = row.get("entry")
                if is_unparsed_judge_entry(key, entry):
                    file_dropped += 1
                    continue
                kept.append(row)
            n_dropped += file_dropped
            if dry_run or not file_dropped:
                continue
            if kept:
                _atomic_write_jsonl(path, kept)
            else:
                path.unlink()
    return n_files, n_rows, n_dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove unparsed run_judges.py verdicts so a resume will re-grade them."
        )
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=DEFAULT_PACK,
        help="Judging pack directory (default: outputs/mmar-judging).",
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Gradee label under models/, or 'all' (default).",
    )
    parser.add_argument(
        "--judge",
        default="",
        help="Comma-separated judge-key prefixes (default: every LLM judge).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Print this many dropped generation snippets.",
    )
    args = parser.parse_args()

    pack_dir = args.pack
    prefixes = _csv_parts(args.judge)
    labels = _discover_labels(pack_dir, args.model)
    print(f"[scrub] pack={pack_dir} models={labels} judges={prefixes or '(all)'}")

    grand_entries = 0
    grand_dropped = 0
    dropped_by_key: Counter[str] = Counter()
    examples: list[tuple[str, str, int, str]] = []

    for label in labels:
        pred_path = pack_dir / "models" / label / "predictions.jsonl"
        if not pred_path.is_file():
            raise SystemExit(f"predictions.jsonl not found: {pred_path}")
        records = load_jsonl(pred_path)
        n_entries = 0
        n_dropped = 0
        for record in records:
            e, d, by_key, rec_examples = scrub_record(record, prefixes=prefixes)
            n_entries += e
            n_dropped += d
            dropped_by_key.update(by_key)
            if args.examples and len(examples) < args.examples:
                room = args.examples - len(examples)
                examples.extend(rec_examples[:room])
        grand_entries += n_entries
        grand_dropped += n_dropped
        rate = (n_entries - n_dropped) / n_entries if n_entries else 1.0
        print(
            f"{label}: parse rate {rate:.4f} "
            f"({n_entries - n_dropped}/{n_entries} parsed), dropped {n_dropped}"
        )
        if args.dry_run or not n_dropped:
            continue
        backup = pred_path.with_suffix(".jsonl.bak")
        shutil.copy2(pred_path, backup)
        _atomic_write_jsonl(pred_path, records)
        print(f"wrote {pred_path}")
        print(f"backup {backup}")

    n_sc_files, n_sc_rows, n_sc_dropped = scrub_sidecars(
        pack_dir, labels, prefixes=prefixes, dry_run=args.dry_run
    )
    sc_rate = (n_sc_rows - n_sc_dropped) / n_sc_rows if n_sc_rows else 1.0
    print(
        f"sidecars: parse rate {sc_rate:.4f} "
        f"({n_sc_rows - n_sc_dropped}/{n_sc_rows} parsed), "
        f"dropped {n_sc_dropped} rows in {n_sc_files} files"
    )
    if dropped_by_key:
        print("dropped by judge key:")
        for key, count in sorted(dropped_by_key.items()):
            print(f"  {key}: {count}")
    print(
        f"all: parse rate "
        f"{((grand_entries - grand_dropped) / grand_entries) if grand_entries else 1.0:.4f} "
        f"({grand_entries - grand_dropped}/{grand_entries} parsed), "
        f"dropped {grand_dropped}"
    )
    for key, qid, shot_index, generation in examples:
        snippet = generation.replace("\n", " ")[:240]
        print(f"\n--- {key} {qid} shot {shot_index} ---")
        print(snippet or "(empty generation)")
    if args.dry_run:
        print("dry-run: no files written")


if __name__ == "__main__":
    main()
