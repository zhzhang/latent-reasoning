"""Re-run the grade extractor on stored judge text. No model calls.

``run_judges.py`` writes ``correct`` / ``verdict`` / ``output`` at grade time.
A later parser change does not move those fields. This script restamps them
from stored ``generation`` (and ``samples`` when present) using the current
``parse_grade_verdict``, then rewrites ``predictions.jsonl`` and matching
``judge_partials/*.jsonl`` sidecars.

Does not change ``manifest.json`` / ``primary_judge``. Does not call an API.

Usage::

    uv run python reparse_judges.py --dry-run
    uv run python reparse_judges.py
    uv run python reparse_judges.py --unparsed-only
    uv run python reparse_judges.py --judge qwen2.5-omni-7b --examples 5
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

from grader import (
    _verdict_fields,
    format_grade_output,
    majority_grade_verdict,
)
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


def _stored_parsed(entry: dict) -> bool:
    return str(entry.get("verdict") or "").strip().lower() in {"pass", "fail"}


def _generation_text(entry: dict) -> str:
    text = str(entry.get("generation") or "")
    if text.strip():
        return text
    return str(entry.get("reasoning") or "")


def _stamp_verdict(dest: dict, verdict: bool | None) -> None:
    dest["correct"] = bool(verdict) if verdict is not None else False
    dest["verdict"] = (
        "pass" if verdict is True else "fail" if verdict is False else None
    )
    dest["output"] = format_grade_output(verdict)


def _verdict_snapshot(entry: dict) -> tuple:
    samples = entry.get("samples")
    sample_t: tuple = ()
    if isinstance(samples, list):
        sample_t = tuple(
            (s.get("correct"), s.get("verdict"), s.get("output"))
            for s in samples
            if isinstance(s, dict)
        )
    return (entry.get("correct"), entry.get("verdict"), entry.get("output"), sample_t)


def reparse_entry(entry: dict) -> bool:
    """Restamp verdict fields from stored text. True when any field changed."""
    before = _verdict_snapshot(entry)
    samples = entry.get("samples")
    if isinstance(samples, list) and samples:
        verdicts: list[bool | None] = []
        for sample in samples:
            if not isinstance(sample, dict):
                verdicts.append(None)
                continue
            fields = _verdict_fields(_generation_text(sample))
            _stamp_verdict(sample, fields["grader_verdict_raw"])
            verdicts.append(fields["grader_verdict_raw"])
        _stamp_verdict(entry, majority_grade_verdict(verdicts))
    else:
        text = _generation_text(entry)
        if not text.strip():
            return False
        fields = _verdict_fields(text)
        _stamp_verdict(entry, fields["grader_verdict_raw"])
    return _verdict_snapshot(entry) != before


def _should_reparse(key: str, entry: object, *, unparsed_only: bool) -> bool:
    if key == STRING_MATCH_JUDGE_LABEL or not isinstance(entry, dict):
        return False
    if unparsed_only and _stored_parsed(entry):
        return False
    return True


def reparse_record(
    record: dict,
    *,
    prefixes: list[str],
    unparsed_only: bool,
) -> tuple[int, int, int, int, Counter[str], list[tuple[str, str, int, str, str]]]:
    """Restamp LLM judge entries. Returns scan/change/recover counts.

    Returns
    ``(n_entries, n_changed, n_unparsed_before, n_unparsed_after,
    recovered_by_key, examples)``.
    """
    n_entries = 0
    n_changed = 0
    n_unparsed_before = 0
    n_unparsed_after = 0
    recovered_by_key: Counter[str] = Counter()
    examples: list[tuple[str, str, int, str, str]] = []
    qid = str(record.get("id") or "")
    changed_any = False
    for shot in record.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        judges = shot.get("judges")
        if not isinstance(judges, dict):
            continue
        shot_index = int(shot.get("shot_index", 0))
        for key, entry in judges.items():
            if not _judge_wanted(str(key), prefixes):
                continue
            n_entries += 1
            if not isinstance(entry, dict):
                continue
            was_parsed = _stored_parsed(entry)
            if not was_parsed:
                n_unparsed_before += 1
            if not _should_reparse(str(key), entry, unparsed_only=unparsed_only):
                if not was_parsed:
                    n_unparsed_after += 1
                continue
            generation = _generation_text(entry)
            changed = reparse_entry(entry)
            now_parsed = _stored_parsed(entry)
            if not now_parsed:
                n_unparsed_after += 1
            if changed:
                n_changed += 1
                changed_any = True
            if not was_parsed and now_parsed:
                recovered_by_key[str(key)] += 1
                examples.append(
                    (
                        str(key),
                        qid,
                        shot_index,
                        generation,
                        str(entry.get("verdict") or ""),
                    )
                )
    if changed_any:
        recompute_multi_judge_scores(record, record.get("primary_judge"))
    return (
        n_entries,
        n_changed,
        n_unparsed_before,
        n_unparsed_after,
        recovered_by_key,
        examples,
    )


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


def reparse_sidecars(
    pack_dir: Path,
    labels: list[str],
    *,
    prefixes: list[str],
    unparsed_only: bool,
    dry_run: bool,
) -> tuple[int, int, int, int, int]:
    """Restamp sidecar ``entry`` objects. Returns file/row/change counts."""
    n_files = 0
    n_rows = 0
    n_changed = 0
    n_unparsed_before = 0
    n_unparsed_after = 0
    for label in labels:
        partials_dir = pack_dir / "models" / label / "judge_partials"
        if not partials_dir.is_dir():
            continue
        for path in sorted(partials_dir.glob("*.jsonl")):
            n_files += 1
            rows = load_jsonl(path)
            file_changed = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                n_rows += 1
                key = str(row.get("judge_key") or path.stem)
                if not _judge_wanted(key, prefixes):
                    continue
                entry = row.get("entry")
                if not isinstance(entry, dict):
                    continue
                was_parsed = _stored_parsed(entry)
                if not was_parsed:
                    n_unparsed_before += 1
                if not _should_reparse(key, entry, unparsed_only=unparsed_only):
                    if not was_parsed:
                        n_unparsed_after += 1
                    continue
                if reparse_entry(entry):
                    file_changed += 1
                if not _stored_parsed(entry):
                    n_unparsed_after += 1
            n_changed += file_changed
            if dry_run or not file_changed:
                continue
            _atomic_write_jsonl(path, rows)
    return n_files, n_rows, n_changed, n_unparsed_before, n_unparsed_after


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-parse stored run_judges.py verdicts with the current extractor."
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
        "--unparsed-only",
        action="store_true",
        help="Skip entries whose stored verdict is already pass/fail.",
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
        help="Print this many newly parsed generation snippets.",
    )
    args = parser.parse_args()

    pack_dir = args.pack
    prefixes = _csv_parts(args.judge)
    labels = _discover_labels(pack_dir, args.model)
    print(
        f"[reparse] pack={pack_dir} models={labels} "
        f"judges={prefixes or '(all)'} unparsed_only={args.unparsed_only}"
    )

    grand_entries = 0
    grand_changed = 0
    grand_unparsed_before = 0
    grand_unparsed_after = 0
    recovered_by_key: Counter[str] = Counter()
    examples: list[tuple[str, str, int, str, str]] = []

    for label in labels:
        pred_path = pack_dir / "models" / label / "predictions.jsonl"
        if not pred_path.is_file():
            raise SystemExit(f"predictions.jsonl not found: {pred_path}")
        records = load_jsonl(pred_path)
        n_entries = 0
        n_changed = 0
        n_unparsed_before = 0
        n_unparsed_after = 0
        for record in records:
            e, c, ub, ua, by_key, rec_examples = reparse_record(
                record,
                prefixes=prefixes,
                unparsed_only=args.unparsed_only,
            )
            n_entries += e
            n_changed += c
            n_unparsed_before += ub
            n_unparsed_after += ua
            recovered_by_key.update(by_key)
            if args.examples and len(examples) < args.examples:
                room = args.examples - len(examples)
                examples.extend(rec_examples[:room])
        grand_entries += n_entries
        grand_changed += n_changed
        grand_unparsed_before += n_unparsed_before
        grand_unparsed_after += n_unparsed_after
        recovered = n_unparsed_before - n_unparsed_after
        print(
            f"{label}: unparsed {n_unparsed_before} -> {n_unparsed_after} "
            f"(recovered {recovered}), changed {n_changed}, n={n_entries}"
        )
        if args.dry_run or not n_changed:
            continue
        backup = pred_path.with_suffix(".jsonl.bak")
        shutil.copy2(pred_path, backup)
        _atomic_write_jsonl(pred_path, records)
        print(f"wrote {pred_path}")
        print(f"backup {backup}")

    n_sc_files, n_sc_rows, n_sc_changed, n_sc_ub, n_sc_ua = reparse_sidecars(
        pack_dir,
        labels,
        prefixes=prefixes,
        unparsed_only=args.unparsed_only,
        dry_run=args.dry_run,
    )
    print(
        f"sidecars: unparsed {n_sc_ub} -> {n_sc_ua} "
        f"(recovered {n_sc_ub - n_sc_ua}), changed {n_sc_changed} rows "
        f"in {n_sc_files} files (n={n_sc_rows})"
    )
    if recovered_by_key:
        print("recovered by judge key:")
        for key, count in sorted(recovered_by_key.items()):
            print(f"  {key}: {count}")
    recovered = grand_unparsed_before - grand_unparsed_after
    print(
        f"all: unparsed {grand_unparsed_before} -> {grand_unparsed_after} "
        f"(recovered {recovered}), changed {grand_changed}, n={grand_entries}"
    )
    for key, qid, shot_index, generation, verdict in examples:
        snippet = generation.replace("\n", " ")[:240]
        print(f"\n--- {key} {qid} shot {shot_index} -> {verdict} ---")
        print(snippet or "(empty generation)")
    if args.dry_run:
        print("dry-run: no files written")


if __name__ == "__main__":
    main()
