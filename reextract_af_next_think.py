"""Re-extract answers in a freeform pack with the current parsers.

Rewrites ``thinking_prediction`` / ``answer_prediction`` from ``model_output``.
Shots whose extracted answer changes have their judge verdicts dropped so they
can be re-graded.

Usage::

    uv run python reextract_af_next_think.py --dry-run
    uv run python reextract_af_next_think.py
    uv run python reextract_af_next_think.py --model all --pack ./outputs/exp-mmar-question-difficulty/20260827T033757Z
    uv run python reextract_af_next_think.py --examples 5
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Callable
from pathlib import Path

from aggregate import aggregate_difficulty, write_jsonl
from mmar_common import (
    load_jsonl,
    parse_freeform_output,
    parse_music_flamingo_output,
    recompute_multi_judge_scores,
    split_last_think_close,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PACK = REPO_ROOT / "outputs" / "mmar-freeform"
DEFAULT_MODEL = "af-next-think"


def _drop_shot_judges(shot: dict) -> int:
    judges = shot.get("judges")
    n = len(judges) if isinstance(judges, dict) else 0
    shot.pop("judges", None)
    shot.pop("grader", None)
    shot.pop("grader_output", None)
    shot["correct"] = None
    shot["pending_grade"] = True
    return n


def parse_fn_for_model(label: str) -> Callable:
    """Match freeform ``_parse_fn_for`` in ``mmar_models``."""
    if label == "music-flamingo":

        def parse(raw_text, choices=None):
            return parse_music_flamingo_output(
                raw_text, choices, fallback=parse_freeform_output
            )

        return parse
    return parse_freeform_output


def reextract_record(
    record: dict,
    *,
    keep_judges: bool,
    parse_fn: Callable | None = None,
) -> dict:
    """Update shot extracts in place. Returns counts for this record."""
    parser = parse_fn or parse_freeform_output
    choices = record.get("choices") or []
    stats = {
        "shots": 0,
        "with_close": 0,
        "changed": 0,
        "judges_dropped": 0,
    }
    for shot in record.get("shots") or []:
        stats["shots"] += 1
        raw = str(shot.get("model_output") or "")
        old = str(shot.get("answer_prediction") or "")
        if split_last_think_close(raw) is not None:
            stats["with_close"] += 1
        thinking, answer = parser(raw, choices)
        shot["thinking_prediction"] = thinking
        shot["answer_prediction"] = answer
        if answer != old:
            stats["changed"] += 1
            if not keep_judges:
                stats["judges_dropped"] += _drop_shot_judges(shot)
    recompute_multi_judge_scores(record)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-extract af-next-think answers after the last </think>."
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=DEFAULT_PACK,
        help="Freeform pack directory (default: outputs/mmar-freeform).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model label under models/, or 'all' (default: af-next-think).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing.",
    )
    parser.add_argument(
        "--keep-judges",
        action="store_true",
        help="Leave existing shot verdicts even when the extracted answer changes.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip regenerating pack difficulty.jsonl / scores.json.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Print this many changed shot before/after pairs.",
    )
    args = parser.parse_args()

    models_root = args.pack / "models"
    if str(args.model).lower() == "all":
        labels = sorted(
            path.parent.name
            for path in models_root.glob("*/predictions.jsonl")
            if path.is_file()
        )
    else:
        labels = [args.model]
    if not labels:
        raise SystemExit(f"no models with predictions.jsonl under {models_root}")

    grand = {"shots": 0, "with_close": 0, "changed": 0, "judges_dropped": 0}
    examples: list[tuple[str, str, int, str, str]] = []

    for label in labels:
        pred_path = models_root / label / "predictions.jsonl"
        if not pred_path.is_file():
            raise SystemExit(f"predictions.jsonl not found: {pred_path}")

        records = load_jsonl(pred_path)
        totals = {"shots": 0, "with_close": 0, "changed": 0, "judges_dropped": 0}
        parse_fn = parse_fn_for_model(label)

        for record in records:
            qid = str(record.get("id") or "")
            before = [
                str(shot.get("answer_prediction") or "")
                for shot in (record.get("shots") or [])
            ]
            stats = reextract_record(
                record, keep_judges=args.keep_judges, parse_fn=parse_fn
            )
            for key, value in stats.items():
                totals[key] += value
            if args.examples and len(examples) < args.examples:
                for shot, old in zip(record.get("shots") or [], before):
                    new = str(shot.get("answer_prediction") or "")
                    if new == old:
                        continue
                    examples.append(
                        (label, qid, int(shot.get("shot_index", 0)), old, new)
                    )
                    if len(examples) >= args.examples:
                        break

        for key, value in totals.items():
            grand[key] += value
        print(
            f"{label}: {len(records)} records, {totals['shots']} shots, "
            f"{totals['with_close']} with </think>, "
            f"{totals['changed']} answers changed"
            + (
                f", {totals['judges_dropped']} judge entries dropped"
                if not args.keep_judges
                else ""
            )
        )

        if args.dry_run:
            continue

        backup = pred_path.with_suffix(".jsonl.bak")
        shutil.copy2(pred_path, backup)
        write_jsonl(pred_path, records)
        print(f"wrote {pred_path}")
        print(f"backup {backup}")

    if len(labels) > 1:
        print(
            f"all: {grand['shots']} shots, {grand['changed']} answers changed"
            + (
                f", {grand['judges_dropped']} judge entries dropped"
                if not args.keep_judges
                else ""
            )
        )
    for label, qid, shot_index, old, new in examples:
        print(f"\n--- {label} {qid} shot {shot_index} ---")
        print(f"OLD: {old}")
        print(f"NEW: {new}")

    if args.dry_run:
        print("dry-run: no files written")
        return

    if not args.no_aggregate:
        aggregate_difficulty(args.pack)
        print(f"re-aggregated {args.pack / 'difficulty.jsonl'}")


if __name__ == "__main__":
    main()
