"""Pick one representative description per model × clip.

For each question and model, scan shots in this order:

    3, 4, 5, …   (the second group)
    0, 1, 2      (the first three)

and take the first that is neither a doom loop nor too short. If every
shot is bad, fall back to the first shot in that same order.

Usage::

    uv run python mmar-descriptions/choose_descriptions.py
    uv run python mmar-descriptions/choose_descriptions.py --pack-dir ./outputs/mmar-descriptions
    uv run python mmar-descriptions/choose_descriptions.py --min-chars 200 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MMAR_DESC_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MMAR_DESC_DIR) not in sys.path:
    sys.path.insert(0, str(MMAR_DESC_DIR))

from detect_doom_loops import (  # noqa: E402
    DEFAULT_MIN_CHARS,
    detect_doom_loop,
    is_too_short,
)
from mmar_common import write_jsonl  # noqa: E402

DEFAULT_PACK_DIR = REPO_ROOT / "outputs" / "mmar-descriptions"
PREFERRED_START = 3


def _shot_index(shot: dict[str, Any]) -> int:
    raw = shot.get("shot_index")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _shot_text(shot: dict[str, Any]) -> str:
    return str(
        shot.get("text")
        or shot.get("answer_prediction")
        or shot.get("model_output")
        or ""
    ).strip()


def candidate_order(
    shots: list[dict[str, Any]],
    *,
    preferred_start: int = PREFERRED_START,
) -> list[dict[str, Any]]:
    """Second group first (shot_index >= 3), then the first three."""
    indexed = sorted(shots, key=_shot_index)
    preferred = [shot for shot in indexed if _shot_index(shot) >= preferred_start]
    earlier = [shot for shot in indexed if _shot_index(shot) < preferred_start]
    return preferred + earlier


def shot_is_valid(text: str, min_chars: int) -> bool:
    return not is_too_short(text, min_chars) and detect_doom_loop(text) is None


def choose_shot(
    shots: list[dict[str, Any]],
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    preferred_start: int = PREFERRED_START,
) -> tuple[dict[str, Any] | None, bool]:
    """Return ``(shot, valid)``. ``valid`` is False when every shot is bad."""
    ordered = candidate_order(shots, preferred_start=preferred_start)
    for shot in ordered:
        if shot_is_valid(_shot_text(shot), min_chars):
            return shot, True
    if ordered:
        return ordered[0], False
    return None, False


def _choice_row(
    qid: str,
    model: str,
    shot: dict[str, Any],
    *,
    valid: bool,
    n_shots: int,
    n_valid: int,
    min_chars: int,
    preferred_start: int,
) -> dict[str, Any]:
    text = _shot_text(shot)
    shot_index = _shot_index(shot)
    doom = detect_doom_loop(text)
    too_short = is_too_short(text, min_chars)
    return {
        "id": qid,
        "model": model,
        "shot_index": shot_index,
        "text": text,
        "n_chars": len(" ".join(text.split())),
        "too_short": too_short,
        "doom_loop": doom is not None,
        "doom": doom,
        "valid": valid,
        "from_second_group": shot_index >= preferred_start,
        "n_shots": n_shots,
        "n_valid": n_valid,
        "min_chars": min_chars,
    }


def choose_pack(
    predictions: dict[str, dict[str, dict[str, Any]]],
    *,
    question_ids: list[str] | None = None,
    model_labels: list[str] | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    preferred_start: int = PREFERRED_START,
) -> dict[str, Any]:
    """Pick one representative shot per model × clip."""
    min_chars = int(min_chars)
    labels = model_labels or sorted(predictions)
    if question_ids is None:
        seen: list[str] = []
        found: set[str] = set()
        for label in labels:
            for qid in predictions.get(label) or {}:
                if qid not in found:
                    found.add(qid)
                    seen.append(qid)
        question_ids = seen

    rows: list[dict[str, Any]] = []
    n_missing = 0
    for qid in question_ids:
        for label in labels:
            record = (predictions.get(label) or {}).get(qid)
            shots = list((record or {}).get("shots") or [])
            if not shots:
                n_missing += 1
                continue
            n_valid = sum(
                1
                for shot in shots
                if shot_is_valid(_shot_text(shot), min_chars)
            )
            shot, valid = choose_shot(
                shots, min_chars=min_chars, preferred_start=preferred_start
            )
            if shot is None:
                n_missing += 1
                continue
            rows.append(
                _choice_row(
                    str(qid),
                    label,
                    shot,
                    valid=valid,
                    n_shots=len(shots),
                    n_valid=n_valid,
                    min_chars=min_chars,
                    preferred_start=preferred_start,
                )
            )

    by_model = Counter(row["model"] for row in rows)
    by_shot = Counter(int(row["shot_index"]) for row in rows)
    n_second = sum(1 for row in rows if row["from_second_group"])
    n_valid = sum(1 for row in rows if row["valid"])
    n_fallback = sum(1 for row in rows if not row["valid"])
    return {
        "min_chars": min_chars,
        "preferred_start": preferred_start,
        "n_questions": len(question_ids),
        "n_models": len(labels),
        "n_chosen": len(rows),
        "n_missing": n_missing,
        "n_valid": n_valid,
        "n_fallback": n_fallback,
        "n_from_second_group": n_second,
        "n_from_first_group": len(rows) - n_second,
        "by_model": dict(by_model),
        "by_shot_index": {str(k): by_shot[k] for k in sorted(by_shot)},
        "model_labels": labels,
        "rows": rows,
    }


def choose_pack_dir(
    pack_dir: Path,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    preferred_start: int = PREFERRED_START,
) -> dict[str, Any]:
    from view_descriptions import load_pack  # noqa: PLC0415

    bundle = load_pack(str(pack_dir.expanduser().resolve()))
    question_ids = [str(row["id"]) for row in bundle.get("questions") or []]
    report = choose_pack(
        bundle["predictions"],
        question_ids=question_ids,
        model_labels=bundle["model_labels"],
        min_chars=min_chars,
        preferred_start=preferred_start,
    )
    report["pack_dir"] = str(pack_dir)
    report["n_shots_pack"] = bundle.get("n_shots")
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print(f"Pack: {report.get('pack_dir')}")
    print(f"Min chars: {report['min_chars']}")
    print(
        f"Chosen: {report['n_chosen']}  "
        f"(valid={report['n_valid']} fallback={report['n_fallback']} "
        f"missing={report['n_missing']})"
    )
    print(
        f"From second group (shot>={report['preferred_start']}): "
        f"{report['n_from_second_group']}  "
        f"from first three: {report['n_from_first_group']}"
    )
    print("By shot index:")
    for index, count in (report.get("by_shot_index") or {}).items():
        print(f"  shot {index}: {count}")
    print("By model:")
    for label in report.get("model_labels") or []:
        n = (report.get("by_model") or {}).get(label, 0)
        n_valid = sum(
            1
            for row in report.get("rows") or []
            if row["model"] == label and row["valid"]
        )
        n_second = sum(
            1
            for row in report.get("rows") or []
            if row["model"] == label and row["from_second_group"]
        )
        print(
            f"  {label:<24} {n} chosen  "
            f"valid={n_valid}  second_group={n_second}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=DEFAULT_PACK_DIR,
        help="Descriptions pack (default: outputs/mmar-descriptions)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL path (default: <pack-dir>/chosen.jsonl)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Reject generations shorter than this (default: 200)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    pack_dir = args.pack_dir.expanduser().resolve()
    if not pack_dir.is_dir():
        raise SystemExit(
            f"Pack not found: {pack_dir}\n"
            "Download with:\n"
            "  uv run modal run download_results.py --volume-name mmar-descriptions"
        )

    report = choose_pack_dir(pack_dir, min_chars=args.min_chars)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else pack_dir / "chosen.jsonl"
    )
    write_jsonl(output, report["rows"], mode="w")
    report["output"] = str(output)

    if args.json:
        printable = {key: value for key, value in report.items() if key != "rows"}
        printable["n_rows"] = len(report["rows"])
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        return

    _print_summary(report)
    print(f"Wrote {len(report['rows'])} rows -> {output}")


if __name__ == "__main__":
    main()
