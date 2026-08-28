"""Detect doom-looped (stuck repeating) description generations.

Heuristics flag outputs where the model appears to repeat lines, phrases,
or tail segments instead of finishing the caption/transcription.

Usage::

    uv run python mmar-descriptions/detect_doom_loops.py
    uv run python mmar-descriptions/detect_doom_loops.py --json
    uv run python mmar-descriptions/detect_doom_loops.py --pack-dir ./outputs/mmar-descriptions
    uv run python mmar-descriptions/detect_doom_loops.py --min-chars 200 --all-bad-only
"""

from __future__ import annotations

import argparse
import json
import re
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

DEFAULT_PACK_DIR = REPO_ROOT / "outputs" / "mmar-descriptions"
DEFAULT_MIN_CHARS = 200

# Minimum text length before any doom-loop heuristic runs.
MIN_CHARS = 80
MIN_WORDS = 24

# A hit needs score >= this (signals are weighted and capped at 1.0).
DOOM_SCORE_THRESHOLD = 0.55


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split())


def _words(text: str) -> list[str]:
    return re.findall(r"\w+(?:['']\w+)?", _normalize(text).lower())


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]


def _snippet(text: str, *, limit: int = 120) -> str:
    cleaned = _normalize(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _best_repeated_phrase(words: list[str]) -> tuple[str, int, int] | None:
    """Return (phrase, word_len, count) for the strongest repeating n-gram."""
    if len(words) < 12:
        return None
    best: tuple[str, int, int, float] | None = None
    n_words = len(words)
    for window in (4, 6, 8, 10, 12):
        if window > n_words // 2:
            continue
        counts: Counter[tuple[str, ...]] = Counter()
        for start in range(n_words - window + 1):
            counts[tuple(words[start : start + window])] += 1
        phrase, count = counts.most_common(1)[0]
        if count < 3:
            continue
        phrase_text = " ".join(phrase)
        if len(phrase_text) < 12:
            continue
        coverage = (count * window) / n_words
        if coverage < 0.28:
            continue
        score = coverage * min(count / 3.0, 2.0)
        if best is None or score > best[3]:
            best = (phrase_text, window, count, score)
    if best is None:
        return None
    return best[0], best[1], best[2]


def _tail_repeat(normalized: str) -> tuple[str, int] | None:
    """Detect a short tail substring copied many times."""
    if len(normalized) < MIN_CHARS:
        return None
    for tail_len in (40, 60, 80):
        if tail_len >= len(normalized):
            continue
        tail = normalized[-tail_len:]
        if len(tail) < 25:
            continue
        count = normalized.count(tail)
        if count >= 3:
            return tail, count
    return None


def _consecutive_line_run(lines: list[str]) -> tuple[str, int] | None:
    """Same line repeated back-to-back (classic stuck loop)."""
    if len(lines) < 3:
        return None
    best: tuple[str, int] | None = None
    run_line = lines[0]
    run_len = 1
    for line in lines[1:]:
        if line == run_line:
            run_len += 1
        else:
            if run_len >= 3 and (best is None or run_len > best[1]):
                best = (run_line, run_len)
            run_line = line
            run_len = 1
    if run_len >= 3 and (best is None or run_len > best[1]):
        best = (run_line, run_len)
    return best


def detect_doom_loop(text: str) -> dict[str, Any] | None:
    """Return doom-loop metadata when ``text`` looks stuck repeating."""
    raw = str(text or "").strip()
    normalized = _normalize(raw)
    if len(normalized) < MIN_CHARS:
        return None

    words = _words(normalized)
    if len(words) < MIN_WORDS:
        return None

    reasons: list[str] = []
    score = 0.0
    snippet = ""
    repeat_count = 0

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.32 and len(words) >= 40:
        reasons.append("low_lexical_diversity")
        score += 0.35
        snippet = snippet or _snippet(" ".join(words[:12]))

    lines = _lines(raw)
    if lines:
        line_counts = Counter(ln.lower() for ln in lines)
        line_text, line_n = line_counts.most_common(1)[0]
        if line_n >= 3 and len(line_text) >= 18:
            reasons.append("repeated_line")
            score += min(0.45, 0.2 + 0.08 * (line_n - 2))
            snippet = snippet or _snippet(line_text)
            repeat_count = max(repeat_count, line_n)

        run = _consecutive_line_run([ln.lower() for ln in lines])
        if run is not None:
            run_text, run_n = run
            reasons.append("consecutive_line_run")
            score += min(0.5, 0.25 + 0.1 * (run_n - 2))
            snippet = snippet or _snippet(run_text)
            repeat_count = max(repeat_count, run_n)

    phrase = _best_repeated_phrase(words)
    if phrase is not None:
        phrase_text, _window, phrase_n = phrase
        reasons.append("phrase_loop")
        score += min(0.55, 0.25 + 0.1 * (phrase_n - 2))
        snippet = snippet or _snippet(phrase_text)
        repeat_count = max(repeat_count, phrase_n)

    tail = _tail_repeat(normalized)
    if tail is not None:
        tail_text, tail_n = tail
        reasons.append("tail_repeat")
        score += min(0.45, 0.2 + 0.08 * (tail_n - 2))
        snippet = snippet or _snippet(tail_text)
        repeat_count = max(repeat_count, tail_n)

    # Long spans of identical punctuation / filler tokens.
    filler = re.search(r"(.)\1{19,}", normalized)
    if filler is not None:
        reasons.append("character_stutter")
        score += 0.35
        snippet = snippet or filler.group(0)[:80]

    score = min(1.0, score)
    if score < DOOM_SCORE_THRESHOLD:
        return None

    return {
        "doom_loop": True,
        "score": round(score, 3),
        "reasons": list(dict.fromkeys(reasons)),
        "snippet": snippet or _snippet(normalized),
        "repeat_count": repeat_count,
        "n_chars": len(normalized),
        "n_words": len(words),
        "unique_word_ratio": round(unique_ratio, 3),
    }


def doom_loop_key(qid: str, model: str, shot_index: int) -> str:
    return f"{qid}|{model}|{shot_index}"


def shot_key(qid: str, model: str, shot_index: int) -> str:
    return doom_loop_key(qid, model, shot_index)


def is_too_short(text: str, min_chars: int) -> bool:
    return len(_normalize(str(text or ""))) < int(min_chars)


def scan_quality(
    predictions: dict[str, dict[str, dict[str, Any]]],
    *,
    model_labels: list[str] | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> dict[str, Any]:
    """Scan predictions for doom loops, short generations, and all-bad pairs."""
    min_chars = int(min_chars)
    doom = scan_predictions(predictions, model_labels=model_labels)
    doom_keys = {
        doom_loop_key(str(hit["id"]), hit["model"], int(hit["shot_index"]))
        for hit in doom.get("hits") or []
    }

    labels = model_labels or sorted(predictions)
    short_hits: list[dict[str, Any]] = []
    short_by_qid: dict[str, int] = {}
    short_by_model: dict[str, int] = {}
    fully_filtered: list[dict[str, Any]] = []
    n_shots = 0

    for label in labels:
        model_preds = predictions.get(label) or {}
        for qid, record in model_preds.items():
            shots = record.get("shots") or []
            if not shots:
                continue

            n_short_shots = 0
            n_doom_shots = 0
            n_bad_shots = 0
            shot_rows: list[dict[str, Any]] = []

            for shot in shots:
                n_shots += 1
                text = str(shot.get("text") or shot.get("answer_prediction") or "")
                normalized = _normalize(text)
                n_char_count = len(normalized)
                shot_index = int(shot.get("shot_index") or 0)
                key = doom_loop_key(str(qid), label, shot_index)

                too_short = n_char_count < min_chars
                is_doom = key in doom_keys
                is_bad = too_short or is_doom

                if too_short:
                    short_hits.append(
                        {
                            "id": str(qid),
                            "model": label,
                            "shot_index": shot_index,
                            "too_short": True,
                            "n_chars": n_char_count,
                            "min_chars": min_chars,
                            "snippet": _snippet(text),
                        }
                    )
                    short_by_qid[str(qid)] = short_by_qid.get(str(qid), 0) + 1
                    short_by_model[label] = short_by_model.get(label, 0) + 1
                    n_short_shots += 1

                if is_doom:
                    n_doom_shots += 1
                if is_bad:
                    n_bad_shots += 1

                shot_rows.append(
                    {
                        "shot_index": shot_index,
                        "n_chars": n_char_count,
                        "too_short": too_short,
                        "doom_loop": is_doom,
                        "bad": is_bad,
                    }
                )

            if n_bad_shots == len(shots):
                fully_filtered.append(
                    {
                        "id": str(qid),
                        "model": label,
                        "n_shots": len(shots),
                        "n_short": n_short_shots,
                        "n_doom": n_doom_shots,
                        "shots": shot_rows,
                    }
                )

    fully_filtered.sort(key=lambda row: (row["id"], row["model"]))
    fully_filtered_by_qid: dict[str, list[str]] = {}
    fully_filtered_by_model: dict[str, int] = {}
    for row in fully_filtered:
        fully_filtered_by_qid.setdefault(row["id"], []).append(row["model"])
        fully_filtered_by_model[row["model"]] = (
            fully_filtered_by_model.get(row["model"], 0) + 1
        )
    for qid in fully_filtered_by_qid:
        fully_filtered_by_qid[qid] = sorted(fully_filtered_by_qid[qid])

    short_hits.sort(
        key=lambda row: (int(row.get("n_chars") or 0), row["id"], row["model"])
    )

    return {
        "min_chars": min_chars,
        "doom": doom,
        "short_hits": short_hits,
        "n_short_hits": len(short_hits),
        "short_by_qid": short_by_qid,
        "short_by_model": short_by_model,
        "n_questions_with_short": len(short_by_qid),
        "fully_filtered": fully_filtered,
        "n_fully_filtered": len(fully_filtered),
        "fully_filtered_by_qid": fully_filtered_by_qid,
        "n_clips_with_fully_filtered_model": len(fully_filtered_by_qid),
        "fully_filtered_by_model": fully_filtered_by_model,
        "n_shots_scanned": n_shots,
    }


def scan_predictions(
    predictions: dict[str, dict[str, dict[str, Any]]],
    *,
    model_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Scan a ``load_pack``-style predictions tree for doom loops."""
    labels = model_labels or sorted(predictions)
    hits: list[dict[str, Any]] = []
    by_qid: dict[str, int] = {}
    by_model: dict[str, int] = {}
    n_shots = 0

    for label in labels:
        model_preds = predictions.get(label) or {}
        for qid, record in model_preds.items():
            for shot in record.get("shots") or []:
                n_shots += 1
                text = shot.get("text") or shot.get("answer_prediction") or ""
                result = detect_doom_loop(str(text))
                if result is None:
                    continue
                shot_index = int(shot.get("shot_index") or 0)
                hit = {
                    "id": str(qid),
                    "model": label,
                    "shot_index": shot_index,
                    **result,
                }
                hits.append(hit)
                by_qid[str(qid)] = by_qid.get(str(qid), 0) + 1
                by_model[label] = by_model.get(label, 0) + 1

    hits.sort(key=lambda row: (-float(row.get("score") or 0), row["id"], row["model"]))
    return {
        "hits": hits,
        "n_hits": len(hits),
        "n_shots_scanned": n_shots,
        "n_questions_with_hits": len(by_qid),
        "by_qid": by_qid,
        "by_model": by_model,
        "threshold": DOOM_SCORE_THRESHOLD,
    }


def scan_pack_dir(pack_dir: Path, *, min_chars: int = DEFAULT_MIN_CHARS) -> dict[str, Any]:
    from view_descriptions import load_pack  # noqa: PLC0415

    bundle = load_pack(str(pack_dir.expanduser().resolve()))
    report = scan_quality(
        bundle["predictions"],
        model_labels=bundle["model_labels"],
        min_chars=min_chars,
    )
    report["pack_dir"] = str(pack_dir)
    report["model_labels"] = bundle["model_labels"]
    report["n_questions"] = bundle["n_questions"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=DEFAULT_PACK_DIR,
        help="Descriptions pack directory",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Flag generations shorter than this many characters (default: 200)",
    )
    parser.add_argument(
        "--all-bad-only",
        action="store_true",
        help="Only print clip×model pairs where every shot is too short or doom-looped",
    )
    args = parser.parse_args()

    pack_dir = args.pack_dir.expanduser().resolve()
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")

    report = scan_pack_dir(pack_dir, min_chars=args.min_chars)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print(f"Pack: {pack_dir}")
    print(f"Min chars: {report['min_chars']}")
    doom = report.get("doom") or {}
    print(
        f"Doom loops: {doom.get('n_hits', 0)} / {report['n_shots_scanned']} shots "
        f"({doom.get('n_questions_with_hits', 0)} clips)"
    )
    print(
        f"Too short: {report['n_short_hits']} / {report['n_shots_scanned']} shots "
        f"({report['n_questions_with_short']} clips)"
    )
    print(
        f"All bad (every shot too short or doom): {report['n_fully_filtered']} "
        f"clip×model pairs ({report['n_clips_with_fully_filtered_model']} clips)"
    )
    for label, count in sorted(
        (report.get("fully_filtered_by_model") or {}).items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {label:<24} {count} all-bad clips")
    print()

    if args.all_bad_only:
        for row in report.get("fully_filtered") or []:
            print(
                f"{row['id']}  {row['model']}  "
                f"{row['n_shots']} shots  "
                f"short={row['n_short']} doom={row['n_doom']}"
            )
        return

    for label, count in sorted(
        (doom.get("by_model") or {}).items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {label:<24} {count} doom loops")
    print()
    for hit in doom.get("hits") or []:
        reasons = ", ".join(hit.get("reasons") or [])
        print(
            f"{hit['score']:.2f}  {hit['id']}  {hit['model']}  "
            f"shot {hit['shot_index']}  [{reasons}]"
        )
        print(f"       {hit.get('snippet') or ''}")
        print()


if __name__ == "__main__":
    main()
