"""Gemini filter for binary MMAR questions the UI does not already tag.

Sends question text only (no gold answer, no model answers, no choices) for
items that ``is_yes_no_question`` leaves unmarked. Gemini decides whether the
wording itself implies a two-way answer, e.g. "in front of or behind".

Writes:

- ``gemini_binary_judgments.jsonl`` — full judge rows (resumable)
- ``gemini_binary_question_ids.csv`` — extra binary IDs for the viewer
- ``open_ended_question_ids.csv`` — remaining open-ended IDs (Hide yes/no)

Usage::

    export GEMINI_API_KEY=...

    uv run python answer-variety/filter_binary_questions.py
    uv run python answer-variety/filter_binary_questions.py --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
for path in (str(PACKAGE_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from answer_variety import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LIMIT,
    DEFAULT_SOURCE_RUN_ID,
    GEMINI_BINARY_IDS_PATH,
    OPEN_ENDED_IDS_PATH,
    GeminiUniformityScorer,
    build_uniformity_items,
    extract_reason,
    is_yes_no_question,
    load_completed_ids,
    parse_uniformity_verdict,
    prune_incomplete_evaluations,
    write_id_csv,
)
from mmar_common import load_jsonl, write_json, write_jsonl  # noqa: E402

DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs"
JUDGMENTS_PATH = PACKAGE_DIR / "gemini_binary_judgments.jsonl"
SUMMARY_PATH = PACKAGE_DIR / "gemini_binary_filter_summary.json"

BINARY_SYSTEM_PROMPT = (
    "You classify audio-understanding questions from question text alone.\n"
    "Decide whether the wording implies a binary answer space: exactly two "
    "mutually exclusive values.\n"
    "\n"
    "Binary examples: yes/no; true/false; presence vs absence of one "
    "attribute; two named alternatives such as \"in front of or behind the "
    "listener\", \"male or female\", \"arriving or departing\", "
    "\"towards or away\", \"major or minor\".\n"
    "\n"
    "Not binary when many answers are plausible: what/which/who/how/why "
    "with an open set, counts, categories, transcription, or description. "
    "A question that lists three or more alternatives is not binary.\n"
    "\n"
    "Do not assume multiple-choice options or a gold answer. Judge only "
    "the question wording.\n"
    "\n"
    "Reason briefly if needed, then end your reply with a single final line "
    "containing only Yes or No. Yes means the question implies a binary "
    "answer. No means it is more open-ended."
)


def create_binary_user_prompt(question: str) -> str:
    text = str(question or "").strip() or "(empty question)"
    return (
        "Does this question imply a binary answer?\n\n"
        f"Question:\n{text}"
    )


def candidates_for_binary_filter(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if is_yes_no_question(item.get("choices") or [], item.get("question") or ""):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        out.append(
            {
                "id": item["id"],
                "question": question,
            }
        )
    return out


def judgment_record(item: dict, *, raw_response: str, verdict: str | None) -> dict[str, Any]:
    return {
        "id": item["id"],
        "question": item.get("question") or "",
        "verdict": verdict,
        "binary": True if verdict == "Yes" else (False if verdict == "No" else None),
        "reason": extract_reason(raw_response, verdict),
        "raw_response": raw_response,
        "scoring": "binary_question_text",
    }


def write_filter_outputs(
    judgments_path: Path,
    *,
    binary_ids_path: Path,
    open_ended_ids_path: Path,
    summary_path: Path,
    judge_model_id: str,
    n_skipped_yes_no: int,
    n_candidates: int,
) -> dict[str, Any]:
    records = load_jsonl(judgments_path) if judgments_path.is_file() else []
    by_id: dict[str, dict] = {}
    for item in records:
        qid = str(item.get("id") or "")
        if qid:
            by_id[qid] = item
    ordered = list(by_id.values())
    binary_ids = [
        str(item["id"]) for item in ordered if item.get("verdict") == "Yes"
    ]
    open_ids = [
        str(item["id"]) for item in ordered if item.get("verdict") == "No"
    ]
    n_unparsed = sum(
        1 for item in ordered if item.get("verdict") not in {"Yes", "No"}
    )
    write_id_csv(binary_ids_path, binary_ids)
    write_id_csv(open_ended_ids_path, open_ids)
    summary = {
        "judge_model_id": judge_model_id,
        "scoring": "binary_question_text",
        "n_skipped_yes_no": n_skipped_yes_no,
        "n_candidates": n_candidates,
        "n_judged": len(ordered),
        "n_binary": len(binary_ids),
        "n_open_ended": len(open_ids),
        "n_unparsed": n_unparsed,
        "judgments_path": str(judgments_path),
        "binary_ids_path": str(binary_ids_path),
        "open_ended_ids_path": str(open_ended_ids_path),
    }
    write_json(summary_path, summary)
    return summary


async def classify_with_gemini(
    items: list[dict[str, Any]],
    judgments_path: Path,
    *,
    model_id: str,
    max_workers: int,
    qps: float,
    timeout: float,
    retries: int,
    retry_interval: float,
    max_output_tokens: int,
    thinking_level: str,
) -> dict[str, Any]:
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    removed = prune_incomplete_evaluations(judgments_path)
    if removed:
        print(f"[gemini] pruned {removed} incomplete rows")
    completed = load_completed_ids(judgments_path)
    pending = [item for item in items if item["id"] not in completed]
    print(
        f"[gemini] judge={model_id} pending={len(pending)} "
        f"completed={len(completed)} -> {judgments_path}"
    )
    if not pending:
        return {"status": "already_done", "n_ok": 0, "n_fail": 0, "n_pending": 0}

    scorer = GeminiUniformityScorer(
        model_name=model_id,
        api_max_retries=retries,
        api_retry_interval=retry_interval,
        qps=qps,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
        thinking_level=thinking_level,
        system_instruction=BINARY_SYSTEM_PROMPT,
    )
    semaphore = asyncio.Semaphore(max_workers)
    write_lock = asyncio.Lock()
    n_ok = 0
    n_fail = 0
    n_unparsed = 0

    async def _one(item: dict) -> None:
        nonlocal n_ok, n_fail, n_unparsed
        user_prompt = create_binary_user_prompt(item.get("question") or "")
        try:
            async with semaphore:
                raw = await scorer.call(item["id"], user_prompt)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"[gemini] failed {item['id']}: {exc}")
            return
        verdict = parse_uniformity_verdict(raw)
        if verdict is None:
            n_unparsed += 1
            print(f"[gemini] unparsed verdict for {item['id']}: {raw[:180]!r}")
        async with write_lock:
            write_jsonl(
                judgments_path,
                [judgment_record(item, raw_response=raw, verdict=verdict)],
                mode="a",
            )
            n_ok += 1
            if n_ok % 10 == 0 or n_ok == len(pending):
                print(f"[gemini] scored {n_ok}/{len(pending)}")

    await asyncio.gather(*[_one(item) for item in pending])
    return {
        "status": "ok",
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_unparsed": n_unparsed,
        "n_pending": len(pending),
    }


async def async_main(args: argparse.Namespace) -> dict:
    results_dir = Path(args.results_dir).expanduser().resolve()
    items, _bundle = build_uniformity_items(
        results_dir,
        args.source_run_id,
        limit=args.limit,
    )
    n_skipped = sum(
        1
        for item in items
        if is_yes_no_question(item.get("choices") or [], item.get("question") or "")
    )
    candidates = candidates_for_binary_filter(items)
    print(
        f"[filter] {len(items)} questions, {n_skipped} already yes/no, "
        f"{len(candidates)} sent to Gemini (question text only)"
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = out_dir / JUDGMENTS_PATH.name
    binary_ids_path = out_dir / GEMINI_BINARY_IDS_PATH.name
    open_ended_ids_path = out_dir / OPEN_ENDED_IDS_PATH.name
    summary_path = out_dir / SUMMARY_PATH.name

    grade = await classify_with_gemini(
        candidates,
        judgments_path,
        model_id=args.gemini_model,
        max_workers=args.max_workers,
        qps=args.qps,
        timeout=args.timeout,
        retries=args.retries,
        retry_interval=args.retry_interval,
        max_output_tokens=args.max_tokens,
        thinking_level=args.thinking_level,
    )
    summary = write_filter_outputs(
        judgments_path,
        binary_ids_path=binary_ids_path,
        open_ended_ids_path=open_ended_ids_path,
        summary_path=summary_path,
        judge_model_id=args.gemini_model,
        n_skipped_yes_no=n_skipped,
        n_candidates=len(candidates),
    )
    result = {**grade, "summary": summary}
    print("[gemini] done:", result)
    print(f"[filter] binary IDs:     {binary_ids_path} ({summary['n_binary']})")
    print(f"[filter] open-ended IDs: {open_ended_ids_path} ({summary['n_open_ended']})")
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max source questions before yes/no skip (0 = all).",
    )
    p.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--out-dir", default=str(PACKAGE_DIR))
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--retries", type=int, default=20)
    p.add_argument("--retry-interval", type=float, default=1.0)
    p.add_argument("--qps", type=float, default=4.0)
    p.add_argument("--max-workers", "-j", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--thinking-level",
        default="medium",
        help="Gemini 3 thinking_level: low, medium, or high.",
    )
    args = p.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
