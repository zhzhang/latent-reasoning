"""Local answer-uniformity eval with Gemini 3.7 Flash.

Reads freeform MMAR traces from a question-difficulty run and asks whether
all extracted answer strings (every model × every shot) name the same
concept. Writes under ``outputs/exp-mmar-answer-variety/``.

Usage::

    export GEMINI_API_KEY=...

    uv run python answer-variety/run_answer_variety.py
    uv run python answer-variety/run_answer_variety.py \\
      --source-run-id 20260807T145000Z \\
      --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
for path in (str(PACKAGE_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from answer_variety import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LIMIT,
    DEFAULT_SOURCE_RUN_ID,
    build_uniformity_items,
    evaluated_path_for,
    grade_items_with_gemini,
    variety_run_dir,
    write_variety_manifest,
    write_variety_scores,
)

DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs"


async def async_main(args: argparse.Namespace) -> dict:
    results_dir = Path(args.results_dir).expanduser().resolve()
    items, bundle = build_uniformity_items(
        results_dir,
        args.source_run_id,
        limit=args.limit,
    )
    selected_ids = [str(item["id"]) for item in items]
    model_labels = list(bundle.get("model_labels") or [])

    out_dir = variety_run_dir(results_dir, args.source_run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluated_path = evaluated_path_for(out_dir)

    write_variety_manifest(
        out_dir,
        source_run_id=args.source_run_id,
        question_ids=selected_ids,
        judge_model_id=args.gemini_model,
        limit=args.limit,
        model_labels=model_labels,
    )

    grade = await grade_items_with_gemini(
        items,
        evaluated_path,
        model_id=args.gemini_model,
        max_workers=args.max_workers,
        qps=args.qps,
        timeout=args.timeout,
        retries=args.retries,
        retry_interval=args.retry_interval,
        max_output_tokens=args.max_tokens,
        thinking_level=args.thinking_level,
    )
    summary = write_variety_scores(
        out_dir,
        evaluated_path,
        judge_model_id=args.gemini_model,
    )
    result = {**grade, "scores": summary, "evaluated_path": str(evaluated_path)}
    print("[gemini] done:", result)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max questions (0 = all).",
    )
    p.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--retries", type=int, default=20)
    p.add_argument("--retry-interval", type=float, default=1.0)
    p.add_argument("--qps", type=float, default=4.0)
    p.add_argument("--max-workers", "-j", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--thinking-level",
        default="medium",
        help="Gemini 3 thinking_level: low, medium, or high (3.7 Flash has no minimal).",
    )
    args = p.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
