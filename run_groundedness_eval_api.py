"""Local audio-groundedness eval with Gemini 3.1 Pro.

The judge hears the wav clip and reads the test-taker's first-shot thinking
trace — no question stem and no answer options. Writes under
``outputs/exp-mmar-groundedness-api/`` so Modal downloads into
``outputs/exp-mmar-groundedness/`` cannot overwrite them.

Usage::

    export GEMINI_API_KEY=...

    uv run python run_groundedness_eval_api.py \\
      --source-run-id 20260807T144946Z \\
      --model af-next-think \\
      --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from mmar_groundedness import (
    DEFAULT_LIMIT,
    DEFAULT_MODEL_LABEL,
    GEMINI_JUDGE_ID,
    GROUNDEDNESS_API_EXPERIMENT,
    SOURCE_EXPERIMENT,
    build_groundedness_input_items,
    grade_items_with_gemini,
    judge_model_dir,
    resolve_gemini_model_id,
    write_groundedness_manifest,
    write_groundedness_scores,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "mmar"
DEFAULT_AUDIO_DIR = DEFAULT_DATA_DIR / "audio"

logger = logging.getLogger(__name__)


async def async_main(args: argparse.Namespace) -> dict:
    results_dir = Path(args.results_dir).expanduser().resolve()
    source_dir = results_dir / SOURCE_EXPERIMENT / args.source_run_id
    if not source_dir.is_dir():
        raise SystemExit(f"Source run not found: {source_dir}")

    data_root = Path(args.data_root).expanduser().resolve()
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    items, selected_ids = build_groundedness_input_items(
        source_dir,
        model_label=args.model,
        data_root=data_root,
        audio_dir=audio_dir,
        limit=args.limit,
    )

    judge_model_id = resolve_gemini_model_id(args.gemini_model)
    out_dir = results_dir / GROUNDEDNESS_API_EXPERIMENT / args.source_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = judge_model_dir(out_dir, judge_model_id, args.model)
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluated_path = model_dir / "predictions.evaluated.jsonl"

    write_groundedness_manifest(
        out_dir,
        source_run_id=args.source_run_id,
        model_label=args.model,
        question_ids=selected_ids,
        judge_label=judge_model_id,
        judge_model_id=judge_model_id,
        limit=args.limit,
        backend="gemini",
        experiment=GROUNDEDNESS_API_EXPERIMENT,
    )

    grade = await grade_items_with_gemini(
        items,
        evaluated_path,
        model_id=judge_model_id,
        max_workers=args.max_workers,
        qps=args.qps,
        timeout=args.timeout,
        retries=args.retries,
        retry_interval=args.retry_interval,
        max_output_tokens=args.max_tokens,
        thinking_level=args.thinking_level,
    )
    summary = write_groundedness_scores(
        model_dir,
        evaluated_path,
        judge_label=judge_model_id,
        judge_model_id=judge_model_id,
    )
    result = {**grade, "scores": summary, "evaluated_path": str(evaluated_path)}
    print("[gemini] done:", result)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run-id", default="20260807T144946Z")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_LABEL,
        help="Test-taker under the source run (default af-next-think).",
    )
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--gemini-model", default=GEMINI_JUDGE_ID)
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--data-root", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--retries", type=int, default=20)
    p.add_argument("--retry-interval", type=float, default=1.0)
    p.add_argument("--qps", type=float, default=4.0)
    p.add_argument("--max-workers", "-j", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--thinking-level",
        default="medium",
        help="Gemini 3 thinking_level: minimal, low, medium, or high.",
    )
    args = p.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
