"""Local MMAR-Rubrics evaluation with OpenAI and/or Anthropic APIs.

Reads an existing MC difficulty run (qwen3-omni or af-next-think first-shot
traces, first 100 ids) and writes judged outputs under
``outputs/exp-mmar-rubrics-api/`` so Modal downloads into
``outputs/exp-mmar-rubrics/`` cannot overwrite them.

Each test-taker is a separate viewer dropdown entry under the same source
run id.

Usage::

    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...

    uv run python run_rubrics_eval_api.py \\
      --source-run-id 20260807T144946Z \\
      --model qwen3-omni \\
      --limit 100 \\
      --providers openai,anthropic

    uv run python run_rubrics_eval_api.py \\
      --source-run-id 20260807T144946Z \\
      --model af-next-think \\
      --limit 100 \\
      --providers openai,anthropic

    # One provider:
    uv run python run_rubrics_eval_api.py --providers openai
    uv run python run_rubrics_eval_api.py --providers anthropic
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from evaluation_rubrics import (
    ANTHROPIC_MODEL_NAME,
    DEFAULT_API_MAX_RETRIES,
    DEFAULT_API_RETRY_INTERVAL,
    DEFAULT_MAX_WORKERS,
    DEFAULT_QPS,
    DEFAULT_TIMEOUT,
    MODEL_NAME,
    AnthropicScorer,
    OpenAIScorer,
    evaluate_one_record,
)
from mmar_rubrics import (
    DEFAULT_LIMIT,
    DEFAULT_MODEL_LABEL,
    RUBRICS_API_EXPERIMENT,
    SOURCE_EXPERIMENT,
    append_evaluated,
    build_rubric_input_items,
    evaluated_record_from_result,
    judge_model_dir,
    load_completed_ids,
    prune_incomplete_evaluations,
    write_judge_scores,
    write_rubrics_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs"
DEFAULT_META = REPO_ROOT / "data" / "mmar" / "MMAR-meta.jsonl"

logger = logging.getLogger(__name__)


def _judge_label_for_model(model_id: str) -> str:
    return str(model_id).strip().replace("/", "-")


async def _run_provider(
    *,
    provider: str,
    items: list[dict],
    selected_ids: list[str],
    out_dir: Path,
    source_run_id: str,
    model_label: str,
    limit: int,
    openai_model: str,
    anthropic_model: str,
    timeout: float,
    retries: int,
    retry_interval: float,
    qps: float,
    max_workers: int,
) -> dict:
    if provider == "openai":
        judge_model_id = openai_model
        scorer = OpenAIScorer(
            api_max_retries=retries,
            api_retry_interval=retry_interval,
            max_workers=max_workers,
            qps=qps,
            timeout=timeout,
            model_name=openai_model,
        )
        backend = "openai"
    elif provider == "anthropic":
        judge_model_id = anthropic_model
        scorer = AnthropicScorer(
            api_max_retries=retries,
            api_retry_interval=retry_interval,
            max_workers=max_workers,
            qps=qps,
            timeout=timeout,
            model_name=anthropic_model,
        )
        backend = "anthropic"
    else:
        raise SystemExit(f"Unknown provider: {provider!r} (expected openai|anthropic)")

    judge_label = _judge_label_for_model(judge_model_id)
    model_dir = judge_model_dir(out_dir, judge_label, model_label)
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluated_path = model_dir / "predictions.evaluated.jsonl"

    write_rubrics_manifest(
        out_dir,
        source_run_id=source_run_id,
        model_label=model_label,
        question_ids=selected_ids,
        judge_label=judge_label,
        judge_model_id=judge_model_id,
        num_raters=1,
        limit=limit,
        backend=backend,
        experiment=RUBRICS_API_EXPERIMENT,
    )

    completed = load_completed_ids(evaluated_path)
    # Re-run prior short-circuited string-match fails (no raw LLM response).
    removed = prune_incomplete_evaluations(evaluated_path)
    if removed:
        print(f"[{provider}] pruned {removed} incomplete rows for rejudge")
        completed = load_completed_ids(evaluated_path)
    pending = [item for item in items if item["id"] not in completed]
    print(
        f"[{provider}] judge={judge_label} pending={len(pending)} "
        f"completed={len(completed)} -> {evaluated_path}"
    )
    if not pending:
        summary = write_judge_scores(
            model_dir,
            evaluated_path,
            judge_label=judge_label,
            judge_model_id=judge_model_id,
        )
        return {"provider": provider, "status": "already_done", "scores": summary}

    semaphore = asyncio.Semaphore(max_workers)
    write_lock = asyncio.Lock()
    n_ok = 0
    n_fail = 0

    async def _one(item: dict) -> None:
        nonlocal n_ok, n_fail
        result = await evaluate_one_record(
            scorer,
            semaphore,
            item["id"],
            item["question"],
            item["answer"],
            item["thinking"],
            item["cue"],
            item["choices"],
            item["rubric"],
            item["thinking_prediction"],
            item["answer_prediction"],
            num_raters=1,
        )
        if result.exception is not None:
            n_fail += 1
            print(f"[{provider}] failed {item['id']}: {result.exception}")
            return
        async with write_lock:
            append_evaluated(evaluated_path, [evaluated_record_from_result(item, result)])
            n_ok += 1
            if n_ok % 10 == 0 or n_ok == len(pending):
                print(f"[{provider}] scored {n_ok}/{len(pending)}")

    await asyncio.gather(*[_one(item) for item in pending])

    summary = write_judge_scores(
        model_dir,
        evaluated_path,
        judge_label=judge_label,
        judge_model_id=judge_model_id,
    )
    return {
        "provider": provider,
        "status": "ok",
        "n_ok": n_ok,
        "n_fail": n_fail,
        "scores": summary,
        "evaluated_path": str(evaluated_path),
    }


async def async_main(args: argparse.Namespace) -> dict:
    results_dir = Path(args.results_dir).expanduser().resolve()
    source_dir = results_dir / SOURCE_EXPERIMENT / args.source_run_id
    if not source_dir.is_dir():
        raise SystemExit(f"Source run not found: {source_dir}")

    meta_path = Path(args.meta).expanduser().resolve()
    if not meta_path.is_file():
        raise SystemExit(f"MMAR meta not found: {meta_path}")

    items, selected_ids = build_rubric_input_items(
        source_dir,
        model_label=args.model,
        meta_path=meta_path,
        limit=args.limit,
    )
    out_dir = results_dir / RUBRICS_API_EXPERIMENT / args.source_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    providers = [
        part.strip().lower()
        for part in str(args.providers).split(",")
        if part.strip()
    ]
    if not providers:
        raise SystemExit("Pass at least one provider via --providers")

    results = {}
    for provider in providers:
        results[provider] = await _run_provider(
            provider=provider,
            items=items,
            selected_ids=selected_ids,
            out_dir=out_dir,
            source_run_id=args.source_run_id,
            model_label=args.model,
            limit=args.limit,
            openai_model=args.openai_model,
            anthropic_model=args.anthropic_model,
            timeout=args.timeout,
            retries=args.retries,
            retry_interval=args.retry_interval,
            qps=args.qps,
            max_workers=args.max_workers,
        )
        print(f"[{provider}] done:", results[provider])
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run-id", default="20260807T144946Z")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_LABEL,
        help="Test-taker under the source run (qwen3-omni or af-next-think).",
    )
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument(
        "--providers",
        default="openai,anthropic",
        help="Comma-separated: openai, anthropic",
    )
    p.add_argument("--openai-model", default=MODEL_NAME)
    p.add_argument("--anthropic-model", default=ANTHROPIC_MODEL_NAME)
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--meta", default=str(DEFAULT_META))
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("--retries", type=int, default=DEFAULT_API_MAX_RETRIES)
    p.add_argument("--retry-interval", type=float, default=DEFAULT_API_RETRY_INTERVAL)
    p.add_argument("--qps", type=float, default=DEFAULT_QPS)
    p.add_argument("--max-workers", "-j", type=int, default=min(32, DEFAULT_MAX_WORKERS))
    args = p.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
