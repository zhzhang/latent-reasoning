"""Throughput tuning harness for the freeform vLLM judge.

Replays real grade prompts from a past freeform run against one judge model
under several vLLM engine configs and concurrency levels, reporting output
tokens/sec plus verdict parity. Writes nothing to predictions / scores.

Target for Qwen3.6-35B-A3B-FP8 on a single H100: ~1200 output tok/s.

Measured on 512 replayed shots from 20260807T145000Z (mean ~600 output
tokens/shot). ``enforce_eager`` is the only lever that matters; widening
``max_num_seqs`` / ``max_num_batched_tokens`` and ``async_scheduling`` were
within noise, so the committed spec keeps them at their defaults:

    spec (eager, batch 64)      332 out tok/s
    graphs, batch 128         3,542
    graphs, batch 256         4,824
    graphs, batch 512         7,044

Usage::

    # Full sweep (one H100 container per engine variant, run in parallel)
    uv run modal run tune_judge.py::main

    # Limited/fast speed check on a single variant
    uv run modal run tune_judge.py::main --variants graphs --n-cases 64

    # Concurrency sweep for the winning engine config
    uv run modal run tune_judge.py::main --variants graphs --batch-sizes 128,256,512

    # Confirm the committed JUDGE_SPECS entry through the real grader path
    uv run modal run tune_judge.py::verify
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_cache import (
    RESULTS_MOUNT,
    VOLUME_MOUNT,
    hf_secret,
    results_volume,
    volume,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = RESULTS_MOUNT / "exp-mmar-question-difficulty"
DEFAULT_RUN_ID = "20260807T145000Z"
DEFAULT_JUDGE_MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
# Stored judge shown in the summary table's agreement column. Agreement
# against a degenerate judge is meaningless, so ``stored_pass_rate`` is
# reported alongside it — 20260807T145000Z's qwen3.6-27b-fp8 verdicts, for
# instance, are all Fail (it was run with too small a token budget to ever
# emit a verdict).
PARITY_JUDGE_LABEL = "qwen2.5-3b-instruct"
TARGET_OUTPUT_TOKS_PER_S = 1200.0

DEFAULT_N_CASES = 96
DEFAULT_BATCH_SIZES = "64,256"
DEFAULT_VARIANTS = "spec,graphs,graphs_async,graphs_async_wide"

# Engine kwarg overrides layered on top of grader.JUDGE_SPECS[...]["engine"].
# ``spec`` is the current committed config (the control).
ENGINE_VARIANTS: dict[str, dict[str, Any]] = {
    "spec": {},
    # MoE decode is kernel-launch bound; CUDA graphs are the main lever.
    "graphs": {
        "enforce_eager": False,
        "gpu_memory_utilization": 0.92,
    },
    "graphs_async": {
        "enforce_eager": False,
        "gpu_memory_utilization": 0.92,
        "async_scheduling": True,
    },
    # Isolates the batching knobs from async_scheduling.
    "graphs_wide": {
        "enforce_eager": False,
        "gpu_memory_utilization": 0.92,
        "max_num_seqs": 512,
        "max_num_batched_tokens": 16384,
    },
    # Also widen the decode batch: 3B active params leave KV room to spare.
    "graphs_async_wide": {
        "enforce_eager": False,
        "gpu_memory_utilization": 0.92,
        "async_scheduling": True,
        "max_num_seqs": 512,
        "max_num_batched_tokens": 16384,
    },
}

app = modal.App("tune-judge")


def _cuda_base_image(python_version: str = "3.12") -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            add_python=python_version,
        )
        .entrypoint([])
        .apt_install("ffmpeg", "git")
    )


judge_image = (
    _cuda_base_image()
    .uv_pip_install(
        "vllm==0.26.0",
        "transformers>=5.5.3",
        "huggingface-hub>=0.30.0",
        "numpy",
        "tqdm>=4.67.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    .add_local_python_source(
        "modal_cache",
        "mmar_common",
        "audio_flamingo_runtime",
        "grader",
    )
)


def _load_cases(*, run_dir: Path, n_cases: int) -> list[dict]:
    """Round-robin real (question, gold, prediction) triples across test models."""
    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        raise SystemExit(f"run models dir not found: {models_dir}")

    per_model: list[list[dict]] = []
    for model_dir in sorted(models_dir.iterdir()):
        pred_path = model_dir / "predictions.jsonl"
        if not pred_path.is_file():
            continue
        rows: list[dict] = []
        with open(pred_path, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                question = str(record.get("question") or "")
                answer = str(record.get("answer") or "")
                for shot in record.get("shots") or []:
                    prediction = str(
                        shot.get("answer_prediction") or shot.get("model_output") or ""
                    )
                    if not prediction.strip():
                        continue
                    stored = {
                        label: (entry or {}).get("correct")
                        for label, entry in (shot.get("judges") or {}).items()
                    }
                    rows.append(
                        {
                            "id": record.get("id"),
                            "model_label": model_dir.name,
                            "question": question,
                            "answer": answer,
                            "prediction": prediction,
                            "stored": stored,
                        }
                    )
        if rows:
            per_model.append(rows)

    if not per_model:
        raise SystemExit(f"No usable freeform shots under {models_dir}")

    cases: list[dict] = []
    index = 0
    while len(cases) < n_cases:
        added = False
        for rows in per_model:
            if index < len(rows):
                cases.append(rows[index])
                added = True
                if len(cases) >= n_cases:
                    break
        if not added:
            break
        index += 1
    return cases


def _parity(
    cases: list[dict],
    verdicts: list[bool | None],
) -> dict[str, dict[str, float | int | None]]:
    """Agreement with each stored judge, plus that judge's own pass rate.

    A stored judge that never passes anything yields agreement == 1 - pass
    rate, which says nothing about quality; ``stored_pass_rate`` makes that
    case visible instead of silently misleading.
    """
    labels: list[str] = []
    for case in cases:
        for label in case.get("stored") or {}:
            if label not in labels:
                labels.append(label)

    out: dict[str, dict[str, float | int | None]] = {}
    for label in labels:
        agree = 0
        comparable = 0
        stored_pass = 0
        for case, verdict in zip(cases, verdicts):
            stored = (case.get("stored") or {}).get(label)
            if stored is None or verdict is None:
                continue
            comparable += 1
            stored_pass += int(bool(stored))
            agree += int(bool(stored) == bool(verdict))
        out[label] = {
            "agreement": round(agree / comparable, 3) if comparable else None,
            "stored_pass_rate": (
                round(stored_pass / comparable, 3) if comparable else None
            ),
            "n_comparable": comparable,
        }
    return out


def _summarize(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "p50": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": round(sum(ordered) / len(ordered), 1),
        "p50": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


@app.function(
    image=judge_image,
    gpu="H100",
    timeout=60 * 90,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=32768,
)
def benchmark_variant(
    variant: str,
    model_id: str = DEFAULT_JUDGE_MODEL_ID,
    run_id: str = DEFAULT_RUN_ID,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    n_cases: int = DEFAULT_N_CASES,
    batch_sizes: str = DEFAULT_BATCH_SIZES,
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
) -> dict:
    """Time one engine config over several concurrency levels."""
    import gc

    from vllm import LLM, SamplingParams

    from audio_flamingo_runtime import resolve_model_dir
    from grader import (
        build_grade_prompt,
        parse_grade_verdict,
        resolve_judge_model_id,
        resolve_judge_spec,
    )

    volume.reload()
    results_volume.reload()

    if variant not in ENGINE_VARIANTS:
        raise SystemExit(
            f"Unknown variant {variant!r}; choose from {sorted(ENGINE_VARIANTS)}"
        )

    model_id = resolve_judge_model_id(model_id)
    spec = resolve_judge_spec(model_id)
    engine = {
        **dict(spec["engine"]),
        # vLLM's own throughput lines make long sweeps observable.
        "disable_log_stats": False,
        **ENGINE_VARIANTS[variant],
    }
    sampling_kwargs = dict(spec["sampling"])
    if max_tokens is not None:
        sampling_kwargs["max_tokens"] = int(max_tokens)

    run_dir = Path(output_dir).expanduser().resolve() / run_id
    cases = _load_cases(run_dir=run_dir, n_cases=n_cases)
    print(f"[{variant}] loaded {len(cases)} cases from {run_dir}")
    print(f"[{variant}] engine={engine}")
    print(f"[{variant}] sampling={sampling_kwargs}")

    local_id = resolve_model_dir(model_id, None)
    init_start = time.perf_counter()
    try:
        llm = LLM(model=local_id, **engine)
    except TypeError as exc:
        # Older vLLM builds may not accept every engine kwarg in the spec.
        if not engine.get("language_model_only"):
            raise
        print(f"[{variant}] language_model_only unsupported ({exc}); retrying without")
        engine.pop("language_model_only", None)
        llm = LLM(model=local_id, **engine)
    except Exception as exc:  # noqa: BLE001 - one bad variant must not kill the sweep
        print(f"[{variant}] engine init FAILED: {type(exc).__name__}: {exc}")
        return {
            "variant": variant,
            "model_id": model_id,
            "engine": {k: str(v) for k, v in engine.items()},
            "status": "engine_init_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "rows": [],
        }
    init_secs = time.perf_counter() - init_start
    tokenizer = llm.get_tokenizer()
    print(f"[{variant}] engine init took {init_secs:.1f}s")

    def _render(case: dict) -> str:
        user_text = build_grade_prompt(
            question=str(case["question"]),
            answer=str(case["answer"]),
            prediction=str(case["prediction"]),
        )
        messages = [{"role": "user", "content": user_text}]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)

    prompts = [_render(case) for case in cases]

    # Warm up kernels / JIT so the first timed batch is not penalized.
    warmup = SamplingParams(temperature=0.0, max_tokens=16, seed=0)
    llm.generate(prompts[:8], sampling_params=warmup, use_tqdm=False)

    sizes = [int(x) for x in str(batch_sizes).split(",") if str(x).strip()]
    rows: list[dict] = []
    for size in sizes:
        sampling = SamplingParams(**sampling_kwargs)
        chunk_outputs: list[Any] = []
        elapsed = 0.0
        for start in range(0, len(prompts), size):
            chunk = prompts[start : start + size]
            began = time.perf_counter()
            outs = llm.generate(chunk, sampling_params=sampling, use_tqdm=False)
            elapsed += time.perf_counter() - began
            chunk_outputs.extend(outs)

        out_lens: list[int] = []
        prompt_lens: list[int] = []
        finish_reasons: dict[str, int] = {}
        texts: list[str] = []
        for out in chunk_outputs:
            prompt_ids = getattr(out, "prompt_token_ids", None) or []
            prompt_lens.append(len(prompt_ids))
            completion = (getattr(out, "outputs", None) or [None])[0]
            if completion is None:
                continue
            token_ids = getattr(completion, "token_ids", None) or []
            out_lens.append(len(token_ids))
            reason = str(getattr(completion, "finish_reason", None))
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
            texts.append(str(getattr(completion, "text", "") or ""))

        verdicts = [parse_grade_verdict(text) for text in texts]
        parsed = [v for v in verdicts if v is not None]
        parity = _parity(cases, verdicts)

        total_out = sum(out_lens)
        row = {
            "variant": variant,
            "batch_size": size,
            "n_prompts": len(chunk_outputs),
            "wall_secs": round(elapsed, 2),
            "output_tokens": total_out,
            "output_toks_per_s": round(total_out / elapsed, 1) if elapsed else None,
            "prompt_tokens": sum(prompt_lens),
            "prompt_toks_per_s": (
                round(sum(prompt_lens) / elapsed, 1) if elapsed else None
            ),
            "secs_per_1k_shots": (
                round(elapsed / len(chunk_outputs) * 1000, 1) if chunk_outputs else None
            ),
            "output_len": _summarize(out_lens),
            "prompt_len": _summarize(prompt_lens),
            "finish_reasons": finish_reasons,
            "parse_rate": (
                round(len(parsed) / len(verdicts), 3) if verdicts else None
            ),
            "pass_rate": (
                round(sum(1 for v in parsed if v) / len(parsed), 3) if parsed else None
            ),
            "parity": parity,
        }
        rows.append(row)
        print(f"[{variant}] {json.dumps(row)}")

    del llm
    gc.collect()

    return {
        "variant": variant,
        "model_id": model_id,
        "engine": {k: str(v) for k, v in engine.items()},
        "sampling": sampling_kwargs,
        "enable_thinking": enable_thinking,
        "status": "ok",
        "init_secs": round(init_secs, 1),
        "n_cases": len(cases),
        "rows": rows,
    }


@app.function(
    image=judge_image,
    gpu="H100",
    timeout=60 * 90,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=32768,
)
def verify_spec(
    model_id: str = DEFAULT_JUDGE_MODEL_ID,
    run_id: str = DEFAULT_RUN_ID,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    n_cases: int = 512,
) -> dict:
    """Time the committed spec through the real grader path (no overrides)."""
    from grader import (
        grade_shot_batch,
        load_grader,
        resolve_judge_batch_size,
        resolve_judge_model_id,
    )

    volume.reload()
    results_volume.reload()

    model_id = resolve_judge_model_id(model_id)
    run_dir = Path(output_dir).expanduser().resolve() / run_id
    cases = _load_cases(run_dir=run_dir, n_cases=n_cases)

    init_start = time.perf_counter()
    handle = load_grader(model_id)
    init_secs = time.perf_counter() - init_start
    batch_size = resolve_judge_batch_size(model_id)
    print(f"[verify] init={init_secs:.1f}s batch_size={batch_size} n_cases={len(cases)}")

    jobs = [
        {
            "question": case["question"],
            "answer": case["answer"],
            "prediction": case["prediction"],
        }
        for case in cases
    ]

    # Mirror grade_predictions_file's chunking exactly.
    results: list[dict] = []
    elapsed = 0.0
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        began = time.perf_counter()
        results.extend(grade_shot_batch(handle, chunk))
        elapsed += time.perf_counter() - began

    tokenizer = handle["tokenizer"]
    out_tokens = sum(
        len(tokenizer.encode(str(r.get("generation") or ""))) for r in results
    )
    parsed = [r for r in results if r.get("grader_verdict_raw") is not None]
    parity = _parity(cases, [r.get("grader_verdict_raw") for r in results])

    summary = {
        "model_id": model_id,
        "batch_size": batch_size,
        "n_shots": len(results),
        "init_secs": round(init_secs, 1),
        "wall_secs": round(elapsed, 2),
        "output_tokens": out_tokens,
        "output_toks_per_s": round(out_tokens / elapsed, 1) if elapsed else None,
        "secs_per_1k_shots": (
            round(elapsed / len(results) * 1000, 1) if results else None
        ),
        "parse_rate": round(len(parsed) / len(results), 3) if results else None,
        "pass_rate": (
            round(sum(1 for r in parsed if r["grader_verdict_raw"]) / len(parsed), 3)
            if parsed
            else None
        ),
        "parity": parity,
    }
    print(f"[verify] {json.dumps(summary)}")
    return summary


@app.local_entrypoint()
def verify(
    model_id: str = DEFAULT_JUDGE_MODEL_ID,
    run_id: str = DEFAULT_RUN_ID,
    n_cases: int = 512,
):
    """Confirm the committed JUDGE_SPECS entry hits the throughput target."""
    summary = verify_spec.remote(model_id=model_id, run_id=run_id, n_cases=n_cases)
    rate = summary.get("output_toks_per_s") or 0.0
    verdict = "MEETS" if rate >= TARGET_OUTPUT_TOKS_PER_S else "BELOW"
    print(json.dumps(summary, indent=2))
    print(f"\n{verdict} target: {rate:.0f} vs {TARGET_OUTPUT_TOKS_PER_S:.0f} out tok/s")


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_JUDGE_MODEL_ID,
    run_id: str = DEFAULT_RUN_ID,
    variants: str = DEFAULT_VARIANTS,
    n_cases: int = DEFAULT_N_CASES,
    batch_sizes: str = DEFAULT_BATCH_SIZES,
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
):
    """Sweep engine variants in parallel (one H100 container each)."""
    names = [x.strip() for x in variants.split(",") if x.strip()]
    unknown = [x for x in names if x not in ENGINE_VARIANTS]
    if unknown:
        raise SystemExit(
            f"Unknown variants {unknown}; choose from {sorted(ENGINE_VARIANTS)}"
        )

    handles = [
        benchmark_variant.spawn(
            variant=name,
            model_id=model_id,
            run_id=run_id,
            n_cases=n_cases,
            batch_sizes=batch_sizes,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        for name in names
    ]
    summaries = [handle.get() for handle in handles]

    rows: list[dict] = []
    for summary in summaries:
        if summary.get("status") != "ok":
            print(f"[{summary.get('variant')}] {summary.get('error')}")
        for row in summary.get("rows") or []:
            rows.append({**row, "init_secs": summary.get("init_secs")})
    rows.sort(key=lambda r: r.get("output_toks_per_s") or 0.0, reverse=True)

    print("\n=== judge throughput sweep ===")
    header = (
        f"{'variant':<20} {'batch':>6} {'out tok/s':>10} {'wall s':>8} "
        f"{'mean out':>9} {'parse':>6} {'pass':>6} {'agree':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        rate = row.get("output_toks_per_s") or 0
        target = "*" if rate >= TARGET_OUTPUT_TOKS_PER_S else " "
        parity = (row.get("parity") or {}).get(PARITY_JUDGE_LABEL) or {}
        agree = parity.get("agreement")
        print(
            f"{row['variant']:<20} {row['batch_size']:>6} "
            f"{rate:>9.1f}{target} "
            f"{row.get('wall_secs') or 0:>8.1f} "
            f"{(row.get('output_len') or {}).get('mean') or 0:>9.1f} "
            f"{row.get('parse_rate') if row.get('parse_rate') is not None else '-':>6} "
            f"{row.get('pass_rate') if row.get('pass_rate') is not None else '-':>6} "
            f"{agree if agree is not None else '-':>6}"
        )
    print(f"\n(* = meets {TARGET_OUTPUT_TOKS_PER_S:.0f} output tok/s target)")
    print(f"(agree = vs stored {PARITY_JUDGE_LABEL} verdicts)")

    # Stored judges that never pass anything make agreement uninformative.
    for summary in summaries:
        for row in summary.get("rows") or []:
            for label, stats in (row.get("parity") or {}).items():
                if stats.get("stored_pass_rate") == 0.0:
                    print(
                        f"warning: stored judge {label!r} passed 0/"
                        f"{stats.get('n_comparable')} shots in this run; "
                        "agreement against it is not a quality signal"
                    )
            break
        break
    print(json.dumps(summaries, indent=2)[:8000])
