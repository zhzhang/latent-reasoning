"""Run MMAR-Rubrics evaluation on Modal with a local vLLM judge (Qwen3.6).

Scores existing MC predictions (qwen3-omni or af-next-think first-shot
traces, first 100 question ids from a difficulty run) with a single judge
pass per sample — no 5-rater trim-mean.

Each test-taker writes under the same source-run folder
(``judges/<judge>/models/<model>/``) and appears as its own viewer dropdown
entry.

Prereq::

    uv run modal run --detach seed_volume.py --datasets none --models qwen3.6-35b-a3b-fp8

Usage::

    uv run modal run --detach run_rubrics_eval.py \\
      --source-run-id 20260807T144946Z \\
      --model qwen3-omni \\
      --limit 100 \\
      --judge-model-id qwen3.6-35b-a3b-fp8

    uv run modal run --detach run_rubrics_eval.py \\
      --source-run-id 20260807T144946Z \\
      --model af-next-think \\
      --limit 100 \\
      --judge-model-id qwen3.6-35b-a3b-fp8

Download::

    uv run modal run download_results.py \\
      --remote-path exp-mmar-rubrics/20260807T144946Z
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import modal

from modal_cache import (
    DEFAULT_MMAR_META,
    RESULTS_MOUNT,
    VOLUME_MOUNT,
    hf_secret,
    results_volume,
    volume,
)
from mmar_rubrics import (
    DEFAULT_LIMIT,
    DEFAULT_MODEL_LABEL,
    RUBRICS_EXPERIMENT,
    SOURCE_EXPERIMENT,
    append_evaluated,
    build_rubric_input_items,
    evaluated_record_from_result,
    judge_model_dir,
    load_completed_ids,
    partition_by_string_match,
    prune_incomplete_evaluations,
    write_judge_scores,
    write_rubrics_manifest,
)

app = modal.App("exp-mmar-rubrics")

DEFAULT_SOURCE_ROOT = RESULTS_MOUNT / SOURCE_EXPERIMENT
DEFAULT_OUTPUT_ROOT = RESULTS_MOUNT / RUBRICS_EXPERIMENT
DEFAULT_JUDGE_MODEL_ID = "qwen3.6-35b-a3b-fp8"


def _cuda_base_image(python_version: str = "3.12") -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            add_python=python_version,
        )
        .entrypoint([])
        .apt_install("ffmpeg", "git")
    )


def _mount_sources(image: modal.Image) -> modal.Image:
    return image.add_local_python_source(
        "modal_cache",
        "mmar_common",
        "mmar_rubrics",
        "evaluation_rubrics",
        "api_batch",
        "audio_flamingo_runtime",
        "grader",
    )


grader_image = _mount_sources(
    _cuda_base_image()
    .uv_pip_install(
        "vllm==0.26.0",
        "transformers>=5.5.3",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "numpy",
        "tqdm>=4.67.0",
        "accelerate>=1.14.0",
        "openai>=1.82.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
)

cpu_image = _mount_sources(
    modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "numpy", "tqdm>=4.67.0", "openai>=1.82.0"
    )
)


def _resolve_run_dir(root: str, run_id: str) -> Path:
    base = Path(root).expanduser().resolve()
    if base.name == run_id:
        return base
    return base / run_id


def _format_rubric_chat(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
    return f"System: {system_prompt}\n\nUser: {user_prompt}\nAssistant:"


@app.function(
    image=cpu_image,
    timeout=10 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def prepare_rubrics_eval(
    source_run_id: str,
    model_label: str = DEFAULT_MODEL_LABEL,
    limit: int = DEFAULT_LIMIT,
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID,
    source_root: str = str(DEFAULT_SOURCE_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    meta_path: str = str(DEFAULT_MMAR_META),
) -> dict:
    from grader import resolve_judge_model_id
    from mmar_common import judge_label

    volume.reload()
    results_volume.reload()

    source_dir = _resolve_run_dir(source_root, source_run_id)
    if not source_dir.is_dir():
        raise SystemExit(f"Source run not found: {source_dir}")

    predictions_path = source_dir / "models" / model_label / "predictions.jsonl"
    if not predictions_path.is_file():
        raise SystemExit(f"Predictions not found: {predictions_path}")

    judge_model_id = resolve_judge_model_id(judge_model_id)
    judge_key = judge_label(judge_model_id)
    if not judge_key:
        raise SystemExit(f"Invalid judge_model_id: {judge_model_id!r}")

    items, selected_ids = build_rubric_input_items(
        source_dir,
        model_label=model_label,
        meta_path=Path(meta_path),
        limit=limit,
    )

    out_dir = _resolve_run_dir(output_root, source_run_id)
    write_rubrics_manifest(
        out_dir,
        source_run_id=source_run_id,
        model_label=model_label,
        question_ids=selected_ids,
        judge_label=judge_key,
        judge_model_id=judge_model_id,
        num_raters=1,
        limit=limit,
        backend="vllm",
        experiment=RUBRICS_EXPERIMENT,
    )
    results_volume.commit()

    return {
        "source_run_id": source_run_id,
        "source_dir": str(source_dir),
        "output_dir": str(out_dir),
        "model_label": model_label,
        "judge_model_id": judge_model_id,
        "judge_label": judge_key,
        "limit": limit,
        "n_items": len(items),
        "n_question_ids": len(selected_ids),
        "meta_path": str(meta_path),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }


@app.function(
    image=grader_image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=32768,
)
def run_qwen_rubrics(
    source_run_id: str,
    model_label: str = DEFAULT_MODEL_LABEL,
    limit: int = DEFAULT_LIMIT,
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID,
    source_root: str = str(DEFAULT_SOURCE_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    meta_path: str = str(DEFAULT_MMAR_META),
    batch_size: int | None = None,
    max_tokens: int = 4096,
) -> dict:
    """Batch-score string-match-passing items with the Qwen vLLM judge."""
    from evaluation_rubrics import (
        EVALUATE_SYS_PROMPT,
        LENGTH_LIMIT,
        create_evaluation_user_prompt,
        evaluation_result_from_raw,
        string_match,
    )
    from grader import (
        judge_sampling_params,
        load_grader,
        resolve_judge_batch_size,
        resolve_judge_model_id,
    )
    from mmar_common import judge_label

    volume.reload()
    results_volume.reload()

    source_dir = _resolve_run_dir(source_root, source_run_id)
    out_dir = _resolve_run_dir(output_root, source_run_id)

    judge_model_id = resolve_judge_model_id(judge_model_id)
    judge_key = judge_label(judge_model_id)
    model_dir = judge_model_dir(out_dir, judge_key, model_label)
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluated_path = model_dir / "predictions.evaluated.jsonl"

    items, selected_ids = build_rubric_input_items(
        source_dir,
        model_label=model_label,
        meta_path=Path(meta_path),
        limit=limit,
    )
    removed = prune_incomplete_evaluations(evaluated_path)
    if removed:
        print(f"[rubrics] pruned {removed} incomplete (pre-LLM short-circuit) rows")
    completed = load_completed_ids(evaluated_path)
    pending = [item for item in items if item["id"] not in completed]
    match_pass, match_fail = partition_by_string_match(pending)
    print(
        f"[rubrics] judge={judge_key} model={model_label} "
        f"items={len(items)} pending={len(pending)} completed={len(completed)} "
        f"(string_match pass={len(match_pass)} fail={len(match_fail)}; both judged)"
    )

    write_rubrics_manifest(
        out_dir,
        source_run_id=source_run_id,
        model_label=model_label,
        question_ids=selected_ids,
        judge_label=judge_key,
        judge_model_id=judge_model_id,
        num_raters=1,
        limit=limit,
        backend="vllm",
        experiment=RUBRICS_EXPERIMENT,
    )

    if not pending:
        summary = write_judge_scores(
            model_dir,
            evaluated_path,
            judge_label=judge_key,
            judge_model_id=judge_model_id,
        )
        results_volume.commit()
        return {
            "status": "already_done",
            "n_pending": 0,
            "scores": summary,
            "evaluated_path": str(evaluated_path),
        }

    # Judge every pending item (including string-match failures).
    llm_jobs = []
    for item in pending:
        thinking = item["thinking_prediction"]
        answer = item["answer_prediction"]
        if len(answer) >= LENGTH_LIMIT or len(thinking) >= LENGTH_LIMIT:
            print(
                f"[rubrics] skipping overlong prediction for {item['id']} "
                f"(thinking={len(thinking)} answer={len(answer)})"
            )
            continue
        user_prompt = create_evaluation_user_prompt(
            item["question"],
            item["answer"],
            item["thinking"],
            item["cue"],
            thinking,
            answer,
            item["rubric"],
        )
        llm_jobs.append((item, user_prompt))

    if not llm_jobs:
        summary = write_judge_scores(
            model_dir,
            evaluated_path,
            judge_label=judge_key,
            judge_model_id=judge_model_id,
        )
        results_volume.commit()
        return {
            "status": "ok",
            "n_llm": 0,
            "scores": summary,
            "evaluated_path": str(evaluated_path),
        }

    handle = load_grader(judge_model_id)
    effective_batch = resolve_judge_batch_size(judge_model_id, batch_size)
    sampling = judge_sampling_params(judge_model_id, max_tokens=max_tokens)
    tokenizer = handle["tokenizer"]

    n_scored = 0
    n_parse_fail = 0
    for start in range(0, len(llm_jobs), effective_batch):
        batch = llm_jobs[start : start + effective_batch]
        prompts = [
            _format_rubric_chat(tokenizer, EVALUATE_SYS_PROMPT, user_prompt)
            for _, user_prompt in batch
        ]
        outputs = handle["llm"].generate(prompts, sampling_params=sampling)
        batch_records = []
        for (item, _), out in zip(batch, outputs):
            text = ""
            outs = getattr(out, "outputs", None) or []
            if outs:
                text = str(getattr(outs[0], "text", "") or "")
            try:
                answer_correct = string_match(
                    item["answer"], item["answer_prediction"], item["choices"]
                )
                result = evaluation_result_from_raw(
                    item["id"],
                    item["rubric"],
                    text,
                    correct=answer_correct,
                )
                batch_records.append(evaluated_record_from_result(item, result))
                n_scored += 1
            except Exception as exc:
                n_parse_fail += 1
                print(f"[rubrics] parse failed for {item['id']}: {exc}")
        if batch_records:
            append_evaluated(evaluated_path, batch_records)
            results_volume.commit()
        print(
            f"[rubrics] batch {start // effective_batch + 1}: "
            f"wrote {len(batch_records)} / {len(batch)}"
        )

    summary = write_judge_scores(
        model_dir,
        evaluated_path,
        judge_label=judge_key,
        judge_model_id=judge_model_id,
    )
    results_volume.commit()
    return {
        "status": "ok",
        "n_string_match_fail_pending": len(match_fail),
        "n_llm_jobs": len(llm_jobs),
        "n_scored": n_scored,
        "n_parse_fail": n_parse_fail,
        "scores": summary,
        "evaluated_path": str(evaluated_path),
        "output_dir": str(out_dir),
    }


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def run_rubrics_pipeline(
    source_run_id: str,
    model_label: str = DEFAULT_MODEL_LABEL,
    limit: int = DEFAULT_LIMIT,
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID,
    source_root: str = str(DEFAULT_SOURCE_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    meta_path: str = str(DEFAULT_MMAR_META),
    batch_size: int | None = None,
    max_tokens: int = 4096,
) -> dict:
    """Remote orchestrator so ``modal run --detach`` survives both phases."""
    prep = prepare_rubrics_eval.remote(
        source_run_id=source_run_id,
        model_label=model_label,
        limit=limit,
        judge_model_id=judge_model_id,
        source_root=source_root,
        output_root=output_root,
        meta_path=meta_path,
    )
    print("Prepared:", prep)
    grade = run_qwen_rubrics.remote(
        source_run_id=source_run_id,
        model_label=model_label,
        limit=limit,
        judge_model_id=judge_model_id,
        source_root=source_root,
        output_root=output_root,
        meta_path=meta_path,
        batch_size=batch_size,
        max_tokens=max_tokens,
    )
    print("Graded:", grade)
    return {"prepare": prep, "grade": grade}


@app.local_entrypoint()
def main(
    source_run_id: str = "20260807T144946Z",
    model: str = DEFAULT_MODEL_LABEL,
    limit: int = DEFAULT_LIMIT,
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID,
    source_root: str = str(DEFAULT_SOURCE_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    meta_path: str = str(DEFAULT_MMAR_META),
    batch_size: int | None = None,
    max_tokens: int = 4096,
):
    """Evaluate MMAR rubrics for one MC model with Qwen3.6 on Modal.

    Args:
        source_run_id: Existing ``exp-mmar-question-difficulty/<id>`` run.
        model: Test-taker label under ``models/`` (qwen3-omni or af-next-think).
        limit: First N question ids from the source run.
        judge_model_id: Local vLLM judge (default Qwen3.6-35B-A3B-FP8).
        source_root: Results volume path to the source experiment.
        output_root: Results volume path for rubrics outputs.
        meta_path: MMAR meta with thinking/rubric/cue.
        batch_size: Optional vLLM generate batch size override.
        max_tokens: Max generation tokens per rubric judgment.
    """
    if not source_run_id or not str(source_run_id).strip():
        raise SystemExit("--source-run-id is required")

    call = run_rubrics_pipeline.spawn(
        source_run_id=source_run_id.strip(),
        model_label=model.strip(),
        limit=int(limit),
        judge_model_id=judge_model_id.strip(),
        source_root=source_root,
        output_root=output_root,
        meta_path=meta_path,
        batch_size=batch_size,
        max_tokens=max_tokens,
    )
    print(f"Spawned rubrics pipeline: {call.object_id}")
    print(
        "Download when finished:\n"
        f"  uv run modal run download_results.py "
        f"--remote-path {RUBRICS_EXPERIMENT}/{source_run_id.strip()}"
    )
    return {"call_id": call.object_id}
