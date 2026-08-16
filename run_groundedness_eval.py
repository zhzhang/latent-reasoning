"""Audio-groundedness eval on Modal with Qwen3-Omni as the judge.

The judge hears the clip and reads the test-taker's first-shot thinking
trace — no question stem and no answer options. Start by grading
af-next-think traces.

Prereq::

    uv run modal run seed_volume.py --datasets mmar --models qwen3-omni

Usage::

    uv run modal run --detach run_groundedness_eval.py \\
      --source-run-id 20260807T144946Z \\
      --model af-next-think \\
      --limit 100

    uv run modal run --detach run_groundedness_eval.py \\
      --source-run-id 20260807T144946Z \\
      --model af-next-think \\
      --judge-model qwen3-omni

Download::

    uv run modal run download_results.py \\
      --remote-path exp-mmar-groundedness/20260807T144946Z

Gemini 3.1 Pro (audio API, local)::

    export GEMINI_API_KEY=...
    uv run python run_groundedness_eval_api.py \\
      --source-run-id 20260807T144946Z \\
      --model af-next-think \\
      --limit 100
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import modal

from evaluation_rubrics import LENGTH_LIMIT
from modal_cache import (
    DEFAULT_MMAR_DATA_ROOT,
    RESULTS_MOUNT,
    VOLUME_MOUNT,
    hf_secret,
    results_volume,
    volume,
)
from mmar_groundedness import (
    DEFAULT_LIMIT,
    DEFAULT_MODEL_LABEL,
    GROUNDEDNESS_EXPERIMENT,
    QWEN3_OMNI_JUDGE_ID,
    QWEN3_OMNI_SAMPLE_RATE,
    SOURCE_EXPERIMENT,
    append_evaluated,
    build_groundedness_input_items,
    create_groundedness_user_prompt,
    evaluated_record_from_verdict,
    format_qwen3_omni_audio_prompt,
    judge_model_dir,
    load_completed_ids,
    parse_groundedness_verdict,
    prune_incomplete_evaluations,
    student_output_from_item,
    write_groundedness_manifest,
    write_groundedness_scores,
)

app = modal.App("exp-mmar-groundedness")

DEFAULT_SOURCE_ROOT = RESULTS_MOUNT / SOURCE_EXPERIMENT
DEFAULT_OUTPUT_ROOT = RESULTS_MOUNT / GROUNDEDNESS_EXPERIMENT
DEFAULT_JUDGE_MODEL = "qwen3-omni"
DEFAULT_MAX_MODEL_LEN = 8192

_SHARED_SOURCES = (
    "modal_cache",
    "mmar_common",
    "mmar_rubrics",
    "mmar_groundedness",
    "mmar_models",
    "audio_flamingo_runtime",
    "evaluation_rubrics",
    "grader",
)

_INPROC_VLLM_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}

_FUSED_MOE_CONFIG_CMD = (
    "D=/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs && "
    "SRC=\"$D/E=128,N=768,device_name=NVIDIA_H200.json\" && "
    "if [ ! -f \"$SRC\" ]; then echo \"fused_moe: no H200 config, skipping\"; "
    "else "
    "for name in NVIDIA_A100_80GB_PCIe NVIDIA_A100-SXM4-80GB; do "
    "DST=\"$D/E=128,N=768,device_name=$name.json\"; "
    "cp -n \"$SRC\" \"$DST\" 2>/dev/null || cp \"$SRC\" \"$DST\"; "
    "echo \"fused_moe: installed $name from H200\"; "
    "done; fi"
)


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
    return image.add_local_python_source(*_SHARED_SOURCES)


qwen3_omni_image = _mount_sources(
    _cuda_base_image()
    .uv_pip_install(
        "vllm[audio]==0.26.0",
        "transformers>=5.5.3",
        "mistral-common[audio]",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "soxr",
        "av",
        "numpy",
        "tqdm>=4.67.0",
        "accelerate>=1.14.0",
        "torch",
        "torchaudio",
        "openai>=1.82.0",
    )
    .run_commands(_FUSED_MOE_CONFIG_CMD)
    .env(_INPROC_VLLM_ENV)
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


def _judge_label(judge_model: str) -> str:
    text = str(judge_model or "").strip()
    if not text:
        return DEFAULT_JUDGE_MODEL
    if "/" in text:
        lower = text.lower()
        if "qwen3-omni" in lower:
            return "qwen3-omni"
        return text.replace("/", "-")
    return text


@app.function(
    image=cpu_image,
    timeout=10 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def prepare_groundedness_eval(
    source_run_id: str,
    model_label: str = DEFAULT_MODEL_LABEL,
    limit: int = DEFAULT_LIMIT,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    source_root: str = str(DEFAULT_SOURCE_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
) -> dict:
    volume.reload()
    results_volume.reload()

    source_dir = _resolve_run_dir(source_root, source_run_id)
    if not source_dir.is_dir():
        raise SystemExit(f"Source run not found: {source_dir}")

    predictions_path = source_dir / "models" / model_label / "predictions.jsonl"
    if not predictions_path.is_file():
        raise SystemExit(f"Predictions not found: {predictions_path}")

    judge_key = _judge_label(judge_model)
    judge_model_id = (
        QWEN3_OMNI_JUDGE_ID if judge_key == "qwen3-omni" else judge_model
    )

    items, selected_ids = build_groundedness_input_items(
        source_dir,
        model_label=model_label,
        data_root=Path(data_root),
        limit=limit,
    )

    out_dir = _resolve_run_dir(output_root, source_run_id)
    write_groundedness_manifest(
        out_dir,
        source_run_id=source_run_id,
        model_label=model_label,
        question_ids=selected_ids,
        judge_label=judge_key,
        judge_model_id=judge_model_id,
        limit=limit,
        backend="vllm",
        experiment=GROUNDEDNESS_EXPERIMENT,
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
        "data_root": data_root,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }


@app.function(
    image=qwen3_omni_image,
    gpu="A100-80GB",
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=65536,
)
def run_qwen3_omni_groundedness(
    source_run_id: str,
    model_label: str = DEFAULT_MODEL_LABEL,
    limit: int = DEFAULT_LIMIT,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    source_root: str = str(DEFAULT_SOURCE_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
    batch_size: int | None = 32,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> dict:
    """Grade thinking traces with Qwen3-Omni given audio only (no question)."""
    from vllm import SamplingParams

    from mmar_models import _load_audio_tuple, load_model

    volume.reload()
    results_volume.reload()

    source_dir = _resolve_run_dir(source_root, source_run_id)
    out_dir = _resolve_run_dir(output_root, source_run_id)
    data_root_path = Path(data_root)

    judge_key = _judge_label(judge_model)
    if judge_key != "qwen3-omni":
        raise SystemExit(
            f"run_groundedness_eval.py Modal path only supports qwen3-omni, "
            f"got {judge_model!r}."
        )
    judge_model_id = QWEN3_OMNI_JUDGE_ID
    model_dir = judge_model_dir(out_dir, judge_key, model_label)
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluated_path = model_dir / "predictions.evaluated.jsonl"

    items, selected_ids = build_groundedness_input_items(
        source_dir,
        model_label=model_label,
        data_root=data_root_path,
        limit=limit,
    )
    removed = prune_incomplete_evaluations(evaluated_path)
    if removed:
        print(f"[groundedness] pruned {removed} incomplete rows")
    completed = load_completed_ids(evaluated_path)
    pending = [item for item in items if item["id"] not in completed]
    print(
        f"[groundedness] judge={judge_key} model={model_label} "
        f"items={len(items)} pending={len(pending)} completed={len(completed)}"
    )

    write_groundedness_manifest(
        out_dir,
        source_run_id=source_run_id,
        model_label=model_label,
        question_ids=selected_ids,
        judge_label=judge_key,
        judge_model_id=judge_model_id,
        limit=limit,
        backend="vllm",
        experiment=GROUNDEDNESS_EXPERIMENT,
    )

    if not pending:
        summary = write_groundedness_scores(
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

    llm_jobs: list[dict] = []
    for item in pending:
        output = student_output_from_item(item)
        if len(output) >= LENGTH_LIMIT:
            print(
                f"[groundedness] skipping overlong trace for {item['id']} "
                f"(chars={len(output)})"
            )
            continue
        audio_path = Path(item["audio_path"])
        if not audio_path.is_file():
            print(f"[groundedness] missing audio for {item['id']}: {audio_path}")
            continue
        llm_jobs.append(item)

    if not llm_jobs:
        summary = write_groundedness_scores(
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

    compile_cache = VOLUME_MOUNT / "vllm" / "qwen3-omni-groundedness"
    compile_cache.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(compile_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache / "torchinductor")

    args = SimpleNamespace(
        model_id=judge_model_id,
        local_model_dir=None,
        max_model_len=int(max_model_len),
        max_num_seqs=None,
        gpu_memory_utilization=None,
        seed=0,
    )
    handle = load_model("qwen3-omni", args)
    sampling = SamplingParams(
        temperature=float(temperature),
        top_p=0.95,
        top_k=20,
        max_tokens=int(max_tokens),
        seed=0,
        n=1,
    )
    effective_batch = max(1, int(batch_size or len(llm_jobs)))

    n_scored = 0
    n_unparsed = 0
    for start in range(0, len(llm_jobs), effective_batch):
        batch = llm_jobs[start : start + effective_batch]
        prompts = []
        for item in batch:
            user_text = create_groundedness_user_prompt(student_output_from_item(item))
            audio = _load_audio_tuple(item["audio_path"], QWEN3_OMNI_SAMPLE_RATE)
            prompts.append(
                {
                    "prompt": format_qwen3_omni_audio_prompt(user_text),
                    "multi_modal_data": {"audio": audio},
                }
            )
        outputs = handle["llm"].generate(prompts, sampling_params=sampling)
        batch_records = []
        for item, out in zip(batch, outputs):
            text = ""
            outs = getattr(out, "outputs", None) or []
            if outs:
                text = str(getattr(outs[0], "text", "") or "")
            verdict = parse_groundedness_verdict(text)
            if verdict is None:
                n_unparsed += 1
                print(
                    f"[groundedness] unparsed verdict for {item['id']}: "
                    f"{text[:180]!r}"
                )
            batch_records.append(
                evaluated_record_from_verdict(
                    item, raw_response=text, verdict=verdict
                )
            )
            n_scored += 1
        append_evaluated(evaluated_path, batch_records)
        results_volume.commit()
        print(
            f"[groundedness] batch {start // effective_batch + 1}: "
            f"wrote {len(batch_records)} / {len(batch)}"
        )

    try:
        volume.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[groundedness] volume.commit after generate failed: {exc}")

    summary = write_groundedness_scores(
        model_dir,
        evaluated_path,
        judge_label=judge_key,
        judge_model_id=judge_model_id,
    )
    results_volume.commit()
    return {
        "status": "ok",
        "n_llm_jobs": len(llm_jobs),
        "n_scored": n_scored,
        "n_unparsed": n_unparsed,
        "scores": summary,
        "evaluated_path": str(evaluated_path),
        "output_dir": str(out_dir),
    }


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def run_groundedness_pipeline(
    source_run_id: str,
    model_label: str = DEFAULT_MODEL_LABEL,
    limit: int = DEFAULT_LIMIT,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    source_root: str = str(DEFAULT_SOURCE_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
    batch_size: int | None = 32,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> dict:
    """Remote orchestrator so ``modal run --detach`` survives both phases."""
    prep = prepare_groundedness_eval.remote(
        source_run_id=source_run_id,
        model_label=model_label,
        limit=limit,
        judge_model=judge_model,
        source_root=source_root,
        output_root=output_root,
        data_root=data_root,
    )
    print("Prepared:", prep)
    grade = run_qwen3_omni_groundedness.remote(
        source_run_id=source_run_id,
        model_label=model_label,
        limit=limit,
        judge_model=judge_model,
        source_root=source_root,
        output_root=output_root,
        data_root=data_root,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
        max_model_len=max_model_len,
    )
    print("Graded:", grade)
    return {"prepare": prep, "grade": grade}


@app.local_entrypoint()
def main(
    source_run_id: str = "20260807T144946Z",
    model: str = DEFAULT_MODEL_LABEL,
    limit: int = DEFAULT_LIMIT,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    source_root: str = str(DEFAULT_SOURCE_ROOT),
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
    batch_size: int | None = 32,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
):
    """Grade first-shot thinking traces for audio groundedness with Qwen3-Omni.

    Args:
        source_run_id: Existing ``exp-mmar-question-difficulty/<id>`` run.
        model: Test-taker whose traces are judged (default af-next-think).
        limit: First N question ids from the source run.
        judge_model: Audio judge (qwen3-omni on Modal).
        source_root: Results volume path to the source experiment.
        output_root: Results volume path for groundedness outputs.
        data_root: MMAR data root with ``audio/*.wav``.
        batch_size: Prompts per vLLM generate() call.
        max_tokens: Max generation tokens per judgment.
        temperature: Judge sampling temperature.
        max_model_len: vLLM context length (audio + trace + verdict).
    """
    if not source_run_id or not str(source_run_id).strip():
        raise SystemExit("--source-run-id is required")

    call = run_groundedness_pipeline.spawn(
        source_run_id=source_run_id.strip(),
        model_label=model.strip(),
        limit=int(limit),
        judge_model=judge_model.strip(),
        source_root=source_root,
        output_root=output_root,
        data_root=data_root,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
        max_model_len=max_model_len,
    )
    print(f"Spawned groundedness pipeline: {call.object_id}")
    print(
        "Download when finished:\n"
        f"  uv run modal run download_results.py "
        f"--remote-path {GROUNDEDNESS_EXPERIMENT}/{source_run_id.strip()}"
    )
    return {"call_id": call.object_id}
