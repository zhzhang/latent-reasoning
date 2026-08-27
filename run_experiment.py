"""MMAR freeform generation on Modal (vLLM / vLLM-Omni).

Runs the full MMAR set (every clip with audio), ``n_shots`` temperature
samples per model (per-model SamplingParams). Questions that already have
``n_shots`` generations are skipped on resume. Grading is a separate
pipeline (``run_judges.py``).

Inference backends:
  - af-next-think: vLLM 0.24 (native MusicFlamingo)
  - music-flamingo: vLLM 0.24 AudioFlamingo3 (HF fallback)
  - mimo-audio-7b: vLLM-Omni 0.24
  - step-audio-2-mini-think: HF StepAudio2 (official examples-think.py text path)
  - interactive-omni-8b: vLLM transformers backend (HF .chat fallback)
  - qwen3-omni: vLLM 0.28 thinker-only (Qwen3-Omni-30B-A3B-Thinking)
  - qwen3-omni-instruct / qwen2.5-omni-7b / phi-4-multimodal / gemma-4-e4b /
    nemotron-3-nano-omni: vLLM 0.28 audio
  - voxtral-small-24b: vLLM 0.28 Mistral-format audio

Results layout on ``latent-reasoning-results``:

    exp-mmar-question-difficulty/<run_id>/
      question_ids.json
      manifest.json
      models/<label>/predictions.jsonl
      difficulty.jsonl
      scores.json

Prereqs:

    uv run modal run seed_volume.py --datasets mmar \\
      --models af-next-think,mimo-audio-7b,interactive-omni-8b,qwen3-omni,voxtral-small-24b

Usage:

    uv run modal run --detach run_experiment.py
    uv run modal run run_experiment.py \\
      --models af-next-think --n-shots 2
    uv run modal run --detach seed_volume.py --datasets none --models music-flamingo
    uv run modal run --detach run_experiment.py \\
      --run-id 20260807T145000Z --models music-flamingo --n-shots 5 --seed 42
    uv run modal run --detach seed_volume.py --datasets none \\
      --models qwen2.5-omni-7b,phi-4-multimodal,gemma-4-e4b,qwen3-omni-instruct,nemotron-3-nano-omni
    uv run modal run --detach run_experiment.py \\
      --models qwen2.5-omni-7b,phi-4-multimodal,gemma-4-e4b,qwen3-omni-instruct,nemotron-3-nano-omni \\
      --n-shots 3
    # Resume after a crash (full MMAR; skip questions that already have
    # the requested number of shots):
    uv run modal run --detach run_experiment.py --run-id <run_id>
    uv run modal run run_experiment.py --aggregate-only --run-id <run_id>
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import modal

from aggregate import aggregate_difficulty
from mmar_models import (
    ALL_MODEL_LABELS,
    MODEL_SPECS,
    backend_duplicates_shots,
    generate_batch,
    load_model,
    parse_model_list,
    resolve_sampling,
)
from mmar_common import (
    aggregate_n_shot_record,
    count_wavs,
    load_jsonl,
    make_run_id,
    resolve_path,
    write_json,
    write_jsonl,
)
from modal_cache import (
    DEFAULT_MMAR_DATA_ROOT,
    DEFAULT_MMAR_META,
    RESULTS_MOUNT,
    VOLUME_MOUNT,
    VLLM_WHEEL_INDEX,
    hf_secret,
    results_volume,
    volume,
)

REPO_ROOT = Path(__file__).resolve().parent
_DEPLOY_MOUNT = "/root/deploy"

DEFAULT_OUTPUT_DIR = RESULTS_MOUNT / "exp-mmar-question-difficulty"
DEFAULT_N_SHOTS = 10
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
# Modal rule: after any ``add_local_*``, no further build steps (apt/pip/run/env).
# Put installs + env first; mount local sources last.

_SHARED_SOURCES = (
    "modal_cache",
    "mmar_common",
    "audio_flamingo_runtime",
    "aggregate",
    "mmar_models",
)


def _mount_local_sources(image: modal.Image) -> modal.Image:
    """Attach Python modules + Omni deploy YAML (must be last image steps)."""
    return image.add_local_python_source(*_SHARED_SOURCES).add_local_dir(
        str(REPO_ROOT / "deploy"), remote_path=_DEPLOY_MOUNT
    )


_VLLM_CACHE_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}

# AF-Next / large multimodal: keep EngineCore in-process. Qwen3-Omni's
# profile_run hits a meta/cuda device mismatch under multiprocess EngineCore
# (Tensor on device meta is not on the expected device cuda:0). AF-Next also
# needs in-process for its symlink model-view load path.
_INPROC_VLLM_ENV = {
    **_VLLM_CACHE_ENV,
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}


def _cuda_base_image(python_version: str = "3.12") -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            add_python=python_version,
        )
        .entrypoint([])
        .apt_install("ffmpeg", "git")
    )


# AF-Next: vLLM 0.24 MusicFlamingo (+ HF fallback deps if weight load fails).
af_next_image = _mount_local_sources(
    _cuda_base_image()
    .uv_pip_install(
        "vllm==0.24.0",
        "transformers>=5.5.3",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "numpy",
        "tqdm>=4.67.0",
        "accelerate>=1.14.0",
        "soxr",
        "torch",
        "torchaudio",
        "peft>=0.15.2",
        "safetensors>=0.8.0",
    )
    .env(_INPROC_VLLM_ENV)
)

# MiMo: vLLM-Omni on the 0.24 line.
omni_image = _mount_local_sources(
    _cuda_base_image()
    .uv_pip_install(
        "vllm==0.24.0",
        "vllm-omni==0.24.0",
        "transformers>=5.5.3",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "numpy",
        "tqdm>=4.67.0",
        "accelerate==1.12.0",
        "einops",
        "torchaudio",
        "onnxruntime",
    )
    .env(_VLLM_CACHE_ENV)
)

# Step-Audio-2-mini-Think: official HF StepAudio2 (no vLLM-Omni).
step_audio_image = _mount_local_sources(
    _cuda_base_image()
    .uv_pip_install(
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "transformers==4.49.0",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "numpy",
        "tqdm>=4.67.0",
        "accelerate==1.12.0",
        "einops",
        "onnxruntime",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/stepfun-ai/Step-Audio2.git /opt/Step-Audio2"
    )
    .env({**_VLLM_CACHE_ENV, "PYTHONPATH": "/opt/Step-Audio2"})
)

# InteractiveOmni: newer vLLM transformers-audio backend + HF chat fallback.
interactive_omni_image = _mount_local_sources(
    _cuda_base_image()
    .uv_pip_install(
        "vllm==0.28.0",
        "transformers>=5.5.3",
        "torch",
        "torchaudio",
        "torchvision",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "safetensors>=0.8.0",
        "einops",
        "decord",
        "onnxruntime",
        "diffusers",
        "Pillow",
        "omegaconf",
        "scipy",
        "timm",
        "numpy",
        "tqdm>=4.67.0",
        "accelerate>=1.14.0",
        extra_index_url=VLLM_WHEEL_INDEX,
    )
    .env(_VLLM_CACHE_ENV)
)

# Qwen3-Omni thinker + Voxtral Small (A100-80GB); needs mistral-common[audio] + PyAV.
# Install E=128,N=768 fused-MoE Triton config under both A100 device names
# Modal may assign (PCIe or SXM4). vLLM 0.28 ships no A100 variant for this
# shape — use H200 bf16 as the best available stand-in vs untuned defaults.
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
large_mm_image = _mount_local_sources(
    _cuda_base_image()
    .uv_pip_install(
        "vllm[audio]==0.28.0",
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
        extra_index_url=VLLM_WHEEL_INDEX,
    )
    .run_commands(_FUSED_MOE_CONFIG_CMD)
    .env(_INPROC_VLLM_ENV)
)

# Lightweight CPU image for manifest / question-id helpers.
cpu_image = _mount_local_sources(
    modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "numpy", "tqdm>=4.67.0"
    )
)

app = modal.App("exp-mmar-question-difficulty")


# ---------------------------------------------------------------------------
# Shared helpers (run inside Modal containers)
# ---------------------------------------------------------------------------


def _run_dir(output_dir: str, run_id: str) -> Path:
    return Path(output_dir).expanduser().resolve() / run_id


def _shot_seed(seed: int, question_id: str, shot_index: int) -> int:
    digest = hashlib.md5(f"{seed}:{question_id}:{shot_index}".encode()).hexdigest()
    return seed + (int(digest[:8], 16) % 1_000_000)


def _n_generated_shots(record: dict | None) -> int:
    """How many shot generations are stored on this prediction record."""
    if not record:
        return 0
    shots = record.get("shots")
    if isinstance(shots, list):
        return len(shots)
    try:
        return max(0, int(record.get("n_shots") or 0))
    except (TypeError, ValueError):
        return 0


def _load_prediction_records(predictions_path: Path) -> dict[str, dict]:
    """Return ``{id: record}`` in file order, repairing a truncated last line."""
    if not predictions_path.exists():
        return {}

    records: dict[str, dict] = {}
    valid_items: list[dict] = []
    corrupt = False
    with open(predictions_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                corrupt = True
                print(f"Skipping corrupt predictions line in {predictions_path}")
                continue
            record_id = item.get("id")
            if record_id:
                records[str(record_id)] = item
            valid_items.append(item)

    if corrupt:
        tmp_path = predictions_path.with_suffix(".jsonl.tmp")
        write_jsonl(tmp_path, valid_items, mode="w")
        tmp_path.replace(predictions_path)
        print(
            f"Repaired predictions file -> {predictions_path} "
            f"({len(valid_items)} lines)"
        )

    return records


def _rewrite_predictions_file(
    predictions_path: Path,
    records: dict[str, dict],
    order: list[str],
) -> None:
    ordered: list[dict] = []
    seen: set[str] = set()
    for qid in order:
        rec = records.get(qid)
        if rec is not None:
            ordered.append(rec)
            seen.add(qid)
    for qid, rec in records.items():
        if qid not in seen:
            ordered.append(rec)
    tmp_path = predictions_path.with_suffix(".jsonl.tmp")
    write_jsonl(tmp_path, ordered, mode="w")
    tmp_path.replace(predictions_path)


def _merge_shot_record(
    item: dict,
    existing: dict | None,
    new_outputs: list[dict],
    *,
    start_index: int,
) -> dict:
    """Append newly generated shots onto an existing record, or build a new one."""
    new_record = aggregate_n_shot_record(
        item, new_outputs, pending_grade=True
    )
    if not existing or start_index <= 0:
        return new_record

    shots = list(existing.get("shots") or [])[:start_index]
    for offset, shot in enumerate(new_record.get("shots") or []):
        merged = dict(shot)
        merged["shot_index"] = start_index + offset
        shots.append(merged)
    record = {**existing, **item}
    record["shots"] = shots
    record["n_shots"] = len(shots)
    record["pending_grade"] = True
    return record


def _model_progress(
    run_dir: Path,
    model_label: str,
    question_ids: list[str],
    n_shots: int,
) -> dict:
    """Per-model completion: questions that already have ``n_shots`` generations."""
    predictions_path = run_dir / "models" / model_label / "predictions.jsonl"
    records = _load_prediction_records(predictions_path)
    n_done = sum(
        1
        for qid in question_ids
        if _n_generated_shots(records.get(qid)) >= n_shots
    )
    n_total = len(question_ids)
    return {
        "model_label": model_label,
        "n_done": n_done,
        "n_total": n_total,
        "n_shots": n_shots,
        "complete": n_total > 0 and n_done >= n_total,
        "predictions_path": str(predictions_path),
    }


def _mmar_ids_with_audio(meta_path: Path, data_root: Path) -> list[str]:
    ids: list[str] = []
    for item in load_jsonl(meta_path):
        audio_path = resolve_path(data_root, item["audio_path"])
        if not os.path.exists(audio_path):
            print(f"Skipping {item['id']}: missing audio at {audio_path}")
            continue
        ids.append(str(item["id"]))
    return ids


def _write_question_ids(ids_path: Path, ids: list[str], *, seed: int) -> None:
    payload = {
        "seed": seed,
        "n": len(ids),
        "ids": ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ids_path, payload)
    results_volume.commit()


def _ensure_question_ids(
    run_dir: Path,
    *,
    meta_path: Path,
    data_root: Path,
    seed: int,
) -> list[str]:
    ids_path = run_dir / "question_ids.json"
    full_ids = _mmar_ids_with_audio(meta_path, data_root)
    if not full_ids:
        raise SystemExit(f"No MMAR items with audio under {data_root}")

    existing: list[str] = []
    if ids_path.exists():
        try:
            payload = json.loads(ids_path.read_text(encoding="utf-8"))
            existing = [str(x) for x in payload.get("ids", [])]
        except json.JSONDecodeError:
            existing = []
    full_set = set(full_ids)
    merged = [qid for qid in existing if qid in full_set]
    merged = list(dict.fromkeys([*merged, *full_ids]))
    if merged == existing and existing and ids_path.exists():
        print(f"Reusing {len(existing)} question ids from {ids_path}")
        return existing

    _write_question_ids(ids_path, merged, seed=seed)
    if existing and len(merged) > len(existing):
        print(
            f"Expanded question set {len(existing)} -> {len(merged)} "
            f"(full MMAR) -> {ids_path}"
        )
    else:
        print(f"Wrote {len(merged)} question ids (full MMAR) -> {ids_path}")
    return merged


def _load_selected_items(
    meta_path: Path,
    data_root: Path,
    question_ids: list[str],
) -> list[dict]:
    by_id = {str(item["id"]): item for item in load_jsonl(meta_path)}
    items: list[dict] = []
    for qid in question_ids:
        item = by_id.get(qid)
        if item is None:
            print(f"Missing meta for id={qid}")
            continue
        audio_path = resolve_path(data_root, item["audio_path"])
        if not os.path.exists(audio_path):
            print(f"Skipping {qid}: missing audio at {audio_path}")
            continue
        items.append({**item, "audio_path": audio_path})
    return items


def _run_model_eval(
    *,
    model_label: str,
    run_id: str,
    output_dir: str,
    meta: str,
    data_root: str,
    n_shots: int,
    seed: int,
    print_every: int,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    greedy_non_thinking: bool = False,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
    model_id: str | None = None,
    tokenizer_id: str | None = None,
) -> dict:
    """Load one model and write n-shot freeform predictions for the full question set."""
    volume.reload()
    results_volume.reload()

    spec = MODEL_SPECS[model_label]
    args = SimpleNamespace(
        model_id=model_id or spec["model_id"],
        tokenizer_id=tokenizer_id or spec.get("tokenizer_id"),
        local_model_dir=None,
        local_tokenizer_dir=None,
        meta=meta,
        data_root=data_root,
        output_dir=output_dir,
        n_shots=n_shots,
        # Optional CLI overrides (None → use MODEL_SPECS[label].sampling).
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        greedy_non_thinking=greedy_non_thinking,
        seed=seed,
        torch_dtype="bfloat16",
        print_every=print_every,
        run_id=run_id,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        prompt_mode="freeform",
    )
    sampling = resolve_sampling(model_label, args)
    # Materialize effective sampling for HF fallbacks / logging.
    args.temperature = float(sampling["temperature"])
    args.top_p = float(sampling.get("top_p", 1.0))
    args.max_new_tokens = int(sampling["max_tokens"])
    args.repetition_penalty = float(sampling.get("repetition_penalty", 1.0))
    args.sampling = sampling

    run_dir = _run_dir(output_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "models" / model_label
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = model_dir / "predictions.jsonl"

    meta_path = Path(meta).expanduser().resolve()
    data_root_path = Path(data_root).expanduser().resolve()
    audio_dir = data_root_path / "audio"
    if not meta_path.exists():
        raise SystemExit(
            f"MMAR metadata not found: {meta_path}\n"
            "Seed first: uv run modal run seed_volume.py --datasets mmar --models none"
        )
    wav_count = count_wavs(audio_dir)
    if wav_count < 100:
        raise SystemExit(f"MMAR audio missing/incomplete in {audio_dir} ({wav_count} wavs)")

    question_ids = _ensure_question_ids(
        run_dir,
        meta_path=meta_path,
        data_root=data_root_path,
        seed=seed,
    )
    items = _load_selected_items(meta_path, data_root_path, question_ids)
    existing_records = _load_prediction_records(predictions_path)
    # Commit after a possible corrupt-line repair inside the loader.
    results_volume.commit()
    pending_items: list[dict] = []
    pending_have: list[int] = []
    n_done = 0
    for item in items:
        n_have = _n_generated_shots(existing_records.get(str(item["id"])))
        if n_have >= n_shots:
            n_done += 1
            continue
        pending_items.append(item)
        pending_have.append(n_have)
    print(
        f"[{model_label}] backend={spec.get('backend')} mode=freeform "
        f"{len(items)} selected, {n_done} done (>= {n_shots} shots), "
        f"{len(pending_items)} pending (n_shots={n_shots}, sampling={sampling})"
    )

    if not pending_items:
        return {
            "status": "already_complete",
            "model_label": model_label,
            "n_predictions": n_done,
            "predictions_path": str(predictions_path),
            "mode": "freeform",
        }

    # Persist torch.compile / Triton JIT caches across cold starts. Per-model
    # subdirs avoid concurrent writers when models run in parallel.
    compile_cache = VOLUME_MOUNT / "vllm" / model_label
    compile_cache.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(compile_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache / "torchinductor")
    print(f"[{model_label}] compile cache -> {compile_cache}")

    handle = load_model(model_label, args)
    try:
        volume.commit()
    except Exception as exc:  # noqa: BLE001 — cache commit is best-effort
        print(f"[{model_label}] volume.commit after load failed: {exc}")
    active_backend = handle.get("backend", spec.get("backend"))
    # Omni / HF cannot fork SamplingParams(n>1); expand question×shot rows.
    # Plain vLLM uses n=n_shots on one prompt per question (shared prefill).
    # Submit all pending in one generate() so vLLM continuous-batches.
    duplicate_shots = backend_duplicates_shots(str(active_backend))

    start_time = time.time()
    n_pending = len(pending_items)
    shot_outputs_by_index: list[list[dict]] = [[] for _ in pending_items]
    all_fresh = all(n_have == 0 for n_have in pending_have)

    if duplicate_shots or not all_fresh:
        gen_samples: list[dict] = []
        seeds: list[int] = []
        owners: list[tuple[int, int]] = []
        for item_index, (item, n_have) in enumerate(
            zip(pending_items, pending_have)
        ):
            for shot_index in range(n_have, n_shots):
                gen_samples.append(item)
                seeds.append(_shot_seed(seed, str(item["id"]), shot_index))
                owners.append((item_index, shot_index))
        n_completions = 1
        n_requests = len(gen_samples)
    else:
        gen_samples = list(pending_items)
        seeds = [
            _shot_seed(seed, str(item["id"]), 0) for item in pending_items
        ]
        owners = [
            (item_index, shot_index)
            for item_index in range(n_pending)
            for shot_index in range(n_shots)
        ]
        n_completions = n_shots
        n_requests = len(gen_samples)

    n_missing = sum(n_shots - n_have for n_have in pending_have)
    print(
        f"[{model_label}] generate n_questions={n_pending} "
        f"n_missing_shots={n_missing} "
        f"n_requests={n_requests} n_completions={n_completions}"
    )
    try:
        outputs = generate_batch(
            model_label,
            handle,
            gen_samples,
            args,
            seeds=seeds,
            n_completions=n_completions,
        )
    except Exception as exc:
        # Offline generate is all-or-nothing; resume retries the same pending set.
        raise RuntimeError(
            f"[{model_label}] generate failed "
            f"n_questions={n_pending} "
            f"n_requests={n_requests} n_completions={n_completions}: {exc}"
        ) from exc
    if len(outputs) != len(owners):
        raise RuntimeError(
            f"[{model_label}] expected {len(owners)} shot outputs, "
            f"got {len(outputs)}"
        )
    for (item_index, _shot_index), output in zip(owners, outputs):
        shot_outputs_by_index[item_index].append(output)

    for item, n_have, new_outputs in zip(
        pending_items, pending_have, shot_outputs_by_index
    ):
        qid = str(item["id"])
        existing_records[qid] = _merge_shot_record(
            item,
            existing_records.get(qid),
            new_outputs,
            start_index=n_have,
        )
    _rewrite_predictions_file(predictions_path, existing_records, question_ids)
    with open(predictions_path, "rb") as pred_file:
        os.fsync(pred_file.fileno())
    results_volume.commit()

    written = n_pending
    elapsed = time.time() - start_time
    try:
        # Persist any Triton JIT / inductor caches written during generate.
        volume.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[{model_label}] volume.commit after generate failed: {exc}")
    if print_every > 0:
        for idx, item in enumerate(pending_items, start=1):
            if idx % print_every == 0 or idx == written:
                record = existing_records[str(item["id"])]
                print(
                    f"[{model_label}] {idx}/{written} "
                    f"id={record['id']} pending_grade ({elapsed:.0f}s)"
                )

    total = sum(
        1
        for qid in question_ids
        if _n_generated_shots(existing_records.get(qid)) >= n_shots
    )
    print(
        f"[{model_label}] done: updated {written} questions "
        f"({n_missing} shots), total={total}/{len(question_ids)} "
        f"with {n_shots} shots ({elapsed:.0f}s)"
    )
    return {
        "status": "ok",
        "model_label": model_label,
        "n_written": written,
        "n_predictions": total,
        "predictions_path": str(predictions_path),
        "backend": active_backend,
        "mode": "freeform",
    }


# ---------------------------------------------------------------------------
# Modal eval workers (one GPU container pool per model_label)
# ---------------------------------------------------------------------------
# Parametrized Cls: each model_label is its own autoscaler pool even when
# models share an image / GPU type. single_use_containers keeps a GPU from
# being reused after that model returns.

_EVAL_KW = dict(
    timeout=12 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=65536,
    single_use_containers=True,
)


@app.cls(image=af_next_image, gpu="L40S", **_EVAL_KW)
class EvalAfNext:
    model_label: str = modal.parameter()

    @modal.method()
    def run(self, **kwargs) -> dict:
        return _run_model_eval(model_label=self.model_label, **kwargs)


@app.cls(image=omni_image, gpu="A100-80GB", **_EVAL_KW)
class EvalMimo:
    model_label: str = modal.parameter()

    @modal.method()
    def run(self, **kwargs) -> dict:
        return _run_model_eval(model_label=self.model_label, **kwargs)


@app.cls(image=step_audio_image, gpu="A100-80GB", **_EVAL_KW)
class EvalStepAudio:
    model_label: str = modal.parameter()

    @modal.method()
    def run(self, **kwargs) -> dict:
        return _run_model_eval(model_label=self.model_label, **kwargs)


@app.cls(image=interactive_omni_image, gpu="A100-80GB", **_EVAL_KW)
class EvalInteractiveOmni:
    model_label: str = modal.parameter()

    @modal.method()
    def run(self, **kwargs) -> dict:
        return _run_model_eval(model_label=self.model_label, **kwargs)


@app.cls(image=large_mm_image, gpu="A100-80GB", **_EVAL_KW)
class EvalLargeMmA100:
    model_label: str = modal.parameter()

    @modal.method()
    def run(self, **kwargs) -> dict:
        return _run_model_eval(model_label=self.model_label, **kwargs)


@app.cls(image=large_mm_image, gpu="L40S", **_EVAL_KW)
class EvalLargeMmL40S:
    model_label: str = modal.parameter()

    @modal.method()
    def run(self, **kwargs) -> dict:
        return _run_model_eval(model_label=self.model_label, **kwargs)


@app.cls(image=large_mm_image, gpu="H100", **_EVAL_KW)
class EvalLargeMmH100:
    model_label: str = modal.parameter()

    @modal.method()
    def run(self, **kwargs) -> dict:
        return _run_model_eval(model_label=self.model_label, **kwargs)


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def prepare_run(
    run_id: str,
    output_dir: str,
    model_labels: list[str],
    n_shots: int,
    seed: int,
    meta: str,
    data_root: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    greedy_non_thinking: bool = False,
) -> dict:
    """Create question_ids + manifest before parallel model workers start.

    Defaults to the full MMAR set (every clip with audio). Re-running with
    the same ``run_id`` expands an older sampled id list to full MMAR,
    merges models into the existing manifest, and reports per-model
    progress (questions that already have ``n_shots`` generations) so the
    pipeline orchestrator can skip already-complete workers.
    """
    volume.reload()
    results_volume.reload()

    run_dir = _run_dir(output_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(meta).expanduser().resolve()
    data_root_path = Path(data_root).expanduser().resolve()
    question_ids = _ensure_question_ids(
        run_dir,
        meta_path=meta_path,
        data_root=data_root_path,
        seed=seed,
    )

    now = datetime.now(timezone.utc).isoformat()
    manifest_path = run_dir / "manifest.json"
    existing: dict = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    prior_models = [str(x) for x in (existing.get("models") or [])]
    merged_models = list(dict.fromkeys([*prior_models, *model_labels]))
    is_resume = bool(existing.get("created_at")) or any(
        (run_dir / "models" / label / "predictions.jsonl").exists()
        for label in merged_models
    )

    progress = {
        label: _model_progress(run_dir, label, question_ids, n_shots)
        for label in merged_models
    }

    override_ns = SimpleNamespace(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        greedy_non_thinking=greedy_non_thinking,
    )
    model_sampling = {
        label: resolve_sampling(label, override_ns) for label in merged_models
    }

    manifest = {
        "run_id": run_id,
        "experiment": "exp-mmar-question-difficulty",
        "mode": "freeform",
        "models": merged_models,
        "n_shots": n_shots,
        "seed": existing.get("seed", seed),
        # Per-model SamplingParams (no global temperature / top_p / max_tokens).
        "model_sampling": model_sampling,
        "sampling_overrides": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "greedy_non_thinking": greedy_non_thinking,
        },
        "inference": "vllm",
        "n_questions": len(question_ids),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "resumed": is_resume,
        "progress": progress,
        "model_specs": {
            label: {
                "model_id": MODEL_SPECS[label]["model_id"],
                "backend": MODEL_SPECS[label].get("backend"),
                "gpu": MODEL_SPECS[label].get("gpu"),
                "sampling": MODEL_SPECS[label].get("sampling"),
                "native_thinking": bool(MODEL_SPECS[label].get("native_thinking")),
            }
            for label in ALL_MODEL_LABELS
        },
    }
    # Preserve judge metadata written by other pathways.
    for key in ("scoring", "grader_model_id", "judges", "primary_judge"):
        if key in existing:
            manifest[key] = existing[key]
    write_json(manifest_path, manifest)
    results_volume.commit()
    return {
        "manifest": manifest,
        "progress": progress,
        "question_ids": question_ids,
        "resumed": is_resume,
        "mode": "freeform",
    }


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def run_aggregate(run_id: str, output_dir: str = str(DEFAULT_OUTPUT_DIR)) -> dict:
    results_volume.reload()
    run_dir = _run_dir(output_dir, run_id)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")
    result = aggregate_difficulty(run_dir)
    # Stamp scoring mode from manifest when present.
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            scores = result.get("scores") or {}
            if manifest.get("scoring"):
                scores["scoring"] = manifest["scoring"]
            if manifest.get("mode"):
                scores["mode"] = manifest["mode"]
            if manifest.get("grader_model_id"):
                scores["grader_model_id"] = manifest["grader_model_id"]
            if manifest.get("judges") is not None:
                scores["judges"] = manifest["judges"]
            if manifest.get("primary_judge"):
                scores["primary_judge"] = manifest["primary_judge"]
            write_json(run_dir / "scores.json", scores)
            result["scores"] = scores
        except json.JSONDecodeError:
            pass
    results_volume.commit()
    print("Aggregated:", result.get("scores"))
    return result


_EVAL_CLS = {
    "af-next-think": EvalAfNext,
    "music-flamingo": EvalAfNext,
    "mimo-audio-7b": EvalMimo,
    "step-audio-2-mini-think": EvalStepAudio,
    "interactive-omni-8b": EvalInteractiveOmni,
    "qwen3-omni": EvalLargeMmA100,
    "voxtral-small-24b": EvalLargeMmA100,
    "qwen2.5-omni-7b": EvalLargeMmL40S,
    "phi-4-multimodal": EvalLargeMmL40S,
    "gemma-4-e4b": EvalLargeMmL40S,
    "qwen3-omni-instruct": EvalLargeMmH100,
    "nemotron-3-nano-omni": EvalLargeMmH100,
}

_missing_eval = [label for label in ALL_MODEL_LABELS if label not in _EVAL_CLS]
if _missing_eval:
    raise RuntimeError(f"No GPU eval worker for models: {_missing_eval}")


def _spawn_model_eval(label: str, **common):
    """Start one dedicated GPU container for ``label`` (does not wait)."""
    cls = _EVAL_CLS.get(label)
    if cls is None:
        raise SystemExit(f"No GPU worker for model {label!r}")
    call = cls(model_label=label).run.spawn(**common)
    print(f"Spawned {label} call_id={call.object_id}")
    return call


def _collect_model_eval(calls: list[tuple[str, object]]) -> list[dict]:
    results: list[dict] = []
    for label, call in calls:
        try:
            result = call.get()
            print(f"Finished {label}:", result)
            results.append(result)
        except Exception as exc:  # noqa: BLE001 — keep sibling workers alive
            print(f"FAILED {label}: {exc}")
            results.append(
                {"status": "error", "model_label": label, "error": str(exc)}
            )
    return results


@app.function(
    image=cpu_image,
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def run_pipeline(
    models: str = "all",
    n_shots: int = DEFAULT_N_SHOTS,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    greedy_non_thinking: bool = False,
    seed: int = DEFAULT_SEED,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
    meta: str = str(DEFAULT_MMAR_META),
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    run_id: str | None = None,
    print_every: int = 5,
    aggregate_only: bool = False,
) -> dict:
    """Remote orchestrator for prepare → generate.

    Runs on Modal so ``modal run --detach`` keeps the app alive across
    phases. Orchestrating from ``@app.local_entrypoint`` fails after the
    last model returns: with detach there are briefly no live inputs, the
    ephemeral app stops, and the next ``.spawn()`` raises ConflictError.

    Each pending model is spawned on its own GPU container; a multi-model
    run launches those containers in parallel. Grading is a separate
    pipeline (``run_judges.py``).
    """
    resolved_run_id = run_id or make_run_id()
    model_labels = parse_model_list(models)

    if aggregate_only:
        result = run_aggregate.remote(run_id=resolved_run_id, output_dir=output_dir)
        print("Done (aggregate-only):", result)
        return {
            "run_id": resolved_run_id,
            "mode": "freeform",
            "aggregate_only": True,
            "aggregate": result,
        }

    common = dict(
        run_id=resolved_run_id,
        output_dir=output_dir,
        meta=meta,
        data_root=data_root,
        n_shots=n_shots,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        greedy_non_thinking=greedy_non_thinking,
        seed=seed,
        print_every=print_every,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    print(
        f"Experiment run_id={resolved_run_id} mode=freeform "
        f"models={model_labels} n_shots={n_shots} "
        f"gpu_containers=per-model parallel_launch=True inference=vllm "
        f"sampling_overrides={{temperature={temperature}, top_p={top_p}, "
        f"max_new_tokens={max_new_tokens}, greedy_non_thinking={greedy_non_thinking}}}"
    )

    prep = prepare_run.remote(
        run_id=resolved_run_id,
        output_dir=output_dir,
        model_labels=model_labels,
        n_shots=n_shots,
        seed=seed,
        meta=meta,
        data_root=data_root,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        greedy_non_thinking=greedy_non_thinking,
    )

    progress = prep.get("progress") or {}
    pending_labels = [
        label
        for label in model_labels
        if not (progress.get(label) or {}).get("complete")
    ]
    skipped_labels = [label for label in model_labels if label not in pending_labels]

    if prep.get("resumed") or skipped_labels:
        print(f"Resume check for run_id={resolved_run_id}:")
        for label in model_labels:
            info = progress.get(label) or {}
            print(
                f"  {label}: {info.get('n_done', 0)}/{info.get('n_total', '?')} "
                f"questions with {n_shots} shots "
                f"{'complete — skip spawn' if label in skipped_labels else 'pending'}"
            )

    results: list[dict] = [
        {
            "status": "already_complete",
            "model_label": label,
            "n_predictions": (progress.get(label) or {}).get("n_done"),
            "predictions_path": (progress.get(label) or {}).get("predictions_path"),
        }
        for label in skipped_labels
    ]

    if pending_labels:
        print(
            f"Launching {len(pending_labels)} dedicated GPU container(s)"
            f"{' in parallel' if len(pending_labels) > 1 else ''}: "
            f"{pending_labels}"
        )
        calls = [
            (label, _spawn_model_eval(label, **common))
            for label in pending_labels
        ]
        results.extend(_collect_model_eval(calls))
    else:
        print("All requested models already complete; skipping inference.")

    return {
        "run_id": resolved_run_id,
        "mode": "freeform",
        "models": results,
        "pending_labels": pending_labels,
        "skipped_labels": skipped_labels,
    }


@app.local_entrypoint()
def main(
    models: str = "all",
    n_shots: int = DEFAULT_N_SHOTS,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    greedy_non_thinking: bool = False,
    seed: int = DEFAULT_SEED,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
    meta: str = str(DEFAULT_MMAR_META),
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    run_id: str | None = None,
    print_every: int = 5,
    aggregate_only: bool = False,
):
    """Launch MMAR freeform generation.

    Args:
        models: Comma-separated labels or ``all``.
        n_shots: Independent temperature samples per question (default 10).
            Questions that already have this many generations are skipped;
            missing shots are filled in. Plain vLLM uses SamplingParams(n=...)
            shared prefill when every pending question is starting from
            zero; Omni/HF (and partial fills) duplicate prompts per shot.
            All pending questions go in one generate() so vLLM
            continuous-batches.
        temperature: Optional override of each model's sampling temperature.
        top_p: Optional override of each model's top_p.
        max_new_tokens: Optional override of each model's max_tokens.
        greedy_non_thinking: Force temperature=0 on models without native
            ``<think>`` / reasoning mode. Thinking models keep card sampling
            unless ``temperature`` is also set.
        seed: RNG seed for per-question sample seeds.
        max_num_seqs: Optional vLLM override (escape hatch; prefer defaults).
        gpu_memory_utilization: Optional vLLM GPU memory fraction override.
        meta: Path to MMAR-meta.jsonl on the data volume.
        data_root: MMAR root used to resolve audio paths.
        output_dir: Results volume directory for run folders.
        run_id: Optional run folder name; default is a UTC timestamp.
            Pass an existing id to resume: questions that already have
            ``n_shots`` generations are skipped; others (including the rest
            of MMAR after an older sampled run) are generated.
        print_every: Progress print interval per model.
        aggregate_only: Skip inference; only build difficulty.jsonl /
            scores.json from existing predictions (after grading via
            ``run_judges.py``).
    """
    # One remote spawn owns prepare→infer so ``--detach`` does not stop
    # the ephemeral app between phases.
    resolved_run_id = run_id or make_run_id()
    out = run_pipeline.spawn(
        models=models,
        n_shots=n_shots,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        greedy_non_thinking=greedy_non_thinking,
        seed=seed,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        meta=meta,
        data_root=data_root,
        output_dir=output_dir,
        run_id=resolved_run_id,
        print_every=print_every,
        aggregate_only=aggregate_only,
    ).get()

    print("Done:", out)
    rid = out.get("run_id") or resolved_run_id
    print(
        "Download with:\n"
        f"  uv run modal run download_results.py "
        f"--remote-path exp-mmar-question-difficulty/{rid}"
    )
    if out.get("pending_labels") or out.get("skipped_labels"):
        print(
            "To resume this run later:\n"
            f"  uv run modal run --detach run_experiment.py "
            f"--run-id {rid}"
        )
