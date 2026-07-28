"""MMAR question-difficulty experiment on Modal (vLLM / vLLM-Omni).

Samples a fixed 200 MMAR questions, runs 10 temperature=1.0 responses per
model (no rubric grading), then aggregates mean string-match success rates.

Inference backends:
  - af-next-think: vLLM 0.24 (native MusicFlamingo)
  - mimo-audio-7b: vLLM-Omni 0.24
  - interactive-omni-8b: vLLM transformers backend (HF .chat fallback)

Results layout on ``latent-reasoning-results``:

    exp-mmar-question-difficulty/<run_id>/
      question_ids.json
      manifest.json
      models/<label>/predictions.jsonl
      difficulty.jsonl
      scores.json

Prereqs:

    uv run modal run seed_volume.py --datasets mmar \\
      --models af-next-think,mimo-audio-7b,interactive-omni-8b

Usage:

    uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py
    uv run modal run exp-mmar-question-difficulty/run_experiment.py \\
      --models af-next-think --num-samples 8 --batch-size 8
    # Resume after a crash (reuse question set; skip finished models /
    # already-written questions):
    uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py \\
      --run-id <run_id>
    uv run modal run exp-mmar-question-difficulty/run_experiment.py \\
      --aggregate-only --run-id <run_id>
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import modal

# Local sibling imports (experiment folder on PYTHONPATH in the image).
EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
_EXP_MOUNT = "/root/exp-mmar-question-difficulty"


def _ensure_exp_path() -> None:
    """Make experiment sibling modules importable inside Modal containers."""
    for path in (_EXP_MOUNT, str(EXP_DIR)):
        if path and path not in sys.path:
            if path == _EXP_MOUNT or Path(path).is_dir():
                sys.path.insert(0, path)


_ensure_exp_path()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aggregate import MODEL_LABELS as DEFAULT_MODEL_LABELS  # noqa: E402
from aggregate import aggregate_difficulty  # noqa: E402
from models import (  # noqa: E402
    ALL_MODEL_LABELS,
    MODEL_SPECS,
    generate_batch,
    load_model,
    parse_model_list,
)
from mmar_common import (  # noqa: E402
    aggregate_n_shot_record,
    count_wavs,
    load_jsonl,
    make_run_id,
    resolve_path,
    write_json,
    write_jsonl,
)
from modal_cache import (  # noqa: E402
    DEFAULT_MMAR_DATA_ROOT,
    DEFAULT_MMAR_META,
    RESULTS_MOUNT,
    VOLUME_MOUNT,
    hf_secret,
    results_volume,
    volume,
)

DEFAULT_OUTPUT_DIR = RESULTS_MOUNT / "exp-mmar-question-difficulty"
DEFAULT_NUM_SAMPLES = 200
DEFAULT_N_SHOTS = 10
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
# Modal rule: after any ``add_local_*``, no further build steps (apt/pip/run/env).
# Put installs + env first; mount local sources last.

_SHARED_SOURCES = (
    "modal_cache",
    "mmar_common",
    "audio_flamingo_runtime",
    "latent_cot",
)

_VLLM_CACHE_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    # Keep EngineCore in-process so AF-Next weight-loader patches apply.
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "PYTHONPATH": _EXP_MOUNT,
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


def _vllm_image(
    *,
    vllm_version: str,
    extra_packages: list[str] | None = None,
) -> modal.Image:
    packages = [
        f"vllm=={vllm_version}",
        "transformers>=5.5.3",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "numpy",
        "tqdm>=4.67.0",
        "accelerate>=1.14.0",
        *(extra_packages or []),
    ]
    return (
        _cuda_base_image()
        .uv_pip_install(*packages)
        .env(_VLLM_CACHE_ENV)
        .add_local_python_source(*_SHARED_SOURCES)
        .add_local_dir(str(EXP_DIR), remote_path=_EXP_MOUNT)
    )


# AF-Next: vLLM 0.24 MusicFlamingo (+ HF fallback deps if weight load fails).
af_next_image = _vllm_image(
    vllm_version="0.24.0",
    extra_packages=[
        "soxr",
        "torch",
        "torchaudio",
        "peft>=0.15.2",
        "safetensors>=0.8.0",
    ],
)

# Step / MiMo: vLLM-Omni on the same 0.24 line.
omni_image = (
    _cuda_base_image()
    .uv_pip_install(
        "vllm==0.24.0",
        "vllm-omni==0.24.0",
        "step-audio2",
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
    .add_local_python_source(*_SHARED_SOURCES)
    .add_local_dir(str(EXP_DIR), remote_path=_EXP_MOUNT)
)

# InteractiveOmni: newer vLLM transformers-audio backend + HF chat fallback.
interactive_omni_image = (
    _cuda_base_image()
    .uv_pip_install(
        "vllm==0.26.0",
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
    )
    .env(_VLLM_CACHE_ENV)
    .add_local_python_source(*_SHARED_SOURCES)
    .add_local_dir(str(EXP_DIR), remote_path=_EXP_MOUNT)
)

# Lightweight CPU image for manifest / question-id helpers.
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "tqdm>=4.67.0")
    .env({"PYTHONPATH": _EXP_MOUNT})
    .add_local_python_source(*_SHARED_SOURCES)
    .add_local_dir(str(EXP_DIR), remote_path=_EXP_MOUNT)
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


def _load_completed_prediction_ids(predictions_path: Path) -> set[str]:
    """Return completed question ids, tolerating a truncated trailing JSONL line.

    A mid-batch crash can leave a partial last line. Skip corrupt lines and
    rewrite the file so the next resume does not trip on them again.
    """
    if not predictions_path.exists():
        return set()

    completed: set[str] = set()
    valid_lines: list[str] = []
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
                completed.add(str(record_id))
            valid_lines.append(stripped)

    if corrupt:
        tmp_path = predictions_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for line in valid_lines:
                handle.write(line + "\n")
        tmp_path.replace(predictions_path)
        print(f"Repaired predictions file -> {predictions_path} ({len(valid_lines)} lines)")

    return completed


def _model_progress(
    run_dir: Path,
    model_label: str,
    question_ids: list[str],
) -> dict:
    """Per-model completion against the fixed question set."""
    predictions_path = run_dir / "models" / model_label / "predictions.jsonl"
    completed = _load_completed_prediction_ids(predictions_path)
    selected = set(question_ids)
    n_done = len(completed & selected)
    n_total = len(question_ids)
    return {
        "model_label": model_label,
        "n_done": n_done,
        "n_total": n_total,
        "complete": n_total > 0 and n_done >= n_total,
        "predictions_path": str(predictions_path),
    }


def _ensure_question_ids(
    run_dir: Path,
    *,
    meta_path: Path,
    data_root: Path,
    num_samples: int,
    seed: int,
    start: int = 0,
) -> list[str]:
    ids_path = run_dir / "question_ids.json"
    if ids_path.exists():
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        ids = [str(x) for x in payload.get("ids", [])]
        if ids:
            print(f"Reusing {len(ids)} question ids from {ids_path}")
            return ids

    meta_items = load_jsonl(meta_path)
    pool = meta_items[start:]
    rng = random.Random(seed)
    if num_samples < 0 or num_samples >= len(pool):
        selected = pool
    else:
        indices = rng.sample(range(len(pool)), num_samples)
        selected = [pool[i] for i in indices]

    ids: list[str] = []
    for item in selected:
        audio_path = resolve_path(data_root, item["audio_path"])
        if not os.path.exists(audio_path):
            print(f"Skipping {item['id']}: missing audio at {audio_path}")
            continue
        ids.append(str(item["id"]))

    if len(ids) < min(num_samples if num_samples > 0 else len(pool), len(selected)):
        print(
            f"Warning: only {len(ids)} items with audio after sampling "
            f"(requested {num_samples})."
        )

    payload = {
        "seed": seed,
        "start": start,
        "num_samples": num_samples,
        "n": len(ids),
        "ids": ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ids_path, payload)
    results_volume.commit()
    print(f"Wrote {len(ids)} question ids -> {ids_path}")
    return ids


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
    num_samples: int,
    n_shots: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    print_every: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
    model_id: str | None = None,
    tokenizer_id: str | None = None,
) -> dict:
    """Load one model and write n-shot predictions for the fixed question set."""
    _ensure_exp_path()

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
        num_samples=num_samples,
        n_shots=n_shots,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        torch_dtype="bfloat16",
        repetition_penalty=(
            1.2
            if model_label == "af-next-think"
            else 1.05
            if model_label.startswith("step")
            else 1.1
        ),
        print_every=print_every,
        run_id=run_id,
        batch_size=batch_size,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
    )

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
        num_samples=num_samples,
        seed=seed,
    )
    items = _load_selected_items(meta_path, data_root_path, question_ids)
    selected_ids = {str(item["id"]) for item in items}
    completed = _load_completed_prediction_ids(predictions_path)
    # Commit after a possible corrupt-line repair inside the loader.
    results_volume.commit()
    n_done = len(completed & selected_ids)
    pending = [item for item in items if str(item["id"]) not in completed]
    print(
        f"[{model_label}] backend={spec.get('backend')} "
        f"{len(items)} selected, {n_done} done, {len(pending)} pending "
        f"(n_shots={n_shots}, temperature={temperature}, batch_size={batch_size})"
    )

    if not pending:
        return {
            "status": "already_complete",
            "model_label": model_label,
            "n_predictions": n_done,
            "predictions_path": str(predictions_path),
        }

    handle = load_model(model_label, args)
    active_backend = handle.get("backend", spec.get("backend"))

    start_time = time.time()
    written = 0
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        shot_outputs_by_index: list[list[dict]] = [[] for _ in batch]

        # Flatten question×shot into one generate queue so vLLM can continuous-
        # batch and reuse prefixes across shots. Omni regroups by shot inside
        # generate_batch (stage SamplingParams are shared per call).
        expanded: list[dict] = []
        seeds: list[int] = []
        owners: list[tuple[int, int]] = []
        for item_index, item in enumerate(batch):
            for shot_index in range(n_shots):
                expanded.append(item)
                seeds.append(_shot_seed(seed, str(item["id"]), shot_index))
                owners.append((item_index, shot_index))
        try:
            outputs = generate_batch(
                model_label, handle, expanded, args, seeds=seeds
            )
        except Exception as exc:
            # Do not swallow OOMs / engine deaths into empty predictions — abort
            # so resume can retry from the last committed batch.
            raise RuntimeError(
                f"[{model_label}] batch failed "
                f"ids={[item['id'] for item in batch]} "
                f"n_requests={len(expanded)}: {exc}"
            ) from exc
        for (item_index, _shot_index), output in zip(owners, outputs):
            shot_outputs_by_index[item_index].append(output)

        records = [
            aggregate_n_shot_record(item, shot_outputs)
            for item, shot_outputs in zip(batch, shot_outputs_by_index)
        ]
        write_jsonl(predictions_path, records, mode="a")
        # Durability before volume commit so a crash mid-batch does not lose
        # already-finished questions (resume skips committed ids).
        with open(predictions_path, "rb") as pred_file:
            os.fsync(pred_file.fileno())
        prev_written = written
        written += len(records)
        if print_every > 0:
            elapsed = time.time() - start_time
            for offset, record in enumerate(records, start=1):
                idx = prev_written + offset
                if idx % print_every == 0 or idx == len(pending):
                    print(
                        f"[{model_label}] {idx}/{len(pending)} "
                        f"id={record['id']} "
                        f"shots={record['n_shot_correct']}/{record['n_shots']} "
                        f"({elapsed:.0f}s)"
                    )
        results_volume.commit()

    total = len(_load_completed_prediction_ids(predictions_path) & selected_ids)
    print(f"[{model_label}] done: wrote {written} new, total={total}")
    return {
        "status": "ok",
        "model_label": model_label,
        "n_written": written,
        "n_predictions": total,
        "predictions_path": str(predictions_path),
        "backend": active_backend,
    }


# ---------------------------------------------------------------------------
# Modal functions (one image/GPU profile per model family)
# ---------------------------------------------------------------------------


@app.function(
    image=af_next_image,
    gpu="L40S",
    timeout=12 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=65536,
)
def run_af_next(**kwargs) -> dict:
    return _run_model_eval(model_label="af-next-think", **kwargs)


@app.function(
    image=omni_image,
    gpu="A100-80GB",
    timeout=12 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=65536,
)
def run_mimo_audio(**kwargs) -> dict:
    return _run_model_eval(model_label="mimo-audio-7b", **kwargs)


@app.function(
    image=interactive_omni_image,
    gpu="A100-80GB",
    timeout=12 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=65536,
)
def run_interactive_omni(**kwargs) -> dict:
    return _run_model_eval(model_label="interactive-omni-8b", **kwargs)


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def prepare_run(
    run_id: str,
    output_dir: str,
    model_labels: list[str],
    num_samples: int,
    n_shots: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    meta: str,
    data_root: str,
    batch_size: int,
) -> dict:
    """Create question_ids + manifest before parallel model workers start.

    Re-running with the same ``run_id`` reuses ``question_ids.json``, merges
    models into the existing manifest, and reports per-model progress so the
    local entrypoint can skip already-complete workers.
    """
    _ensure_exp_path()
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
        num_samples=num_samples,
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
        label: _model_progress(run_dir, label, question_ids)
        for label in merged_models
    }

    manifest = {
        "run_id": run_id,
        "experiment": "exp-mmar-question-difficulty",
        "models": merged_models,
        "num_samples": existing.get("num_samples", num_samples),
        "n_shots": existing.get("n_shots", n_shots),
        "temperature": existing.get("temperature", temperature),
        "top_p": existing.get("top_p", top_p),
        "max_new_tokens": existing.get("max_new_tokens", max_new_tokens),
        "seed": existing.get("seed", seed),
        "batch_size": batch_size,
        "scoring": "string_match_no_rubric",
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
            }
            for label in ALL_MODEL_LABELS
        },
    }
    write_json(manifest_path, manifest)
    results_volume.commit()
    return {
        "manifest": manifest,
        "progress": progress,
        "question_ids": question_ids,
        "resumed": is_resume,
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
    result = aggregate_difficulty(run_dir, model_labels=DEFAULT_MODEL_LABELS)
    results_volume.commit()
    print("Aggregated:", result.get("scores"))
    return result


_MODEL_FNS = {
    "af-next-think": run_af_next,
    "mimo-audio-7b": run_mimo_audio,
    "interactive-omni-8b": run_interactive_omni,
}


@app.local_entrypoint()
def main(
    models: str = "all",
    num_samples: int = DEFAULT_NUM_SAMPLES,
    n_shots: int = DEFAULT_N_SHOTS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = 1.0,
    max_new_tokens: int = 512,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
    meta: str = str(DEFAULT_MMAR_META),
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    run_id: str | None = None,
    print_every: int = 5,
    aggregate_only: bool = False,
    skip_aggregate: bool = False,
    parallel_models: bool = True,
):
    """Launch the MMAR question-difficulty experiment.

    Args:
        models: Comma-separated labels or ``all``.
        num_samples: Fixed question sample size (default 200).
        n_shots: Independent generations per question (default 10).
        temperature: Sampling temperature (default 1.0).
        top_p: Nucleus sampling parameter.
        max_new_tokens: Generation length cap.
        seed: RNG seed for question sampling and per-shot reseeding.
        batch_size: Questions per checkpoint wave; all n_shots for those
            questions go in one generate() queue (Omni regroups by shot).
        max_num_seqs: Optional override for vLLM continuous-batch width.
        gpu_memory_utilization: Optional override for vLLM GPU memory fraction.
        meta: Path to MMAR-meta.jsonl on the data volume.
        data_root: MMAR root used to resolve audio paths.
        output_dir: Results volume directory for run folders.
        run_id: Optional run folder name; default is a UTC timestamp.
            Pass an existing id to resume: finished models are skipped and
            incomplete models continue from written ``predictions.jsonl``.
        print_every: Progress print interval per model.
        aggregate_only: Skip inference; only build difficulty.jsonl.
        skip_aggregate: Run models but skip final aggregation.
        parallel_models: Spawn all model workers concurrently (default True).
    """
    resolved_run_id = run_id or make_run_id()
    model_labels = parse_model_list(models)

    if aggregate_only:
        result = run_aggregate.spawn(
            run_id=resolved_run_id, output_dir=output_dir
        ).get()
        print("Done (aggregate-only):", result)
        return

    common = dict(
        run_id=resolved_run_id,
        output_dir=output_dir,
        meta=meta,
        data_root=data_root,
        num_samples=num_samples,
        n_shots=n_shots,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        print_every=print_every,
        batch_size=batch_size,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    print(
        f"Experiment run_id={resolved_run_id} models={model_labels} "
        f"num_samples={num_samples} n_shots={n_shots} temperature={temperature} "
        f"batch_size={batch_size} parallel_models={parallel_models} inference=vllm"
    )

    prep = prepare_run.spawn(
        run_id=resolved_run_id,
        output_dir=output_dir,
        model_labels=model_labels,
        num_samples=num_samples,
        n_shots=n_shots,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        meta=meta,
        data_root=data_root,
        batch_size=batch_size,
    ).get()

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
        if parallel_models and len(pending_labels) > 1:
            calls = []
            for label in pending_labels:
                call = _MODEL_FNS[label].spawn(**common)
                print(f"Spawned {label} call_id={call.object_id}")
                calls.append((label, call))
            for label, call in calls:
                result = call.get()
                print(f"Finished {label}:", result)
                results.append(result)
        else:
            for label in pending_labels:
                call = _MODEL_FNS[label].spawn(**common)
                print(f"Spawned {label} call_id={call.object_id}")
                result = call.get()
                print(f"Finished {label}:", result)
                results.append(result)
    else:
        print("All requested models already complete; skipping inference.")

    if not skip_aggregate:
        agg = run_aggregate.spawn(
            run_id=resolved_run_id, output_dir=output_dir
        ).get()
        print("Aggregated:", agg)
    else:
        agg = None

    print("Done:", {"run_id": resolved_run_id, "models": results, "aggregate": agg})
    print(
        "Download with:\n"
        f"  uv run modal run download_results.py "
        f"--remote-path exp-mmar-question-difficulty/{resolved_run_id}"
    )
    if pending_labels or skipped_labels:
        print(
            "To resume this run later:\n"
            f"  uv run modal run --detach "
            f"exp-mmar-question-difficulty/run_experiment.py "
            f"--run-id {resolved_run_id}"
        )
