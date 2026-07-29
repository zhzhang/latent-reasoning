"""MMAR question-difficulty experiment on Modal (vLLM / vLLM-Omni).

Samples a fixed 200 MMAR questions, runs 10 temperature samples per
model (per-model SamplingParams), then aggregates mean success rates.

Modes:
  - ``mc`` (default): multiple-choice prompts + string-match scoring
  - ``freeform``: question-only prompts; Qwen2.5-3B-Instruct grades each shot
    against the gold answer

Inference backends:
  - af-next-think: vLLM 0.24 (native MusicFlamingo)
  - mimo-audio-7b: vLLM-Omni 0.24
  - interactive-omni-8b: vLLM transformers backend (HF .chat fallback)
  - qwen3-omni: vLLM 0.26 thinker-only (Qwen3-Omni-30B-A3B-Thinking)
  - voxtral-small-24b: vLLM 0.26 Mistral-format audio

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
    # Freeform grading also needs:
    uv run modal run seed_volume.py --datasets none --models qwen2.5-3b

Usage:

    uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py
    # Free-form + Qwen judge on the same 200 ids as a prior MC run:
    uv run modal run --detach exp-mmar-question-difficulty/run_experiment.py \\
      --mode freeform --source-run-id 20260727T154400Z
    uv run modal run exp-mmar-question-difficulty/run_experiment.py \\
      --models af-next-think --num-samples 8 --n-shots 2
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
    backend_duplicates_shots,
    generate_batch,
    load_model,
    parse_model_list,
    resolve_sampling,
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
DEFAULT_SEED = 42
DEFAULT_MODE = "mc"
DEFAULT_SOURCE_RUN_ID = "20260727T154400Z"
DEFAULT_GRADER_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

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

# Qwen3-Omni thinker + Voxtral Small (A100-80GB); needs mistral-common[audio] + PyAV.
large_mm_image = (
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

# Text-only freeform grader (Qwen2.5-3B-Instruct via vLLM).
grader_image = _vllm_image(vllm_version="0.24.0")

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


def _normalize_mode(mode: str | None) -> str:
    value = str(mode or DEFAULT_MODE).strip().lower()
    if value in {"freeform", "free_form", "free-form", "open"}:
        return "freeform"
    if value in {"mc", "multiple_choice", "multiple-choice", "choice"}:
        return "mc"
    raise ValueError(f"Unknown mode {mode!r}; expected 'mc' or 'freeform'")


def _scoring_label(mode: str) -> str:
    return (
        "qwen_freeform_judge"
        if mode == "freeform"
        else "string_match_no_rubric"
    )


def _normalize_source_run_id(
    source_run_id: str | None,
    *,
    mode: str,
    num_samples: int | None = None,
) -> str | None:
    """Resolve source-run-id; freeform defaults to the fixed MC difficulty set."""
    if source_run_id is not None:
        value = str(source_run_id).strip()
        if not value or value.lower() in {"none", "null", "-"}:
            return None
        return value
    # Only auto-pin the prior 200-id set for full freeform runs.
    if mode == "freeform" and (
        num_samples is None or int(num_samples) == DEFAULT_NUM_SAMPLES
    ):
        return DEFAULT_SOURCE_RUN_ID
    return None


def _ensure_question_ids(
    run_dir: Path,
    *,
    meta_path: Path,
    data_root: Path,
    num_samples: int,
    seed: int,
    start: int = 0,
    source_run_id: str | None = None,
    output_dir: str | None = None,
) -> list[str]:
    ids_path = run_dir / "question_ids.json"
    if ids_path.exists():
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        ids = [str(x) for x in payload.get("ids", [])]
        if ids:
            print(f"Reusing {len(ids)} question ids from {ids_path}")
            return ids

    # Copy the fixed question set from a prior run when requested.
    if source_run_id:
        source_root = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else run_dir.parent
        )
        source_ids_path = source_root / source_run_id / "question_ids.json"
        if not source_ids_path.exists():
            raise SystemExit(
                f"source-run-id={source_run_id} question_ids not found at "
                f"{source_ids_path}"
            )
        payload = json.loads(source_ids_path.read_text(encoding="utf-8"))
        ids = [str(x) for x in payload.get("ids", [])]
        if not ids:
            raise SystemExit(f"No ids in source question set: {source_ids_path}")
        copied = {
            "seed": payload.get("seed", seed),
            "start": payload.get("start", start),
            "num_samples": payload.get("num_samples", num_samples),
            "n": len(ids),
            "ids": ids,
            "source_run_id": source_run_id,
            "source_path": str(source_ids_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(ids_path, copied)
        results_volume.commit()
        print(
            f"Copied {len(ids)} question ids from {source_run_id} -> {ids_path}"
        )
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
    seed: int,
    print_every: int,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
    model_id: str | None = None,
    tokenizer_id: str | None = None,
    mode: str = DEFAULT_MODE,
    source_run_id: str | None = None,
) -> dict:
    """Load one model and write n-shot predictions for the fixed question set."""
    _ensure_exp_path()

    volume.reload()
    results_volume.reload()

    prompt_mode = _normalize_mode(mode)
    pending_grade = prompt_mode == "freeform"

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
        # Optional CLI overrides (None → use MODEL_SPECS[label].sampling).
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        torch_dtype="bfloat16",
        print_every=print_every,
        run_id=run_id,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        prompt_mode=prompt_mode,
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
        num_samples=num_samples,
        seed=seed,
        source_run_id=source_run_id,
        output_dir=output_dir,
    )
    items = _load_selected_items(meta_path, data_root_path, question_ids)
    selected_ids = {str(item["id"]) for item in items}
    completed = _load_completed_prediction_ids(predictions_path)
    # Commit after a possible corrupt-line repair inside the loader.
    results_volume.commit()
    n_done = len(completed & selected_ids)
    pending = [item for item in items if str(item["id"]) not in completed]
    print(
        f"[{model_label}] backend={spec.get('backend')} mode={prompt_mode} "
        f"{len(items)} selected, {n_done} done, {len(pending)} pending "
        f"(n_shots={n_shots}, sampling={sampling})"
    )

    if not pending:
        return {
            "status": "already_complete",
            "model_label": model_label,
            "n_predictions": n_done,
            "predictions_path": str(predictions_path),
            "mode": prompt_mode,
        }

    handle = load_model(model_label, args)
    active_backend = handle.get("backend", spec.get("backend"))
    # Omni / HF cannot fork SamplingParams(n>1); expand question×shot rows.
    # Plain vLLM uses n=n_shots on one prompt per question (shared prefill).
    # Submit all pending in one generate() so vLLM continuous-batches.
    duplicate_shots = backend_duplicates_shots(str(active_backend))

    start_time = time.time()
    shot_outputs_by_index: list[list[dict]] = [[] for _ in pending]

    if duplicate_shots:
        gen_samples: list[dict] = []
        seeds: list[int] = []
        owners: list[tuple[int, int]] = []
        for item_index, item in enumerate(pending):
            for shot_index in range(n_shots):
                gen_samples.append(item)
                seeds.append(_shot_seed(seed, str(item["id"]), shot_index))
                owners.append((item_index, shot_index))
        n_completions = 1
        n_requests = len(gen_samples)
    else:
        gen_samples = list(pending)
        seeds = [_shot_seed(seed, str(item["id"]), 0) for item in pending]
        owners = [
            (item_index, shot_index)
            for item_index in range(len(pending))
            for shot_index in range(n_shots)
        ]
        n_completions = n_shots
        n_requests = len(gen_samples)

    print(
        f"[{model_label}] generate n_questions={len(pending)} "
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
            f"n_questions={len(pending)} "
            f"n_requests={n_requests} n_completions={n_completions}: {exc}"
        ) from exc
    if len(outputs) != len(owners):
        raise RuntimeError(
            f"[{model_label}] expected {len(owners)} shot outputs, "
            f"got {len(outputs)}"
        )
    for (item_index, _shot_index), output in zip(owners, outputs):
        shot_outputs_by_index[item_index].append(output)

    records = [
        aggregate_n_shot_record(
            item,
            shot_outputs,
            pending_grade=pending_grade,
        )
        for item, shot_outputs in zip(pending, shot_outputs_by_index)
    ]
    write_jsonl(predictions_path, records, mode="a")
    with open(predictions_path, "rb") as pred_file:
        os.fsync(pred_file.fileno())
    results_volume.commit()

    written = len(records)
    elapsed = time.time() - start_time
    if print_every > 0:
        for idx, record in enumerate(records, start=1):
            if idx % print_every == 0 or idx == written:
                if pending_grade:
                    score_msg = "pending_grade"
                else:
                    score_msg = (
                        f"shots={record['n_shot_correct']}/{record['n_shots']}"
                    )
                print(
                    f"[{model_label}] {idx}/{written} "
                    f"id={record['id']} {score_msg} ({elapsed:.0f}s)"
                )

    total = len(_load_completed_prediction_ids(predictions_path) & selected_ids)
    print(
        f"[{model_label}] done: wrote {written} new, total={total} "
        f"({elapsed:.0f}s)"
    )
    return {
        "status": "ok",
        "model_label": model_label,
        "n_written": written,
        "n_predictions": total,
        "predictions_path": str(predictions_path),
        "backend": active_backend,
        "mode": prompt_mode,
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
    image=large_mm_image,
    gpu="A100-80GB",
    timeout=12 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=65536,
)
def run_qwen3_omni(**kwargs) -> dict:
    return _run_model_eval(model_label="qwen3-omni", **kwargs)


@app.function(
    image=large_mm_image,
    gpu="A100-80GB",
    timeout=12 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=65536,
)
def run_voxtral(**kwargs) -> dict:
    return _run_model_eval(model_label="voxtral-small-24b", **kwargs)


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
    seed: int,
    meta: str,
    data_root: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    mode: str = DEFAULT_MODE,
    source_run_id: str | None = None,
    grader_model_id: str = DEFAULT_GRADER_MODEL_ID,
) -> dict:
    """Create question_ids + manifest before parallel model workers start.

    Re-running with the same ``run_id`` reuses ``question_ids.json``, merges
    models into the existing manifest, and reports per-model progress so the
    pipeline orchestrator can skip already-complete workers.
    """
    _ensure_exp_path()
    volume.reload()
    results_volume.reload()

    prompt_mode = _normalize_mode(mode)
    # Freeform defaults to the fixed MC difficulty question set unless overridden.
    effective_source = _normalize_source_run_id(
        source_run_id, mode=prompt_mode, num_samples=num_samples
    )

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
        source_run_id=effective_source,
        output_dir=output_dir,
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

    override_ns = SimpleNamespace(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    model_sampling = {
        label: resolve_sampling(label, override_ns) for label in merged_models
    }

    manifest = {
        "run_id": run_id,
        "experiment": "exp-mmar-question-difficulty",
        "mode": existing.get("mode") or prompt_mode,
        "models": merged_models,
        "num_samples": existing.get("num_samples", num_samples),
        "n_shots": existing.get("n_shots", n_shots),
        "seed": existing.get("seed", seed),
        # Per-model SamplingParams (no global temperature / top_p / max_tokens).
        "model_sampling": model_sampling,
        "sampling_overrides": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        "scoring": existing.get("scoring") or _scoring_label(prompt_mode),
        "grader_model_id": (
            existing.get("grader_model_id")
            or (grader_model_id if prompt_mode == "freeform" else None)
        ),
        "source_run_id": existing.get("source_run_id") or effective_source,
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
        "mode": prompt_mode,
        "source_run_id": effective_source,
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
            if manifest.get("source_run_id"):
                scores["source_run_id"] = manifest["source_run_id"]
            write_json(run_dir / "scores.json", scores)
            result["scores"] = scores
        except json.JSONDecodeError:
            pass
    results_volume.commit()
    print("Aggregated:", result.get("scores"))
    return result


@app.function(
    image=grader_image,
    gpu="L40S",
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=32768,
)
def run_freeform_grade(
    run_id: str,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    model_labels: list[str] | None = None,
    grader_model_id: str = DEFAULT_GRADER_MODEL_ID,
    batch_size: int = 64,
    force: bool = False,
) -> dict:
    """Grade free-form predictions with Qwen2.5-3B-Instruct."""
    _ensure_exp_path()
    from grader import grade_predictions_file, load_grader

    volume.reload()
    results_volume.reload()

    run_dir = _run_dir(output_dir, run_id)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    labels = list(model_labels or DEFAULT_MODEL_LABELS)
    handle = load_grader(grader_model_id)
    per_model: dict[str, dict] = {}
    for label in labels:
        predictions_path = run_dir / "models" / label / "predictions.jsonl"
        print(f"[grader] grading {label} -> {predictions_path}")
        per_model[label] = grade_predictions_file(
            predictions_path,
            handle,
            batch_size=batch_size,
            force=force,
        )
        results_volume.commit()
        print(f"[grader] {label}:", per_model[label])

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        manifest["scoring"] = "qwen_freeform_judge"
        manifest["grader_model_id"] = grader_model_id
        manifest["graded_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)

    results_volume.commit()
    return {
        "status": "ok",
        "run_id": run_id,
        "grader_model_id": grader_model_id,
        "by_model": per_model,
    }


_MODEL_FNS = {
    "af-next-think": run_af_next,
    "mimo-audio-7b": run_mimo_audio,
    "interactive-omni-8b": run_interactive_omni,
    "qwen3-omni": run_qwen3_omni,
    "voxtral-small-24b": run_voxtral,
}


@app.function(
    image=cpu_image,
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def run_pipeline(
    models: str = "all",
    num_samples: int = DEFAULT_NUM_SAMPLES,
    n_shots: int = DEFAULT_N_SHOTS,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    seed: int = DEFAULT_SEED,
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
    mode: str = DEFAULT_MODE,
    source_run_id: str | None = None,
    grader_model_id: str = DEFAULT_GRADER_MODEL_ID,
    grade_only: bool = False,
    force_grade: bool = False,
    grader_batch_size: int = 64,
    skip_grade: bool = False,
) -> dict:
    """Remote orchestrator for prepare → models → grade → aggregate.

    Runs on Modal so ``modal run --detach`` keeps the app alive across
    phases. Orchestrating from ``@app.local_entrypoint`` fails after the
    last model returns: with detach there are briefly no live inputs, the
    ephemeral app stops, and the next ``.spawn()`` raises ConflictError.
    """
    resolved_run_id = run_id or make_run_id()
    model_labels = parse_model_list(models)
    prompt_mode = _normalize_mode(mode)
    effective_source = _normalize_source_run_id(
        source_run_id, mode=prompt_mode, num_samples=num_samples
    )

    if aggregate_only:
        result = run_aggregate.remote(run_id=resolved_run_id, output_dir=output_dir)
        print("Done (aggregate-only):", result)
        return {
            "run_id": resolved_run_id,
            "mode": prompt_mode,
            "aggregate_only": True,
            "aggregate": result,
        }

    if grade_only:
        grade = run_freeform_grade.remote(
            run_id=resolved_run_id,
            output_dir=output_dir,
            model_labels=model_labels,
            grader_model_id=grader_model_id,
            batch_size=grader_batch_size,
            force=force_grade,
        )
        print("Graded:", grade)
        agg = None
        if not skip_aggregate:
            agg = run_aggregate.remote(
                run_id=resolved_run_id, output_dir=output_dir
            )
            print("Aggregated:", agg)
        return {
            "run_id": resolved_run_id,
            "mode": prompt_mode,
            "grade_only": True,
            "grade": grade,
            "aggregate": agg,
        }

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
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        mode=prompt_mode,
        source_run_id=effective_source,
    )

    print(
        f"Experiment run_id={resolved_run_id} mode={prompt_mode} "
        f"models={model_labels} num_samples={num_samples} n_shots={n_shots} "
        f"parallel_models={parallel_models} inference=vllm "
        f"source_run_id={effective_source} "
        f"sampling_overrides={{temperature={temperature}, top_p={top_p}, "
        f"max_new_tokens={max_new_tokens}}}"
    )

    prep = prepare_run.remote(
        run_id=resolved_run_id,
        output_dir=output_dir,
        model_labels=model_labels,
        num_samples=num_samples,
        n_shots=n_shots,
        seed=seed,
        meta=meta,
        data_root=data_root,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        mode=prompt_mode,
        source_run_id=effective_source,
        grader_model_id=grader_model_id,
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

    grade = None
    if prompt_mode == "freeform" and not skip_grade:
        grade = run_freeform_grade.remote(
            run_id=resolved_run_id,
            output_dir=output_dir,
            model_labels=model_labels,
            grader_model_id=grader_model_id,
            batch_size=grader_batch_size,
            force=force_grade,
        )
        print("Graded:", grade)

    if not skip_aggregate:
        agg = run_aggregate.remote(run_id=resolved_run_id, output_dir=output_dir)
        print("Aggregated:", agg)
    else:
        agg = None

    return {
        "run_id": resolved_run_id,
        "mode": prompt_mode,
        "models": results,
        "grade": grade,
        "aggregate": agg,
        "pending_labels": pending_labels,
        "skipped_labels": skipped_labels,
    }


@app.local_entrypoint()
def main(
    models: str = "all",
    num_samples: int = DEFAULT_NUM_SAMPLES,
    n_shots: int = DEFAULT_N_SHOTS,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    seed: int = DEFAULT_SEED,
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
    mode: str = DEFAULT_MODE,
    source_run_id: str | None = None,
    grader_model_id: str = DEFAULT_GRADER_MODEL_ID,
    grade_only: bool = False,
    force_grade: bool = False,
    grader_batch_size: int = 64,
    skip_grade: bool = False,
):
    """Launch the MMAR question-difficulty experiment.

    Args:
        models: Comma-separated labels or ``all``.
        num_samples: Fixed question sample size (default 200).
        n_shots: Independent temperature samples per question (default 10).
            Plain vLLM uses SamplingParams(n=...) shared prefill; Omni/HF
            duplicate prompts per shot. All pending questions go in one
            offline generate() so vLLM continuous-batches.
        temperature: Optional override of each model's sampling temperature.
        top_p: Optional override of each model's top_p.
        max_new_tokens: Optional override of each model's max_tokens.
        seed: RNG seed for question sampling and per-question sample seeds.
        max_num_seqs: Optional vLLM override (escape hatch; prefer defaults).
        gpu_memory_utilization: Optional vLLM GPU memory fraction override.
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
        mode: ``mc`` (choices + string match) or ``freeform`` (no choices;
            Qwen2.5-3B grades each shot).
        source_run_id: Copy ``question_ids.json`` from this prior run.
            Freeform defaults to ``20260727T154400Z``.
        grader_model_id: Hugging Face id for the freeform judge.
        grade_only: Skip generation; only run the freeform grader.
        force_grade: Re-grade shots even if already graded.
        grader_batch_size: Shots per grader generate() call.
        skip_grade: Freeform generation without the Qwen grading pass.
    """
    # One remote spawn owns the full prepare→infer→grade→aggregate chain so
    # ``--detach`` does not stop the ephemeral app between phases.
    resolved_run_id = run_id or make_run_id()
    out = run_pipeline.spawn(
        models=models,
        num_samples=num_samples,
        n_shots=n_shots,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        meta=meta,
        data_root=data_root,
        output_dir=output_dir,
        run_id=resolved_run_id,
        print_every=print_every,
        aggregate_only=aggregate_only,
        skip_aggregate=skip_aggregate,
        parallel_models=parallel_models,
        mode=mode,
        source_run_id=source_run_id,
        grader_model_id=grader_model_id,
        grade_only=grade_only,
        force_grade=force_grade,
        grader_batch_size=grader_batch_size,
        skip_grade=skip_grade,
    ).get()

    print("Done:", out)
    rid = out.get("run_id") or resolved_run_id
    prompt_mode = out.get("mode") or _normalize_mode(mode)
    print(
        "Download with:\n"
        f"  uv run modal run download_results.py "
        f"--remote-path exp-mmar-question-difficulty/{rid}"
    )
    if out.get("pending_labels") or out.get("skipped_labels"):
        print(
            "To resume this run later:\n"
            f"  uv run modal run --detach "
            f"exp-mmar-question-difficulty/run_experiment.py "
            f"--run-id {rid} --mode {prompt_mode}"
        )
