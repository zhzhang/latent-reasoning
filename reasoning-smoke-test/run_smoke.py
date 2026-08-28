"""Tiny reasoning smoke test for every MMAR test-taker in ``MODEL_SPECS``.

Loads one model, runs one freeform MMAR clip, and writes the prompt/output
token ids the model actually consumed. vLLM boots with ``enforce_eager`` by
default so compile / CUDA-graph warmup is skipped. Pass ``--no-enforce-eager``
to exercise ``profile_run`` with torch.compile / CUDA graphs.

Usage::

    uv run modal run reasoning-smoke-test/run_smoke.py
    uv run modal run reasoning-smoke-test/run_smoke.py --models gemma-4-e4b
    uv run modal run reasoning-smoke-test/run_smoke.py --models music-flamingo
    uv run modal run reasoning-smoke-test/run_smoke.py --models qwen3-omni --no-enforce-eager
    uv run modal run reasoning-smoke-test/run_smoke.py \\
      --question-id GJ6r_T6ckc4_00-00-00_00-00-06 --max-new-tokens 2048

View locally (after the run returns, or after downloading the volume)::

    uv run python reasoning-smoke-test/view_smoke.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import modal

from modal_cache import (
    DEFAULT_MMAR_DATA_ROOT,
    DEFAULT_MMAR_META,
    RESULTS_MOUNT,
    VOLUME_MOUNT,
    VLLM_WHEEL_INDEX,
    configure_compile_cache,
    hf_secret,
    results_volume,
    volume,
)
from mmar_common import load_jsonl, make_run_id, resolve_path, write_json
from mmar_models import (
    ALL_MODEL_LABELS,
    MODEL_SPECS,
    generate_raw_trace,
    load_model,
    parse_model_list,
    resolve_sampling,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEPLOY_MOUNT = "/root/deploy"
DEFAULT_QUESTION_ID = "GJ6r_T6ckc4_00-00-00_00-00-06"
DEFAULT_OUTPUT_DIR = RESULTS_MOUNT / "reasoning-smoke-test"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_SEED = 0


def _write_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def _bound_preview(pieces: list[dict[str, Any]], limit: int = 80) -> str:
    parts: list[str] = []
    for piece in pieces[:limit]:
        text = piece.get("text") or piece.get("token") or "?"
        parts.append(str(text).replace("\n", "\\n"))
    extra = f" │…(+{len(pieces) - limit})" if len(pieces) > limit else ""
    return "│".join(parts) + extra

_VLLM_CACHE_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}
_INPROC_VLLM_ENV = {
    **_VLLM_CACHE_ENV,
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}
_SHARED_SOURCES = (
    "modal_cache",
    "mmar_common",
    "audio_flamingo_runtime",
    "mmar_models",
)
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


def _cuda_base_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            add_python="3.12",
        )
        .entrypoint([])
        .apt_install("ffmpeg", "git")
    )


def _mount_local_sources(image: modal.Image) -> modal.Image:
    return image.add_local_python_source(*_SHARED_SOURCES).add_local_dir(
        str(REPO_ROOT / "deploy"), remote_path=_DEPLOY_MOUNT
    )


# AF-Next / Music Flamingo: vLLM 0.24 MusicFlamingo (+ HF fallback).
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

app = modal.App("reasoning-smoke-test")

_FN_KW = dict(
    timeout=45 * 60,
    memory=65536,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
)


def _pick_item(question_id: str | None) -> dict:
    meta_path = Path(DEFAULT_MMAR_META)
    data_root = Path(DEFAULT_MMAR_DATA_ROOT)
    if not meta_path.is_file():
        raise SystemExit(
            f"MMAR meta missing at {meta_path}. Seed first:\n"
            "  uv run modal run seed_volume.py --datasets mmar --models none"
        )
    items = load_jsonl(meta_path)
    wanted = str(question_id or DEFAULT_QUESTION_ID)
    match = next((item for item in items if str(item.get("id")) == wanted), None)
    ordered = ([match] if match else []) + [
        item for item in items if item is not match
    ]
    for item in ordered:
        audio_path = resolve_path(data_root, item.get("audio_path") or "")
        if audio_path and Path(audio_path).is_file():
            if str(item.get("id")) != wanted:
                print(f"[smoke] {wanted} missing; using {item.get('id')}")
            return {**item, "audio_path": str(audio_path)}
    raise SystemExit(f"No MMAR wav found under {data_root / 'audio'}")


def _parse_models(value: str) -> list[str]:
    try:
        return parse_model_list(value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _smoke_one(
    *,
    model_label: str,
    run_id: str,
    output_dir: str,
    question_id: str | None,
    max_new_tokens: int,
    seed: int,
    enforce_eager: bool = True,
) -> dict[str, Any]:
    volume.reload()
    results_volume.reload()

    spec = MODEL_SPECS[model_label]
    configure_compile_cache(model_label)
    item = _pick_item(question_id)
    args = SimpleNamespace(
        model_id=spec["model_id"],
        tokenizer_id=spec.get("tokenizer_id"),
        local_model_dir=None,
        local_tokenizer_dir=None,
        temperature=None,
        top_p=None,
        max_new_tokens=int(max_new_tokens),
        greedy_non_thinking=False,
        seed=int(seed),
        max_num_seqs=1,
        gpu_memory_utilization=None,
        prompt_mode="freeform",
        enforce_eager=bool(enforce_eager),
    )
    sampling = resolve_sampling(model_label, args)
    args.temperature = float(sampling["temperature"])
    args.top_p = float(sampling.get("top_p", 1.0))
    args.max_new_tokens = int(sampling["max_tokens"])
    args.repetition_penalty = float(sampling.get("repetition_penalty", 1.0))
    args.sampling = sampling

    # Skip DeepGEMM kernel warmup; isolate compile / CUDA-graph profile_run.
    os.environ["VLLM_DEEP_GEMM_WARMUP"] = "skip"

    try:
        import vllm

        vllm_version = getattr(vllm, "__version__", "unknown")
    except Exception:  # noqa: BLE001 — version is diagnostic only
        vllm_version = "unknown"
    print(
        f"[smoke {model_label}] id={item.get('id')} "
        f"vllm={vllm_version} "
        f"native_thinking={spec.get('native_thinking')} "
        f"enable_thinking={spec.get('enable_thinking')} "
        f"enforce_eager={bool(enforce_eager)} sampling={sampling}"
    )
    handle = load_model(model_label, args)
    try:
        volume.commit()
    except Exception as exc:  # noqa: BLE001 — cache commit is best-effort
        print(f"[smoke {model_label}] volume.commit after load failed: {exc}")
    trace = generate_raw_trace(model_label, handle, item, args)
    try:
        volume.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke {model_label}] volume.commit after generate failed: {exc}")

    record = {
        "model_label": model_label,
        "model_id": spec["model_id"],
        "question_id": item.get("id"),
        "question": item.get("question"),
        "audio_path": Path(item["audio_path"]).name,
        "modality": item.get("modality"),
        "category": item.get("category"),
        "sampling": sampling,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **trace,
    }

    run_dir = Path(output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{model_label}.json"
    _write_trace(out_path, record)
    results_volume.commit()

    prompt_text = record["prompt"]["text"]
    output_text = record["output"]["text"]
    print(f"[smoke {model_label}] prompt_tokens={record['prompt']['n_tokens']}")
    print(f"[smoke {model_label}] output_tokens={record['output']['n_tokens']}")
    print(f"[smoke {model_label}] prompt_text=\n{prompt_text}")
    print(f"[smoke {model_label}] output_text=\n{output_text}")
    print(f"[smoke {model_label}] prompt_bounded=\n{_bound_preview(record['prompt']['pieces'])}")
    print(f"[smoke {model_label}] output_bounded=\n{_bound_preview(record['output']['pieces'])}")
    print(f"[smoke {model_label}] wrote {out_path}")
    return record


@app.function(image=af_next_image, gpu="L40S", **_FN_KW)
def smoke_af_l40s(model_label: str, **kwargs) -> dict:
    return _smoke_one(model_label=model_label, **kwargs)


@app.function(image=omni_image, gpu="A100-80GB", **_FN_KW)
def smoke_omni_a100(model_label: str, **kwargs) -> dict:
    return _smoke_one(model_label=model_label, **kwargs)


@app.function(image=interactive_omni_image, gpu="A100-80GB", **_FN_KW)
def smoke_interactive_a100(model_label: str, **kwargs) -> dict:
    return _smoke_one(model_label=model_label, **kwargs)


@app.function(image=large_mm_image, gpu="A100-80GB", **_FN_KW)
def smoke_large_a100(model_label: str, **kwargs) -> dict:
    return _smoke_one(model_label=model_label, **kwargs)


@app.function(image=large_mm_image, gpu="L40S", **_FN_KW)
def smoke_large_l40s(model_label: str, **kwargs) -> dict:
    return _smoke_one(model_label=model_label, **kwargs)


@app.function(image=large_mm_image, gpu="H100", **_FN_KW)
def smoke_large_h100(model_label: str, **kwargs) -> dict:
    return _smoke_one(model_label=model_label, **kwargs)


def _smoke_worker_for(label: str):
    spec = MODEL_SPECS[label]
    backend = str(spec.get("backend") or "")
    gpu = str(spec.get("gpu") or "")
    if label in {"af-next-think", "music-flamingo"}:
        return smoke_af_l40s
    if backend == "vllm_omni":
        return smoke_omni_a100
    if label == "interactive-omni-8b":
        return smoke_interactive_a100
    if gpu == "H100":
        return smoke_large_h100
    if gpu == "L40S":
        return smoke_large_l40s
    if gpu == "A100-80GB":
        return smoke_large_a100
    raise RuntimeError(f"No smoke worker for {label!r} backend={backend} gpu={gpu}")


_WORKERS = {label: _smoke_worker_for(label) for label in ALL_MODEL_LABELS}
_missing_smoke = [label for label in ALL_MODEL_LABELS if label not in _WORKERS]
if _missing_smoke:
    raise RuntimeError(f"No smoke worker for models: {_missing_smoke}")


def _write_local(run_id: str, records: list[dict[str, Any]]) -> Path:
    local_dir = REPO_ROOT / "outputs" / "reasoning-smoke-test" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(rec["model_label"]) for rec in records]
    write_json(
        local_dir / "manifest.json",
        {
            "run_id": run_id,
            "models": labels,
            "question_id": records[0].get("question_id") if records else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    for rec in records:
        _write_trace(local_dir / f"{rec['model_label']}.json", rec)
    return local_dir


@app.local_entrypoint()
def main(
    models: str = "all",
    question_id: str = DEFAULT_QUESTION_ID,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = DEFAULT_SEED,
    run_id: str | None = None,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    enforce_eager: bool = True,
) -> None:
    labels = _parse_models(models)
    resolved_run_id = run_id or make_run_id()
    kwargs = dict(
        run_id=resolved_run_id,
        output_dir=output_dir,
        question_id=question_id,
        max_new_tokens=max_new_tokens,
        seed=seed,
        enforce_eager=enforce_eager,
    )
    print(
        f"[smoke] run_id={resolved_run_id} models={labels} "
        f"question_id={question_id} max_new_tokens={max_new_tokens} "
        f"enforce_eager={enforce_eager}"
    )
    handles = [
        _WORKERS[label].spawn(model_label=label, **kwargs) for label in labels
    ]
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for label, handle in zip(labels, handles):
        try:
            records.append(handle.get())
        except Exception as exc:  # noqa: BLE001 — keep sibling traces
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"[smoke] {label} FAILED: {type(exc).__name__}: {exc}")
    if records:
        local_dir = _write_local(resolved_run_id, records)
        print(f"[smoke] local copy -> {local_dir}")
        print("View with:")
        print(f"  uv run python reasoning-smoke-test/view_smoke.py --run-id {resolved_run_id}")
    if failures:
        raise SystemExit(
            f"{len(failures)}/{len(labels)} model(s) failed:\n  " + "\n  ".join(failures)
        )
