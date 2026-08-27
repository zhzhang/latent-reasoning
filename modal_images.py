"""Shared Modal container images for MMAR eval and LALM judging.

Pip / CUDA / env stacks match ``run_experiment.py``. GPU choice still
comes from ``MODEL_SPECS[label]["gpu"]``. Which labels get a worker is
``ALL_MODEL_LABELS`` (the uncommented keys of ``MODEL_SPECS``).
"""

from __future__ import annotations

from pathlib import Path

import modal

from modal_cache import VLLM_WHEEL_INDEX

REPO_ROOT = Path(__file__).resolve().parent
_DEPLOY_MOUNT = "/root/deploy"

# Modal rule: after any ``add_local_*``, no further build steps (apt/pip/run/env).
# Put installs + env first; mount local sources last.
_SHARED_SOURCES = (
    "modal_images",
    "modal_cache",
    "mmar_common",
    "mmar_api",
    "audio_flamingo_runtime",
    "aggregate",
    "grader",
    "mmar_models",
)


def mount_local_sources(image: modal.Image) -> modal.Image:
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


def cuda_base_image(python_version: str = "3.12") -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            add_python=python_version,
        )
        .entrypoint([])
        .apt_install("ffmpeg", "git")
    )


# AF-Next: vLLM 0.24 MusicFlamingo (+ HF fallback deps if weight load fails).
af_next_image = mount_local_sources(
    cuda_base_image()
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
omni_image = mount_local_sources(
    cuda_base_image()
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

# Step-Audio-R1.1: StepFun custom vLLM fork (registers step_audio_2).
step_audio_image = mount_local_sources(
    modal.Image.from_registry("stepfun2025/vllm:step-audio-2-v20250909")
    .entrypoint([])
    .apt_install("ffmpeg", "git")
    .uv_pip_install(
        "librosa>=0.11.0",
        "soundfile",
        "numpy",
        "tqdm>=4.67.0",
        "huggingface-hub>=0.30.0",
    )
    .env(_INPROC_VLLM_ENV)
)

# InteractiveOmni: newer vLLM transformers-audio backend + HF chat fallback.
interactive_omni_image = mount_local_sources(
    cuda_base_image()
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
large_mm_image = mount_local_sources(
    cuda_base_image()
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
cpu_image = mount_local_sources(
    modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "numpy", "tqdm>=4.67.0"
    )
)

# Image for each known label (including specs currently commented out).
# GPU comes from MODEL_SPECS[label]["gpu"].
EVAL_IMAGES: dict[str, modal.Image] = {
    "af-next-think": af_next_image,
    "music-flamingo": af_next_image,
    "mimo-audio-7b": omni_image,
    "step-audio-r1.1": step_audio_image,
    "interactive-omni-8b": interactive_omni_image,
    "qwen3-omni": large_mm_image,
    "voxtral-small-24b": large_mm_image,
    "qwen2.5-omni-7b": large_mm_image,
    "phi-4-multimodal": large_mm_image,
    "gemma-4-e4b": large_mm_image,
    "qwen3-omni-instruct": large_mm_image,
    "nemotron-3-nano-omni": large_mm_image,
}
