"""Shared Modal container images for MMAR eval and LALM judging.

All eval / suite-judge GPU workers share one vLLM 0.28.0 image. GPU choice
still comes from ``MODEL_SPECS[label]["gpu"]``. Which labels get a worker is
``ALL_MODEL_LABELS`` (the uncommented keys of ``MODEL_SPECS``).
"""

from __future__ import annotations

import modal

from modal_cache import VLLM_WHEEL_INDEX

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
    """Attach Python modules (must be last image steps)."""
    return image.add_local_python_source(*_SHARED_SOURCES)


# Keep EngineCore in-process. Qwen3-Omni's profile_run hits a meta/cuda device
# mismatch under multiprocess EngineCore (Tensor on device meta is not on the
# expected device cuda:0). AF-Next also needs in-process for its symlink
# model-view load path.
_EVAL_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
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


# Install E=128,N=768 fused-MoE Triton config under B200. Skip the copy when
# the wheel already has a native file for that device.
_FUSED_MOE_CONFIG_CMD = (
    "D=/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs && "
    "SRC=\"$D/E=128,N=768,device_name=NVIDIA_H200.json\" && "
    "if [ ! -f \"$SRC\" ]; then echo \"fused_moe: no H200 config, skipping\"; "
    "else "
    "DST=\"$D/E=128,N=768,device_name=NVIDIA_B200.json\"; "
    "if [ -f \"$DST\" ]; then echo \"fused_moe: NVIDIA_B200 already present\"; "
    "else cp \"$SRC\" \"$DST\" && echo \"fused_moe: installed NVIDIA_B200 from H200\"; fi; "
    "fi"
)

eval_image = mount_local_sources(
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
        "peft>=0.15.2",
        "safetensors>=0.8.0",
        extra_index_url=VLLM_WHEEL_INDEX,
    )
    .run_commands(_FUSED_MOE_CONFIG_CMD)
    .env(_EVAL_ENV)
)

# Lightweight CPU image for manifest / question-id helpers.
cpu_image = mount_local_sources(
    modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "numpy", "tqdm>=4.67.0"
    )
)
