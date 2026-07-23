"""Shared Modal Volume mounts and secrets for latent-reasoning evals."""

from __future__ import annotations

from pathlib import Path

import modal

VOLUME_NAME = "latent-reasoning"
RESULTS_VOLUME_NAME = "latent-reasoning-results"
VOLUME_MOUNT = Path("/cache")
RESULTS_MOUNT = Path("/results")
DATA_ROOT = VOLUME_MOUNT / "data"
MODELS_ROOT = VOLUME_MOUNT / "models"

DEFAULT_MMAR_DATA_ROOT = DATA_ROOT / "mmar"
DEFAULT_MMAR_META = DEFAULT_MMAR_DATA_ROOT / "MMAR-meta.jsonl"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
openai_secret = modal.Secret.from_name(
    "openai-secret", required_keys=["OPENAI_API_KEY"]
)


def mmar_eval_image(*extra_python_sources: str) -> modal.Image:
    """Debian image with AF audio deps + local modules for MMAR eval scripts."""
    sources = (
        "evaluation_rubrics",
        "modal_cache",
        "mmar_common",
        "audio_flamingo_runtime",
        "latent_cot",
        *extra_python_sources,
    )
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg")
        .uv_pip_install(
            "torch",
            "torchaudio",
            "transformers>=5.12.1",
            "accelerate>=1.14.0",
            "peft>=0.15.2",
            "huggingface-hub>=0.30.0",
            "librosa>=0.11.0",
            "soundfile",
            "safetensors>=0.8.0",
            "openai>=1.82.0",
            "tqdm>=4.67.0",
            "numpy",
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
        .add_local_python_source(*sources)
    )
