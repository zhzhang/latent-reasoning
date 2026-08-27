"""Shared Modal Volume mounts and secrets for latent-reasoning evals."""

from __future__ import annotations

from pathlib import Path

import modal

VOLUME_NAME = "latent-reasoning"
RESULTS_VOLUME_NAME = "latent-reasoning-results"
JUDGING_VOLUME_NAME = "mmar-judging"
FREEFORM_THINKING_VOLUME_NAME = "mmar-freeform-5-shot-thinking"
MMAR_FREEFORM_THINKING_VOLUME_NAME = "mmar-freeform-thinking"
VOLUME_MOUNT = Path("/cache")
RESULTS_MOUNT = Path("/results")
JUDGING_MOUNT = Path("/judging")
FREEFORM_THINKING_MOUNT = Path("/mmar-freeform-5-shot-thinking")
MMAR_FREEFORM_THINKING_MOUNT = Path("/mmar-freeform-thinking")
# Snapshot of a local ``outputs/mmar-freeform-thinking`` download, when present.
LOCAL_MMAR_FREEFORM_THINKING_MOUNT = Path("/local-mmar-freeform-thinking")
DATA_ROOT = VOLUME_MOUNT / "data"
MODELS_ROOT = VOLUME_MOUNT / "models"

DEFAULT_MMAR_DATA_ROOT = DATA_ROOT / "mmar"
DEFAULT_MMAR_META = DEFAULT_MMAR_DATA_ROOT / "MMAR-meta.jsonl"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
judging_volume = modal.Volume.from_name(JUDGING_VOLUME_NAME, create_if_missing=True)
freeform_thinking_volume = modal.Volume.from_name(
    FREEFORM_THINKING_VOLUME_NAME, create_if_missing=True
)
mmar_freeform_thinking_volume = modal.Volume.from_name(
    MMAR_FREEFORM_THINKING_VOLUME_NAME, create_if_missing=True
)

hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# GitHub tag v0.28.0 is not on PyPI yet. vLLM publishes that tag's wheels at
# wheels.vllm.ai/<full sha> (uv prefers extra-index-url over PyPI).
VLLM_VERSION = "0.28.0"
VLLM_GIT_SHA = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
VLLM_WHEEL_INDEX = f"https://wheels.vllm.ai/{VLLM_GIT_SHA}"

