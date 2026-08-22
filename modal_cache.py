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

