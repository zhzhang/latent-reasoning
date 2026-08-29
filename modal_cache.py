"""Shared Modal Volume mounts and secrets for latent-reasoning evals."""

from __future__ import annotations

import os
from pathlib import Path

import modal

VOLUME_NAME = "latent-reasoning"
RESULTS_VOLUME_NAME = "latent-reasoning-results"
JUDGING_VOLUME_NAME = "mmar-judging"
FREEFORM_THINKING_VOLUME_NAME = "mmar-freeform-5-shot-thinking"
MMAR_FREEFORM_THINKING_VOLUME_NAME = "mmar-freeform-thinking"
MMAR_DESCRIPTIONS_VOLUME_NAME = "mmar-descriptions"
VOLUME_MOUNT = Path("/cache")
RESULTS_MOUNT = Path("/results")
JUDGING_MOUNT = Path("/judging")
FREEFORM_THINKING_MOUNT = Path("/mmar-freeform-5-shot-thinking")
MMAR_FREEFORM_THINKING_MOUNT = Path("/mmar-freeform-thinking")
MMAR_DESCRIPTIONS_MOUNT = Path("/mmar-descriptions")
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
mmar_descriptions_volume = modal.Volume.from_name(
    MMAR_DESCRIPTIONS_VOLUME_NAME, create_if_missing=True
)

hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# GitHub tag v0.28.0 is not on PyPI yet. vLLM publishes that tag's wheels at
# wheels.vllm.ai/<full sha> (uv prefers extra-index-url over PyPI).
VLLM_VERSION = "0.28.0"
VLLM_GIT_SHA = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
VLLM_WHEEL_INDEX = f"https://wheels.vllm.ai/{VLLM_GIT_SHA}"

# Per-model torch.compile / Triton / nvcc / vLLM JIT artifacts. Subdirs avoid
# concurrent writers when models run in parallel. Must be configured before
# importing torch / vLLM / triton.
COMPILE_CACHE_ROOT = VOLUME_MOUNT / "vllm"


def compile_cache_dir(model_label: str) -> Path:
    return COMPILE_CACHE_ROOT / model_label


def configure_compile_cache(model_label: str) -> Path | None:
    """Point inductor / Triton / nvcc / vLLM caches at a volume subdir.

    Returns the cache root, or ``None`` when ``/cache`` is not writable
    (local runs without the Modal volume). Later GPU containers call this
    before ``load_model`` so they reuse artifacts from ``compile_cache.py``.
    """
    root = compile_cache_dir(model_label)
    subdirs = {
        "torchinductor": root / "torchinductor",
        "triton": root / "triton",
        "cuda": root / "cuda",
        "torch_extensions": root / "torch_extensions",
        "flashinfer": root / "flashinfer",
    }
    try:
        root.mkdir(parents=True, exist_ok=True)
        for path in subdirs.values():
            path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[{model_label}] compile cache {root} unavailable ({exc})")
        return None

    os.environ["VLLM_CACHE_ROOT"] = str(root)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(subdirs["torchinductor"])
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TRITON_CACHE_DIR"] = str(subdirs["triton"])
    os.environ["TRITON_HOME"] = str(subdirs["triton"])
    os.environ["CUDA_CACHE_PATH"] = str(subdirs["cuda"])
    os.environ["TORCH_EXTENSIONS_DIR"] = str(subdirs["torch_extensions"])
    os.environ["FLASHINFER_CACHE_DIR"] = str(subdirs["flashinfer"])
    os.environ["FLASHINFER_WORKSPACE_DIR"] = str(subdirs["flashinfer"])
    print(f"[{model_label}] compile cache -> {root}")
    return root


def compile_cache_stats(model_label: str) -> dict[str, object]:
    """File count + byte size of a model's compile-cache tree."""
    root = compile_cache_dir(model_label)
    n_files = 0
    n_bytes = 0
    newest_mtime = 0.0
    if root.is_dir():
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                n_files += 1
                try:
                    st = (Path(dirpath) / name).stat()
                except OSError:
                    continue
                n_bytes += st.st_size
                if st.st_mtime > newest_mtime:
                    newest_mtime = st.st_mtime
    return {
        "path": str(root),
        "n_files": n_files,
        "n_bytes": n_bytes,
        "n_mib": round(n_bytes / (1024 * 1024), 1),
        "newest_mtime": newest_mtime,
    }


def commit_compile_cache(model_label: str | None = None) -> None:
    """Persist compile-cache writes on the ``latent-reasoning`` volume."""
    tag = f"[{model_label}] " if model_label else ""
    try:
        volume.commit()
    except Exception as exc:  # noqa: BLE001 — cache commit is best-effort
        print(f"{tag}volume.commit compile cache failed: {exc}")

