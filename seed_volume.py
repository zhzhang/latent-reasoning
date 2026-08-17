"""Seed a shared Modal Volume with MMAR data and model weights.

Layout on the ``latent-reasoning`` Volume:

    /data/mmar/
      audio/*.wav
      MMAR-meta.jsonl
      data/mmar-audio.tar.gz
    /models/nvidia/audio-flamingo-3-hf/
    /models/nvidia/audio-flamingo-next-think-hf/
    /models/nvidia/audio-flamingo-2/
    /models/Qwen/Qwen3-Omni-30B-A3B-Thinking/
    /models/Qwen/Qwen3.6-35B-A3B-FP8/
    /models/mistralai/Voxtral-Small-24B-2507/
    ...

Consumers can mount the whole Volume, or use subpaths::

    volume.with_mount_options(sub_path="data")   # -> /data
    volume.with_mount_options(sub_path="models") # -> /models

Usage:

    uv run modal run seed_volume.py
    uv run modal run seed_volume.py --datasets mmar --models af3
    uv run modal run seed_volume.py --datasets mmar --models eval
    uv run modal run seed_volume.py --datasets mmar --models none
    uv run modal run seed_volume.py --datasets none --models af3,af2 --force
    uv run modal run seed_volume.py --repo-id nvidia/audio-flamingo-3-hf --datasets none
    uv run modal run seed_volume.py --list-only
    uv run modal run --detach seed_volume.py --datasets none --models af-next-think

Requires a Modal Secret named ``huggingface-secret`` with key ``HF_TOKEN``
(for higher HF rate limits / gated repos).
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
from pathlib import Path

import modal

VOLUME_NAME = "latent-reasoning"
VOLUME_MOUNT = Path("/cache")
DATA_ROOT = VOLUME_MOUNT / "data"
MODELS_ROOT = VOLUME_MOUNT / "models"

MMAR_REPO = "BoJack/MMAR"
MMAR_AUDIO_ARCHIVE = "mmar-audio.tar.gz"
# HF MMAR-meta.json omits thinking/rubric/cue required by MMAR-Rubrics scoring.
# Use the GitHub release that includes instance rubrics + GT CoT.
MMAR_META_URL = "https://raw.githubusercontent.com/ddlBoJack/MMAR/main/MMAR-meta.jsonl"
MMAR_RUBRIC_KEYS = ("thinking", "rubric", "cue")
MIN_MMAR_WAVS = 1000
MIN_DISK_SIZE = 524288

# Short aliases used by this repo's MMAR eval scripts.
MODEL_ALIASES: dict[str, str] = {
    "af3": "nvidia/audio-flamingo-3-hf",
    "audio-flamingo-3": "nvidia/audio-flamingo-3-hf",
    "audio-flamingo-3-hf": "nvidia/audio-flamingo-3-hf",
    "af-next-think": "nvidia/audio-flamingo-next-think-hf",
    "afnext-think": "nvidia/audio-flamingo-next-think-hf",
    "audio-flamingo-next-think": "nvidia/audio-flamingo-next-think-hf",
    "audio-flamingo-next-think-hf": "nvidia/audio-flamingo-next-think-hf",
    "af2": "nvidia/audio-flamingo-2",
    "audio-flamingo-2": "nvidia/audio-flamingo-2",
    "qwen3-omni": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    "qwen3-omni-thinking": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    "qwen3-omni-instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen2.5-omni-7b": "Qwen/Qwen2.5-Omni-7B",
    "qwen2.5-omni": "Qwen/Qwen2.5-Omni-7B",
    "phi-4-multimodal": "microsoft/Phi-4-multimodal-instruct",
    "phi-4-mm": "microsoft/Phi-4-multimodal-instruct",
    "gemma-4-e4b": "google/gemma-4-E4B-it",
    "gemma-4-e4b-it": "google/gemma-4-E4B-it",
    "nemotron-3-nano-omni": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
    "nemotron-3-nano-omni-fp8": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3.6-35b-a3b-fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
    "qwen3.6-35b-a3b": "Qwen/Qwen3.6-35B-A3B-FP8",
    "qwen3.6-35b": "Qwen/Qwen3.6-35B-A3B-FP8",
    "qwen3.6": "Qwen/Qwen3.6-35B-A3B-FP8",
    "step-audio-2-mini": "stepfun-ai/Step-Audio-2-mini",
    "step-audio-2": "stepfun-ai/Step-Audio-2-mini",
    "mimo-audio-7b": "XiaomiMiMo/MiMo-Audio-7B-Instruct",
    "mimo-audio-7b-instruct": "XiaomiMiMo/MiMo-Audio-7B-Instruct",
    "mimo-audio-tokenizer": "XiaomiMiMo/MiMo-Audio-Tokenizer",
    "interactive-omni-8b": "sensenova/InteractiveOmni-8B",
    "interactive-omni": "sensenova/InteractiveOmni-8B",
    "voxtral-small-24b": "mistralai/Voxtral-Small-24B-2507",
    "voxtral-small": "mistralai/Voxtral-Small-24B-2507",
    "voxtral": "mistralai/Voxtral-Small-24B-2507",
}

# Aliases that should also seed companion repos (e.g. MiMo tokenizer).
MODEL_SEED_EXTRAS: dict[str, tuple[str, ...]] = {
    "mimo-audio-7b": ("XiaomiMiMo/MiMo-Audio-Tokenizer",),
    "mimo-audio-7b-instruct": ("XiaomiMiMo/MiMo-Audio-Tokenizer",),
    "XiaomiMiMo/MiMo-Audio-7B-Instruct": ("XiaomiMiMo/MiMo-Audio-Tokenizer",),
}

DEFAULT_DATASETS = ("mmar",)
DEFAULT_MODELS = ("af3",)
ALL_MODELS = (
    "af3",
    "af-next-think",
    "af2",
    "qwen3-omni",
    "qwen3-omni-instruct",
    "qwen2.5-omni-7b",
    "phi-4-multimodal",
    "gemma-4-e4b",
    "nemotron-3-nano-omni",
    "qwen2.5-3b",
    "qwen3.6-35b-a3b-fp8",
    "step-audio-2-mini",
    "mimo-audio-7b",
    "interactive-omni-8b",
    "voxtral-small-24b",
)

# Checkpoints used by ``run_experiment.py --models all`` (no judges / unused AF2/AF3).
EVAL_MODELS = (
    "af-next-think",
    "mimo-audio-7b",
    "interactive-omni-8b",
    "qwen3-omni",
    "voxtral-small-24b",
    "qwen2.5-omni-7b",
    "phi-4-multimodal",
    "gemma-4-e4b",
    "qwen3-omni-instruct",
    "nemotron-3-nano-omni",
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "huggingface-hub>=0.30.0",
        "tqdm>=4.67.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("seed-volume", image=image)

hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])


def resolve_repo_id(name: str) -> str:
    key = name.strip()
    if not key:
        raise ValueError("Empty model name")
    return MODEL_ALIASES.get(key.lower(), key)


def expand_seed_targets(names: list[str]) -> list[str]:
    """Resolve aliases and append companion repos (e.g. MiMo tokenizer)."""
    targets: list[str] = []
    for name in names:
        repo_id = resolve_repo_id(name)
        targets.append(repo_id)
        extras = MODEL_SEED_EXTRAS.get(name.strip()) or MODEL_SEED_EXTRAS.get(
            name.strip().lower()
        ) or MODEL_SEED_EXTRAS.get(repo_id)
        if extras:
            targets.extend(extras)
    return list(dict.fromkeys(targets))


def model_dir_for(repo_id: str) -> Path:
    return MODELS_ROOT / repo_id


def _count_wavs(audio_dir: Path) -> int:
    if not audio_dir.is_dir():
        return 0
    return sum(1 for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav")


def _meta_has_rubrics(meta_path: Path) -> bool:
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        return False
    with open(meta_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            return all(key in record for key in MMAR_RUBRIC_KEYS)
    return False


def _download_mmar_rubric_meta(dest: Path) -> int:
    """Download MMAR-meta.jsonl with thinking/rubric/cue from GitHub."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"Downloading rubric meta from {MMAR_META_URL} ...")
    urllib.request.urlretrieve(MMAR_META_URL, tmp)
    if not _meta_has_rubrics(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded MMAR meta is missing required keys {MMAR_RUBRIC_KEYS}: {tmp}"
        )
    tmp.replace(dest)
    with open(dest, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _dir_nonempty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _looks_seeded(path: Path) -> bool:
    if not _dir_nonempty(path):
        return False
    if (path / ".seed_complete").exists():
        return True
    has_config = (path / "config.json").exists() or any(path.glob("*/config.json"))
    has_weights = any(path.rglob("*.safetensors")) or any(path.rglob("*.bin"))
    return bool(has_config and has_weights)


def _parse_noneable(value: str) -> str | None:
    """Return None when the CLI value means 'skip this category'."""
    stripped = value.strip()
    if not stripped or stripped.lower() in {"none", "skip", "-"}:
        return None
    return stripped


@app.function(
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
    timeout=2 * 60 * 60,
    ephemeral_disk=MIN_DISK_SIZE,  # MiB; MMAR archive is ~3 GiB before extract
)
def seed_mmar(force: bool = False) -> dict:
    """Download MMAR audio + metadata into ``/cache/data/mmar``."""
    from huggingface_hub import hf_hub_download

    dest_root = DATA_ROOT / "mmar"
    audio_dir = dest_root / "audio"
    meta_path = dest_root / "MMAR-meta.jsonl"
    archive_cache = dest_root / "data" / MMAR_AUDIO_ARCHIVE

    wav_count = _count_wavs(audio_dir)
    meta_ok = _meta_has_rubrics(meta_path)
    if wav_count >= MIN_MMAR_WAVS and meta_ok and not force:
        summary = {
            "dataset": "mmar",
            "status": "skipped",
            "wav_files": wav_count,
            "meta": str(meta_path),
            "meta_has_rubrics": True,
        }
        print(summary)
        return summary

    # Audio is present but meta lacks rubrics: refresh meta only (no full re-download).
    if wav_count >= MIN_MMAR_WAVS and not meta_ok and not force:
        n_meta = _download_mmar_rubric_meta(meta_path)
        volume.commit()
        summary = {
            "dataset": "mmar",
            "status": "meta_refreshed",
            "wav_files": wav_count,
            "meta_records": n_meta,
            "meta_has_rubrics": True,
            "path": str(dest_root),
        }
        print(summary)
        return summary

    tmp_root = Path("/tmp/mmar")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    print(f"Downloading {MMAR_AUDIO_ARCHIVE} from {MMAR_REPO} ...")
    archive_tmp = Path(
        hf_hub_download(
            repo_id=MMAR_REPO,
            filename=MMAR_AUDIO_ARCHIVE,
            repo_type="dataset",
            local_dir=str(tmp_root / "download"),
            token=os.environ.get("HF_TOKEN"),
        )
    )

    print(f"Extracting {archive_tmp} ...")
    extract_dir = tmp_root / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_tmp, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    # Archive may unpack as ./audio or ./mmar/audio; normalize to dest_root/audio.
    candidate_audio = extract_dir / "audio"
    if not candidate_audio.is_dir():
        matches = [path for path in extract_dir.rglob("audio") if path.is_dir()]
        if not matches:
            raise RuntimeError(f"No audio/ directory found after extracting {archive_tmp}")
        candidate_audio = matches[0]

    if dest_root.exists() and force:
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    (dest_root / "data").mkdir(parents=True, exist_ok=True)

    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    shutil.copytree(candidate_audio, audio_dir)
    shutil.copy2(archive_tmp, archive_cache)
    n_meta = _download_mmar_rubric_meta(meta_path)

    wav_count = _count_wavs(audio_dir)
    if wav_count < MIN_MMAR_WAVS:
        raise RuntimeError(
            f"Expected at least {MIN_MMAR_WAVS} wav files in {audio_dir}, found {wav_count}."
        )

    volume.commit()
    summary = {
        "dataset": "mmar",
        "status": "ok",
        "wav_files": wav_count,
        "meta_records": n_meta,
        "meta_has_rubrics": True,
        "path": str(dest_root),
    }
    print(summary)
    return summary


@app.function(
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
    # Scratch for hub temp files; weights are written onto the Volume itself.
    ephemeral_disk=MIN_DISK_SIZE,
)
def seed_model(
    repo_id: str,
    force: bool = False,
    revision: str | None = None,
    ignore_patterns: list[str] | None = None,
    hub_cache_layout: bool = False,
) -> dict:
    """Download one Hub model into the shared Volume.

    Default layout: ``/cache/models/<repo_id>`` (explicit local snapshot).
    With ``hub_cache_layout=True``: standard HF hub cache under ``/cache/models``
    (consumers should set ``HF_HUB_CACHE=/models`` when mounting the models subpath).
    """
    from huggingface_hub import snapshot_download

    repo_id = resolve_repo_id(repo_id)
    token = os.environ.get("HF_TOKEN")
    # Prefer safetensors; skip redundant full pytorch dumps. Keep think/*.bin
    # (AF3 non_lora_trainables) and other small adapter bins. InteractiveOmni
    # needs campplus.onnx at load time, so do not ignore *.onnx for that repo.
    patterns = ignore_patterns or [
        "*.pt",
        "*.gguf",
        "*.h5",
        "pytorch_model*.bin",
    ]
    if "InteractiveOmni" not in repo_id and "interactive-omni" not in repo_id.lower():
        patterns = ["*.onnx", *patterns]

    if hub_cache_layout:
        os.environ["HF_HUB_CACHE"] = str(MODELS_ROOT)
        marker = MODELS_ROOT / ".seeded" / repo_id.replace("/", "__")
        already = marker.exists() and not force
        dest_label = str(MODELS_ROOT)
    else:
        dest = model_dir_for(repo_id)
        marker = dest / ".seed_complete"
        already = _looks_seeded(dest) and not force
        dest_label = str(dest)

    if already:
        summary = {
            "repo_id": repo_id,
            "status": "skipped",
            "path": dest_label,
            "layout": "hub-cache" if hub_cache_layout else "local-dir",
        }
        print(summary)
        return summary

    print(f"Downloading {repo_id} (revision={revision!r}) ...")
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    if hub_cache_layout:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            force_download=force,
            ignore_patterns=patterns,
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{snapshot_path}\n", encoding="utf-8")
        path_out = snapshot_path
    else:
        dest = model_dir_for(repo_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(dest),
            token=token,
            force_download=force,
            ignore_patterns=patterns,
        )
        marker.write_text("ok\n", encoding="utf-8")
        path_out = str(dest)

    volume.commit()
    summary = {
        "repo_id": repo_id,
        "status": "ok",
        "path": path_out,
        "layout": "hub-cache" if hub_cache_layout else "local-dir",
    }
    print(summary)
    return summary


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=10 * 60)
def list_volume() -> dict:
    """Summarize datasets and models currently on the Volume."""
    volume.reload()
    summary: dict[str, object] = {
        "volume": VOLUME_NAME,
        "mount": str(VOLUME_MOUNT),
        "datasets": {},
        "models": [],
    }

    datasets: dict[str, object] = {}
    path = DATA_ROOT / "mmar"
    if not path.exists():
        datasets["mmar"] = {"present": False}
    else:
        info: dict[str, object] = {
            "present": True,
            "path": str(path),
            "wav_files": _count_wavs(path / "audio"),
            "meta": (path / "MMAR-meta.jsonl").exists(),
        }
        datasets["mmar"] = info
        print(f"dataset mmar: {info}")
    summary["datasets"] = datasets

    entries: list[dict] = []
    seen: set[str] = set()
    if MODELS_ROOT.is_dir():
        for org_dir in sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir()):
            if org_dir.name.startswith(".") or org_dir.name.startswith("models--"):
                continue
            for model_dir in sorted(p for p in org_dir.iterdir() if p.is_dir()):
                repo_id = f"{org_dir.name}/{model_dir.name}"
                seen.add(repo_id)
                entries.append(
                    {
                        "repo_id": repo_id,
                        "path": str(model_dir),
                        "seeded": _looks_seeded(model_dir),
                        "layout": "local-dir",
                    }
                )

        seeded_dir = MODELS_ROOT / ".seeded"
        if seeded_dir.is_dir():
            for marker in sorted(seeded_dir.iterdir()):
                if not marker.is_file():
                    continue
                repo_id = marker.name.replace("__", "/")
                if repo_id in seen:
                    continue
                entries.append(
                    {
                        "repo_id": repo_id,
                        "path": marker.read_text(encoding="utf-8").strip(),
                        "seeded": True,
                        "layout": "hub-cache",
                    }
                )

        for cache_dir in sorted(MODELS_ROOT.glob("models--*")):
            if not cache_dir.is_dir():
                continue
            repo_id = cache_dir.name.removeprefix("models--").replace("--", "/")
            if repo_id in seen:
                continue
            entries.append(
                {
                    "repo_id": repo_id,
                    "path": str(cache_dir),
                    "seeded": True,
                    "layout": "hub-cache-dir",
                }
            )

    summary["models"] = entries
    for item in entries:
        print(f"model: {item}")
    if not entries:
        print("No models found on volume.")
    return summary


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DEFAULT_DATASETS),
    models: str = ",".join(DEFAULT_MODELS),
    repo_id: str | None = None,
    force: bool = False,
    revision: str | None = None,
    hub_cache_layout: bool = False,
    list_only: bool = False,
):
    """Commit MMAR datasets and/or model weights into the shared Modal Volume.

    Args:
        datasets: ``mmar`` or ``all``. Pass ``none`` to skip datasets.
        models: Comma-separated aliases (af3, af-next-think, af2, qwen3-omni, ...)
            or Hub repo ids.
            Pass ``eval`` / ``mmar-eval`` for the ``run_experiment.py`` suite.
            Pass ``none`` to skip models. Ignored when ``repo_id`` is set.
        repo_id: Optional single Hub repo id (overrides --models when set).
        force: Re-download / overwrite even if data already exists.
        revision: Optional git revision / commit pin for snapshot_download.
        hub_cache_layout: Store models in HF hub cache format under /cache/models
            instead of /cache/models/<repo_id>.
        list_only: Only print what is already on the Volume.
    """
    if list_only:
        list_volume.remote()
        return

    results: list[dict] = []

    datasets_arg = _parse_noneable(datasets)
    if datasets_arg is not None:
        wanted = {item.strip().lower() for item in datasets_arg.split(",") if item.strip()}
        if "all" in wanted:
            wanted = {"mmar"}

        unknown = wanted - {"mmar"}
        if unknown:
            raise SystemExit(f"Unknown datasets: {sorted(unknown)}")
        if not wanted:
            raise SystemExit("Pass at least one dataset via --datasets (or none to skip)")

        # Use .spawn().get() (not .remote()) so `modal run --detach` keeps
        # long downloads alive after the local client disconnects.
        if "mmar" in wanted:
            results.append(seed_mmar.spawn(force=force).get())

    if repo_id:
        targets = expand_seed_targets([repo_id])
    else:
        models_arg = _parse_noneable(models)
        if models_arg is None:
            targets = []
        else:
            raw = [item.strip() for item in models_arg.split(",") if item.strip()]
            if not raw:
                raise SystemExit("Pass at least one model via --models or --repo-id (or none to skip)")
            if any(item.lower() == "all" for item in raw):
                targets = expand_seed_targets(list(ALL_MODELS))
            elif any(item.lower() in {"eval", "mmar-eval"} for item in raw):
                targets = expand_seed_targets(list(EVAL_MODELS))
            else:
                targets = expand_seed_targets(raw)

    # De-dupe while preserving order.
    targets = list(dict.fromkeys(targets))
    for target in targets:
        call = seed_model.spawn(
            repo_id=target,
            force=force,
            revision=revision,
            hub_cache_layout=hub_cache_layout,
        )
        print(f"Spawned seed_model({target}) call_id={call.object_id}")
        results.append(call.get())

    if not results:
        raise SystemExit(
            "Nothing to seed. Pass --datasets and/or --models (or --repo-id), "
            "or use --list-only."
        )

    print("Done:", results)
    list_volume.remote()
