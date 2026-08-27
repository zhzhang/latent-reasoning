"""Smoke-test Step-Audio-2-mini-Think on one MMAR clip.

Standalone Modal app (not ``run_experiment.py``). Uses the official
Hugging Face ``StepAudio2`` path from https://github.com/stepfun-ai/Step-Audio2
and https://huggingface.co/stepfun-ai/Step-Audio-2-mini-Think.

Prereq: MMAR audio on the ``latent-reasoning`` volume
(``uv run modal run seed_volume.py --datasets mmar --models none``).
Weights download onto the volume on first run.

Usage::

    uv run modal run smoke_step_audio.py
    uv run modal run smoke_step_audio.py --question-id <mmar_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

from audio_flamingo_runtime import resolve_model_dir
from modal_cache import (
    DEFAULT_MMAR_DATA_ROOT,
    DEFAULT_MMAR_META,
    VOLUME_MOUNT,
    hf_secret,
    volume,
)
from mmar_common import ASSISTANT_THINK_OPEN, load_jsonl, resolve_path

MODEL_ID = "stepfun-ai/Step-Audio-2-mini-Think"
_STEP_AUDIO2_ROOT = "/opt/Step-Audio2"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("ffmpeg", "git")
    .uv_pip_install(
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "transformers==4.49.0",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "numpy",
        "accelerate==1.12.0",
        "einops",
        "onnxruntime",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .run_commands(
        f"git clone --depth 1 https://github.com/stepfun-ai/Step-Audio2.git {_STEP_AUDIO2_ROOT}"
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONPATH": _STEP_AUDIO2_ROOT,
        }
    )
    .add_local_python_source(
        "modal_cache",
        "mmar_common",
        "audio_flamingo_runtime",
    )
)

app = modal.App("smoke-step-audio-2-mini-think", image=image)


def _pick_item(question_id: str | None) -> dict:
    meta_path = Path(DEFAULT_MMAR_META)
    data_root = Path(DEFAULT_MMAR_DATA_ROOT)
    if not meta_path.is_file():
        raise SystemExit(
            f"MMAR meta missing at {meta_path}. Seed first:\n"
            "  uv run modal run seed_volume.py --datasets mmar --models none"
        )
    items = load_jsonl(meta_path)
    if question_id:
        wanted = str(question_id)
        match = next((item for item in items if str(item.get("id")) == wanted), None)
        if match is None:
            raise SystemExit(f"Question id not in MMAR meta: {wanted}")
        items = [match]
    for item in items:
        audio_path = resolve_path(data_root, item.get("audio_path") or "")
        if audio_path and Path(audio_path).is_file():
            return {**item, "audio_path": str(audio_path)}
    raise SystemExit(f"No MMAR wav found under {data_root / 'audio'}")


@app.function(
    gpu="A100-80GB",
    timeout=45 * 60,
    memory=65536,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
)
def smoke(question_id: str | None = None) -> dict:
    volume.reload()
    if _STEP_AUDIO2_ROOT not in sys.path:
        sys.path.insert(0, _STEP_AUDIO2_ROOT)
    from stepaudio2 import StepAudio2

    item = _pick_item(question_id)
    question = str(item.get("question") or "").strip()
    print(f"[smoke] {MODEL_ID} id={item.get('id')} audio={item['audio_path']}")
    print(f"[smoke] question={question}")

    local_id = resolve_model_dir(MODEL_ID, None)
    model = StepAudio2(local_id)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in audio analysis. Activate deep thinking: "
                "reason step by step about what you hear, then answer accurately."
            ),
        },
        {
            "role": "human",
            "content": [
                {"type": "audio", "audio": item["audio_path"]},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": f"\n{ASSISTANT_THINK_OPEN}", "eot": False},
    ]
    _, text, _ = model(
        messages,
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    raw = str(text or "")
    preview = raw if len(raw) <= 2000 else raw[:2000] + "\n…[truncated]"
    print(f"[smoke] raw_chars={len(raw)}")
    print(f"[smoke] raw=\n{preview}")
    if not raw.strip():
        raise SystemExit("Empty model_output — StepAudio2 text decode still failing")
    return {
        "id": item.get("id"),
        "question": question,
        "raw_chars": len(raw),
        "empty": False,
    }


@app.local_entrypoint()
def main(question_id: str | None = None):
    result = smoke.remote(question_id=question_id)
    print("Done:", result)
