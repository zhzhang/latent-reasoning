"""Smoke-test HF MusicFlamingo (Audio Flamingo Next) forward I/O.

Checks that the native transformers implementation:
  1. Accepts ``input_features`` / ``input_features_mask``
  2. Returns decoder ``attentions`` when ``output_attentions=True``
  3. Returns ``hidden_states`` when ``output_hidden_states=True``

Uses synthetic mel features (no audio file) and ``attn_implementation="eager"``
so attention weights are materializable.

Usage:

    uv run modal run test_af_next_hf_outputs.py
    uv run modal run test_af_next_hf_outputs.py --synthetic-frames 3000
"""

from __future__ import annotations

import json

import modal

from modal_cache import VOLUME_MOUNT, hf_secret, mmar_eval_image, volume

DEFAULT_MODEL_ID = "nvidia/audio-flamingo-next-think-hf"
# AF-Next audio tower uses Whisper-style fixed positions (1500 after conv1),
# so mel length must be 3000 frames (30s @ 16kHz / hop 160).
DEFAULT_SYNTHETIC_FRAMES = 3000

image = mmar_eval_image()
app = modal.App("test-af-next-hf-outputs", image=image)


def _build_synthetic_inputs(processor, *, frames: int, seed: int, device, dtype):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    input_features = torch.randn(1, 128, frames, generator=g, dtype=torch.float32)
    input_features_mask = torch.ones(1, frames, dtype=torch.bool)

    mid = (frames - 1) // 2 + 1
    post = (mid - 2) // 2 + 1
    audio_token = getattr(processor, "audio_token", "<sound>")
    audio_bos = getattr(processor, "audio_bos_token", "<|sound_bos|>")
    audio_eos = getattr(processor, "audio_eos_token", "<|sound_eos|>")
    sound_span = audio_bos + (audio_token * post) + audio_eos
    text = (
        f"<|im_start|>user\nDescribe the audio.{sound_span}"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )
    enc = processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)

    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc.get(
            "attention_mask", torch.ones_like(enc["input_ids"])
        ).to(device),
        "input_features": input_features.to(device=device, dtype=dtype),
        "input_features_mask": input_features_mask.to(device),
    }


@app.function(
    gpu="L40S",
    timeout=60 * 30,
    memory=65536,
    volumes={VOLUME_MOUNT: volume},
    secrets=[hf_secret],
)
def check_outputs(
    model_id: str = DEFAULT_MODEL_ID,
    synthetic_frames: int = DEFAULT_SYNTHETIC_FRAMES,
    seed: int = 0,
) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoProcessor

    from audio_flamingo_runtime import cast_model_floating_tensors, resolve_model_dir

    model_dir = resolve_model_dir(model_id, None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        dtype=dtype,
        attn_implementation="eager",
        device_map={"": device} if device.type == "cuda" else "cpu",
    )
    # Audio-tower LayerNorms can remain float32 after dtype=bfloat16 load.
    cast_model_floating_tensors(model, dtype)
    model.eval()

    inputs = _build_synthetic_inputs(
        processor,
        frames=synthetic_frames,
        seed=seed,
        device=device,
        dtype=dtype,
    )

    accepts_input_features = "input_features" in inputs
    report: dict = {
        "model_class": type(model).__name__,
        "model_dir": model_dir,
        "accepts_input_features": accepts_input_features,
        "input_features_shape": list(inputs["input_features"].shape),
        "input_ids_shape": list(inputs["input_ids"].shape),
    }

    with torch.inference_mode():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            input_features_mask=inputs["input_features_mask"],
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = out.attentions
    if attentions is None:
        attentions = getattr(out, "decoder_attentions", None)
    hidden_states = out.hidden_states

    has_attentions = attentions is not None and len(attentions) > 0
    has_hidden_states = hidden_states is not None and len(hidden_states) > 0

    report.update(
        {
            "has_attentions": has_attentions,
            "n_attention_layers": len(attentions) if has_attentions else 0,
            "attention_shape": list(attentions[0].shape) if has_attentions else None,
            "has_hidden_states": has_hidden_states,
            "n_hidden_states": len(hidden_states) if has_hidden_states else 0,
            "hidden_state_shape": (
                list(hidden_states[0].shape) if has_hidden_states else None
            ),
            "has_logits": getattr(out, "logits", None) is not None,
            "logits_shape": (
                list(out.logits.shape) if getattr(out, "logits", None) is not None else None
            ),
            "pass": bool(
                accepts_input_features and has_attentions and has_hidden_states
            ),
        }
    )
    return report


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL_ID,
    synthetic_frames: int = DEFAULT_SYNTHETIC_FRAMES,
    seed: int = 0,
):
    report = check_outputs.remote(
        model_id=model_id,
        synthetic_frames=synthetic_frames,
        seed=seed,
    )
    print(json.dumps(report, indent=2))
    if not report.get("pass"):
        raise SystemExit(1)
    print(
        "OK: HF MusicFlamingo accepts input_features and returns "
        "attentions + hidden_states."
    )
