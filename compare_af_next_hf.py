"""Compare local AudioFlamingoNextForConditionalGeneration vs HuggingFace MusicFlamingo.

Loads both models with the same AF-Next checkpoint, runs a short audio+text
forward pass (eager attention), and reports:

  - LM layer attention diffs
  - output logit diffs / argmax match rate

Usage:

    uv run python compare_af_next_hf.py \\
        --model_dir models/audio-flamingo-next-think-hf \\
        --dtype bfloat16 --sequential

    # Optional: path to a mono wav (otherwise synthetic mel features are used)
    uv run python compare_af_next_hf.py --audio_path path/to.wav
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoProcessor

from audio_flamingo_next import AudioFlamingoNextForConditionalGeneration
from audio_flamingo_runtime import cast_model_floating_tensors, drop_empty_audio_windows


@dataclass
class TensorDiff:
    max_abs: float
    mean_abs: float
    cosine: float

    def __str__(self) -> str:
        return (
            f"max_abs={self.max_abs:.6e}  mean_abs={self.mean_abs:.6e}  "
            f"cosine={self.cosine:.8f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_dir",
        type=str,
        default="models/audio-flamingo-next-think-hf",
        help="Local snapshot of nvidia/audio-flamingo-next-think-hf (or instruct).",
    )
    p.add_argument(
        "--model_id",
        type=str,
        default="nvidia/audio-flamingo-next-think-hf",
        help="Hub id used when model_dir is missing (download into model_dir).",
    )
    p.add_argument("--audio_path", type=str, default=None)
    p.add_argument("--prompt", type=str, default="Transcribe the input speech.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu / mps / cuda. Default: cuda > mps > cpu.",
    )
    p.add_argument(
        "--sequential",
        action="store_true",
        help="Run one model at a time to reduce peak memory.",
    )
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--cosine_atol", type=float, default=0.999)
    p.add_argument(
        "--max_layers",
        type=int,
        default=None,
        help="Only compare the first N attention layers (debug).",
    )
    p.add_argument(
        "--synthetic_frames",
        type=int,
        default=3000,
        help="Mel frames for synthetic audio (3000 = 30s @ hop 160 / 16kHz).",
    )
    return p.parse_args()


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float32


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> TensorDiff:
    a32 = a.detach().float().reshape(-1)
    b32 = b.detach().float().reshape(-1)
    if a32.numel() != b32.numel():
        raise ValueError(f"Shape mismatch for diff: {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (a32 - b32).abs()
    # Normalize before the dot product so large tensors don't overflow float32.
    a_n = a32.norm()
    b_n = b32.norm()
    if a_n > 0 and b_n > 0:
        cosine = float(((a32 / a_n) @ (b32 / b_n)).clamp(-1.0, 1.0).item())
    else:
        cosine = 1.0
    return TensorDiff(
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        cosine=cosine,
    )


def argmax_match_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item())


def ensure_model_dir(model_dir: Path, model_id: str) -> Path:
    weights = model_dir / "model.safetensors"
    index = model_dir / "model.safetensors.index.json"
    if model_dir.is_dir() and (weights.is_file() or index.is_file()):
        return model_dir
    print(f"Weights not found under {model_dir}; downloading {model_id} ...")
    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        ignore_patterns=["*.docx", "*.webp", ".gitattributes"],
    )
    return model_dir


def build_inputs(
    processor,
    *,
    prompt: str,
    audio_path: str | None,
    synthetic_frames: int,
    seed: int,
    device: torch.device,
    feature_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if audio_path:
        conversation = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "path": audio_path},
                    ],
                }
            ]
        ]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
    else:
        # Synthetic mel: (1, 128, T) with a full-ones mask. Build text with the
        # matching number of <sound> placeholders via the processor tokenizer.
        g = torch.Generator(device="cpu").manual_seed(seed)
        input_features = torch.randn(1, 128, synthetic_frames, generator=g, dtype=torch.float32)
        input_features_mask = torch.ones(1, synthetic_frames, dtype=torch.bool)

        # Encoder post length after conv2 + avgpool: ((T-1)//2+1 - 2)//2 + 1
        mid = (synthetic_frames - 1) // 2 + 1
        post = (mid - 2) // 2 + 1
        audio_token = getattr(processor, "audio_token", "<sound>")
        audio_bos = getattr(processor, "audio_bos_token", "<|sound_bos|>")
        audio_eos = getattr(processor, "audio_eos_token", "<|sound_eos|>")
        sound_span = audio_bos + (audio_token * post) + audio_eos
        text = f"<|im_start|>user\n{prompt}{sound_span}<|im_end|>\n<|im_start|>assistant\n"
        tok = processor.tokenizer
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        inputs = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask", torch.ones_like(enc["input_ids"])),
            "input_features": input_features,
            "input_features_mask": input_features_mask,
        }

    inputs = drop_empty_audio_windows(inputs)
    prepared = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            prepared[key] = value
            continue
        if key == "input_features":
            prepared[key] = value.to(device=device, dtype=feature_dtype)
        else:
            prepared[key] = value.to(device=device)
    return prepared


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def run_local(model, inputs: dict) -> dict[str, torch.Tensor | tuple]:
    out = model(
        input_ids=inputs["input_ids"],
        input_features=inputs.get("input_features"),
        input_features_mask=inputs.get("input_features_mask"),
        attention_mask=inputs.get("attention_mask"),
        output_attentions=True,
        output_hidden_states=True,
    )
    return {
        "logits": out["logits"].detach().cpu(),
        "attentions": tuple(a.detach().cpu() for a in out["attentions"]),
        "hidden_states": tuple(h.detach().cpu() for h in out["hidden_states"]),
        "audio_hidden_states": (
            out["audio_hidden_states"].detach().cpu()
            if out["audio_hidden_states"] is not None
            else None
        ),
    }


@torch.no_grad()
def run_hf(model, inputs: dict) -> dict[str, torch.Tensor | tuple]:
    out = model(
        input_ids=inputs["input_ids"],
        input_features=inputs.get("input_features"),
        input_features_mask=inputs.get("input_features_mask"),
        attention_mask=inputs.get("attention_mask"),
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    attentions = out.attentions
    if attentions is None:
        attentions = getattr(out, "decoder_attentions", None)
    return {
        "logits": out.logits.detach().cpu(),
        "attentions": tuple(a.detach().cpu() for a in attentions),
        "hidden_states": tuple(h.detach().cpu() for h in out.hidden_states),
        "audio_hidden_states": (
            out.audio_hidden_states.detach().cpu()
            if getattr(out, "audio_hidden_states", None) is not None
            else None
        ),
    }


def load_local(model_dir: Path, dtype: torch.dtype, device: torch.device):
    print(f"Loading local AudioFlamingoNext from {model_dir} ...")
    return AudioFlamingoNextForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, device=device, strict=True
    )


def load_hf(model_dir: Path, dtype: torch.dtype, device: torch.device):
    print(f"Loading HuggingFace MusicFlamingo from {model_dir} ...")
    kwargs = {
        "dtype": dtype,
        "attn_implementation": "eager",
    }
    if device.type == "cuda":
        kwargs["device_map"] = {"": device}
    else:
        kwargs["device_map"] = "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), **kwargs)
    cast_model_floating_tensors(model, dtype)
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model


def print_report(
    logits_diff: TensorDiff,
    logits_argmax: float,
    attn_diffs: list[tuple[str, TensorDiff]],
    audio_diff: TensorDiff | None,
    atol: float,
    cosine_atol: float,
) -> bool:
    print("\n=== Logits ===")
    print(logits_diff)
    print(f"argmax match rate: {logits_argmax:.6f}")

    if audio_diff is not None:
        print("\n=== Audio projected embeds ===")
        print(audio_diff)

    print("\n=== Layer attentions (LM) ===")
    attn_ok = True
    for name, diff in attn_diffs:
        ok = diff.max_abs <= atol and diff.cosine >= cosine_atol
        flag = "OK" if ok else "DIFF"
        print(f"  [{flag}] {name:>16}: {diff}")
        attn_ok = attn_ok and ok

    logits_ok = logits_diff.max_abs <= atol and logits_argmax == 1.0
    print("\n=== Summary ===")
    if logits_ok and attn_ok:
        print(
            f"PASS: logits max_abs <= {atol:g}, argmax match, "
            f"attentions within atol/cosine>={cosine_atol:g}."
        )
        return True

    reasons = []
    if not logits_ok:
        reasons.append("logits")
    if not attn_ok:
        reasons.append("attentions")
    print(f"FAIL: mismatch in {', '.join(reasons)}.")
    return False


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    dtype = torch_dtype(args.dtype)
    atol = args.atol if args.atol is not None else (2.0 if dtype == torch.bfloat16 else 1e-3)
    model_dir = ensure_model_dir(Path(args.model_dir), args.model_id)

    print(f"model_dir={model_dir}")
    print(f"device={device}  dtype={dtype}  atol={atol:g}  sequential={args.sequential}")

    processor = AutoProcessor.from_pretrained(str(model_dir))
    inputs = build_inputs(
        processor,
        prompt=args.prompt,
        audio_path=args.audio_path,
        synthetic_frames=args.synthetic_frames,
        seed=args.seed,
        device=device,
        feature_dtype=dtype,
    )
    print(
        f"input_ids={tuple(inputs['input_ids'].shape)}  "
        f"features={tuple(inputs['input_features'].shape)}  "
        f"n_sound={(inputs['input_ids'] == 151667).sum().item()}"
    )

    if args.sequential:
        local = load_local(model_dir, dtype, device)
        local_out = run_local(local, inputs)
        free_model(local)

        hf = load_hf(model_dir, dtype, device)
        # Move inputs onto HF device (may differ if HF stayed on cpu via device_map)
        hf_device = next(hf.parameters()).device
        hf_inputs = {
            k: v.to(hf_device) if torch.is_tensor(v) else v for k, v in inputs.items()
        }
        if "input_features" in hf_inputs:
            hf_inputs["input_features"] = hf_inputs["input_features"].to(dtype=dtype)
        hf_out = run_hf(hf, hf_inputs)
        free_model(hf)
    else:
        local = load_local(model_dir, dtype, device)
        hf = load_hf(model_dir, dtype, device)
        local_out = run_local(local, inputs)
        hf_out = run_hf(hf, inputs)
        free_model(local)
        free_model(hf)

    logits_diff = tensor_diff(local_out["logits"], hf_out["logits"])
    logits_argmax = argmax_match_rate(local_out["logits"], hf_out["logits"])

    audio_diff = None
    if local_out["audio_hidden_states"] is not None and hf_out["audio_hidden_states"] is not None:
        audio_diff = tensor_diff(local_out["audio_hidden_states"], hf_out["audio_hidden_states"])

    n_layers = len(local_out["attentions"])
    if len(hf_out["attentions"]) != n_layers:
        raise RuntimeError(
            f"Attention layer count mismatch: local={n_layers} hf={len(hf_out['attentions'])}"
        )
    limit = n_layers if args.max_layers is None else min(n_layers, args.max_layers)
    attn_diffs = []
    for i in range(limit):
        attn_diffs.append((f"layer_{i}", tensor_diff(local_out["attentions"][i], hf_out["attentions"][i])))

    ok = print_report(logits_diff, logits_argmax, attn_diffs, audio_diff, atol, args.cosine_atol)

    report = {
        "logits": {
            "max_abs": logits_diff.max_abs,
            "mean_abs": logits_diff.mean_abs,
            "cosine": logits_diff.cosine,
            "argmax_match": logits_argmax,
        },
        "audio_embeds": None
        if audio_diff is None
        else {
            "max_abs": audio_diff.max_abs,
            "mean_abs": audio_diff.mean_abs,
            "cosine": audio_diff.cosine,
        },
        "attentions": {
            name: {"max_abs": d.max_abs, "mean_abs": d.mean_abs, "cosine": d.cosine}
            for name, d in attn_diffs
        },
        "pass": ok,
    }
    out_path = Path("outputs") / "af_next_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
