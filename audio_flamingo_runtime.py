"""Shared Audio Flamingo model I/O helpers for Modal MMAR evals."""

from __future__ import annotations

import os
from pathlib import Path

from modal_cache import MODELS_ROOT, volume


def torch_dtype_value(torch_module, dtype_name):
    return {
        "auto": "auto",
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[dtype_name]


def generation_kwargs(args, *, extra: dict | None = None):
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
    }
    if extra:
        kwargs.update(extra)
    return {key: value for key, value in kwargs.items() if value is not None}


def model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def model_param_dtype(model):
    return next(model.parameters()).dtype


def unwrap_model(model):
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    return model


def audio_tower_dtype(model):
    """Dtype of the Whisper-style audio encoder (may differ from the LLM)."""
    root = unwrap_model(model)
    inner = getattr(root, "model", root)
    tower = getattr(inner, "audio_tower", None) or getattr(root, "audio_tower", None)
    if tower is None:
        return model_param_dtype(model)
    return next(tower.parameters()).dtype


def cast_floating_state_dict(state_dict, dtype):
    import torch

    out = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.is_floating_point():
            out[key] = value.to(dtype=dtype)
        else:
            out[key] = value
    return out


def cast_model_floating_tensors(model, dtype):
    """Cast parameters/buffers to ``dtype`` in-place without disturbing device_map."""
    import torch

    if dtype is None or dtype == "auto":
        return model
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None or not param.is_floating_point() or param.dtype == dtype:
                continue
            module._parameters[name] = torch.nn.Parameter(
                param.data.to(dtype=dtype),
                requires_grad=param.requires_grad,
            )
        for name, buf in list(module._buffers.items()):
            if buf is None or not torch.is_tensor(buf):
                continue
            if not buf.is_floating_point() or buf.dtype == dtype:
                continue
            module._buffers[name] = buf.to(dtype=dtype)
    return model


def drop_empty_audio_windows(inputs, model=None):
    """Drop audio windows whose encoder post-length would be 0.

    AF-Next / MusicFlamingo builds per-window timestamps and indexes them by
    audio-sample spans derived from placeholder tokens. A near-empty trailing
    window (e.g. mask_sum=1 -> post_length=0) still occupies a feature row, so
    ``sample_indices`` can equal ``n_audio_samples`` and OOB on
    ``sample_start_rows[sample_indices]``. Tokens already omit empty windows
    (token count == sum(post_lengths)), so filtering feature rows is enough.
    """
    import torch

    if "input_features" not in inputs or "input_features_mask" not in inputs:
        return inputs

    features = inputs["input_features"]
    mask = inputs["input_features_mask"]
    if features.ndim != 3 or mask.ndim != 2 or features.shape[0] != mask.shape[0]:
        return inputs

    lengths = mask.sum(dim=-1).to(torch.long)
    post_lengths = None
    root = unwrap_model(model) if model is not None else None
    inner = getattr(root, "model", root) if root is not None else None
    tower = getattr(inner, "audio_tower", None) if inner is not None else None
    if tower is not None and hasattr(tower, "_get_feat_extract_output_lengths"):
        _, post_lengths = tower._get_feat_extract_output_lengths(lengths)
    else:
        # Mirror AudioFlamingo3Encoder._get_feat_extract_output_lengths.
        mid = (lengths - 1) // 2 + 1
        post_lengths = (mid - 2) // 2 + 1

    keep = post_lengths > 0
    if bool(keep.all().item()):
        return inputs

    kept = int(keep.sum().item())
    if kept == 0:
        raise RuntimeError(
            "All audio windows have zero encoder post-length; cannot run AF-Next."
        )

    print(
        f"Dropping {int(features.shape[0]) - kept} empty audio window(s) "
        f"(mask_sums={lengths.detach().cpu().tolist()}, "
        f"post_lengths={post_lengths.detach().cpu().tolist()})"
    )
    out = dict(inputs)
    out["input_features"] = features[keep]
    out["input_features_mask"] = mask[keep]
    return out


def prepare_model_inputs(inputs, model):
    """Move processor outputs onto the model device with per-tower dtypes.

    The audio encoder conv stack must see ``input_features`` in the encoder's
    dtype. Casting *all* floats to the LLM dtype (or leaving features as float32
    against a bf16 encoder) triggers Float/BFloat16 mismatches.
    Masks stay on-device without a dtype cast.
    """
    inputs = drop_empty_audio_windows(inputs, model=model)
    device = model_input_device(model)
    feature_dtype = audio_tower_dtype(model)
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


def resolve_model_dir(model_id: str, local_model_dir: str | None) -> str:
    """Prefer a seeded Volume snapshot; otherwise download into the models subpath."""
    from huggingface_hub import snapshot_download

    if local_model_dir:
        path = Path(local_model_dir).expanduser()
        if path.is_dir() and any(path.iterdir()):
            return str(path.resolve())
        raise SystemExit(f"local_model_dir not found or empty: {path}")

    seeded = MODELS_ROOT / model_id
    if seeded.is_dir() and (
        (seeded / "config.json").exists()
        or any(seeded.glob("*.safetensors"))
        or any(seeded.rglob("*.safetensors"))
    ):
        print(f"Using seeded model snapshot at {seeded}")
        return str(seeded)

    print(f"Model not found at {seeded}; downloading {model_id} onto Volume ...")
    seeded.parent.mkdir(parents=True, exist_ok=True)
    # Keep think/*.bin (non_lora_trainables); skip redundant full pytorch dumps.
    snapshot_download(
        repo_id=model_id,
        local_dir=str(seeded),
        token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.pt", "*.gguf", "*.onnx", "*.h5", "pytorch_model*.bin"],
    )
    marker = seeded / ".seed_complete"
    marker.write_text("ok\n", encoding="utf-8")
    volume.commit()
    return str(seeded)


def processor_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def format_raw_token(tokenizer, token_id: int) -> str:
    """Surface the vocabulary piece for one id (specials kept as named tokens)."""
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    if token_id in special_ids:
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece is not None:
            return str(piece)
    piece = tokenizer.convert_ids_to_tokens(token_id)
    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    if piece is not None and (decoded == "" or decoded.isspace()):
        return str(piece)
    if decoded:
        return decoded
    if piece is not None:
        return str(piece)
    return f"[{token_id}]"


def build_raw_tokens(tokenizer, token_ids, role: str) -> list[dict]:
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    tokens = []
    for token_id in token_ids:
        tid = int(token_id)
        tokens.append(
            {
                "id": tid,
                "token": format_raw_token(tokenizer, tid),
                "role": role,
                "special": tid in special_ids,
            }
        )
    return tokens


def generate_batch(
    model,
    processor,
    samples,
    args,
    *,
    build_prompt,
    parse_output,
    generation_extra: dict | None = None,
):
    """Run chat-template generation for a batch of MMAR samples."""
    import torch

    conversations = []
    for sample in samples:
        prompt_text = build_prompt(sample)
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "audio", "path": sample["audio_path"]},
                    ],
                }
            ]
        )

    inputs = prepare_model_inputs(
        processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ),
        model,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, **generation_kwargs(args, extra=generation_extra)
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, prompt_len:]
    decoded_clean = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    tokenizer = processor_tokenizer(processor)
    attention_mask = inputs.get("attention_mask")
    results = []
    for index, (sample, clean_output) in enumerate(zip(samples, decoded_clean)):
        if attention_mask is not None:
            mask = attention_mask[index].bool()
            input_ids = inputs["input_ids"][index][mask].tolist()
        else:
            input_ids = inputs["input_ids"][index].tolist()
        gen_ids = generated_ids[index].tolist()
        raw_tokens = build_raw_tokens(tokenizer, input_ids, "input") + build_raw_tokens(
            tokenizer, gen_ids, "generated"
        )
        full_ids = input_ids + gen_ids
        raw_text = tokenizer.decode(full_ids, skip_special_tokens=False)

        thinking_prediction, answer_prediction = parse_output(
            clean_output,
            sample["choices"],
        )
        results.append(
            {
                "model_output": raw_text,
                "raw_tokens": raw_tokens,
                "thinking_prediction": thinking_prediction,
                "answer_prediction": answer_prediction,
            }
        )
    return results
