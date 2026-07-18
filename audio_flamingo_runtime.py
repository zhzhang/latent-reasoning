"""Shared Audio Flamingo model I/O helpers for Modal MMAR evals."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# AF3 / AF-Next text stack (Qwen2).
DEFAULT_NUM_LAYERS = 28
DEFAULT_NUM_HEADS = 28
ATTENTION_BYTES_PER_ELEM = 2  # float16
MAX_ATTENTION_ARTIFACT_BYTES = 2 * 1024**3  # 2 GiB


def torch_dtype_value(torch_module, dtype_name):
    return {
        "auto": "auto",
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[dtype_name]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible greedy decode."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def estimate_attention_bytes(
    prompt_len: int,
    gen_len: int,
    *,
    layers: int = DEFAULT_NUM_LAYERS,
    heads: int = DEFAULT_NUM_HEADS,
    bytes_per_elem: int = ATTENTION_BYTES_PER_ELEM,
) -> int:
    """Packed size for per-generated-token layer×head attention (ragged keys)."""
    if prompt_len < 0 or gen_len < 0:
        raise ValueError("prompt_len and gen_len must be non-negative")
    # Σ_{t=1..T} (P + t) = T*P + T*(T+1)/2
    key_positions = gen_len * prompt_len + gen_len * (gen_len + 1) // 2
    return int(key_positions * layers * heads * bytes_per_elem)


def attention_artifact_id(sample_id: str) -> str:
    """Filesystem-safe id for attention artifact filenames."""
    cleaned = re.sub(r"[^\w.\-]+", "_", sample_id.strip())
    return cleaned or "sample"


def compare_generated_token_ids(
    generated_ids: list[int],
    stored_raw_tokens: list[dict] | None,
) -> dict:
    """Compare newly generated ids to stored ``raw_tokens`` with role=generated."""
    stored: list[int] = []
    if stored_raw_tokens:
        stored = [
            int(tok["id"])
            for tok in stored_raw_tokens
            if tok.get("role") == "generated"
        ]
    match = generated_ids == stored
    mismatch_index = None
    if not match:
        limit = min(len(generated_ids), len(stored))
        mismatch_index = next(
            (i for i in range(limit) if generated_ids[i] != stored[i]),
            limit if len(generated_ids) != len(stored) else None,
        )
    return {
        "token_match": match,
        "mismatch_index": mismatch_index,
        "generated_len": len(generated_ids),
        "stored_generated_len": len(stored),
    }


def pack_decoder_attentions_for_generated(
    attentions,
    *,
    prompt_len: int,
    gen_len: int,
    batch_index: int = 0,
):
    """Slice full-seq decoder attentions into per-generated-token (L, H, K) arrays.

    ``attentions`` is the HF forward tuple: one ``(batch, heads, seq, seq)``
    tensor per decoder layer.
    """
    import numpy as np

    if not attentions:
        raise ValueError("Model returned no attentions; use attn_implementation=eager")

    packed = []
    for step in range(gen_len):
        query_pos = prompt_len + step
        key_len = query_pos + 1
        layers = []
        for layer_attn in attentions:
            # (H, key_len)
            row = (
                layer_attn[batch_index, :, query_pos, :key_len]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float16, copy=False)
            )
            layers.append(row)
        packed.append(np.stack(layers, axis=0))
    return packed


def pack_generate_step_attentions(step_attentions, *, batch_index: int = 0):
    """Pack one generate-step attention tuple into (L, H, key_len) float16."""
    import numpy as np

    layers = []
    for layer_attn in step_attentions:
        # Prefer the last query row (prefill step has q_len=prompt_len).
        row = layer_attn[batch_index, :, -1, :].detach().float().cpu().numpy()
        layers.append(row.astype(np.float16, copy=False))
    return np.stack(layers, axis=0)


def save_attention_artifact(
    run_dir: Path | str,
    sample_id: str,
    attentions: list,
    meta: dict,
) -> dict:
    """Write ``attentions/<id>.npz`` + ``attentions/<id>.json`` under ``run_dir``."""
    import numpy as np

    run_path = Path(run_dir)
    attn_dir = run_path / "attentions"
    attn_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = attention_artifact_id(sample_id)
    npz_path = attn_dir / f"{artifact_id}.npz"
    json_path = attn_dir / f"{artifact_id}.json"

    arrays = {f"t{i}": np.asarray(arr) for i, arr in enumerate(attentions)}
    if not arrays:
        raise ValueError("No attention steps to save")
    np.savez_compressed(npz_path, **arrays)

    payload = {
        **meta,
        "sample_id": sample_id,
        "artifact_id": artifact_id,
        "npz_path": str(npz_path),
        "num_steps": len(attentions),
        "num_layers": int(attentions[0].shape[0]),
        "num_heads": int(attentions[0].shape[1]),
        "dtype": "float16",
        "layout": "per_generated_token: (num_layers, num_heads, key_len)",
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_attention_meta(run_dir: Path | str, sample_id: str) -> dict | None:
    path = Path(run_dir) / "attentions" / f"{attention_artifact_id(sample_id)}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_attention_vector(
    run_dir: Path | str,
    sample_id: str,
    *,
    gen_index: int,
    layer: int,
    head: int,
) -> list[float]:
    """Load one (generated-token, layer, head) attention vector as plain floats."""
    import numpy as np

    artifact_id = attention_artifact_id(sample_id)
    npz_path = Path(run_dir) / "attentions" / f"{artifact_id}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Attention artifact not found: {npz_path}")
    with np.load(npz_path) as data:
        key = f"t{gen_index}"
        if key not in data:
            raise KeyError(f"Missing attention step {gen_index} in {npz_path.name}")
        arr = data[key]
        if layer < 0 or layer >= arr.shape[0]:
            raise IndexError(f"layer {layer} out of range for shape {arr.shape}")
        if head < 0 or head >= arr.shape[1]:
            raise IndexError(f"head {head} out of range for shape {arr.shape}")
        return arr[layer, head].astype(np.float32).tolist()


def load_attention_token_layer_matrix(
    run_dir: Path | str,
    sample_id: str,
    *,
    gen_index: int,
    head: int,
) -> dict:
    """Load attention over keys × layers for one generated token and head.

    Returns a matrix shaped ``(key_len, num_layers)`` so callers can render
    tokens as rows and layers as columns.
    """
    import numpy as np

    artifact_id = attention_artifact_id(sample_id)
    npz_path = Path(run_dir) / "attentions" / f"{artifact_id}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Attention artifact not found: {npz_path}")
    with np.load(npz_path) as data:
        key = f"t{gen_index}"
        if key not in data:
            raise KeyError(f"Missing attention step {gen_index} in {npz_path.name}")
        arr = data[key]
        if head < 0 or head >= arr.shape[1]:
            raise IndexError(f"head {head} out of range for shape {arr.shape}")
        # arr: (num_layers, num_heads, key_len) -> (key_len, num_layers)
        matrix = arr[:, head, :].astype(np.float32).T
        return {
            "matrix": matrix.tolist(),
            "key_len": int(matrix.shape[0]),
            "num_layers": int(matrix.shape[1]),
            "num_heads": int(arr.shape[1]),
        }


SOUND_TOKEN = "<sound>"
ATTENTION_AUDIO_REDUCE_OPS = ("avg", "sum", "max")


def audio_token_indices(raw_tokens: list[dict] | None) -> list[int]:
    """Return key positions of ``<sound>`` audio tokens in the full sequence."""
    if not raw_tokens:
        return []
    return [
        i
        for i, tok in enumerate(raw_tokens)
        if str(tok.get("token") or "") == SOUND_TOKEN
    ]


def load_attention_audio_gen_layer_matrix(
    run_dir: Path | str,
    sample_id: str,
    *,
    audio_indices: list[int],
    reduce: str = "avg",
) -> dict:
    """Load generated-token × layer matrix of audio attention probability mass.

    Stored attentions are post-softmax weights (each head sums to ~1 over keys).
    For each generated step and layer we first sum over audio key positions to
    get per-head probability mass on audio in ``[0, 1]``, then reduce across
    heads with ``avg``, ``sum``, or ``max``.

    Returns a matrix shaped ``(num_steps, num_layers)``.
    """
    import numpy as np

    op = (reduce or "avg").strip().lower()
    if op not in ATTENTION_AUDIO_REDUCE_OPS:
        raise ValueError(
            f"reduce must be one of {ATTENTION_AUDIO_REDUCE_OPS}, got {reduce!r}"
        )
    if not audio_indices:
        raise ValueError("No audio token indices provided")

    artifact_id = attention_artifact_id(sample_id)
    npz_path = Path(run_dir) / "attentions" / f"{artifact_id}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Attention artifact not found: {npz_path}")

    audio_idx = np.asarray(sorted(set(int(i) for i in audio_indices)), dtype=np.int64)
    rows: list[list[float]] = []
    num_layers = 0
    num_heads = 0
    with np.load(npz_path) as data:
        step = 0
        while f"t{step}" in data:
            arr = data[f"t{step}"]  # (L, H, K)
            num_layers = int(arr.shape[0])
            num_heads = int(arr.shape[1])
            key_len = int(arr.shape[2])
            valid = audio_idx[audio_idx < key_len]
            if valid.size == 0:
                rows.append([0.0] * num_layers)
                step += 1
                continue
            # Post-softmax probs → per-head mass on audio: (L, H)
            mass = arr[:, :, valid].astype(np.float32, copy=False).sum(axis=-1)
            if op == "avg":
                values = mass.mean(axis=-1)
            elif op == "sum":
                values = mass.sum(axis=-1)
            else:
                values = mass.max(axis=-1)
            rows.append(values.tolist())
            step += 1

    if not rows:
        raise KeyError(f"No attention steps in {npz_path.name}")

    return {
        "matrix": rows,
        "num_steps": len(rows),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_audio_keys": int(audio_idx.size),
        "reduce": op,
    }


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

    from modal_cache import MODELS_ROOT, volume

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


def _build_conversations(samples, build_prompt):
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
    return conversations


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

    conversations = _build_conversations(samples, build_prompt)
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


def generate_one_with_attentions(
    model,
    processor,
    sample,
    args,
    *,
    build_prompt,
    parse_output,
    generation_extra: dict | None = None,
    stored_raw_tokens: list[dict] | None = None,
    max_artifact_bytes: int = MAX_ATTENTION_ARTIFACT_BYTES,
):
    """Greedy-generate one sample, then capture layer-wise attentions via forward.

    Requires ``attn_implementation=\"eager\"`` (or another backend that returns
    attention weights). Uses a second teacher-forced forward over
    prompt+generation so each generated token has a clean query row.
    """
    import torch

    conversations = _build_conversations([sample], build_prompt)
    inputs = prepare_model_inputs(
        processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ),
        model,
    )

    prompt_len = int(inputs["input_ids"].shape[1])
    stored_gen_len = None
    if stored_raw_tokens:
        stored_gen_len = sum(
            1 for tok in stored_raw_tokens if tok.get("role") == "generated"
        )
    estimate_gen = stored_gen_len if stored_gen_len else int(args.max_new_tokens)
    estimated_bytes = estimate_attention_bytes(prompt_len, estimate_gen)
    if estimated_bytes > max_artifact_bytes:
        raise RuntimeError(
            f"Estimated attention artifact {estimated_bytes / 1e9:.2f} GB exceeds "
            f"limit {max_artifact_bytes / 1e9:.2f} GB "
            f"(prompt_len={prompt_len}, gen_len≈{estimate_gen})."
        )

    gen_kwargs = generation_kwargs(args, extra=generation_extra)
    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated = outputs[:, prompt_len:]
    gen_ids = generated[0].tolist()
    gen_len = len(gen_ids)
    actual_bytes = estimate_attention_bytes(prompt_len, gen_len)
    if actual_bytes > max_artifact_bytes:
        raise RuntimeError(
            f"Attention artifact {actual_bytes / 1e9:.2f} GB exceeds "
            f"limit {max_artifact_bytes / 1e9:.2f} GB "
            f"(prompt_len={prompt_len}, gen_len={gen_len})."
        )

    # Teacher-force prompt + generated tokens to get per-position attentions.
    full_ids = torch.cat([inputs["input_ids"], generated], dim=1)
    forward_inputs = dict(inputs)
    forward_inputs["input_ids"] = full_ids
    if "attention_mask" in forward_inputs and forward_inputs["attention_mask"] is not None:
        ones = torch.ones_like(generated, dtype=forward_inputs["attention_mask"].dtype)
        forward_inputs["attention_mask"] = torch.cat(
            [forward_inputs["attention_mask"], ones], dim=1
        )

    with torch.inference_mode():
        forward_out = model(
            **forward_inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = getattr(forward_out, "attentions", None)
    if attentions is None:
        attentions = getattr(forward_out, "decoder_attentions", None)
    if attentions is None:
        raise RuntimeError(
            "Model forward returned no attentions. "
            "Load with attn_implementation='eager'."
        )

    packed = pack_decoder_attentions_for_generated(
        attentions,
        prompt_len=prompt_len,
        gen_len=gen_len,
        batch_index=0,
    )
    # Free GPU attention tensors promptly.
    del forward_out, attentions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = processor_tokenizer(processor)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        mask = attention_mask[0].bool()
        input_ids = inputs["input_ids"][0][mask].tolist()
    else:
        input_ids = inputs["input_ids"][0].tolist()

    # Align packed keys to unpadded input length when left/right padding differs.
    # For batch_size=1 MMAR capture, prompt_len from input_ids shape is correct.
    raw_tokens = build_raw_tokens(tokenizer, input_ids, "input") + build_raw_tokens(
        tokenizer, gen_ids, "generated"
    )
    clean_output = processor.batch_decode(generated, skip_special_tokens=True)[0]
    thinking_prediction, answer_prediction = parse_output(
        clean_output,
        sample["choices"],
    )
    full_text_ids = input_ids + gen_ids
    raw_text = tokenizer.decode(full_text_ids, skip_special_tokens=False)
    match_info = compare_generated_token_ids(gen_ids, stored_raw_tokens)

    return {
        "model_output": raw_text,
        "raw_tokens": raw_tokens,
        "thinking_prediction": thinking_prediction,
        "answer_prediction": answer_prediction,
        "attentions": packed,
        "prompt_len": prompt_len,
        "generated_ids": gen_ids,
        "estimated_bytes": actual_bytes,
        **match_info,
    }
