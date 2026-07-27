"""Latent chain-of-thought decode for HF MusicFlamingo / Audio Flamingo Next.

The native ``MusicFlamingoForConditionalGeneration.generate`` always feeds
sampled token ids back through the embedding layer. Latent CoT instead feeds
remapped final-layer hidden states as ``inputs_embeds`` while inside
``<think>...</think>``. This module owns:

  - ridge remapping matrix compute / disk cache
  - a thin decode loop that drives a loaded HF model via ``forward``
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import load_file, save_file


LATENT_W_REMAP_FILENAME = "latent_w_remap.safetensors"
LATENT_W_REMAP_KEY = "weight"
LATENT_AVG_EMBED_NORM_KEY = "avg_embed_norm"
# Ridge penalty for (W_out^T W_out + λ I)^{-1}.
LATENT_W_REMAP_LAMBDA = 1e-4
# Floor for ||h @ W|| when applying 1/β at decode time.
_LATENT_H_NORM_EPS = 1e-8
# Any prior remap artifacts under the model dir (stable name + legacy variants).
_LATENT_W_REMAP_GLOB = "latent_w_remap*.safetensors"

# Hub / remapped state-dict keys for latent remapping matrices.
_LM_HEAD_WEIGHT_KEYS = (
    "lm_head.weight",
    "language_model.lm_head.weight",
)
_EMBED_WEIGHT_KEYS = (
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "language_model.embed_tokens.weight",
)


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _apply_repetition_penalty(
    logits: torch.Tensor, token_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """HF-style repetition penalty over all tokens seen so far."""
    if penalty is None or penalty == 1.0:
        return logits
    logits = logits.clone()
    for batch_idx in range(logits.shape[0]):
        unique = torch.unique(token_ids[batch_idx])
        score = logits[batch_idx, unique]
        logits[batch_idx, unique] = torch.where(
            score < 0, score * penalty, score / penalty
        )
    return logits


def _sample_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
) -> torch.Tensor:
    """Return next-token ids of shape (batch, 1)."""
    if not do_sample or temperature is None or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(
            remove, torch.finfo(sorted_logits.dtype).min
        )
        logits = torch.full_like(logits, torch.finfo(logits.dtype).min).scatter(
            -1, sorted_idx, sorted_logits
        )
    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _suffix_matches(generated: torch.LongTensor, suffix_ids: list[int]) -> torch.Tensor:
    """Return a (batch,) bool mask: True when each row ends with ``suffix_ids``."""
    if not suffix_ids:
        return torch.zeros(
            generated.shape[0], dtype=torch.bool, device=generated.device
        )
    suffix = torch.tensor(suffix_ids, device=generated.device, dtype=generated.dtype)
    tail = generated[:, -len(suffix_ids) :]
    return (tail == suffix.unsqueeze(0)).all(dim=1)


def latent_w_remap_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / LATENT_W_REMAP_FILENAME


def _is_latent_w_remap_cache(path: Path) -> bool:
    return path.name.startswith("latent_w_remap") and path.suffix == ".safetensors"


def clear_latent_w_remap_cache(model_dir: str | Path) -> list[Path]:
    """Remove any latent remap cache files under ``model_dir`` (idempotent)."""
    model_dir = Path(model_dir)
    removed: list[Path] = []
    if not model_dir.is_dir():
        return removed
    for path in sorted(model_dir.glob(_LATENT_W_REMAP_GLOB)):
        if not path.is_file():
            continue
        path.unlink()
        removed.append(path)
    return removed


def compute_w_remap_from_weights(
    w_out: torch.Tensor,
    w_in: torch.Tensor,
    *,
    ridge_lambda: float = LATENT_W_REMAP_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute unscaled ridge remapping and mean input-embed norm.

    Full map at decode time is::

        W = (W_out^T W_out + λ I)^{-1} W_out^T W_in
        h' = h @ W
        β = ||h'|| / mean_i(||W_in[i]||)
        remapped = (1/β) h'

    Returns ``(W, mean_i(||W_in[i]||))``. For tall ``W_out`` (V×H, V≫H) the
    solve is on an H×H Gram matrix — avoids ``torch.linalg.pinv`` on V×H.
    """
    w_out = w_out.float()
    w_in = w_in.float()
    hidden = w_out.shape[1]
    gram = w_out.T @ w_out
    if ridge_lambda != 0.0:
        gram = gram + ridge_lambda * torch.eye(
            hidden, device=gram.device, dtype=gram.dtype
        )
    ridge = torch.linalg.solve(gram, w_out.T @ w_in)
    avg_embed_norm = w_in.norm(dim=-1).mean()
    return ridge, avg_embed_norm.detach()


def apply_latent_w_remap(
    h: torch.Tensor,
    w_remap: torch.Tensor,
    avg_embed_norm: torch.Tensor | float,
) -> torch.Tensor:
    """Apply cached ridge remap with β from the post-remap hidden norm.

    ``β = ||h @ W|| / avg_embed_norm``, so the returned vector has norm
    ``avg_embed_norm``.
    """
    mapped = h.float() @ w_remap
    mapped_norm = mapped.norm(dim=-1, keepdim=True).clamp_min(_LATENT_H_NORM_EPS)
    scale = avg_embed_norm / mapped_norm
    return (mapped * scale).to(dtype=h.dtype)


def _compute_w_remap(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute remapping from a loaded model's lm_head and input embeddings."""
    return compute_w_remap_from_weights(
        model.lm_head.weight, model.get_input_embeddings().weight
    )


def load_latent_w_remap(
    model_dir: str | Path,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load cached ``(w_remap, avg_embed_norm)``, or ``None`` if absent/stale."""
    path = latent_w_remap_path(model_dir)
    if not path.is_file():
        # Drop any leftover legacy remap filenames.
        clear_latent_w_remap_cache(model_dir)
        return None
    tensors = load_file(str(path))
    if (
        LATENT_W_REMAP_KEY not in tensors
        or LATENT_AVG_EMBED_NORM_KEY not in tensors
    ):
        # Incomplete/old schema: wipe and treat as cache miss.
        clear_latent_w_remap_cache(model_dir)
        return None
    # Drop any leftover legacy remap filenames beside the canonical cache.
    for leftover in Path(model_dir).glob(_LATENT_W_REMAP_GLOB):
        if leftover != path and leftover.is_file():
            leftover.unlink()
    return tensors[LATENT_W_REMAP_KEY], tensors[LATENT_AVG_EMBED_NORM_KEY]


def save_latent_w_remap(
    model_dir: str | Path,
    w_remap: torch.Tensor,
    avg_embed_norm: torch.Tensor | float,
) -> Path:
    """Persist remapping cache, replacing any previous remap artifacts."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    clear_latent_w_remap_cache(model_dir)
    path = latent_w_remap_path(model_dir)
    avg = torch.as_tensor(avg_embed_norm, dtype=torch.float32).reshape(()).cpu()
    save_file(
        {
            LATENT_W_REMAP_KEY: w_remap.detach().cpu().contiguous(),
            LATENT_AVG_EMBED_NORM_KEY: avg.contiguous(),
        },
        str(path),
    )
    return path


def compute_and_cache_latent_w_remap(
    model_dir: str | Path,
    *,
    force: bool = False,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """Compute ridge remapping from checkpoint weights and cache on disk.

    Loads only the lm_head / embed_tokens tensors (not the full model). Suitable
    for the model-seeding container so eval containers can load the cache.
    Writing always replaces any prior remap cache files under ``model_dir``.
    """
    model_dir = Path(model_dir)
    out_path = latent_w_remap_path(model_dir)
    if not force:
        cached = load_latent_w_remap(model_dir)
        if cached is not None:
            w_remap, avg_embed_norm = cached
            return {
                "status": "skipped",
                "path": str(out_path),
                "shape": list(w_remap.shape),
                "avg_embed_norm": float(avg_embed_norm),
            }

    w_out, w_in = load_latent_remap_source_weights(model_dir)
    if device is not None:
        w_out = w_out.to(device=device)
        w_in = w_in.to(device=device)

    print(
        f"Computing latent w_remap (ridge λ={LATENT_W_REMAP_LAMBDA}) from {model_dir} "
        f"(W_out={tuple(w_out.shape)}, W_in={tuple(w_in.shape)}, "
        f"device={w_out.device}) ..."
    )
    w_remap, avg_embed_norm = compute_w_remap_from_weights(w_out, w_in)
    saved = save_latent_w_remap(model_dir, w_remap, avg_embed_norm)
    print(
        f"Cached latent w_remap at {saved} shape={tuple(w_remap.shape)} "
        f"avg_embed_norm={float(avg_embed_norm):.6g}"
    )
    return {
        "status": "ok",
        "path": str(saved),
        "shape": list(w_remap.shape),
        "avg_embed_norm": float(avg_embed_norm),
    }


def ensure_latent_w_remap(
    model,
    model_dir: str | Path | None = None,
    *,
    persist: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attach ``model._latent_w_remap`` / ``_latent_avg_embed_norm`` from cache."""
    cached_w = getattr(model, "_latent_w_remap", None)
    cached_norm = getattr(model, "_latent_avg_embed_norm", None)
    if cached_w is not None and cached_norm is not None:
        return cached_w, cached_norm

    w_remap: torch.Tensor | None = None
    avg_embed_norm: torch.Tensor | None = None
    if model_dir is not None:
        loaded = load_latent_w_remap(model_dir)
        if loaded is not None:
            w_remap, avg_embed_norm = loaded
            print(
                f"Loaded cached latent w_remap from {latent_w_remap_path(model_dir)} "
                f"shape={tuple(w_remap.shape)} "
                f"avg_embed_norm={float(avg_embed_norm):.6g}"
            )

    if w_remap is None or avg_embed_norm is None:
        print(
            f"Computing latent w_remap (ridge λ={LATENT_W_REMAP_LAMBDA}) ..."
        )
        w_remap, avg_embed_norm = _compute_w_remap(model)
        if persist and model_dir is not None:
            try:
                saved = save_latent_w_remap(model_dir, w_remap, avg_embed_norm)
                print(
                    f"Cached latent w_remap at {saved} shape={tuple(w_remap.shape)} "
                    f"avg_embed_norm={float(avg_embed_norm):.6g}"
                )
            except OSError as exc:
                print(
                    f"Warning: could not cache latent w_remap under {model_dir}: {exc}"
                )

    device = _model_device(model)
    model._latent_w_remap = w_remap.detach().to(device=device).contiguous()
    model._latent_avg_embed_norm = (
        torch.as_tensor(avg_embed_norm, dtype=torch.float32, device=device)
        .reshape(())
        .detach()
    )
    return model._latent_w_remap, model._latent_avg_embed_norm


def _hub_keys_for_remapped(remapped_key: str) -> tuple[str, ...]:
    """Possible hub/raw keys that remap onto ``remapped_key``."""
    keys = [remapped_key]
    if remapped_key.startswith("lm_head."):
        keys.append("language_model." + remapped_key)
    elif remapped_key.startswith("model.language_model."):
        suffix = remapped_key[len("model.language_model.") :]
        keys.append("language_model.model." + suffix)
        keys.append("language_model." + suffix)
    elif remapped_key.startswith("model."):
        keys.append(remapped_key[len("model.") :])
    return tuple(dict.fromkeys(keys))


def load_latent_remap_source_weights(
    model_dir: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load only lm_head + embed_tokens weights needed for latent remapping."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    wanted_remapped = {
        "lm_head.weight": _LM_HEAD_WEIGHT_KEYS[0],
        "model.language_model.embed_tokens.weight": _EMBED_WEIGHT_KEYS[0],
    }
    raw_to_remapped: dict[str, str] = {}
    for remapped in wanted_remapped:
        for raw_key in _hub_keys_for_remapped(remapped):
            raw_to_remapped[raw_key] = remapped
        raw_to_remapped[remapped] = remapped

    found: dict[str, torch.Tensor] = {}

    def _consume(path: Path, keys: set[str] | None = None) -> None:
        with safe_open(str(path), framework="pt") as handle:
            available = set(handle.keys())
            targets = available if keys is None else available & keys
            for key in targets:
                dest = raw_to_remapped.get(key)
                if dest is None or dest in found:
                    continue
                found[dest] = handle.get_tensor(key)

    single = model_dir / "model.safetensors"
    index_path = model_dir / "model.safetensors.index.json"
    if single.is_file():
        _consume(single)
    elif index_path.is_file():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_to_keys: dict[str, set[str]] = {}
        for raw_key, remapped in raw_to_remapped.items():
            if remapped in found:
                continue
            shard = weight_map.get(raw_key)
            if shard is None:
                continue
            shard_to_keys.setdefault(shard, set()).add(raw_key)
        for shard, keys in shard_to_keys.items():
            _consume(model_dir / shard, keys)
            if len(found) == len(wanted_remapped):
                break
    else:
        files = sorted(
            p
            for p in model_dir.glob("*.safetensors")
            if not _is_latent_w_remap_cache(p)
        )
        if not files:
            raise FileNotFoundError(f"No safetensors weights found under {model_dir}")
        for path in files:
            _consume(path)
            if len(found) == len(wanted_remapped):
                break

    missing = [k for k in wanted_remapped if k not in found]
    if missing:
        raise KeyError(
            f"Could not locate latent remapping source weights in {model_dir}: "
            f"missing {missing}"
        )
    return found["lm_head.weight"], found["model.language_model.embed_tokens.weight"]


def _last_hidden(outputs) -> torch.Tensor:
    """Post-norm final-layer hidden states from an HF CausalLM forward."""
    if getattr(outputs, "hidden_states", None) is not None:
        return outputs.hidden_states[-1]
    # Fall back if a custom/wrapper model exposes last_hidden_state.
    if getattr(outputs, "last_hidden_state", None) is not None:
        return outputs.last_hidden_state
    raise RuntimeError(
        "Forward returned neither hidden_states nor last_hidden_state; "
        "call with output_hidden_states=True"
    )


def _resolve_eos_pad(model, eos_token_id, pad_token_id) -> tuple[list[int], int | None]:
    generation_config = getattr(model, "generation_config", None)
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    if eos_token_id is None:
        if generation_config is not None and generation_config.eos_token_id is not None:
            eos_token_id = generation_config.eos_token_id
        elif text_config is not None:
            eos_token_id = getattr(text_config, "eos_token_id", None)
        else:
            eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if pad_token_id is None:
        if generation_config is not None and generation_config.pad_token_id is not None:
            pad_token_id = generation_config.pad_token_id
        elif text_config is not None:
            pad_token_id = getattr(text_config, "pad_token_id", None)
        else:
            pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if isinstance(eos_token_id, int):
        eos_ids = [eos_token_id]
    else:
        eos_ids = list(eos_token_id or [])
    return eos_ids, pad_token_id


@torch.inference_mode()
def latent_generate(
    model,
    *,
    input_ids: torch.LongTensor,
    input_features: torch.FloatTensor | None = None,
    input_features_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    eos_token_id: int | list[int] | None = None,
    pad_token_id: int | None = None,
    think_start_ids: list[int] | None = None,
    think_end_ids: list[int] | None = None,
    model_dir: str | Path | None = None,
    **kwargs: Any,
) -> SimpleNamespace:
    """Autoregressive latent-CoT decode against a native HF MusicFlamingo model.

    After ``think_start_ids``, feed remapped residuals as ``inputs_embeds`` until
    ``</think>`` is the argmax next token. Returns ``sequences`` and ``is_latent``
    masks compatible with ``audio_flamingo_runtime._unpack_generate_output``.
    """
    _ = kwargs  # tolerate HF generate extras
    if not think_start_ids or not think_end_ids:
        raise ValueError(
            "latent_generate requires non-empty think_start_ids and think_end_ids"
        )
    think_start_ids = [int(t) for t in think_start_ids]
    think_end_ids = [int(t) for t in think_end_ids]
    think_end_first = think_end_ids[0]
    eos_ids, pad_token_id = _resolve_eos_pad(model, eos_token_id, pad_token_id)

    w_remap, avg_embed_norm = ensure_latent_w_remap(model, model_dir, persist=False)
    embed_layer = model.get_input_embeddings()

    batch_size = input_ids.shape[0]
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    # Prefill (audio fused only when input_features + input_ids are both set).
    outputs = model(
        input_ids=input_ids,
        input_features=input_features,
        input_features_mask=input_features_mask,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        logits_to_keep=1,
    )
    past = outputs.past_key_values
    generated = input_ids
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    in_latent = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    is_latent = torch.zeros(
        batch_size,
        input_ids.shape[1],
        dtype=torch.bool,
        device=input_ids.device,
    )

    for _ in range(max_new_tokens):
        step_logits = outputs.logits[:, -1, :].float()
        argmax_ids = step_logits.argmax(dim=-1)

        latent_exit = in_latent & (argmax_ids == think_end_first)
        latent_continue = in_latent & ~latent_exit

        penalized = _apply_repetition_penalty(
            step_logits,
            generated,
            repetition_penalty if repetition_penalty is not None else 1.0,
        )
        next_token = _sample_next_token(
            penalized,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        if latent_continue.any():
            # Record argmax under the latent mask; next-step input still
            # uses the remapped hidden state, not this token's embedding.
            next_token = torch.where(
                latent_continue.unsqueeze(1),
                argmax_ids.unsqueeze(1),
                next_token,
            )
        if latent_exit.any():
            next_token = torch.where(
                latent_exit.unsqueeze(1),
                torch.full_like(next_token, think_end_first),
                next_token,
            )
        in_latent = in_latent & ~latent_exit

        if eos_ids:
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(
                    next_token,
                    pad_token_id if pad_token_id is not None else eos_ids[0],
                ),
                next_token,
            )
            for eid in eos_ids:
                hit_eos = next_token.squeeze(1) == eid
                # Latent steps only record argmax; they must not end the sequence.
                hit_eos = hit_eos & ~latent_continue
                finished = finished | hit_eos

        step_is_latent = latent_continue.unsqueeze(1)
        is_latent = torch.cat([is_latent, step_is_latent], dim=1)

        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )

        enter_latent = (
            _suffix_matches(generated, think_start_ids) & ~in_latent & ~finished
        )
        in_latent = in_latent | enter_latent

        if bool(finished.all()):
            break

        if latent_continue.any():
            h = _last_hidden(outputs)[:, -1, :]
            remapped = apply_latent_w_remap(h, w_remap, avg_embed_norm)
            token_embeds = embed_layer(next_token.squeeze(1))
            use_remapped = latent_continue & ~finished
            next_embed = torch.where(use_remapped.unsqueeze(1), remapped, token_embeds)
            outputs = model(
                inputs_embeds=next_embed.unsqueeze(1),
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=1,
            )
        else:
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=1,
            )
        past = outputs.past_key_values

    return SimpleNamespace(sequences=generated, is_latent=is_latent)
