# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import os
import json
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F


# 0.6 billion parameters
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,  # Vocabulary size
    "context_length": 40_960,  # Context length that was used to train the model
    "emb_dim": 1024,  # Embedding dimension
    "n_heads": 16,  # Number of attention heads
    "n_layers": 28,  # Number of layers
    "hidden_dim": 3072,  # Size of the intermediate dimension in FeedForward
    "head_dim": 128,  # Size of the heads in GQA
    "qk_norm": True,  # Whether to normalize queries and keys in GQA
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
}

# 1.7 billion parameters
QWEN3_CONFIG_1_7B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2048,  # 2x larger than above
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 6144,  # 2x larger than above
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 4 billion parameters
QWEN3_CONFIG_4B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2560,  # 25% larger than above
    "n_heads": 32,  # 2x larger than above
    "n_layers": 36,  # 29% larger than above
    "hidden_dim": 9728,  # ~3x larger than above
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 8 billion parameters
QWEN3_CONFIG_8B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 4096,  # 60% larger than above
    "n_heads": 32,
    "n_layers": 36,
    "hidden_dim": 12288,  # 26% larger than above
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 14 billion parameters
QWEN3_CONFIG_14B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 5120,  # 25% larger than above
    "n_heads": 40,  # 25% larger than above
    "n_layers": 40,  # 11% larger than above
    "hidden_dim": 17408,  # 42% larger than above
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

QWEN3_CONFIG_32B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 5120,
    "n_heads": 64,  # 60% larger than above
    "n_layers": 64,  # 60% larger than above
    "hidden_dim": 25600,  # 47% larger than above
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# Mixture of Experts Model
QWEN3_CONFIG_30B_A3B = {
    "vocab_size": 151_936,
    "context_length": 262_144,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 48,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 4,
    "rope_base": 10_000_000.0,
    "dtype": torch.bfloat16,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
}


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )

        # Reusable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(
        self,
        in_idx=None,
        input_embeds=None,
        cache=None,
        start_pos=0,
        return_cache=False,
        key_pad=None,
        thinking_max_layer=None,
    ):
        # Forward pass
        if (in_idx is None) == (input_embeds is None):
            raise ValueError("Provide exactly one of in_idx or input_embeds.")

        if input_embeds is not None:
            x = input_embeds
        else:
            tok_embeds = self.tok_emb(in_idx)
            x = tok_embeds

        num_tokens = x.shape[1]
        total_len = start_pos + num_tokens

        # When there is no padding we pass mask=None so attention can use the fast
        # causal SDPA path. For padded batches we build an explicit boolean keep-mask
        # combining the causal constraint with the per-sequence key padding.
        #
        # key_pad: (batch, total_len) bool, True where a key position is PADDING.
        if key_pad is None:
            mask = None
        else:
            q_pos = torch.arange(start_pos, total_len, device=x.device).unsqueeze(
                1
            )  # (q, 1)
            k_pos = torch.arange(total_len, device=x.device).unsqueeze(0)  # (1, k)
            causal_disallow = k_pos > q_pos  # (q, k)
            disallow = causal_disallow.unsqueeze(0) | key_pad.unsqueeze(1)  # (b, q, k)
            # Always let a position attend itself so fully-padded query rows don't
            # produce NaNs (softmax over an all-masked row). These rows belong to pad
            # tokens whose outputs are never read.
            eye = q_pos == k_pos  # (q, k)
            disallow = disallow & ~eye.unsqueeze(0)
            mask = (~disallow).unsqueeze(1)  # (b, 1, q, k)

        new_caches = []
        output_embed = None
        for i, block in enumerate(self.trf_blocks):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = block(
                x, mask, self.cos, self.sin, start_pos, layer_cache
            )
            if thinking_max_layer is not None and i + 1 == thinking_max_layer:
                output_embed = x
            new_caches.append(layer_new_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))

        if return_cache:
            return logits, new_caches, output_embed
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=512,
        eos_token_id=None,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        thinking_max_layer=None,
        latent_thinking=False,
        thinking_end_token_id=None,
    ):
        """Autoregressively generate tokens using KV-caching.

        input_ids: LongTensor of shape (batch, seq_len). For batched generation the
            inputs must be LEFT-padded and ``attention_mask`` (1 = real, 0 = pad)
            must be provided so padding tokens are excluded from attention.
        Returns the full sequence (prompt + generated) of shape (batch, seq_len + new).
        Generation stops early only when every sequence in the batch has emitted eos.
        """
        self.eval()
        max_ctx = self.cfg["context_length"]

        # Prefill: run the whole prompt once and cache its keys/values.
        prompt_len = input_ids.shape[1]
        if prompt_len > max_ctx:
            input_ids = input_ids[:, -max_ctx:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -max_ctx:]
            prompt_len = max_ctx

        # Only build padding masks when padding is actually present (keeps the
        # single-sequence / full-batch case on the fast causal path).
        key_pad = None
        if attention_mask is not None and bool((attention_mask == 0).any()):
            key_pad = attention_mask == 0

        effective_thinking_layer = thinking_max_layer
        if latent_thinking and effective_thinking_layer is None:
            effective_thinking_layer = len(self.trf_blocks)
        if latent_thinking and (
            effective_thinking_layer < 1
            or effective_thinking_layer > len(self.trf_blocks)
        ):
            raise ValueError(
                f"thinking_max_layer must be in [1, {len(self.trf_blocks)}], "
                f"got {effective_thinking_layer}."
            )

        logits, cache, output_embed = self.forward(
            input_ids,
            cache=None,
            start_pos=0,
            return_cache=True,
            key_pad=key_pad,
            thinking_max_layer=effective_thinking_layer if latent_thinking else None,
        )
        start_pos = prompt_len
        generated = input_ids
        next_logits = logits[:, -1, :]
        next_latent_embed = output_embed[:, -1:, :] if output_embed is not None else None

        batch_size = input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        in_latent_thinking = latent_thinking

        for _ in range(max_new_tokens):
            if in_latent_thinking:
                if thinking_end_token_id is None:
                    raise ValueError(
                        "thinking_end_token_id is required when latent_thinking is enabled."
                    )
                if next_latent_embed is None:
                    raise ValueError(
                        "thinking_max_layer did not produce embeddings for latent thinking."
                    )

                # Stop latent updates once </think> is the most likely next token.
                if bool((next_logits.argmax(dim=-1) == thinking_end_token_id).all()):
                    in_latent_thinking = False
                else:
                    if start_pos >= max_ctx:
                        break

                    if key_pad is not None:
                        key_pad = torch.cat(
                            [
                                key_pad,
                                torch.zeros(
                                    batch_size,
                                    1,
                                    dtype=torch.bool,
                                    device=key_pad.device,
                                ),
                            ],
                            dim=1,
                        )

                    logits, cache, output_embed = self.forward(
                        input_embeds=next_latent_embed,
                        cache=cache,
                        start_pos=start_pos,
                        return_cache=True,
                        key_pad=key_pad,
                        thinking_max_layer=effective_thinking_layer,
                    )
                    start_pos += 1
                    next_logits = logits[:, -1, :]
                    next_latent_embed = (
                        output_embed[:, -1:, :] if output_embed is not None else None
                    )
                    continue

            next_token = self._sample_next(next_logits, temperature, top_p, top_k)

            # Once a sequence has finished, keep emitting eos so the batch stays aligned.
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if bool(finished.all()):
                    break

            if start_pos >= max_ctx:
                break

            # Newly generated tokens are always real (not padding).
            if key_pad is not None:
                key_pad = torch.cat(
                    [
                        key_pad,
                        torch.zeros(
                            batch_size, 1, dtype=torch.bool, device=key_pad.device
                        ),
                    ],
                    dim=1,
                )

            logits, cache, _ = self.forward(
                next_token,
                cache=cache,
                start_pos=start_pos,
                return_cache=True,
                key_pad=key_pad,
            )
            start_pos += 1
            next_logits = logits[:, -1, :]

        return generated

    @staticmethod
    def _sample_next(logits, temperature=0.0, top_p=1.0, top_k=None):
        # logits: (batch, vocab). Returns next tokens of shape (batch, 1).
        logits = logits.float()

        if temperature is None or temperature <= 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth_vals, -torch.inf)

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
            logits = torch.full_like(logits, -torch.inf).scatter(
                -1, sorted_idx, sorted_logits
            )

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"],
        )
        if "num_experts" in cfg and cfg["num_experts"] > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, new_cache = self.att(
            x, mask, cos, sin, start_pos, cache
        )  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x, new_cache


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc2 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc3 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
        )

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_experts = cfg["num_experts"]
        self.emb_dim = cfg["emb_dim"]
        self.gate = nn.Linear(
            cfg["emb_dim"], cfg["num_experts"], bias=False, dtype=cfg["dtype"]
        )

        self.fc1 = nn.ModuleList(
            [
                nn.Linear(
                    cfg["emb_dim"],
                    cfg["moe_intermediate_size"],
                    bias=False,
                    dtype=cfg["dtype"],
                )
                for _ in range(cfg["num_experts"])
            ]
        )
        self.fc2 = nn.ModuleList(
            [
                nn.Linear(
                    cfg["emb_dim"],
                    cfg["moe_intermediate_size"],
                    bias=False,
                    dtype=cfg["dtype"],
                )
                for _ in range(cfg["num_experts"])
            ]
        )
        self.fc3 = nn.ModuleList(
            [
                nn.Linear(
                    cfg["moe_intermediate_size"],
                    cfg["emb_dim"],
                    bias=False,
                    dtype=cfg["dtype"],
                )
                for _ in range(cfg["num_experts"])
            ]
        )

    def forward(self, x):
        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, -1)
        out_flat = torch.zeros(
            batch * seq_len, self.emb_dim, device=x.device, dtype=x.dtype
        )

        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(-1, self.num_experts_per_tok)

        unique_experts = torch.unique(topk_indices_flat)

        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())
            mask = topk_indices_flat == expert_id
            if not mask.any():
                continue

            token_mask = mask.any(dim=-1)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            expert_input = x_flat.index_select(0, selected_idx)
            hidden = torch.nn.functional.silu(
                self.fc1[expert_id](expert_input)
            ) * self.fc2[expert_id](expert_input)
            expert_out = self.fc3[expert_id](hidden)

            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(
                topk_probs_flat.index_select(0, selected_idx),
                dim=-1,
                index=slot_indices,
            ).squeeze(-1)

            out_flat.index_add_(
                0, selected_idx, expert_out * selected_probs.unsqueeze(-1)
            )

        return out_flat.reshape(batch, seq_len, self.emb_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, (
                "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            )
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)  # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )
        values = values.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE at the correct absolute positions (offset by start_pos for cached decoding)
        cos_slice = cos[start_pos : start_pos + num_tokens]
        sin_slice = sin[start_pos : start_pos + num_tokens]
        queries = apply_rope(queries, cos_slice, sin_slice)
        keys = apply_rope(keys, cos_slice, sin_slice)

        # Append the new keys/values to the running cache (incremental decoding)
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys], dim=2)
            values = torch.cat([prev_v, values], dim=2)
        new_cache = (keys, values)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Fused (flash / mem-efficient) scaled-dot-product attention.
        # `mask` is None for unpadded inputs, which lets the flash kernel fire via
        # the causal fast paths:
        #   - prefill (q_len == k_len): standard causal attention
        #   - single-token decode: the new token may attend to all cached keys
        # For padded batches, `mask` is an explicit boolean keep-mask
        # (shape (b, 1, q_len, kv_len), True = attend) combining causal + padding.
        if mask is not None:
            context = F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=mask
            )
        else:
            kv_len = keys.shape[2]
            if num_tokens == kv_len:
                context = F.scaled_dot_product_attention(
                    queries, keys, values, is_causal=True
                )
            else:
                context = F.scaled_dot_product_attention(queries, keys, values)

        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), new_cache


# ==============================================================================
# RoPE implementation summary
#
#
# There are two common styles to implement RoPE, which are
# mathematically equivalent;
# they mainly differ in how the rotation matrix pairs dimensions.
#
# 1) Split-halves style (this repo, Hugging Face Transformers):
#
#   For hidden dim d = 8 (example):
#
#       [ x0   x1   x2   x3   x4   x5   x6   x7 ]
#         │    │    │    │    │    │    │    │
#         ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
#        cos  cos  cos  cos  sin  sin  sin  sin
#
#   Rotation matrix:
#
#       [ cosθ   -sinθ    0      0   ... ]
#       [ sinθ    cosθ    0      0   ... ]
#       [  0       0    cosθ   -sinθ ... ]
#       [  0       0    sinθ    cosθ ... ]
#        ...
#
#   Here, the embedding dims are split into two halves and then
#   each one is rotated in blocks.
#
#
# 2) Interleaved (even/odd) style (original paper, Llama repo):
#
#   For hidden dim d = 8 (example):
#
#       [ x0   x1   x2   x3   x4   x5   x6   x7 ]
#         │    │    │    │    │    │    │    │
#         ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
#        cos  sin  cos  sin  cos  sin  cos  sin
#
#   Rotation matrix:
#       [ cosθ  -sinθ    0      0   ... ]
#       [ sinθ   cosθ    0      0   ... ]
#       [  0      0    cosθ   -sinθ ... ]
#       [  0      0    sinθ    cosθ ... ]
#        ...
#
#   Here, embedding dims are interleaved as even/odd cosine/sine pairs.
#
# Both layouts encode the same relative positions; the only difference is how
# dimensions are paired.
# ==============================================================================


def compute_rope_params(
    head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
            / head_dim
        )
    )

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(
        0
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


def load_weights_into_qwen(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
            )

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    model.tok_emb.weight = assign(
        model.tok_emb.weight,
        params["model.embed_tokens.weight"],
        "model.embed_tokens.weight",
    )

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att

        # Q, K, V projections
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight",
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight",
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight",
        )

        # Output projection
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight",
        )

        # QK norms
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight",
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight",
            )

        # Attention layernorm
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight",
        )

        # Feedforward weights
        if param_config.get("num_experts", 0) > 0:
            # Load router (gating) weights
            block.ff.gate.weight = assign(
                block.ff.gate.weight,
                params[f"model.layers.{l}.mlp.gate.weight"],
                f"model.layers.{l}.mlp.gate.weight",
            )
            # Load expert weights
            for e in range(param_config["num_experts"]):
                prefix = f"model.layers.{l}.mlp.experts.{e}"
                block.ff.fc1[e].weight = assign(
                    block.ff.fc1[e].weight,
                    params[f"{prefix}.gate_proj.weight"],
                    f"{prefix}.gate_proj.weight",
                )
                block.ff.fc2[e].weight = assign(
                    block.ff.fc2[e].weight,
                    params[f"{prefix}.up_proj.weight"],
                    f"{prefix}.up_proj.weight",
                )
                block.ff.fc3[e].weight = assign(
                    block.ff.fc3[e].weight,
                    params[f"{prefix}.down_proj.weight"],
                    f"{prefix}.down_proj.weight",
                )

        else:
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight",
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight",
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight",
            )

        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight",
        )

    # Final normalization and output head
    model.final_norm.scale = assign(
        model.final_norm.scale, params["model.norm.weight"], "model.norm.weight"
    )

    if "lm_head.weight" in params:
        model.out_head.weight = assign(
            model.out_head.weight, params["lm_head.weight"], "lm_head.weight"
        )
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")


def download_from_huggingface(repo_id, filename, local_dir, revision="main"):
    base_url = "https://huggingface.co"
    url = f"{base_url}/{repo_id}/resolve/{revision}/{filename}"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    dest_path = os.path.join(local_dir, filename)

    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
    else:
        print(f"Downloading {url} to {dest_path}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return dest_path


def download_from_huggingface_from_snapshots(repo_id, local_dir):
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file  # or your preferred loader

    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)

    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    single_file_path = os.path.join(repo_dir, "model.safetensors")

    if os.path.exists(index_path):
        # Multi-shard model
        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)
    elif os.path.exists(single_file_path):
        # Single-shard model
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights_dict = load_file(weights_file)
    else:
        raise FileNotFoundError(
            "No model.safetensors or model.safetensors.index.json found."
        )

    return weights_dict
