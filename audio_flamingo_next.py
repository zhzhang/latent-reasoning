"""Local nn.Module implementation of Audio Flamingo Next (MusicFlamingo).

Mirrors the HuggingFace ``MusicFlamingoForConditionalGeneration`` architecture
(Whisper-style AF3 audio encoder + RoTE + MLP projector + Qwen2 LM) so that
``nvidia/audio-flamingo-next-*`` safetensors checkpoints load via
``load_state_dict`` with matching module names.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from math import pi
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AudioEncoderConfig:
    num_mel_bins: int = 128
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    intermediate_size: int = 5120
    max_source_positions: int = 1500
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    layerdrop: float = 0.0
    activation_function: str = "gelu"
    scale_embedding: bool = False

    @property
    def d_model(self) -> int:
        return self.hidden_size

    @property
    def encoder_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def encoder_attention_heads(self) -> int:
        return self.num_attention_heads

    @property
    def encoder_ffn_dim(self) -> int:
        return self.intermediate_size

    @property
    def encoder_layerdrop(self) -> float:
        return self.layerdrop


@dataclass
class TextConfig:
    vocab_size: int = 151672
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 15_300_000.0
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    pad_token_id: int = 151669
    bos_token_id: int = 151668
    eos_token_id: int = 151645
    head_dim: int | None = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class AudioFlamingoNextConfig:
    audio_config: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    text_config: TextConfig = field(default_factory=TextConfig)
    audio_token_id: int = 151667
    audio_bos_token_id: int = 151670
    audio_eos_token_id: int = 151671
    audio_frame_step: float = 0.01
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True
    # RoTE (audio time) parameters
    rope_theta: float = 1200.0
    partial_rotary_factor: float = 0.2
    max_position_embeddings: int = 1200  # for RoTE window axis
    dtype: torch.dtype = torch.bfloat16

    @property
    def head_dim(self) -> int:
        return self.audio_config.hidden_size

    @classmethod
    def from_hf_config(cls, cfg: dict[str, Any], dtype: torch.dtype | None = None) -> AudioFlamingoNextConfig:
        audio_raw = cfg.get("audio_config", {})
        text_raw = cfg.get("text_config", {})
        rope = cfg.get("rope_parameters") or {}
        text_rope = text_raw.get("rope_parameters") or {}

        audio = AudioEncoderConfig(
            num_mel_bins=audio_raw.get("num_mel_bins", 128),
            hidden_size=audio_raw.get("hidden_size", 1280),
            num_hidden_layers=audio_raw.get("num_hidden_layers", 32),
            num_attention_heads=audio_raw.get("num_attention_heads", 20),
            intermediate_size=audio_raw.get("intermediate_size", 5120),
            max_source_positions=audio_raw.get("max_source_positions", 1500),
            dropout=float(audio_raw.get("dropout", 0.0)),
            attention_dropout=float(audio_raw.get("attention_dropout", 0.0)),
            activation_dropout=float(audio_raw.get("activation_dropout", 0.0)),
            layerdrop=float(audio_raw.get("layerdrop", 0.0)),
            activation_function=audio_raw.get("activation_function", "gelu"),
            scale_embedding=bool(audio_raw.get("scale_embedding", False)),
        )
        text = TextConfig(
            vocab_size=text_raw.get("vocab_size", 151672),
            hidden_size=text_raw.get("hidden_size", 3584),
            intermediate_size=text_raw.get("intermediate_size", 18944),
            num_hidden_layers=text_raw.get("num_hidden_layers", 28),
            num_attention_heads=text_raw.get("num_attention_heads", 28),
            num_key_value_heads=text_raw.get("num_key_value_heads", 4),
            max_position_embeddings=text_raw.get("max_position_embeddings", 131072),
            rms_norm_eps=float(text_raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(text_rope.get("rope_theta", 15_300_000.0)),
            attention_dropout=float(text_raw.get("attention_dropout", 0.0)),
            hidden_act=text_raw.get("hidden_act", "silu"),
            pad_token_id=text_raw.get("pad_token_id", 151669),
            bos_token_id=text_raw.get("bos_token_id", 151668),
            eos_token_id=text_raw.get("eos_token_id", 151645),
        )
        resolved_dtype = dtype
        if resolved_dtype is None:
            dtype_name = cfg.get("dtype", "bfloat16")
            resolved_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }.get(dtype_name, torch.bfloat16)

        return cls(
            audio_config=audio,
            text_config=text,
            audio_token_id=cfg.get("audio_token_id", 151667),
            audio_bos_token_id=cfg.get("audio_bos_token_id", 151670),
            audio_eos_token_id=cfg.get("audio_eos_token_id", 151671),
            audio_frame_step=float(cfg.get("audio_frame_step", 0.01)),
            projector_hidden_act=cfg.get("projector_hidden_act", "gelu"),
            projector_bias=bool(cfg.get("projector_bias", True)),
            rope_theta=float(rope.get("rope_theta", 1200.0)),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", 0.2)),
            max_position_embeddings=int(cfg.get("max_position_embeddings", rope.get("rope_theta", 1200))),
            dtype=resolved_dtype,
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, dtype: torch.dtype | None = None) -> AudioFlamingoNextConfig:
        path = Path(path)
        with open(path / "config.json" if path.is_dir() else path) as f:
            return cls.from_hf_config(json.load(f), dtype=dtype)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def _additive_mask_from_bool(keep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert boolean keep-mask to additive 0 / -inf mask."""
    return torch.zeros_like(keep, dtype=dtype).masked_fill(~keep, torch.finfo(dtype).min)


def build_bidirectional_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """HF-eager style bidirectional padding mask: (B, 1, 1, S) additive."""
    if attention_mask is None:
        return None
    if attention_mask.ndim == 4:
        return attention_mask.to(dtype=dtype, device=device)
    # (B, S) with 1 = keep
    keep = attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
    if bool(keep.all()):
        return None
    return _additive_mask_from_bool(keep.expand(batch_size, 1, 1, seq_len), dtype)


def build_causal_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    q_len: int,
    dtype: torch.dtype,
    device: torch.device,
    past_seen_tokens: int = 0,
    kv_len: int | None = None,
) -> torch.Tensor | None:
    """HF-eager style causal + padding mask: (B, 1, q_len, kv_len) additive."""
    if kv_len is None:
        kv_len = past_seen_tokens + q_len
    q = torch.arange(past_seen_tokens, past_seen_tokens + q_len, device=device)
    k = torch.arange(kv_len, device=device)
    causal_keep = k[None, :] <= q[:, None]  # (q_len, kv_len)
    if attention_mask is None:
        keep = causal_keep[None, None, :, :].expand(batch_size, 1, q_len, kv_len)
        return _additive_mask_from_bool(keep, dtype)

    if attention_mask.ndim == 4:
        return attention_mask.to(dtype=dtype, device=device)

    pad_keep = attention_mask.to(device=device, dtype=torch.bool)  # (B, kv_len)
    keep = causal_keep[None, None, :, :] & pad_keep[:, None, None, :]
    return _additive_mask_from_bool(keep, dtype)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def _get_activation(name: str):
    if name == "gelu":
        return F.gelu
    if name == "silu" or name == "swish":
        return F.silu
    raise ValueError(f"Unsupported activation: {name}")


# ---------------------------------------------------------------------------
# Audio encoder (AF3 / Whisper-style)
# ---------------------------------------------------------------------------


class AudioFlamingo3Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Match HF: scale Q before reshape; pass scaling=1.0 into attention math.
        query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        kv_shape = (input_shape[0], -1, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2).contiguous()
        value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2).contiguous()

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None


class AudioFlamingo3EncoderLayer(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = AudioFlamingo3Attention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = _get_activation(config.activation_function)
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        if self.dropout > 0.0 and self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=True)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        if self.activation_dropout > 0.0 and self.training:
            hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=True)
        hidden_states = self.fc2(hidden_states)
        if self.dropout > 0.0 and self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=True)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states, attn_weights


class AudioFlamingo3Encoder(nn.Module):
    """Whisper-style encoder: conv front-end, transformer stack, avg-pool, LayerNorm."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        self.layers = nn.ModuleList(
            [AudioFlamingo3EncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        seq_len = (input_features.shape[-1] - 1) // 2 + 1
        if input_features_mask is not None:
            input_features_lengths = input_features_mask.sum(-1)
            input_features_lengths = (input_features_lengths - 1) // 2 + 1
            input_features_mask = (
                torch.arange(seq_len, device=input_features.device) < input_features_lengths[:, None]
            )

        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        hidden_states = inputs_embeds + self.embed_positions.weight
        if self.dropout > 0.0 and self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=True)

        attention_mask = build_bidirectional_mask(
            input_features_mask,
            batch_size=hidden_states.shape[0],
            seq_len=hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        all_attentions: list[torch.Tensor] = []
        for layer in self.layers:
            if self.training and self.layerdrop > 0.0 and torch.rand([]) < self.layerdrop:
                continue
            hidden_states, attn_weights = layer(
                hidden_states, attention_mask, output_attentions=output_attentions
            )
            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)

        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states).permute(0, 2, 1)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states, tuple(all_attentions) if output_attentions else None


# ---------------------------------------------------------------------------
# RoTE (rotary time embedding on audio features)
# ---------------------------------------------------------------------------


def rotate_half_pairs(x: torch.Tensor) -> torch.Tensor:
    """MusicFlamingo RoTE rotate: pair-wise (x0,x1)->(-x1,x0)."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_time_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    original_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float64)
    cos = cos.to(hidden_states)
    sin = sin.to(hidden_states)
    rot_dim = cos.shape[-1]
    passthrough = hidden_states[..., rot_dim:]
    rotated = hidden_states[..., :rot_dim]
    rotated = (rotated * cos) + (rotate_half_pairs(rotated) * sin)
    return torch.cat((rotated, passthrough), dim=-1).to(original_dtype)


class MusicFlamingoRotaryEmbedding(nn.Module):
    def __init__(self, config: AudioFlamingoNextConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.config = config
        head_dim = config.head_dim
        dim = int(head_dim * config.partial_rotary_factor)
        base = config.rope_theta
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        position_angles = self._compute_position_angles(inv_freq)
        self.register_buffer("position_angles", position_angles, persistent=False)

    def _compute_position_angles(self, inv_freq: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(int(self.max_seq_len_cached), device=inv_freq.device, dtype=inv_freq.dtype)
        positions = positions / self.max_seq_len_cached * (2 * pi)
        position_angles = positions.unsqueeze(-1) * inv_freq
        position_angles = torch.repeat_interleave(position_angles, 2, dim=-1)
        return position_angles.to(dtype=inv_freq.dtype)

    @torch.no_grad()
    def forward(self, timestamps: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        window_starts = timestamps[:, 0].to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        window_duration = self.config.audio_frame_step * 4 * seq_len
        window_positions = torch.round(window_starts / window_duration) / self.max_seq_len_cached
        window_freqs = window_positions.unsqueeze(-1) * self.inv_freq
        window_freqs = torch.repeat_interleave(window_freqs, 2, dim=-1)

        window_freqs = window_freqs[:, None, :]
        time_freqs = self.position_angles[:seq_len][None, :, :]
        window_freqs, time_freqs = torch.broadcast_tensors(window_freqs, time_freqs)
        freqs = torch.cat((window_freqs, time_freqs), dim=-1)
        angle = (-timestamps * 2 * pi).to(freqs)
        freqs = freqs * angle.unsqueeze(-1)
        return freqs.cos(), freqs.sin()


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------


class MultiModalProjector(nn.Module):
    def __init__(self, config: AudioFlamingoNextConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )
        self.act = _get_activation(config.projector_hidden_act)
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(audio_features)))


# ---------------------------------------------------------------------------
# Qwen2 language model
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = _get_activation(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard Qwen2/LLaMA rotate_half (split halves, not pairs)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 matmul (HF disables autocast; MPS falls back to CPU path).
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        present = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.attention_dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=True)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None, present


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, present = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights, present


class Qwen2Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, Any]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape
        past_seen = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_seen = past_key_values[0][0].shape[2]
        kv_len = past_seen + seq_len
        if position_ids is None:
            position_ids = (
                torch.arange(past_seen, past_seen + seq_len, device=inputs_embeds.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        causal_mask = build_causal_mask(
            attention_mask,
            batch_size=batch_size,
            q_len=seq_len,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_seen_tokens=past_seen,
            kv_len=kv_len,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        all_hidden_states: list[torch.Tensor] = []
        all_attentions: list[torch.Tensor] = []
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_past = past_key_values[layer_idx] if past_key_values is not None else None
            hidden_states, attn_weights, present = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_value=layer_past,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            if use_cache and present is not None:
                next_cache.append(present)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            # Replace last pre-norm state with post-norm to match HF convention
            # where the final tuple entry is post-RMSNorm.
            all_hidden_states[-1] = hidden_states

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": tuple(all_hidden_states) if output_hidden_states else None,
            "attentions": tuple(all_attentions) if output_attentions else None,
        }


# ---------------------------------------------------------------------------
# Full AF-Next model
# ---------------------------------------------------------------------------


class AudioFlamingoNextModel(nn.Module):
    def __init__(self, config: AudioFlamingoNextConfig):
        super().__init__()
        self.config = config
        self.audio_tower = AudioFlamingo3Encoder(config.audio_config)
        self.language_model = Qwen2Model(config.text_config)
        self.multi_modal_projector = MultiModalProjector(config)
        self.pos_emb = MusicFlamingoRotaryEmbedding(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def _build_audio_timestamps(
        self,
        input_ids: torch.LongTensor,
        post_lengths: torch.LongTensor,
        max_post_length: int,
    ) -> torch.FloatTensor:
        audio_token_mask = input_ids == self.config.audio_token_id
        diff = torch.diff(F.pad(audio_token_mask.int(), (1, 1), value=0), dim=1)
        _, starts = torch.where(diff == 1)
        _, ends = torch.where(diff == -1)
        sample_lengths = (ends - starts).to(torch.long)

        n_audio_tokens = audio_token_mask.sum()
        n_audio_features = post_lengths.sum()
        if n_audio_tokens != n_audio_features:
            raise ValueError(
                f"Audio features and audio tokens do not match, "
                f"tokens: {n_audio_tokens}, features: {n_audio_features}"
            )

        audio_embed_frame_step = self.config.audio_frame_step * 4
        frame_offsets = (
            torch.arange(max_post_length, device=post_lengths.device, dtype=torch.float32)
            * audio_embed_frame_step
        )

        cumsum_post = torch.cat(
            [torch.zeros(1, device=post_lengths.device), torch.cumsum(post_lengths, dim=0)[:-1]]
        )
        cumsum_samples = torch.cumsum(sample_lengths, dim=0)
        sample_indices = torch.searchsorted(cumsum_samples, cumsum_post, right=True)

        sample_start_rows = torch.searchsorted(
            sample_indices, torch.arange(sample_lengths.shape[0], device=post_lengths.device)
        )
        window_indices = (
            torch.arange(post_lengths.shape[0], device=post_lengths.device) - sample_start_rows[sample_indices]
        )
        return window_indices.unsqueeze(1) * max_post_length * audio_embed_frame_step + frame_offsets

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        input_ids: torch.LongTensor,
        output_attentions: bool = False,
    ) -> dict[str, Any]:
        hidden_states, encoder_attentions = self.audio_tower(
            input_features,
            input_features_mask=input_features_mask,
            output_attentions=output_attentions,
        )
        _, post_lengths = self.audio_tower._get_feat_extract_output_lengths(
            input_features_mask.sum(-1).to(torch.long)
        )
        audio_timestamps = self._build_audio_timestamps(
            input_ids, post_lengths, hidden_states.shape[-2]
        )
        cos, sin = self.pos_emb(audio_timestamps.to(hidden_states.device), seq_len=hidden_states.shape[-2])
        hidden_states = apply_rotary_time_emb(hidden_states, cos, sin)
        audio_embeds = self.multi_modal_projector(hidden_states)

        valid_mask = torch.arange(audio_embeds.shape[1], device=post_lengths.device)[None, :] < post_lengths[:, None]
        pooled = audio_embeds[valid_mask.to(audio_embeds.device)]
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled,
            "attentions": encoder_attentions,
        }

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        audio_features: torch.FloatTensor,
    ) -> torch.Tensor:
        special_audio_mask = input_ids == self.config.audio_token_id
        n_audio_tokens = special_audio_mask.sum()
        n_audio_features = audio_features.shape[0]
        if n_audio_tokens != n_audio_features:
            raise ValueError(
                f"Audio features and audio tokens do not match, "
                f"tokens: {n_audio_tokens}, features: {n_audio_features}"
            )
        return special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, Any]:
        # Audio features are only fused on the prefill step (no past cache).
        fuse_audio = (
            input_features is not None
            and input_ids is not None
            and past_key_values is None
        )
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_embeds = None
        encoder_attentions = None
        if fuse_audio:
            audio_out = self.get_audio_features(
                input_features,
                input_features_mask,
                input_ids=input_ids,
                output_attentions=output_attentions,
            )
            audio_embeds = audio_out["pooler_output"]
            encoder_attentions = audio_out["attentions"]
            special_audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, audio_features=audio_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_audio_mask, audio_embeds.to(inputs_embeds.device)
            )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return {
            "last_hidden_state": outputs["last_hidden_state"],
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
            "audio_hidden_states": audio_embeds,
            "encoder_attentions": encoder_attentions,
        }


def _apply_repetition_penalty(
    logits: torch.Tensor, token_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """HF-style repetition penalty over all tokens seen so far."""
    if penalty is None or penalty == 1.0:
        return logits
    logits = logits.clone()
    # token_ids: (batch, seq)
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
        sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(sorted_logits.dtype).min)
        logits = torch.full_like(logits, torch.finfo(logits.dtype).min).scatter(
            -1, sorted_idx, sorted_logits
        )
    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _compute_w_remap(model: "AudioFlamingoNextForConditionalGeneration") -> torch.Tensor:
    """Precompute ``pinv(W_out) @ W_emb`` for latent-CoT residual remapping."""
    w_out = model.lm_head.weight.float()
    w_emb = model.get_input_embeddings().weight.float()
    return torch.linalg.pinv(w_out) @ w_emb


def _suffix_matches(generated: torch.LongTensor, suffix_ids: list[int]) -> torch.Tensor:
    """Return a (batch,) bool mask: True when each row ends with ``suffix_ids``."""
    if not suffix_ids:
        return torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
    suffix = torch.tensor(suffix_ids, device=generated.device, dtype=generated.dtype)
    tail = generated[:, -len(suffix_ids) :]
    return (tail == suffix.unsqueeze(0)).all(dim=1)


class AudioFlamingoNextForConditionalGeneration(nn.Module):
    """Local AF-Next / MusicFlamingo causal LM with audio conditioning."""

    def __init__(self, config: AudioFlamingoNextConfig):
        super().__init__()
        self.config = config
        self.model = AudioFlamingoNextModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def get_audio_features(self, *args, **kwargs):
        return self.model.get_audio_features(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        # Ignore HF-only kwargs (e.g. logits_to_keep) so call sites stay compatible.
        _ = kwargs
        outputs = self.model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(outputs["last_hidden_state"])
        result = {
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
            "audio_hidden_states": outputs["audio_hidden_states"],
            "encoder_attentions": outputs["encoder_attentions"],
            "last_hidden_state": outputs["last_hidden_state"],
        }
        if return_dict:
            from types import SimpleNamespace

            return SimpleNamespace(**result)
        return result

    @torch.inference_mode()
    def generate(
        self,
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
        latent_cot: bool = False,
        think_start_ids: list[int] | None = None,
        think_end_ids: list[int] | None = None,
        **kwargs,
    ):
        """Autoregressive decode with KV-cache; audio features used on prefill only."""
        _ = kwargs  # tolerate HF generate extras
        if eos_token_id is None:
            eos_token_id = self.config.text_config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.text_config.pad_token_id
        if isinstance(eos_token_id, int):
            eos_ids = [eos_token_id]
        else:
            eos_ids = list(eos_token_id or [])

        if latent_cot:
            if not think_start_ids or not think_end_ids:
                raise ValueError(
                    "latent_cot=True requires non-empty think_start_ids and think_end_ids"
                )
            think_start_ids = [int(t) for t in think_start_ids]
            think_end_ids = [int(t) for t in think_end_ids]
            think_end_first = think_end_ids[0]
            w_remap = _compute_w_remap(self)
            embed_layer = self.get_input_embeddings()
        else:
            think_start_ids = []
            think_end_ids = []
            think_end_first = None
            w_remap = None
            embed_layer = None

        batch_size = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Prefill
        outputs = self.forward(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
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

            if latent_cot:
                latent_exit = in_latent & (argmax_ids == think_end_first)
                latent_continue = in_latent & ~latent_exit

                penalized = _apply_repetition_penalty(
                    step_logits,
                    generated,
                    repetition_penalty if repetition_penalty is not None else 1.0,
                )
                sampled = _sample_next_token(
                    penalized,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
                next_token = sampled
                if latent_continue.any():
                    next_token = torch.where(
                        latent_continue.unsqueeze(1),
                        torch.full_like(next_token, pad_token_id),
                        next_token,
                    )
                if latent_exit.any():
                    next_token = torch.where(
                        latent_exit.unsqueeze(1),
                        torch.full_like(next_token, think_end_first),
                        next_token,
                    )
                in_latent = in_latent & ~latent_exit
            else:
                latent_continue = None
                step_logits = _apply_repetition_penalty(
                    step_logits,
                    generated,
                    repetition_penalty if repetition_penalty is not None else 1.0,
                )
                next_token = _sample_next_token(
                    step_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )

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
                    finished = finished | (next_token.squeeze(1) == eid)

            if latent_cot:
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

            if latent_cot:
                enter_latent = (
                    _suffix_matches(generated, think_start_ids) & ~in_latent & ~finished
                )
                in_latent = in_latent | enter_latent

            if bool(finished.all()):
                break

            if latent_cot and latent_continue is not None and latent_continue.any():
                h = outputs.last_hidden_state[:, -1, :]
                remapped = (h.float() @ w_remap).to(dtype=h.dtype)
                token_embeds = embed_layer(next_token.squeeze(1))
                use_remapped = latent_continue & ~finished
                next_embed = torch.where(use_remapped.unsqueeze(1), remapped, token_embeds)
                outputs = self.forward(
                    inputs_embeds=next_embed.unsqueeze(1),
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                outputs = self.forward(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = outputs.past_key_values

        if latent_cot:
            from types import SimpleNamespace

            return SimpleNamespace(sequences=generated, is_latent=is_latent)
        return generated

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        strict: bool = True,
    ) -> AudioFlamingoNextForConditionalGeneration:
        model_dir = Path(model_dir)
        config = AudioFlamingoNextConfig.from_pretrained(model_dir, dtype=dtype)
        model = cls(config)
        state = load_hf_weights(model_dir)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # pos_emb buffers (inv_freq / position_angles) are not in the checkpoint.
        ignorable_missing = {
            k
            for k in missing
            if k.startswith("model.pos_emb.")
            or k.endswith("inv_freq")
            or "rotary_emb.inv_freq" in k
        }
        real_missing = [k for k in missing if k not in ignorable_missing]
        if strict and (real_missing or unexpected):
            raise RuntimeError(
                f"Weight load mismatch.\n"
                f"missing ({len(real_missing)}): {real_missing[:20]}\n"
                f"unexpected ({len(unexpected)}): {unexpected[:20]}"
            )
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        model.eval()
        return model


def remap_hf_state_dict(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map hub MusicFlamingo keys onto MusicFlamingoForConditionalGeneration names.

    Hub checkpoints store:
      audio_tower.*
      multi_modal_projector.*
      language_model.model.*   -> model.language_model.*
      language_model.lm_head.* -> lm_head.*

    while the module tree is ``model.{audio_tower,language_model,...}`` + ``lm_head``.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if key.startswith("language_model.lm_head."):
            new_key = "lm_head." + key[len("language_model.lm_head.") :]
        elif key.startswith("language_model.model."):
            new_key = "model.language_model." + key[len("language_model.model.") :]
        elif key.startswith("language_model."):
            # Already a flat Qwen2Model under language_model
            new_key = "model.language_model." + key[len("language_model.") :]
        elif key.startswith("model."):
            new_key = key
        elif key.startswith("lm_head."):
            new_key = key
        else:
            # audio_tower.*, multi_modal_projector.*, pos_emb.*
            new_key = "model." + key
        out[new_key] = value
    return out


def load_hf_weights(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load a single-file or sharded safetensors checkpoint (with key remap)."""
    model_dir = Path(model_dir)
    single = model_dir / "model.safetensors"
    if single.is_file():
        raw = load_file(str(single))
        return remap_hf_state_dict(raw)

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shards = sorted(set(weight_map.values()))
        raw: dict[str, torch.Tensor] = {}
        for shard in shards:
            raw.update(load_file(str(model_dir / shard)))
        return remap_hf_state_dict(raw)

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights found under {model_dir}")
    raw = {}
    for path in files:
        raw.update(load_file(str(path)))
    return remap_hf_state_dict(raw)
