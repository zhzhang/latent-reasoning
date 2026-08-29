"""Per-model vLLM adapters for the MMAR difficulty experiment."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from audio_flamingo_runtime import resolve_model_dir
from mmar_common import (
    AF3_THINK_SUFFIX,
    AF_NEXT_THINK_SUFFIX,
    ASSISTANT_THINK_OPEN,
    MUSIC_FLAMINGO_THINK_SUFFIX,
    PREFIX_ASSISTANT_THINK_LABELS,
    build_mmar_description_prompt,
    build_mmar_freeform_prompt,
    build_mmar_prompt,
    ensure_assistant_think_open,
    join_vllm_reasoning,
    parse_choice_output,
    parse_description_output,
    parse_freeform_output,
    parse_music_flamingo_output,
    parse_think_tagged_output,
    vllm_reasoning_text,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Backends that cannot use SamplingParams(n>1) shared prefill (HF has no n=
# fork). Callers must duplicate question×shot prompt rows for these.
_DUPLICATE_SHOT_BACKENDS = frozenset({"hf_af_next"})


def backend_duplicates_shots(backend: str) -> bool:
    """True when n_shots must be expanded into duplicate prompts."""
    return backend in _DUPLICATE_SHOT_BACKENDS


MODEL_SPECS: dict[str, dict[str, Any]] = {
    # https://huggingface.co/nvidia/audio-flamingo-next-think-hf
    "af-next-think": {
        "model_id": "nvidia/audio-flamingo-next-think-hf",
        "gpu": "B200",
        "backend": "vllm",
        # Native MusicFlamingo path in vLLM 0.28. Some AF-Next checkpoints
        # include avg_embed_norm weights that vLLM does not model; we skip them
        # at load time (see load_af_next).
        "engine": {
            "dtype": "bfloat16",
            # transformers>=5.5 advertises MusicFlamingoForConditionalGeneration.
            # vLLM 0.28 only registers AudioFlamingo3; model_impl=auto then
            # picks TransformersMultiModalForCausalLM, which calls
            # get_audio_features without input_ids and dies in profile_run.
            "hf_overrides": {
                "architectures": ["AudioFlamingo3ForConditionalGeneration"],
            },
            # Cap context for long audio+text prompts (not a KV/throughput lever;
            # PagedAttention allocates on demand).
            "max_model_len": 8192,
            # Qwen2.5-7B GQA (28L × 4 kv × 128 × 2 × 2B) ≈ 56 KiB/tok. B200
            # 180 GiB × 0.95 − ~20 GiB weights/encoder ⇒ ~2.6M KV tokens.
            # 512 seqs × ~3k typical MMAR leaves the pool ~40% free for audio
            # outliers; 32k batched tokens covers ~4 concurrent 8k prefills.
            "max_num_seqs": 512,
            "max_num_batched_tokens": 32768,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            "attention_backend": "flashinfer",  # best for throughput
            "async_scheduling": True,  # usually faster, but not all features supported
        },
        # Card generate(): repetition_penalty=1.2; generation_config.json
        # max_new_tokens=2048 and no do_sample (greedy). T=0.2 is for n-shot
        # variance (official generate() is greedy; README example uses 4096).
        "native_thinking": True,
        "enable_thinking": True,
        "sampling": {
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 2048,
            "repetition_penalty": 1.2,
        },
    },
    # CONFIRMED
    # MoE thinker-only (~3B active); 30B-A3B bf16 on one B200.
    # https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking/blob/main/generation_config.json
    "qwen3-omni": {
        "model_id": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "gpu": "B200",
        "backend": "vllm",
        "engine": {
            "dtype": "bfloat16",
            # Sized for measured MMAR Thinking outputs (~850 tok avg, p99 ~780,
            # max ~870). Oversized max_model_len deflates reported concurrency.
            "max_model_len": 8192,
            # 48L × 4 kv × 128 × 2 × 2B ≈ 96 KiB/tok. B200 180 GiB × 0.95 −
            # ~70 GiB weights/encoders ⇒ ~1.0M KV tokens. 256 seqs × ~4k
            # audio+CoT uses ~1.0M; 32k batched tokens for MoE prefill.
            "max_num_seqs": 256,
            "max_num_batched_tokens": 32768,
            "limit_mm_per_prompt": {"audio": 1},
            # vLLM 0.28: torch.compile + CUDA graphs succeed on profile_run.
            "enforce_eager": False,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            "attention_backend": "flashinfer",  # best for throughput
            "async_scheduling": True,  # usually faster, but not all features supported
        },
        "native_thinking": True,
        # chat_template.json: empty <think></think> only when enable_thinking
        # is false. The generate() prompt omits that skip block.
        # Card: Thinking models must use generation_config.json (T=0.6,
        # top_p=0.95, top_k=20, max_new_tokens=32768). Cap max_tokens for MMAR.
        "enable_thinking": True,
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            # Measured max output ~870 tokens; 2048 stops runaway CoT from
            # monopolizing KV without truncating real answers.
            "max_tokens": 16384,
            "repetition_penalty": 1.0,
        },
    },
    # CONFIRMED non-thinking
    # Dense 24B; ~48 GiB bf16 + encoder. Mistral tokenizer/format.
    # https://huggingface.co/mistralai/Voxtral-Small-24B-2507
    "voxtral-small-24b": {
        "model_id": "mistralai/Voxtral-Small-24B-2507",
        "gpu": "B200",
        "backend": "vllm_voxtral",
        "engine": {
            "dtype": "bfloat16",
            "max_model_len": 8192,
            # 40L × 8 kv × 128 × 2 × 2B ≈ 160 KiB/tok. B200 180 GiB × 0.95 −
            # ~55 GiB weights ⇒ ~700k KV tokens. 256 seqs × ~2.5k typical
            # MMAR (non-thinking, max_tokens=2048) sits under the pool.
            "max_num_seqs": 256,
            "max_num_batched_tokens": 32768,
            "limit_mm_per_prompt": {"audio": 1},
            "config_format": "mistral",
            "load_format": "mistral",
            "tokenizer_mode": "mistral",
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            "attention_backend": "flashinfer",  # best for throughput
            "async_scheduling": True,  # usually faster, but not all features supported
        },
        # Card: temperature=0.2 and top_p=0.95 for audio-understanding chat.
        "sampling": {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 2048,
            "repetition_penalty": 1.0,
        },
    },
    # CONFIRMED non-thinking
    # 5.6B; speech LoRA lives next to the checkpoint. Card uses
    # GenerationConfig.from_pretrained + max_new_tokens=1000 (greedy).
    # T=0.2 keeps n-shot variance.
    # https://huggingface.co/microsoft/Phi-4-multimodal-instruct
    "phi-4-multimodal": {
        "model_id": "microsoft/Phi-4-multimodal-instruct",
        "gpu": "L40S",
        "backend": "vllm",
        "engine": {
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 8192,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": False,
            "trust_remote_code": True,
            "enable_lora": True,
            "max_lora_rank": 320,
            "max_loras": 1,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            "attention_backend": "flashinfer",
            "async_scheduling": True,
        },
        "sampling": {
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 1000,
            "repetition_penalty": 1.0,
        },
    },
    # CONFIRMED
    # Effective 4B; native audio. Thinking → B200 (KV is tiny vs 180 GiB).
    # https://huggingface.co/google/gemma-4-E4B-it#best-practices
    "gemma-4-e4b": {
        "model_id": "google/gemma-4-E4B-it",
        "gpu": "B200",
        "backend": "vllm_chat",
        "engine": {
            "dtype": "bfloat16",
            "max_model_len": 8192,
            # Hybrid sliding/full + 18 shared-KV layers; even the full-attn
            # upper bound is ~84 KiB/tok. 512 seqs is decode occupancy, not
            # a KV limit. 32k batched tokens for audio prefill.
            "max_num_seqs": 512,
            "max_num_batched_tokens": 32768,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            # Gemma-4 head_size is unsupported by FLASH_ATTN. FlashInfer JIT
            # targets sm100+ (B200); keep Triton — hybrid k_eq_v / global
            # head_dim=512 is the path that loaded on both SM versions.
            "attention_backend": "TRITON_ATTN",
            "async_scheduling": True,
            "allowed_local_media_path": "/",
        },
        # chat_template.jinja: enable_thinking | default(false). Pass True so
        # the template injects <|think|>.
        # Best Practices / generation_config.json: T=1.0, top_p=0.95, top_k=64.
        # README generate() example uses max_new_tokens=1024.
        "native_thinking": True,
        "enable_thinking": True,
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 1024,
            "repetition_penalty": 1.0,
        },
    },
    # CONFIRMED
    # Dense 12B unified (encoder-free audio+vision).
    # https://huggingface.co/google/gemma-4-12B-it#best-practices
    "gemma-4-12b": {
        "model_id": "google/gemma-4-12B-it",
        "gpu": "B200",
        "backend": "vllm_chat",
        "engine": {
            "dtype": "bfloat16",
            "max_model_len": 8192,
            # 8 full (1 kv × 512) + 40 sliding (8 kv × 256, window 1024) ≈
            # 0.40–0.47 GiB/seq at 4–8k. B200 180×0.95 − ~35 GiB weights ⇒
            # ~136 GiB KV ≈ 290 seqs. 256 leaves room for 16k-token prefills.
            "max_num_seqs": 256,
            "max_num_batched_tokens": 16384,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            # Gemma-4 head_size is unsupported by FLASH_ATTN. FlashInfer JIT
            # targets sm100+ (B200); keep Triton — hybrid k_eq_v / global
            # head_dim=512 is the path that loaded on both SM versions.
            "attention_backend": "TRITON_ATTN",
            "async_scheduling": True,
            "allowed_local_media_path": "/",
        },
        # chat_template.jinja: enable_thinking | default(false). Pass True so
        # the template injects <|think|>. vLLM gemma4 parser splits thought.
        # Best Practices / generation_config.json: T=1.0, top_p=0.95, top_k=64.
        # README generate() example uses max_new_tokens=1024.
        "native_thinking": True,
        "enable_thinking": True,
        "reasoning_parser": "gemma4",
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 4096,
            "repetition_penalty": 1.1,
        },
    },
    # # Block-wise FP8 of Qwen3-Omni Instruct (thinker+talker MoE); H100 native FP8.
    # # Official Instruct eval is greedy.
    # "qwen3-omni-instruct": {
    #     "model_id": "marksverdhei/Qwen3-Omni-30B-A3B-FP8",
    #     "gpu": "H100",
    #     "backend": "vllm",
    #     "engine": {
    #         "dtype": "auto",
    #         "max_model_len": 4096,
    #         "max_num_seqs": 64,
    #         "max_num_batched_tokens": 8192,
    #         # Unspecified modalities default to 999. Dummy video profiling then
    #         # runs the FP8 vision MLP (hidden 4304) which is not divisible by
    #         # the 128-wide block size and asserts in per_token_group_quant_fp8.
    #         "limit_mm_per_prompt": {"audio": 1, "image": 0, "video": 0},
    #         "enforce_eager": True,
    #         "trust_remote_code": True,
    #         "enable_prefix_caching": True,
    #         "gpu_memory_utilization": 0.95,
    #         "disable_log_stats": False,
    #         "attention_backend": "flashinfer",
    #         "async_scheduling": True,
    #     },
    #     "sampling": {
    #         "temperature": 0.2,
    #         "top_p": 1.0,
    #         "max_tokens": 2048,
    #         "repetition_penalty": 1.0,
    #     },
    # },
    # Native FP8 MoE. Thinking-mode card Best Practices.
    # https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8#best-practices
    "nemotron-3-nano-omni": {
        "model_id": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        "gpu": "B200",
        "backend": "vllm_chat",
        "engine": {
            "dtype": "auto",
            "max_model_len": 8192,
            # Hybrid Mamba/attn MoE, FP8 weights ~30 GiB, FP8 KV on ~23 attn
            # layers × 2 kv × 128 ≈ 23 KiB/tok. KV is not the limiter; 512
            # seqs fills decode occupancy, 32k batched tokens for MoE prefill.
            "max_num_seqs": 512,
            "max_num_batched_tokens": 32768,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": False,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            "attention_backend": "flashinfer",
            # FlashInfer fp8_gemm autotune segfaults on B200 (cuBLAS bmm_fp8;
            # vllm#39814). CUTLASS SM100 default is used instead.
            "enable_flashinfer_autotune": False,
            "async_scheduling": True,
            "allowed_local_media_path": "/",
        },
        "native_thinking": True,
        "enable_thinking": True,
        # Card: extra_body thinking_token_budget = reasoning_budget + grace_period.
        "reasoning_budget": 2048,
        "reasoning_parser": "nemotron_v3",
        "grace_period": 512,
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 4096,
            "repetition_penalty": 1.0,
        },
    },
    # Text-only VL MoE (122B / 10B active), native block-wise FP8. No audio
    # encoder; language_model_only skips the vision tower so a single B200
    # (~180 GiB) can hold the ~122 GiB weights plus KV.
    # https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8
    # Card thinking/general: T=1.0, top_p=0.95, top_k=20, presence_penalty=1.5.
    # generation_config.json is T=0.6 (coding); we use the general recipe.
    # "qwen3.5-122b-a10b-fp8": {
    #     "model_id": "Qwen/Qwen3.5-122B-A10B-FP8",
    #     "gpu": "B200",
    #     "backend": "vllm_chat",
    #     "text_only": True,
    #     "engine": {
    #         "dtype": "auto",
    #         "max_model_len": 16384,
    #         # 12 full-attn layers × 2 kv × 256 × 2 × 2B ≈ 24 KiB/tok. B200
    #         # 180×0.90 − ~122 GiB FP8 weights ⇒ ~40 GiB KV ≈ 1.6M tokens.
    #         # 64 seqs × 16k sits under that; 16k batched tokens for prefill.
    #         "max_num_seqs": 64,
    #         "max_num_batched_tokens": 16384,
    #         "enforce_eager": False,
    #         "trust_remote_code": True,
    #         "enable_prefix_caching": True,
    #         "gpu_memory_utilization": 0.90,
    #         "disable_log_stats": False,
    #         "attention_backend": "flashinfer",
    #         # FlashInfer fp8_gemm autotune segfaults on B200 (cuBLAS bmm_fp8;
    #         # vllm#39814). CUTLASS SM100 default is used instead.
    #         "enable_flashinfer_autotune": False,
    #         "async_scheduling": True,
    #         "language_model_only": True,
    #     },
    #     "native_thinking": True,
    #     "enable_thinking": True,
    #     "reasoning_parser": "qwen3",
    #     "sampling": {
    #         "temperature": 1.0,
    #         "top_p": 0.95,
    #         "top_k": 20,
    #         "presence_penalty": 1.5,
    #         "max_tokens": 8192,
    #         "repetition_penalty": 1.0,
    #     },
    # },
}

ALL_MODEL_LABELS = tuple(MODEL_SPECS.keys())
NATIVE_THINKING_LABELS = frozenset(
    label for label, spec in MODEL_SPECS.items() if spec.get("native_thinking")
)
THINKING_ENABLED_LABELS = tuple(
    label for label, spec in MODEL_SPECS.items() if spec.get("enable_thinking")
)


def has_native_thinking(label: str) -> bool:
    """True when the checkpoint emits native ``<think>`` / reasoning traces."""
    return label in NATIVE_THINKING_LABELS


def thinking_enabled(label: str, args: SimpleNamespace | None = None) -> bool:
    """Whether native thinking is active for this request.

    When ``args.enable_thinking`` is set (``True`` or ``False``), it wins.
    Otherwise fall back to ``MODEL_SPECS[label].enable_thinking``, then
    ``native_thinking``.
    """
    if args is not None:
        override = getattr(args, "enable_thinking", None)
        if override is not None:
            return bool(override)
    spec = MODEL_SPECS.get(label) or {}
    if "enable_thinking" in spec:
        return bool(spec["enable_thinking"])
    return bool(spec.get("native_thinking"))


def _maybe_assistant_think_open(
    label: str,
    prompt: str,
    args: SimpleNamespace | None = None,
) -> str:
    if not thinking_enabled(label, args):
        return prompt
    return ensure_assistant_think_open(label, prompt)


def chat_kwargs_for(
    label: str,
    args: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """vLLM ``LLM.chat`` kwargs when the chat template defines ``enable_thinking``."""
    spec = MODEL_SPECS.get(label) or {}
    if "enable_thinking" not in spec:
        return {}
    enabled = thinking_enabled(label, args)
    template_kwargs: dict[str, Any] = {
        "enable_thinking": enabled,
    }
    if enabled and spec.get("reasoning_budget") is not None:
        template_kwargs["reasoning_budget"] = int(spec["reasoning_budget"])
    return {"chat_template_kwargs": template_kwargs}


def engine_kwargs_for(label: str, args: SimpleNamespace) -> dict[str, Any]:
    """vLLM ``LLM(...)`` kwargs: spec ``engine`` plus optional ``reasoning_parser``."""
    spec = MODEL_SPECS.get(label) or {}
    engine = _apply_engine_overrides(dict(spec.get("engine") or {}), args)
    if thinking_enabled(label, args):
        parser = spec.get("reasoning_parser")
        if parser:
            engine["reasoning_parser"] = str(parser)
    return engine


def parse_model_list(value: str) -> list[str]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw or any(item.lower() == "all" for item in raw):
        return list(ALL_MODEL_LABELS)
    unknown = [item for item in raw if item not in MODEL_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown model label(s): {unknown}. "
            f"Choose from {list(ALL_MODEL_LABELS)} or 'all'."
        )
    return list(dict.fromkeys(raw))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _output_dict(
    raw_text: str,
    choices: list,
    *,
    parse_fn: Callable = parse_choice_output,
) -> dict:
    thinking, answer = parse_fn(raw_text, choices)
    return {
        "model_output": raw_text,
        "raw_tokens": None,
        "thinking_prediction": thinking,
        "answer_prediction": answer,
    }


def _load_audio_tuple(
    path: str,
    sampling_rate: int,
    *,
    max_samples: int | None = None,
) -> tuple[Any, int]:
    import librosa
    import numpy as np

    audio, sr = librosa.load(path, sr=sampling_rate, mono=True)
    audio = audio.astype(np.float32)
    if max_samples is not None and audio.shape[0] > max_samples:
        audio = audio[: int(max_samples)]
    return audio, int(sr)


def resolve_sampling(
    label: str,
    args: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """Return SamplingParams kwargs for ``label`` (per-model; no global defaults).

    Optional CLI overrides on ``args`` (``temperature``, ``top_p``,
    ``max_new_tokens``) replace the model values when not ``None``.
    ``greedy_non_thinking`` forces ``temperature=0`` on models without native
    ``<think>`` / reasoning mode; thinking models keep their card sampling
    unless ``temperature`` is also set.
    """
    spec = MODEL_SPECS.get(label)
    if not spec or "sampling" not in spec:
        raise ValueError(f"Model {label!r} has no per-model sampling config")
    out = dict(spec["sampling"])
    reasoning_budget = spec.get("reasoning_budget")
    if reasoning_budget is not None and (args is None or thinking_enabled(label, args)):
        thinking = int(reasoning_budget)
        grace_period = spec.get("grace_period")
        if grace_period is not None:
            thinking += int(grace_period)
        out.setdefault("thinking_token_budget", thinking)
    if args is None:
        return out
    if getattr(args, "greedy_non_thinking", False) and not spec.get("native_thinking"):
        out["temperature"] = 0.0
    if getattr(args, "temperature", None) is not None:
        out["temperature"] = float(args.temperature)
    if getattr(args, "top_p", None) is not None:
        out["top_p"] = float(args.top_p)
    if getattr(args, "max_new_tokens", None) is not None:
        out["max_tokens"] = int(args.max_new_tokens)
    return out


def _sampling_params_for_request(
    label: str,
    args: SimpleNamespace,
    seed: int,
    *,
    n: int = 1,
    stop_token_ids: list[int] | None = None,
    repetition_penalty: float | None = None,
):
    from vllm import SamplingParams

    kwargs = resolve_sampling(label, args)
    temperature = float(kwargs.get("temperature", 0.0))
    kwargs["temperature"] = temperature if temperature > 0 else 0.0
    kwargs["seed"] = int(seed)
    kwargs["n"] = int(n)
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = float(repetition_penalty)
    if stop_token_ids:
        kwargs["stop_token_ids"] = list(stop_token_ids)
    return SamplingParams(**kwargs)


def _completion_texts(output: Any) -> list[str]:
    """Extract all completion strings from a vLLM / Omni generate output."""
    if output is None:
        return [""]
    if isinstance(output, str):
        return [output]
    # Omni stage wrapper: request_output may be one RequestOutput or a list.
    request_output = getattr(output, "request_output", None)
    if request_output is not None:
        if isinstance(request_output, (list, tuple)):
            parts = [_extract_text(item) for item in request_output]
            joined = "\n".join(part for part in parts if part)
            return [joined]
        output = request_output
    outputs = getattr(output, "outputs", None)
    if outputs:
        texts: list[str] = []
        for item in outputs:
            text = getattr(item, "text", None)
            texts.append(
                join_vllm_reasoning(item, str(text) if text is not None else "")
            )
        if texts:
            return texts
    text = getattr(output, "text", None)
    if text is not None or vllm_reasoning_text(output):
        return [join_vllm_reasoning(output, str(text) if text is not None else "")]
    return [str(output)]


def _extract_text(output: Any) -> str:
    """Normalize vLLM / Omni generate outputs to a decoded string."""
    texts = _completion_texts(output)
    return texts[0] if texts else ""


def _prompt_mode(args: SimpleNamespace) -> str:
    mode = str(getattr(args, "prompt_mode", "mc") or "mc").lower()
    if mode in {"freeform", "free_form", "open"}:
        return "freeform"
    if mode in {"description", "describe", "caption"}:
        return "description"
    return "mc"


def _build_prompt(
    sample: dict,
    args: SimpleNamespace,
    *,
    think_suffix: str | None = None,
    with_timestamps: bool = False,
) -> str:
    if _prompt_mode(args) == "description":
        return build_mmar_description_prompt()
    if _prompt_mode(args) == "freeform":
        # AF-Next: bake timestamps into the reason sentence instead of appending
        # AF_NEXT_THINK_SUFFIX (which would duplicate "reason step by step").
        if think_suffix == AF_NEXT_THINK_SUFFIX:
            with_timestamps = True
            think_suffix = None
        return build_mmar_freeform_prompt(
            sample,
            think_suffix=think_suffix,
            with_timestamps=with_timestamps,
        )
    return build_mmar_prompt(sample, think_suffix=think_suffix)


def _parse_fn_for(
    args: SimpleNamespace,
    default: Callable = parse_choice_output,
    *,
    label: str = "",
) -> Callable:
    if default is parse_music_flamingo_output:
        fallback = (
            parse_freeform_output
            if _prompt_mode(args) == "freeform"
            else parse_think_tagged_output
        )

        def parse(raw_text, choices=None):
            return parse_music_flamingo_output(raw_text, choices, fallback=fallback)

        return parse

    if _prompt_mode(args) == "description":
        return parse_description_output
    if _prompt_mode(args) == "freeform":
        if default is parse_think_tagged_output:
            # Free-form still strips <think> blocks; choice matching is skipped.
            return parse_freeform_output
        return parse_freeform_output
    return default


def _vllm_prompt_fn(label: str, args: SimpleNamespace) -> Callable[[dict], str]:
    """String prompt builder used by the plain vLLM ``generate`` path."""
    builders: dict[str, Callable] = {
        "af-next-think": _af_next_prompt,
        "music-flamingo": _music_flamingo_prompt,
        "qwen3-omni": _qwen3_omni_prompt,
        "qwen3-omni-instruct": _qwen3_omni_prompt,
        "phi-4-multimodal": _phi4_prompt,
    }
    builder = builders.get(label, _build_prompt)

    def prompt_fn(sample: dict) -> str:
        return builder(sample, args)

    return prompt_fn


def _format_chat_messages(messages: list[dict]) -> str:
    """Readable dump of ``LLM.chat`` / HF ``.chat`` message payloads."""
    chunks: list[str] = []
    for message in messages:
        role = message.get("role") or "user"
        chunks.append(f"role={role}")
        chunks.append(_content_to_text(message.get("content")))
        chunks.append("")
    return "\n".join(chunks).rstrip() + "\n"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(str(item.get("text") or ""))
        elif kind == "audio_url":
            url = (item.get("audio_url") or {}).get("url") or ""
            parts.append(f"<audio_url>{url}</audio_url>")
        elif kind == "audio":
            audio = item.get("audio") or item.get("audio_url") or ""
            parts.append(f"<audio>{audio}</audio>")
        else:
            parts.append(str(item))
    return "\n".join(parts)


def render_prompt(
    label: str,
    sample: dict,
    args: SimpleNamespace | None = None,
) -> str:
    """Return the text ``generate_batch`` would send for ``label``.

    Audio is represented by the same placeholders used at generate time
    (``<sound>``, ``<|audio_pad|>``, ``file://`` URLs, etc.). Chat backends
    (``vllm_chat``) dump the messages ``LLM.chat`` receives; the model's
    Jinja chat template is applied later by vLLM.
    """
    if label not in MODEL_SPECS:
        raise ValueError(f"Unknown model label {label!r}")
    ns = args or SimpleNamespace(prompt_mode="mc")
    backend = str(MODEL_SPECS[label].get("backend") or "")

    if backend == "vllm_chat":
        return _format_chat_messages(_chat_messages_for(label, sample, ns))
    if backend == "vllm_voxtral":
        return _build_prompt(sample, ns)

    prompt_fn = _vllm_prompt_fn(label, ns)
    return _maybe_assistant_think_open(label, prompt_fn(sample), ns)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


# From nvidia/audio-flamingo-next-think-hf chat_template.jinja (injected when
# the conversation has no system turn).
AF_NEXT_SYSTEM = (
    "You are Audio Flamingo-Next, a multimodal assistant for language and "
    "audio. On each turn you receive an optional audio clip which may contain "
    "speech, music, or ambient sounds and optional text, you will receive at "
    "least one or both; use your world knowledge and reasoning to help the "
    "user with any task. Interpret the entirety of the content of any input "
    "audio—regardless of whether the user calls it audio, speech, music, or "
    "sound."
)

# Injected by nvidia/music-flamingo-hf chat_template.jinja when there is no
# system turn.
MUSIC_FLAMINGO_SYSTEM = "You are a helpful assistant."


def _af_next_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    ns = args or SimpleNamespace(prompt_mode="mc")
    mode = _prompt_mode(ns)
    freeform = mode == "freeform"
    think_suffix = None
    if mode != "description" and thinking_enabled("af-next-think", ns):
        think_suffix = None if freeform else AF_NEXT_THINK_SUFFIX
    question = _build_prompt(
        sample,
        ns,
        think_suffix=think_suffix,
        with_timestamps=freeform and thinking_enabled("af-next-think", ns),
    )
    assistant = ASSISTANT_THINK_OPEN if thinking_enabled("af-next-think", ns) else ""
    # MusicFlamingo / AF-Next chat format (same placeholder family as AF3).
    return (
        f"<|im_start|>system\n{AF_NEXT_SYSTEM}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<sound>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant}"
    )


def _music_flamingo_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    ns = args or SimpleNamespace(prompt_mode="mc")
    think_suffix = None
    if _prompt_mode(ns) != "description" and thinking_enabled("music-flamingo", ns):
        think_suffix = MUSIC_FLAMINGO_THINK_SUFFIX
    question = _build_prompt(sample, ns, think_suffix=think_suffix)
    assistant = ASSISTANT_THINK_OPEN if thinking_enabled("music-flamingo", ns) else ""
    return (
        f"<|im_start|>system\n{MUSIC_FLAMINGO_SYSTEM}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<sound>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant}"
    )


def _qwen3_omni_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    # Qwen3-Omni Thinking primary examples omit a system turn; the chat
    # template only emits one when messages[0].role == "system". Instruct
    # eval notes also say no system prompt. Thinking stays on: the template
    # injects empty <think></think> only when enable_thinking is false.
    ns = args or SimpleNamespace(prompt_mode="mc")
    question = _build_prompt(sample, ns)
    think_disable = "" if thinking_enabled("qwen3-omni", ns) else "<think></think>"
    return (
        "<|im_start|>user\n"
        f"<|audio_start|><|audio_pad|><|audio_end|>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{think_disable}"
    )


def _phi4_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    question = _build_prompt(sample, args or SimpleNamespace(prompt_mode="mc"))
    return f"<|user|><|audio_1|>{question}<|end|><|assistant|>"


def _audio_text_messages(
    sample: dict, args: SimpleNamespace | None = None
) -> list[dict]:
    prompt = _build_prompt(sample, args or SimpleNamespace(prompt_mode="mc"))
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"file://{sample['audio_path']}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _chat_messages_for(
    label: str,
    sample: dict,
    args: SimpleNamespace | None = None,
) -> list[dict]:
    """Chat messages for ``vllm_chat``: audio+text, or text only."""
    spec = MODEL_SPECS.get(label) or {}
    if spec.get("text_only"):
        prompt = _build_prompt(sample, args or SimpleNamespace(prompt_mode="mc"))
        return [{"role": "user", "content": prompt}]
    return _audio_text_messages(sample, args)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _vllm_model_view_without_latent_cache(model_dir: str) -> Path:
    """Build a temp view of ``model_dir`` that omits latent CoT remap caches.

    vLLM loads every ``*.safetensors`` under the model path. Our
    ``latent_w_remap.safetensors`` cache uses keys ``weight`` /
    ``avg_embed_norm`` and breaks MusicFlamingo weight loading.
    """
    import tempfile

    src = Path(model_dir)
    view = Path(tempfile.mkdtemp(prefix="af_next_vllm_"))
    for path in src.iterdir():
        if path.name.startswith("latent_w_remap") and path.suffix == ".safetensors":
            continue
        # Also skip previously hidden caches.
        if path.name.endswith(".vllm_hide"):
            continue
        target = view / path.name
        try:
            target.symlink_to(path)
        except OSError:
            if path.is_dir():
                os.symlink(path, target, target_is_directory=True)
            else:
                # Fallback: hardlink when possible, else skip directories.
                try:
                    os.link(path, target)
                except OSError:
                    import shutil

                    if path.is_file():
                        shutil.copy2(path, target)
    return view


def load_af_next(args: SimpleNamespace):
    from vllm import LLM

    spec = MODEL_SPECS["af-next-think"]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = _apply_engine_overrides(spec["engine"], args)

    # Prefer native MusicFlamingo (vLLM 0.28).
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    vllm_model_path = _vllm_model_view_without_latent_cache(local_id)
    try:
        llm = LLM(model=str(vllm_model_path), **engine)
        print(
            f"AF-Next vLLM ready from {local_id} "
            f"(view={vllm_model_path}) engine={engine}"
        )
        return {"backend": "vllm", "llm": llm, "parse_fn": parse_think_tagged_output}
    except Exception as exc:  # noqa: BLE001
        print(f"AF-Next vLLM load failed ({exc}); falling back to HF generate_batch")
        return _load_af_hf(
            local_id,
            args,
            parse_fn=parse_think_tagged_output,
            think_suffix=AF_NEXT_THINK_SUFFIX,
        )


def load_music_flamingo(args: SimpleNamespace):
    from vllm import LLM

    spec = MODEL_SPECS["music-flamingo"]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = _apply_engine_overrides(spec["engine"], args)

    # Native AudioFlamingo3 / MusicFlamingo path in vLLM 0.28.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    try:
        llm = LLM(model=local_id, **engine)
        print(f"Music Flamingo vLLM ready from {local_id} engine={engine}")
        return {"backend": "vllm", "llm": llm, "parse_fn": parse_music_flamingo_output}
    except Exception as exc:  # noqa: BLE001
        print(
            f"Music Flamingo vLLM load failed ({exc}); falling back to HF generate_batch"
        )
        return _load_af_hf(
            local_id,
            args,
            parse_fn=parse_music_flamingo_output,
            think_suffix=MUSIC_FLAMINGO_THINK_SUFFIX,
        )


def _load_af_hf(
    local_id: str,
    args: SimpleNamespace,
    *,
    parse_fn: Callable,
    think_suffix: str | None,
):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoProcessor

    from audio_flamingo_runtime import (
        audio_tower_dtype,
        cast_model_floating_tensors,
        model_input_device,
        model_param_dtype,
        torch_dtype_value,
    )

    target_dtype = torch_dtype_value(torch, getattr(args, "torch_dtype", "bfloat16"))
    if target_dtype == "auto":
        target_dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(local_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_id, dtype=target_dtype, device_map="auto"
    )
    cast_model_floating_tensors(model, target_dtype)
    model.eval()
    print(
        f"Audio Flamingo HF ready: class={type(model).__name__} "
        f"param_dtype={model_param_dtype(model)} "
        f"audio_tower_dtype={audio_tower_dtype(model)} "
        f"device={model_input_device(model)}"
    )
    return {
        "backend": "hf_af_next",
        "model": model,
        "processor": processor,
        "parse_fn": parse_fn,
        "think_suffix": think_suffix,
    }


def _apply_engine_overrides(engine: dict, args: SimpleNamespace) -> dict:
    out = dict(engine)
    if getattr(args, "max_num_seqs", None):
        out["max_num_seqs"] = int(args.max_num_seqs)
    if getattr(args, "gpu_memory_utilization", None):
        out["gpu_memory_utilization"] = float(args.gpu_memory_utilization)
    if getattr(args, "max_model_len", None):
        out["max_model_len"] = int(args.max_model_len)
    if getattr(args, "enforce_eager", None) is not None:
        if bool(args.enforce_eager):
            out.update(_eager_startup_kwargs())
        else:
            out["enforce_eager"] = False
    return out


def _engine_arg_names() -> frozenset[str]:
    """Top-level ``LLM(...)`` / ``EngineArgs`` field names for this vLLM."""
    try:
        from vllm.engine.arg_utils import EngineArgs
    except ImportError:
        return frozenset()
    fields = getattr(EngineArgs, "__dataclass_fields__", None)
    if fields:
        return frozenset(fields)
    model_fields = getattr(EngineArgs, "model_fields", None)
    if model_fields:
        return frozenset(model_fields)
    return frozenset()


def _eager_startup_kwargs() -> dict[str, Any]:
    """Skip torch.compile, CUDA graphs, and kernel autotune warmup.

    ``enforce_eager`` is the portable knob (vLLM 0.24+). Newer EngineArgs also
    expose FlashInfer / JIT warmup flags and ``optimization_level``.
    """
    known = _engine_arg_names()
    wanted: dict[str, Any] = {
        "enforce_eager": True,
        "enable_flashinfer_autotune": False,
        "enable_jit_warmup": False,
        "enable_cutedsl_warmup": False,
        "optimization_level": 0,
    }
    if not known:
        return {"enforce_eager": True}
    return {key: value for key, value in wanted.items() if key in known}


def load_qwen3_omni(args: SimpleNamespace):
    """Qwen3-Omni Thinking via plain vLLM thinker-only path."""
    return _load_qwen3_family(
        args, label="qwen3-omni", parse_fn=parse_think_tagged_output
    )


def load_qwen3_omni_instruct(args: SimpleNamespace):
    """Qwen3-Omni Instruct via the same thinker-only vLLM path."""
    return _load_qwen3_family(
        args, label="qwen3-omni-instruct", parse_fn=parse_choice_output
    )


def _load_qwen3_family(args: SimpleNamespace, *, label: str, parse_fn: Callable):
    from vllm import LLM

    spec = MODEL_SPECS[label]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = _apply_engine_overrides(spec["engine"], args)
    llm = LLM(model=local_id, **engine)
    print(f"{label} vLLM thinker ready from {local_id} engine={engine}")
    return {
        "backend": "vllm",
        "llm": llm,
        "parse_fn": parse_fn,
    }


def load_phi4_multimodal(args: SimpleNamespace):
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    spec = MODEL_SPECS["phi-4-multimodal"]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = _apply_engine_overrides(spec["engine"], args)
    speech_lora = Path(local_id) / "speech-lora"
    if not speech_lora.is_dir():
        raise SystemExit(f"Phi-4 speech-lora not found at {speech_lora}")
    llm = LLM(model=local_id, **engine)
    print(f"Phi-4-multimodal vLLM ready from {local_id} speech_lora={speech_lora}")
    return {
        "backend": "vllm",
        "llm": llm,
        "parse_fn": parse_choice_output,
        "lora_request": LoRARequest("speech", 1, str(speech_lora)),
    }


def load_gemma_4(label: str, args: SimpleNamespace):
    from vllm import LLM

    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = engine_kwargs_for(label, args)
    llm = LLM(model=local_id, **engine)
    print(f"{label} vLLM chat ready from {local_id} engine={engine}")
    return {
        "backend": "vllm_chat",
        "llm": llm,
        "parse_fn": parse_think_tagged_output,
        "chat_kwargs": chat_kwargs_for(label),
    }


def load_gemma_4_e4b(args: SimpleNamespace):
    return load_gemma_4("gemma-4-e4b", args)


def load_gemma_4_12b(args: SimpleNamespace):
    return load_gemma_4("gemma-4-12b", args)


def load_nemotron_omni(args: SimpleNamespace):
    from vllm import LLM

    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = engine_kwargs_for("nemotron-3-nano-omni", args)
    llm = LLM(model=local_id, **engine)
    print(f"Nemotron-3-Nano-Omni vLLM chat ready from {local_id} engine={engine}")
    return {
        "backend": "vllm_chat",
        "llm": llm,
        "parse_fn": parse_think_tagged_output,
        "chat_kwargs": chat_kwargs_for("nemotron-3-nano-omni"),
    }


def load_qwen35_122b(args: SimpleNamespace):
    """Qwen3.5-122B-A10B-FP8 text-only (skip vision tower)."""
    from vllm import LLM

    label = "qwen3.5-122b-a10b-fp8"
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = engine_kwargs_for(label, args)
    llm_kwargs = dict(engine)
    try:
        llm = LLM(model=local_id, **llm_kwargs)
    except TypeError as exc:
        if not llm_kwargs.get("language_model_only"):
            raise
        print(f"[{label}] language_model_only unsupported ({exc}); retrying without it")
        llm_kwargs.pop("language_model_only", None)
        llm = LLM(model=local_id, **llm_kwargs)
    print(f"{label} vLLM chat ready from {local_id} engine={llm_kwargs}")
    return {
        "backend": "vllm_chat",
        "llm": llm,
        "parse_fn": parse_think_tagged_output,
        "chat_kwargs": chat_kwargs_for(label),
    }


def _load_voxtral_tokenizer(local_id: str):
    """Load Mistral tokenizer from a seeded local dir, else Hub."""
    try:
        from vllm.tokenizers.mistral import MistralTokenizer
    except ImportError:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    for name in ("tekken.json", "tokenizer.model.v3", "tokenizer.model"):
        candidate = Path(local_id) / name
        if candidate.is_file() and hasattr(MistralTokenizer, "from_file"):
            return MistralTokenizer.from_file(str(candidate))
    if hasattr(MistralTokenizer, "from_pretrained"):
        return MistralTokenizer.from_pretrained(local_id)
    return MistralTokenizer.from_hf_hub(local_id)


def load_voxtral(args: SimpleNamespace):
    """Voxtral Small 24B via vLLM with Mistral audio tokenization."""
    from vllm import LLM

    spec = MODEL_SPECS["voxtral-small-24b"]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = _apply_engine_overrides(spec["engine"], args)
    llm = LLM(model=local_id, **engine)
    tokenizer = _load_voxtral_tokenizer(local_id)
    print(f"Voxtral vLLM ready from {local_id} engine={engine}")
    return {
        "backend": "vllm_voxtral",
        "llm": llm,
        "tokenizer": tokenizer,
        "model_id": local_id,
        "parse_fn": parse_choice_output,
    }


# ---------------------------------------------------------------------------
# Batch generators
# ---------------------------------------------------------------------------


def _build_vllm_audio_inputs(
    label: str,
    samples: list[dict],
    *,
    prompt_fn: Callable[[dict], str],
    sampling_rate: int,
    args: SimpleNamespace,
    seeds: list[int],
    n_completions: int = 1,
    stop_token_ids: list[int] | None = None,
    repetition_penalty: float | None = None,
    max_audio_samples: int | None = None,
) -> tuple[list[dict], list[Any]]:
    prompts: list[dict] = []
    sampling: list[Any] = []
    for sample, seed in zip(samples, seeds):
        waveform, sr = _load_audio_tuple(
            sample["audio_path"],
            sampling_rate,
            max_samples=max_audio_samples,
        )
        mm_audio = (waveform, sr)
        prompt_text = _maybe_assistant_think_open(label, prompt_fn(sample), args)
        prompts.append(
            {
                "prompt": prompt_text,
                "multi_modal_data": {"audio": mm_audio},
            }
        )
        sampling.append(
            _sampling_params_for_request(
                label,
                args,
                seed,
                n=n_completions,
                stop_token_ids=stop_token_ids,
                repetition_penalty=repetition_penalty,
            )
        )
    return prompts, sampling


def _expand_n_outputs(
    samples: list[dict],
    outputs: list[Any],
    *,
    n_completions: int,
    parse_fn: Callable,
) -> list[dict]:
    """Unpack SamplingParams(n=...) completions into question-major shot rows."""
    results: list[dict] = []
    for sample, out in zip(samples, outputs):
        texts = _completion_texts(out)
        if len(texts) < n_completions:
            texts = texts + [""] * (n_completions - len(texts))
        choices = sample.get("choices") or []
        for text in texts[:n_completions]:
            results.append(_output_dict(text, choices, parse_fn=parse_fn))
    return results


def generate_batch(
    label: str,
    handle: dict,
    samples: list[dict],
    args: SimpleNamespace,
    *,
    seeds: list[int] | None = None,
    n_completions: int = 1,
) -> list[dict]:
    """Generate completions for ``samples``.

    Plain vLLM backends may pass unique questions with ``n_completions=n_shots``
    so SamplingParams(n=...) shares prefill. Omni / HF callers should pass
    already-expanded question×shot rows with ``n_completions=1``.

    Returns a flat list in question-major order (length
    ``len(samples) * n_completions``).
    """
    if not samples:
        return []
    if seeds is None:
        seeds = [int(args.seed) + i for i in range(len(samples))]
    if len(seeds) != len(samples):
        raise ValueError("seeds length must match samples length")
    n_completions = max(1, int(n_completions))

    backend = handle["backend"]
    parse_fn = _parse_fn_for(
        args, handle.get("parse_fn", parse_choice_output), label=label
    )

    if backend == "vllm":
        prompt_fn = _vllm_prompt_fn(label, args)
        prompts, sampling = _build_vllm_audio_inputs(
            label,
            samples,
            prompt_fn=prompt_fn,
            sampling_rate=16000,
            args=args,
            seeds=seeds,
            n_completions=n_completions,
        )
        generate_kwargs: dict[str, Any] = {}
        if handle.get("lora_request") is not None:
            generate_kwargs["lora_request"] = handle["lora_request"]
        outputs = handle["llm"].generate(
            prompts, sampling_params=sampling, **generate_kwargs
        )
        return _expand_n_outputs(
            samples, outputs, n_completions=n_completions, parse_fn=parse_fn
        )

    if backend == "hf_af_next":
        from audio_flamingo_runtime import generate_batch as af_generate_batch
        from audio_flamingo_runtime import seed_everything

        if n_completions != 1:
            raise ValueError("hf_af_next requires expanded shot rows (n_completions=1)")
        sampling = resolve_sampling(label, args)
        think_suffix = None
        if thinking_enabled(label, args):
            think_suffix = handle.get("think_suffix")
            if think_suffix is None:
                if label == "af-next-think":
                    think_suffix = (
                        None
                        if _prompt_mode(args) == "freeform"
                        else AF_NEXT_THINK_SUFFIX
                    )
                elif label == "music-flamingo":
                    think_suffix = (
                        None
                        if _prompt_mode(args) == "description"
                        else MUSIC_FLAMINGO_THINK_SUFFIX
                    )
                else:
                    think_suffix = AF3_THINK_SUFFIX
        assistant_prefill = (
            ASSISTANT_THINK_OPEN
            if thinking_enabled(label, args) and label in PREFIX_ASSISTANT_THINK_LABELS
            else None
        )
        # HF path has no per-row seeds in one generate call; run one sample at a
        # time so flattened question×shot rows keep distinct seeds.
        results: list[dict] = []
        for sample, row_seed in zip(samples, seeds):
            seed_everything(int(row_seed))
            results.extend(
                af_generate_batch(
                    handle["model"],
                    handle["processor"],
                    [sample],
                    args,
                    build_prompt=lambda item, suffix=think_suffix: _build_prompt(
                        item, args, think_suffix=suffix
                    ),
                    parse_output=parse_fn,
                    assistant_prefill=assistant_prefill,
                    generation_extra={
                        "repetition_penalty": float(
                            sampling.get("repetition_penalty", 1.0)
                        ),
                        "max_new_tokens": int(sampling["max_tokens"]),
                        "temperature": float(sampling["temperature"]),
                        "top_p": float(sampling.get("top_p", 1.0)),
                        "do_sample": float(sampling["temperature"]) > 0,
                    },
                )
            )
        return results

    if backend == "vllm_chat":
        messages = [_chat_messages_for(label, sample, args) for sample in samples]
        sampling = [
            _sampling_params_for_request(label, args, seed, n=n_completions)
            for seed in seeds
        ]
        chat_kwargs = dict(chat_kwargs_for(label, args))
        outputs = handle["llm"].chat(messages, sampling_params=sampling, **chat_kwargs)
        return _expand_n_outputs(
            samples, outputs, n_completions=n_completions, parse_fn=parse_fn
        )

    if backend == "vllm_voxtral":
        return _generate_voxtral_batch(
            handle,
            samples,
            args,
            seeds,
            parse_fn=parse_fn,
            n_completions=n_completions,
            label=label,
        )

    raise ValueError(f"Unknown backend {backend!r} for {label}")


def _build_voxtral_request(
    tokenizer: Any,
    sample: dict,
    args: SimpleNamespace,
) -> dict:
    """Build a Voxtral offline prompt (token ids + audio arrays)."""
    from mistral_common.protocol.instruct.chunk import AudioChunk, TextChunk
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.tokens.tokenizers.audio import Audio

    question = _build_prompt(sample, args)
    audio = Audio.from_file(sample["audio_path"], strict=False)
    messages = [
        UserMessage(
            content=[AudioChunk.from_audio(audio), TextChunk(text=question)]
        ).to_openai()
    ]
    prompt_token_ids = tokenizer.apply_chat_template(messages=messages)
    return {
        "prompt_token_ids": prompt_token_ids,
        "multi_modal_data": {"audio": [(audio.audio_array, audio.sampling_rate)]},
    }


def _generate_voxtral_batch(
    handle: dict,
    samples: list[dict],
    args: SimpleNamespace,
    seeds: list[int],
    *,
    parse_fn: Callable,
    n_completions: int = 1,
    label: str = "voxtral-small-24b",
) -> list[dict]:
    tokenizer = handle["tokenizer"]
    prompts = [_build_voxtral_request(tokenizer, sample, args) for sample in samples]
    sampling = [
        _sampling_params_for_request(label, args, seed, n=n_completions)
        for seed in seeds
    ]
    outputs = handle["llm"].generate(prompts, sampling_params=sampling)
    return _expand_n_outputs(
        samples, outputs, n_completions=n_completions, parse_fn=parse_fn
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_LOADERS = {
    "af-next-think": load_af_next,
    "music-flamingo": load_music_flamingo,
    "qwen3-omni": load_qwen3_omni,
    "qwen3-omni-instruct": load_qwen3_omni_instruct,
    "phi-4-multimodal": load_phi4_multimodal,
    "gemma-4-e4b": load_gemma_4_e4b,
    "gemma-4-12b": load_gemma_4_12b,
    "nemotron-3-nano-omni": load_nemotron_omni,
    "voxtral-small-24b": load_voxtral,
    "qwen3.5-122b-a10b-fp8": load_qwen35_122b,
}


def load_model(label: str, args: SimpleNamespace):
    if label not in _LOADERS:
        raise ValueError(f"No loader for model label {label!r}")
    # Point inductor / Triton / vLLM JIT at the per-model volume cache before
    # those libraries are imported inside the loader.
    try:
        from modal_cache import configure_compile_cache

        configure_compile_cache(label)
    except Exception as exc:  # noqa: BLE001 — local runs have no /cache volume
        print(f"[{label}] compile cache setup skipped: {exc}")
    return _LOADERS[label](args)


def generate_one(label: str, handle, sample: dict, args: SimpleNamespace) -> dict:
    """Single-sample convenience wrapper around ``generate_batch``."""
    return generate_batch(label, handle, [sample], args, seeds=[int(args.seed)])[0]


def _unwrap_tokenizer(tokenizer: Any) -> Any:
    for attr in ("tokenizer", "_tokenizer"):
        inner = getattr(tokenizer, attr, None)
        if inner is not None and inner is not tokenizer:
            return inner
    return tokenizer


def _int_ids(raw: Any) -> list[int]:
    if raw is None:
        return []
    # tokenizers.Encoding (rust fast tokenizer.encode)
    ids = getattr(raw, "ids", None)
    if ids is not None and not isinstance(raw, (bytes, str)):
        raw = ids
    # transformers BatchEncoding / BatchFeature
    elif hasattr(raw, "input_ids"):
        raw = raw["input_ids"] if hasattr(raw, "__getitem__") else raw.input_ids
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, (str, bytes)):
        raise TypeError(f"cannot coerce {type(raw).__name__} to token ids")
    # Nested batch: [[ids...]]
    if raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return [int(x) for x in list(raw)]


def _token_pieces(tokenizer: Any, token_ids: list[int]) -> list[dict[str, Any]]:
    """Per-id vocab piece + decoded text (skip_special_tokens=False)."""
    tok = _unwrap_tokenizer(tokenizer) if tokenizer is not None else None
    convert = getattr(tok, "convert_ids_to_tokens", None) if tok is not None else None
    decode = getattr(tok, "decode", None) if tok is not None else None
    pieces: list[dict[str, Any]] = []
    for tid in token_ids:
        token_name: str | None = None
        if callable(convert):
            try:
                converted = convert(tid)
                if isinstance(converted, list):
                    converted = converted[0] if converted else None
                if converted is not None:
                    token_name = str(converted)
            except Exception:
                token_name = None
        text = ""
        if callable(decode):
            try:
                text = decode([tid], skip_special_tokens=False)
            except TypeError:
                try:
                    text = decode([tid])
                except Exception:
                    text = ""
            except Exception:
                text = ""
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if not text and token_name:
            text = token_name
        pieces.append({"id": int(tid), "token": token_name, "text": text})
    return pieces


def _decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    tok = _unwrap_tokenizer(tokenizer) if tokenizer is not None else None
    decode = getattr(tok, "decode", None) if tok is not None else None
    if not callable(decode) or not token_ids:
        return "".join(p["text"] for p in _token_pieces(tokenizer, token_ids))
    try:
        text = decode(token_ids, skip_special_tokens=False)
    except TypeError:
        text = decode(token_ids)
    return "" if text is None else str(text)


def _tokenizer_of(handle: dict) -> Any:
    if handle.get("tokenizer") is not None:
        return handle["tokenizer"]
    llm = handle.get("llm")
    if llm is not None and hasattr(llm, "get_tokenizer"):
        try:
            return llm.get_tokenizer()
        except Exception:
            return None
    return None


def _trace_from_ids(
    tokenizer: Any,
    token_ids: list[int],
    *,
    text_fallback: str | None = None,
) -> dict[str, Any]:
    ids = _int_ids(token_ids)
    pieces = _token_pieces(tokenizer, ids)
    decoded = _decode_ids(tokenizer, ids) if ids else (text_fallback or "")
    concat = "".join(piece["text"] for piece in pieces)
    return {
        "ids": ids,
        "pieces": pieces,
        "text": decoded or concat or (text_fallback or ""),
        "text_concat": concat,
        "n_tokens": len(ids),
    }


def _request_output_ids(output: Any) -> tuple[list[int], list[int], str, str | None]:
    """Return (prompt_ids, output_ids, output_text, finish_reason) from vLLM."""
    prompt_ids = _int_ids(getattr(output, "prompt_token_ids", None))
    completions = getattr(output, "outputs", None) or []
    first = completions[0] if completions else None
    output_ids = _int_ids(
        getattr(first, "token_ids", None) if first is not None else None
    )
    output_text = ""
    if first is not None and getattr(first, "text", None) is not None:
        output_text = str(first.text)
    elif getattr(output, "text", None) is not None:
        output_text = str(output.text)
    finish = getattr(first, "finish_reason", None) if first is not None else None
    return prompt_ids, output_ids, output_text, str(finish) if finish else None


def generate_raw_trace(
    label: str,
    handle: dict,
    sample: dict,
    args: SimpleNamespace,
) -> dict[str, Any]:
    """One-sample generate that keeps prompt/output token ids as consumed.

    ``prompt.text`` / ``output.text`` are ``tokenizer.decode`` of those ids
    (special tokens kept). ``pieces`` is the per-token split.
    """
    backend = handle["backend"]
    seed = int(getattr(args, "seed", 0) or 0)
    tokenizer = _tokenizer_of(handle)
    prompt_ids: list[int] = []
    output_ids: list[int] = []
    output_text = ""
    finish_reason: str | None = None
    prompt_fallback = ""

    if backend == "vllm":
        prompt_fn = _vllm_prompt_fn(label, args)
        prompt_fallback = _maybe_assistant_think_open(label, prompt_fn(sample), args)
        prompts, sampling = _build_vllm_audio_inputs(
            label,
            [sample],
            prompt_fn=prompt_fn,
            sampling_rate=16000,
            args=args,
            seeds=[seed],
            n_completions=1,
        )
        generate_kwargs: dict[str, Any] = {}
        if handle.get("lora_request") is not None:
            generate_kwargs["lora_request"] = handle["lora_request"]
        outputs = handle["llm"].generate(
            prompts, sampling_params=sampling, **generate_kwargs
        )
        prompt_ids, output_ids, output_text, finish_reason = _request_output_ids(
            outputs[0]
        )
        if getattr(outputs[0], "prompt", None):
            prompt_fallback = str(outputs[0].prompt)

    elif backend == "vllm_chat":
        messages = [_chat_messages_for(label, sample, args)]
        sampling = [_sampling_params_for_request(label, args, seed, n=1)]
        chat_kwargs = dict(chat_kwargs_for(label, args))
        outputs = handle["llm"].chat(messages, sampling_params=sampling, **chat_kwargs)
        prompt_ids, output_ids, output_text, finish_reason = _request_output_ids(
            outputs[0]
        )
        if getattr(outputs[0], "prompt", None):
            prompt_fallback = str(outputs[0].prompt)

    elif backend == "hf_af_next":
        parsed = generate_batch(
            label, handle, [sample], args, seeds=[seed], n_completions=1
        )[0]
        output_text = str(parsed.get("model_output") or "")
        if label == "af-next-think":
            prompt_fallback = _af_next_prompt(sample, args)
        else:
            prompt_fallback = _music_flamingo_prompt(sample, args)
        prompt_fallback = _maybe_assistant_think_open(label, prompt_fallback, args)
        finish_reason = "hf_af_next"

    elif backend == "vllm_voxtral":
        request = _build_voxtral_request(handle["tokenizer"], sample, args)
        prompt_fallback = _build_prompt(sample, args)
        prompt_ids = _int_ids(request.get("prompt_token_ids"))
        sampling = [_sampling_params_for_request(label, args, seed, n=1)]
        outputs = handle["llm"].generate([request], sampling_params=sampling)
        _, output_ids, output_text, finish_reason = _request_output_ids(outputs[0])
        if getattr(outputs[0], "prompt", None):
            prompt_fallback = str(outputs[0].prompt)

    else:
        raise ValueError(f"generate_raw_trace has no path for backend {backend!r}")

    prompt_trace = _trace_from_ids(tokenizer, prompt_ids, text_fallback=prompt_fallback)
    output_trace = _trace_from_ids(tokenizer, output_ids, text_fallback=output_text)
    parse_fn = _parse_fn_for(
        args, handle.get("parse_fn", parse_choice_output), label=label
    )
    parsed = _output_dict(
        output_trace["text"] or output_text,
        sample.get("choices") or [],
        parse_fn=parse_fn,
    )
    return {
        "backend": backend,
        "finish_reason": finish_reason,
        "enable_thinking": thinking_enabled(label, args),
        "chat_kwargs": dict(chat_kwargs_for(label, args)),
        "prompt": prompt_trace,
        "output": output_trace,
        "thinking_prediction": parsed.get("thinking_prediction"),
        "answer_prediction": parsed.get("answer_prediction"),
    }
