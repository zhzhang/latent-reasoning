"""Per-model vLLM / vLLM-Omni adapters for the MMAR difficulty experiment."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from audio_flamingo_runtime import resolve_model_dir
from mmar_common import (
    AF_NEXT_THINK_SUFFIX,
    build_mmar_freeform_prompt,
    build_mmar_prompt,
    parse_choice_output,
    parse_freeform_output,
    parse_think_tagged_output,
)

REPO_ROOT = Path(__file__).resolve().parent
DEPLOY_DIR = REPO_ROOT / "deploy"
_DEPLOY_MOUNT = Path("/root/deploy")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Backends that cannot use SamplingParams(n>1) shared prefill (Omni shares one
# stage SamplingParams list per generate call; HF has no n= fork). Callers must
# duplicate question×shot prompt rows for these.
_DUPLICATE_SHOT_BACKENDS = frozenset({"vllm_omni", "hf_af_next", "hf_chat"})


def backend_duplicates_shots(backend: str) -> bool:
    """True when n_shots must be expanded into duplicate prompts."""
    return backend in _DUPLICATE_SHOT_BACKENDS


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "af-next-think": {
        "model_id": "nvidia/audio-flamingo-next-think-hf",
        "gpu": "L40S",
        "backend": "vllm",
        # Native MusicFlamingo path in vLLM 0.24.x. Some AF-Next checkpoints
        # include avg_embed_norm weights that vLLM does not model; we skip them
        # at load time (see load_af_next).
        "engine": {
            "dtype": "bfloat16",
            # Cap context for long audio+text prompts (not a KV/throughput lever;
            # PagedAttention allocates on demand).
            "max_model_len": 8192,
            # Prefill-heavy: prefer batched tokens over decode-side max_num_seqs.
            "max_num_batched_tokens": 8192,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            "attention_backend": "flashinfer",  # best for throughput
            "async_scheduling": True,  # usually faster, but not all features supported
        },
        # HF generation_config.json: max_new_tokens=2048 + README repetition_penalty
        # 1.2 (card omits do_sample ⇒ greedy). We use T=0.2 like Voxtral so
        # n-shot difficulty runs keep mild sample variance.
        "native_thinking": True,
        "sampling": {
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 2048,
            "repetition_penalty": 1.2,
        },
    },
    "mimo-audio-7b": {
        "model_id": "XiaomiMiMo/MiMo-Audio-7B-Instruct",
        "tokenizer_id": "XiaomiMiMo/MiMo-Audio-Tokenizer",
        "gpu": "A100-80GB",
        "backend": "vllm_omni",
        "deploy_config": str(DEPLOY_DIR / "mimo_audio_understand_throughput.yaml"),
        "sampling_rate": 24000,
        # Official MiMo-Audio audio_understanding global sampler (HF card →
        # XiaomiMiMo/MiMo-Audio inference): T=0.3, top_p=0.95. repetition_penalty
        # 1.1 matches vLLM-Omni's mimo_audio deploy defaults.
        "sampling": {
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 512,
            "repetition_penalty": 1.1,
        },
    },
    "interactive-omni-8b": {
        "model_id": "sensenova/InteractiveOmni-8B",
        "gpu": "A100-80GB",
        # No dedicated vLLM / Omni registry entry; transformers modeling backend
        # with trust_remote_code is the closest path.
        "backend": "vllm_transformers",
        "engine": {
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            "limit_mm_per_prompt": {"audio": 1},
            # Transformers modeling backend may not support CUDA graphs; if load
            # fails, load_interactive_omni falls back to HF .chat().
            "enforce_eager": False,
            "trust_remote_code": True,
            "model_impl": "transformers",
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            "attention_backend": "flashinfer",  # best for throughput
            "async_scheduling": True,  # usually faster, but not all features supported
        },
        # HF README: generation_config = dict(max_new_tokens=1024, do_sample=True)
        # (no temp/top_p/rep → transformers sampling defaults T=1.0, top_p=1.0).
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "repetition_penalty": 1.0,
        },
    },
    # MoE thinker-only (~3B active); fits one A100-80GB via plain vLLM.
    "qwen3-omni": {
        "model_id": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "gpu": "A100-80GB",
        "backend": "vllm",
        "engine": {
            "dtype": "bfloat16",
            # Sized for measured MMAR Thinking outputs (~850 tok avg, p99 ~780,
            # max ~870). Oversized max_model_len deflates reported concurrency.
            "max_model_len": 4096,
            # Cap concurrent seqs from measured avg seq length vs KV cache size
            # (~155k tokens on A100-80GB ⇒ room for ~180 seqs of ~850).
            "max_num_seqs": 64,
            "max_num_batched_tokens": 8192,
            "limit_mm_per_prompt": {"audio": 1},
            # torch.compile fails on Qwen3-Omni MoE (Dynamo meta/cuda mismatch
            # during profile_run). Keep eager; throughput comes from max_num_seqs.
            "enforce_eager": True,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            "attention_backend": "flashinfer",  # best for throughput
            "async_scheduling": True,  # usually faster, but not all features supported
        },
        "native_thinking": True,
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            # Measured max output ~870 tokens; 2048 stops runaway CoT from
            # monopolizing KV without truncating real answers.
            "max_tokens": 2048,
            "repetition_penalty": 1.0,
        },
    },
    # Dense 24B; ~55GB bf16 — needs A100-80GB. Mistral tokenizer/format.
    "voxtral-small-24b": {
        "model_id": "mistralai/Voxtral-Small-24B-2507",
        "gpu": "A100-80GB",
        "backend": "vllm_voxtral",
        "engine": {
            "dtype": "bfloat16",
            "max_model_len": 8192,
            # Dense 24B leaves ~26 GiB KV; max_tokens=512 ⇒ plenty of room.
            "max_num_seqs": 64,
            "max_num_batched_tokens": 8192,
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
        "sampling": {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 512,
            "repetition_penalty": 1.0,
        },
    },
    # Dense 7B thinker-only; HF generation_config.json has no sampler.
    # Official eval is greedy; T=0.2 keeps n-shot variance.
    "qwen2.5-omni-7b": {
        "model_id": "Qwen/Qwen2.5-Omni-7B",
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
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            "attention_backend": "flashinfer",
            "async_scheduling": True,
        },
        "sampling": {
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 2048,
            "repetition_penalty": 1.0,
        },
    },
    # 5.6B; speech LoRA lives next to the checkpoint. Card uses
    # GenerationConfig.from_pretrained + max_new_tokens=1000 (greedy).
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
    # Effective 4B; native audio. Card Best Practices: T=1.0, top_p=0.95, top_k=64.
    "gemma-4-e4b": {
        "model_id": "google/gemma-4-E4B-it",
        "gpu": "L40S",
        "backend": "vllm_chat",
        "engine": {
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 8192,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            # Gemma-4 head_size is unsupported by FLASH_ATTN; FlashInfer JIT
            # on L40S (sm89) requests sm100+ kernels. Triton handles both.
            "attention_backend": "TRITON_ATTN",
            "async_scheduling": True,
            "allowed_local_media_path": "/",
        },
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 1024,
            "repetition_penalty": 1.0,
        },
    },
    # Block-wise FP8 of Qwen3-Omni Instruct (thinker+talker MoE); H100 native FP8.
    # Official Instruct eval is greedy.
    "qwen3-omni-instruct": {
        "model_id": "marksverdhei/Qwen3-Omni-30B-A3B-FP8",
        "gpu": "H100",
        "backend": "vllm",
        "engine": {
            "dtype": "auto",
            "max_model_len": 4096,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 8192,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": True,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            "attention_backend": "flashinfer",
            "async_scheduling": True,
        },
        "sampling": {
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 2048,
            "repetition_penalty": 1.0,
        },
    },
    # Native FP8 MoE; thinking-mode card: T=0.6, top_p=0.95. Cap max_tokens for MMAR.
    "nemotron-3-nano-omni": {
        "model_id": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        "gpu": "H100",
        "backend": "vllm_chat",
        "engine": {
            "dtype": "auto",
            "max_model_len": 4096,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 8192,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": False,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
            "disable_log_stats": False,
            "attention_backend": "flashinfer",
            "async_scheduling": True,
            "allowed_local_media_path": "/",
        },
        "native_thinking": True,
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 2048,
            "repetition_penalty": 1.0,
        },
    },
}

ALL_MODEL_LABELS = tuple(MODEL_SPECS.keys())
NATIVE_THINKING_LABELS = frozenset(
    label for label, spec in MODEL_SPECS.items() if spec.get("native_thinking")
)


def has_native_thinking(label: str) -> bool:
    """True when the checkpoint emits native ``<think>`` / reasoning traces."""
    return label in NATIVE_THINKING_LABELS


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
    return raw


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


# Step-Audio2 encoder pos-embed is n_audio_ctx=1500 after stride-2 conv, so mel
# frames must be <= 3000. At 16 kHz that is just under 30s (exactly 30s → 3002).
STEP_AUDIO_MAX_SAMPLES = 479_680


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
            texts.append(str(text) if text is not None else "")
        if texts:
            return texts
    text = getattr(output, "text", None)
    if text is not None:
        return [str(text)]
    return [str(output)]


def _extract_text(output: Any) -> str:
    """Normalize vLLM / Omni generate outputs to a decoded string."""
    texts = _completion_texts(output)
    return texts[0] if texts else ""


def _prompt_mode(args: SimpleNamespace) -> str:
    mode = str(getattr(args, "prompt_mode", "mc") or "mc").lower()
    return "freeform" if mode in {"freeform", "free_form", "open"} else "mc"


def _build_prompt(sample: dict, args: SimpleNamespace, *, think_suffix: str | None = None) -> str:
    if _prompt_mode(args) == "freeform":
        return build_mmar_freeform_prompt(sample, think_suffix=think_suffix)
    return build_mmar_prompt(sample, think_suffix=think_suffix)


def _parse_fn_for(args: SimpleNamespace, default: Callable = parse_choice_output) -> Callable:
    if _prompt_mode(args) == "freeform":
        if default is parse_think_tagged_output:
            # Free-form still strips <think> blocks; choice matching is skipped.
            return parse_freeform_output
        return parse_freeform_output
    return default


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


def _af_next_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    question = _build_prompt(
        sample,
        args or SimpleNamespace(prompt_mode="mc"),
        think_suffix=AF_NEXT_THINK_SUFFIX,
    )
    # MusicFlamingo / AF-Next chat format (same placeholder family as AF3).
    return (
        f"<|im_start|>system\n{AF_NEXT_SYSTEM}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<sound>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _step_audio_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    ns = args or SimpleNamespace(prompt_mode="mc")
    question = _build_prompt(sample, ns)
    if _prompt_mode(ns) == "freeform":
        system = (
            "You are an expert in audio analysis. "
            "Listen carefully and answer the question accurately.\n"
            f"{question}"
        )
    else:
        system = (
            "You are an expert in audio analysis. "
            "Listen carefully and answer the multiple-choice question accurately.\n"
            f"{question}"
        )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n<audio_patch><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _mimo_audio_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    question = _build_prompt(sample, args or SimpleNamespace(prompt_mode="mc"))
    # Placeholder audio span; waveform is supplied via multi_modal_data.
    return (
        "<|im_start|>user\n"
        f"<|sosp|><|empty|><|eosp|>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n"
    )


def _interactive_omni_messages(sample: dict, args: SimpleNamespace | None = None) -> list[dict]:
    prompt = _build_prompt(sample, args or SimpleNamespace(prompt_mode="mc"))
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"file://{sample['audio_path']}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _qwen3_omni_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    # Qwen3-Omni Thinking primary examples omit a system turn; the chat
    # template only emits one when messages[0].role == "system". Instruct
    # eval notes also say no system prompt.
    question = _build_prompt(sample, args or SimpleNamespace(prompt_mode="mc"))
    return (
        "<|im_start|>user\n"
        f"<|audio_start|><|audio_pad|><|audio_end|>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


QWEN25_OMNI_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


def _qwen25_omni_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    question = _build_prompt(sample, args or SimpleNamespace(prompt_mode="mc"))
    return (
        f"<|im_start|>system\n{QWEN25_OMNI_SYSTEM}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|audio_bos|><|AUDIO|><|audio_eos|>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _phi4_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    question = _build_prompt(sample, args or SimpleNamespace(prompt_mode="mc"))
    return f"<|user|><|audio_1|>{question}<|end|><|assistant|>"


def _audio_text_messages(sample: dict, args: SimpleNamespace | None = None) -> list[dict]:
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
    engine = dict(spec["engine"])
    if getattr(args, "max_num_seqs", None):
        engine["max_num_seqs"] = int(args.max_num_seqs)
    if getattr(args, "gpu_memory_utilization", None):
        engine["gpu_memory_utilization"] = float(args.gpu_memory_utilization)

    # Prefer native MusicFlamingo (vLLM 0.24).
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
        return _load_af_next_hf(local_id, args)


def _load_af_next_hf(local_id: str, args: SimpleNamespace):
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
        f"AF-Next HF ready: class={type(model).__name__} "
        f"param_dtype={model_param_dtype(model)} "
        f"audio_tower_dtype={audio_tower_dtype(model)} "
        f"device={model_input_device(model)}"
    )
    return {
        "backend": "hf_af_next",
        "model": model,
        "processor": processor,
        "parse_fn": parse_think_tagged_output,
    }


def _resolve_deploy_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    alt = _DEPLOY_MOUNT / candidate.name
    if alt.is_file():
        return str(alt)
    return str(candidate)


def load_step_audio(args: SimpleNamespace):
    from vllm_omni.entrypoints.omni import Omni

    spec = MODEL_SPECS["step-audio-2-mini"]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    stage_configs_path = _resolve_deploy_path(
        getattr(args, "deploy_config", None) or spec["deploy_config"]
    )
    omni = Omni(
        model=local_id,
        stage_configs_path=stage_configs_path,
        trust_remote_code=True,
    )
    print(
        f"Step-Audio-2-mini Omni ready from {local_id} "
        f"stage_configs={stage_configs_path}"
    )
    return {
        "backend": "vllm_omni",
        "llm": omni,
        "sampling_rate": int(spec["sampling_rate"]),
        "parse_fn": parse_choice_output,
        "stages": 1,
    }


def load_mimo_audio(args: SimpleNamespace):
    from vllm_omni.entrypoints.omni import Omni

    spec = MODEL_SPECS["mimo-audio-7b"]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    tokenizer_id = getattr(args, "tokenizer_id", None) or spec["tokenizer_id"]
    tokenizer_dir = resolve_model_dir(
        tokenizer_id, getattr(args, "local_tokenizer_dir", None)
    )
    os.environ["MIMO_AUDIO_TOKENIZER_PATH"] = tokenizer_dir

    deploy_config = _resolve_deploy_path(
        getattr(args, "deploy_config", None) or spec["deploy_config"]
    )
    omni = Omni(
        model=local_id,
        deploy_config=deploy_config,
        trust_remote_code=True,
        async_chunk=False,
    )
    print(
        f"MiMo-Audio Omni ready from {local_id} "
        f"(tokenizer={tokenizer_dir}) deploy={deploy_config}"
    )
    return {
        "backend": "vllm_omni",
        "llm": omni,
        "sampling_rate": int(spec["sampling_rate"]),
        "parse_fn": parse_choice_output,
        "stages": 2,
    }


def _ensure_transformers_onnx_shim() -> None:
    """InteractiveOmni's remote WhisperConfig still imports transformers.onnx.

    Newer transformers removed that submodule; stub just enough for config import.
    """
    import sys
    import types

    if "transformers.onnx" in sys.modules:
        return
    try:
        import transformers.onnx  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    onnx_mod = types.ModuleType("transformers.onnx")

    class _OnnxConfig:  # noqa: D401 — stub
        pass

    class _OnnxSeq2SeqConfigWithPast:  # noqa: D401 — stub
        pass

    onnx_mod.OnnxConfig = _OnnxConfig
    onnx_mod.OnnxSeq2SeqConfigWithPast = _OnnxSeq2SeqConfigWithPast
    sys.modules["transformers.onnx"] = onnx_mod


def load_interactive_omni(args: SimpleNamespace):
    from vllm import LLM

    _ensure_transformers_onnx_shim()
    spec = MODEL_SPECS["interactive-omni-8b"]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = dict(spec["engine"])
    if getattr(args, "max_num_seqs", None):
        engine["max_num_seqs"] = int(args.max_num_seqs)
    if getattr(args, "gpu_memory_utilization", None):
        engine["gpu_memory_utilization"] = float(args.gpu_memory_utilization)
    try:
        llm = LLM(model=local_id, **engine)
        print(f"InteractiveOmni vLLM(transformers) ready from {local_id}")
        return {
            "backend": "vllm_chat",
            "llm": llm,
            "parse_fn": parse_choice_output,
        }
    except Exception as exc:  # noqa: BLE001 — fall back to HF chat path
        print(
            f"InteractiveOmni vLLM load failed ({exc}); "
            "falling back to Transformers .chat()"
        )
        return _load_interactive_omni_hf(local_id)


def _ensure_interactive_omni_assets(local_id: str) -> None:
    """Fetch onnx assets that may have been skipped by the volume seeder."""
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    model_dir = Path(local_id)
    needed = ("campplus.onnx",)
    for name in needed:
        dest = model_dir / name
        if dest.is_file():
            continue
        print(f"Downloading missing InteractiveOmni asset {name} ...")
        hf_hub_download(
            repo_id="sensenova/InteractiveOmni-8B",
            filename=name,
            local_dir=str(model_dir),
            token=os.environ.get("HF_TOKEN"),
        )


def _load_interactive_omni_hf(local_id: str):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    _ensure_transformers_onnx_shim()
    _ensure_interactive_omni_assets(local_id)
    config = AutoConfig.from_pretrained(local_id, trust_remote_code=True)
    # Remote config defaults to FlashAttention2; image may not ship flash_attn.
    if hasattr(config, "llm_config") and config.llm_config is not None:
        config.llm_config._attn_implementation = "eager"
    for attr in ("vision_config", "audio_config", "voicelm_config"):
        sub = getattr(config, attr, None)
        if sub is not None and hasattr(sub, "_attn_implementation"):
            sub._attn_implementation = "eager"
    if hasattr(config, "use_flash_attn"):
        config.use_flash_attn = False

    model = (
        AutoModel.from_pretrained(
            local_id,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        local_id, trust_remote_code=True, use_fast=True
    )
    print(f"InteractiveOmni HF ready from {local_id}")
    return {
        "backend": "hf_chat",
        "model": model,
        "tokenizer": tokenizer,
        "parse_fn": parse_choice_output,
    }


def _apply_engine_overrides(engine: dict, args: SimpleNamespace) -> dict:
    out = dict(engine)
    if getattr(args, "max_num_seqs", None):
        out["max_num_seqs"] = int(args.max_num_seqs)
    if getattr(args, "gpu_memory_utilization", None):
        out["gpu_memory_utilization"] = float(args.gpu_memory_utilization)
    if getattr(args, "max_model_len", None):
        out["max_model_len"] = int(args.max_model_len)
    return out


def load_qwen3_omni(args: SimpleNamespace):
    """Qwen3-Omni Thinking via plain vLLM thinker-only path."""
    return _load_qwen3_family(args, label="qwen3-omni", parse_fn=parse_think_tagged_output)


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


def load_qwen25_omni(args: SimpleNamespace):
    from vllm import LLM

    spec = MODEL_SPECS["qwen2.5-omni-7b"]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = _apply_engine_overrides(spec["engine"], args)
    llm = LLM(model=local_id, **engine)
    print(f"Qwen2.5-Omni vLLM thinker ready from {local_id} engine={engine}")
    return {
        "backend": "vllm",
        "llm": llm,
        "parse_fn": parse_choice_output,
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


def load_gemma_4_e4b(args: SimpleNamespace):
    from vllm import LLM

    spec = MODEL_SPECS["gemma-4-e4b"]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = _apply_engine_overrides(spec["engine"], args)
    llm = LLM(model=local_id, **engine)
    print(f"Gemma-4-E4B vLLM chat ready from {local_id} engine={engine}")
    return {
        "backend": "vllm_chat",
        "llm": llm,
        "parse_fn": parse_choice_output,
        "messages_fn": "audio_text",
        "chat_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def load_nemotron_omni(args: SimpleNamespace):
    from vllm import LLM

    spec = MODEL_SPECS["nemotron-3-nano-omni"]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = _apply_engine_overrides(spec["engine"], args)
    llm = LLM(model=local_id, **engine)
    print(f"Nemotron-3-Nano-Omni vLLM chat ready from {local_id} engine={engine}")
    return {
        "backend": "vllm_chat",
        "llm": llm,
        "parse_fn": parse_think_tagged_output,
        "messages_fn": "audio_text",
        "chat_kwargs": {"chat_template_kwargs": {"enable_thinking": True}},
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
        audio = _load_audio_tuple(
            sample["audio_path"],
            sampling_rate,
            max_samples=max_audio_samples,
        )
        prompts.append(
            {
                "prompt": prompt_fn(sample),
                "multi_modal_data": {"audio": audio},
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
    parse_fn = _parse_fn_for(args, handle.get("parse_fn", parse_choice_output))

    if backend == "vllm":
        if label == "af-next-think":
            prompt_fn = lambda s: _af_next_prompt(s, args)  # noqa: E731
        elif label in {"qwen3-omni", "qwen3-omni-instruct"}:
            prompt_fn = lambda s: _qwen3_omni_prompt(s, args)  # noqa: E731
        elif label == "qwen2.5-omni-7b":
            prompt_fn = lambda s: _qwen25_omni_prompt(s, args)  # noqa: E731
        elif label == "phi-4-multimodal":
            prompt_fn = lambda s: _phi4_prompt(s, args)  # noqa: E731
        else:
            prompt_fn = lambda s: _build_prompt(s, args)  # noqa: E731
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
                    build_prompt=lambda item: _build_prompt(
                        item, args, think_suffix=AF_NEXT_THINK_SUFFIX
                    ),
                    parse_output=parse_fn,
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

    if backend == "vllm_omni":
        if n_completions != 1:
            raise ValueError("vllm_omni requires expanded shot rows (n_completions=1)")
        if label == "step-audio-2-mini":
            prompt_fn = lambda s: _step_audio_prompt(s, args)  # noqa: E731
            # Step-Audio2 Thinker can emit audio tokens (ids >= 151696). Without
            # stopping at Qwen <|im_end|>, ASR-style text answers detokenize to
            # empty after the thinker filters to text-only tokens (< 151688).
            stop_token_ids = [151645]
            repetition_penalty = 1.05
        elif label == "mimo-audio-7b":
            prompt_fn = lambda s: _mimo_audio_prompt(s, args)  # noqa: E731
            stop_token_ids = None
            repetition_penalty = None
        else:
            raise ValueError(f"No Omni prompt builder for {label}")
        prompts, sampling = _build_vllm_audio_inputs(
            label,
            samples,
            prompt_fn=prompt_fn,
            sampling_rate=int(handle["sampling_rate"]),
            args=args,
            seeds=seeds,
            n_completions=1,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
            max_audio_samples=(
                STEP_AUDIO_MAX_SAMPLES if label == "step-audio-2-mini" else None
            ),
        )
        # Text-only MMAR answers: prefer text modality so audio-token stages
        # are skipped / discouraged when the pipeline supports it.
        if label in {"mimo-audio-7b", "step-audio-2-mini"}:
            for prompt in prompts:
                prompt["modalities"] = ["text"]
        texts = _omni_generate_texts(
            handle["llm"],
            prompts,
            sampling,
            stages=int(handle.get("stages", 1)),
            n_shots=int(getattr(args, "n_shots", 1) or 1),
            debug_label=label,
        )
        return [
            _output_dict(text, sample.get("choices") or [], parse_fn=parse_fn)
            for sample, text in zip(samples, texts)
        ]

    if backend == "vllm_chat":
        if handle.get("messages_fn") == "audio_text":
            messages = [_audio_text_messages(sample, args) for sample in samples]
        else:
            messages = [_interactive_omni_messages(sample, args) for sample in samples]
        sampling = [
            _sampling_params_for_request(label, args, seed, n=n_completions)
            for seed in seeds
        ]
        chat_kwargs = dict(handle.get("chat_kwargs") or {})
        outputs = handle["llm"].chat(
            messages, sampling_params=sampling, **chat_kwargs
        )
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

    if backend == "hf_chat":
        if n_completions != 1:
            raise ValueError("hf_chat requires expanded shot rows (n_completions=1)")
        return [
            _generate_interactive_omni_hf(handle, sample, args, seed, label=label)
            for sample, seed in zip(samples, seeds)
        ]

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


def _omni_stage_params(stage0, *, stages: int):
    """Build Omni per-stage SamplingParams (shared across one generate call)."""
    if stages <= 1:
        return [stage0]
    from vllm import SamplingParams

    return [
        stage0,
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            seed=int(stage0.seed) if stage0.seed is not None else 0,
        ),
    ]


def _omni_generate_texts(
    omni: Any,
    prompts: list[dict],
    sampling: list[Any],
    *,
    stages: int,
    n_shots: int,
    debug_label: str | None = None,
) -> list[str]:
    """Run Omni generate over a flattened question×shot request list.

    Omni applies one stage SamplingParams list to every request in a call, so
    we regroup question-major flattened rows by shot and issue one generate per
    shot (all questions in that shot share the first row's seed — same as the
    previous outer shot loop). This keeps question batching while accepting the
    runner's flattened API.
    """
    n = len(prompts)
    if n == 0:
        return []
    texts = [""] * n

    def _run_group(indices: list[int]) -> None:
        sub_prompts = [prompts[i] for i in indices]
        stage_params = _omni_stage_params(sampling[indices[0]], stages=stages)
        omni_outputs = omni.generate(sub_prompts, stage_params)
        # Materialize once — Omni may return a single-pass iterator.
        stage_list = list(omni_outputs) if omni_outputs is not None else []
        sub_texts = _collect_omni_texts(stage_list, n=len(sub_prompts))
        if debug_label and (not any(sub_texts) or len(stage_list) == 0):
            _debug_omni_stages(debug_label, stage_list, n=len(sub_prompts))
        for local_i, global_i in enumerate(indices):
            texts[global_i] = sub_texts[local_i]

    if n_shots > 1 and n % n_shots == 0:
        n_questions = n // n_shots
        for shot_index in range(n_shots):
            indices = [q * n_shots + shot_index for q in range(n_questions)]
            _run_group(indices)
        return texts

    # Fallback: group by seed so rows that share a seed stay batched.
    groups: dict[int, list[int]] = {}
    order: list[int] = []
    for index, params in enumerate(sampling):
        seed_key = int(params.seed) if getattr(params, "seed", None) is not None else index
        if seed_key not in groups:
            order.append(seed_key)
            groups[seed_key] = []
        groups[seed_key].append(index)
    for seed_key in order:
        _run_group(groups[seed_key])
    return texts


def _debug_omni_stages(label: str, stage_list: list[Any], *, n: int) -> None:
    """Print Omni stage structure when text extraction yields empties."""
    print(
        f"[{label}] Omni debug: n_requests={n} n_stage_outputs={len(stage_list)}"
    )
    for index, stage_out in enumerate(stage_list[:4]):
        final_type = getattr(stage_out, "final_output_type", None)
        raw = getattr(stage_out, "request_output", stage_out)
        outputs_list = raw if isinstance(raw, list) else [raw]
        print(
            f"[{label}] stage[{index}] type={type(stage_out).__name__} "
            f"final_output_type={final_type!r} n_request_outputs={len(outputs_list)}"
        )
        for j, req in enumerate(outputs_list[:2]):
            outs = getattr(req, "outputs", None) or []
            first = outs[0] if outs else None
            token_ids = getattr(first, "token_ids", None) if first is not None else None
            text = getattr(first, "text", None) if first is not None else None
            tid_preview = list(token_ids)[:12] if token_ids is not None else None
            n_audio = (
                sum(1 for tid in token_ids if int(tid) >= 151696)
                if token_ids is not None
                else None
            )
            print(
                f"[{label}]   req[{j}] id={getattr(req, 'request_id', None)!r} "
                f"text={text!r} n_tokens={len(token_ids) if token_ids is not None else 0} "
                f"n_audio_tokens={n_audio} token_ids_head={tid_preview}"
            )


def _collect_omni_texts(omni_outputs: Any, *, n: int) -> list[str]:
    """Map Omni stage iterator / list outputs back to one text per request."""
    texts = [""] * n
    if omni_outputs is None:
        return texts

    stage_list = list(omni_outputs)
    filled = 0
    for index, stage_out in enumerate(stage_list):
        final_type = getattr(stage_out, "final_output_type", None)
        if final_type is not None and final_type != "text":
            continue
        raw = getattr(stage_out, "request_output", stage_out)
        outputs_list = raw if isinstance(raw, list) else [raw]
        for local_i, req in enumerate(outputs_list):
            text = _extract_text(req)
            if not text:
                continue
            req_id = str(getattr(req, "request_id", ""))
            req_index: int | None = None
            if req_id:
                head = req_id.split("_", 1)[0]
                if head.isdigit():
                    req_index = int(head)
            if req_index is None:
                req_index = local_i if local_i < n else (filled if filled < n else index)
            if 0 <= req_index < n:
                texts[req_index] = text
                filled += 1
    # Fallback: zip stage outputs in order when request_ids were missing.
    if not any(texts):
        for index, out in enumerate(stage_list[:n]):
            texts[index] = _extract_text(out)
    return texts


def _generate_interactive_omni_hf(
    handle: dict,
    sample: dict,
    args: SimpleNamespace,
    seed: int,
    *,
    label: str = "interactive-omni-8b",
) -> dict:
    import torch

    from audio_flamingo_runtime import seed_everything

    seed_everything(seed)
    prompt = _build_prompt(sample, args)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": sample["audio_path"]},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    sampling = resolve_sampling(label, args)
    temperature = float(sampling["temperature"])
    generation_config = {
        "max_new_tokens": int(sampling["max_tokens"]),
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": float(sampling.get("top_p", 1.0)),
        "repetition_penalty": float(sampling.get("repetition_penalty", 1.0)),
    }
    generation_config = {
        key: value for key, value in generation_config.items() if value is not None
    }
    with torch.inference_mode():
        response = handle["model"].chat(
            handle["tokenizer"],
            generation_config,
            messages,
        )
    if isinstance(response, tuple):
        response = response[0]
    return _output_dict(
        str(response or ""),
        sample.get("choices") or [],
        parse_fn=_parse_fn_for(args, parse_choice_output),
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_LOADERS = {
    "af-next-think": load_af_next,
    "mimo-audio-7b": load_mimo_audio,
    "interactive-omni-8b": load_interactive_omni,
    "qwen3-omni": load_qwen3_omni,
    "qwen3-omni-instruct": load_qwen3_omni_instruct,
    "qwen2.5-omni-7b": load_qwen25_omni,
    "phi-4-multimodal": load_phi4_multimodal,
    "gemma-4-e4b": load_gemma_4_e4b,
    "nemotron-3-nano-omni": load_nemotron_omni,
    "voxtral-small-24b": load_voxtral,
}


def load_model(label: str, args: SimpleNamespace):
    if label not in _LOADERS:
        raise ValueError(f"No loader for model label {label!r}")
    return _LOADERS[label](args)


def generate_one(label: str, handle, sample: dict, args: SimpleNamespace) -> dict:
    """Single-sample convenience wrapper around ``generate_batch``."""
    return generate_batch(label, handle, [sample], args, seeds=[int(args.seed)])[0]
