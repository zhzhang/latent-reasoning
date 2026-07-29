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

EXP_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = EXP_DIR / "deploy"

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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
            "max_model_len": 8192,
            "max_num_seqs": 32,
            "gpu_memory_utilization": 0.92,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": True,
            "enable_prefix_caching": True,
        },
    },
    "mimo-audio-7b": {
        "model_id": "XiaomiMiMo/MiMo-Audio-7B-Instruct",
        "tokenizer_id": "XiaomiMiMo/MiMo-Audio-Tokenizer",
        "gpu": "A100-80GB",
        "backend": "vllm_omni",
        "deploy_config": str(DEPLOY_DIR / "mimo_audio_understand_throughput.yaml"),
        "sampling_rate": 24000,
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
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.9,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": True,
            "trust_remote_code": True,
            "model_impl": "transformers",
            "enable_prefix_caching": True,
        },
    },
    # MoE thinker-only (~3B active); fits one A100-80GB via plain vLLM.
    "qwen3-omni": {
        "model_id": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "gpu": "A100-80GB",
        "backend": "vllm",
        "engine": {
            "dtype": "bfloat16",
            # Prompt audio + long Thinking CoT (max_tokens=16384).
            "max_model_len": 32768,
            "max_num_seqs": 4,
            "gpu_memory_utilization": 0.9,
            "limit_mm_per_prompt": {"audio": 1},
            "enforce_eager": True,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
        },
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 16384,
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
            "max_num_seqs": 4,
            "gpu_memory_utilization": 0.9,
            "limit_mm_per_prompt": {"audio": 1},
            "config_format": "mistral",
            "load_format": "mistral",
            "tokenizer_mode": "mistral",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
        },
        "sampling": {
            "temperature": 0.2,
            "top_p": 0.95,
        },
    },
}

ALL_MODEL_LABELS = tuple(MODEL_SPECS.keys())


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


def _sampling_params_for_request(
    args: SimpleNamespace,
    seed: int,
    *,
    stop_token_ids: list[int] | None = None,
    repetition_penalty: float | None = None,
    overrides: dict[str, Any] | None = None,
):
    from vllm import SamplingParams

    temperature = float(args.temperature)
    kwargs: dict[str, Any] = {
        "temperature": temperature if temperature > 0 else 0.0,
        "top_p": float(getattr(args, "top_p", 1.0)),
        "max_tokens": int(getattr(args, "max_new_tokens", 512)),
        "seed": int(seed),
        "repetition_penalty": float(
            repetition_penalty
            if repetition_penalty is not None
            else getattr(args, "repetition_penalty", 1.0)
        ),
    }
    if stop_token_ids:
        kwargs["stop_token_ids"] = list(stop_token_ids)
    if overrides:
        kwargs.update(overrides)
    return SamplingParams(**kwargs)


def _extract_text(output: Any) -> str:
    """Normalize vLLM / Omni generate outputs to a decoded string."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    # Omni stage wrapper: request_output may be one RequestOutput or a list.
    request_output = getattr(output, "request_output", None)
    if request_output is not None:
        if isinstance(request_output, (list, tuple)):
            parts = [_extract_text(item) for item in request_output]
            return "\n".join(part for part in parts if part)
        output = request_output
    outputs = getattr(output, "outputs", None)
    if outputs:
        first = outputs[0]
        text = getattr(first, "text", None)
        if text is not None:
            return str(text)
    text = getattr(output, "text", None)
    if text is not None:
        return str(text)
    return str(output)


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


def _af_next_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    question = _build_prompt(
        sample,
        args or SimpleNamespace(prompt_mode="mc"),
        think_suffix=AF_NEXT_THINK_SUFFIX,
    )
    # MusicFlamingo / AF-Next chat format (same placeholder family as AF3).
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
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


QWEN3_OMNI_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def _qwen3_omni_prompt(sample: dict, args: SimpleNamespace | None = None) -> str:
    question = _build_prompt(sample, args or SimpleNamespace(prompt_mode="mc"))
    return (
        f"<|im_start|>system\n{QWEN3_OMNI_SYSTEM}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|audio_start|><|audio_pad|><|audio_end|>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


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
    from latent_cot import ensure_latent_w_remap

    target_dtype = torch_dtype_value(torch, getattr(args, "torch_dtype", "bfloat16"))
    if target_dtype == "auto":
        target_dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(local_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_id, dtype=target_dtype, device_map="auto"
    )
    cast_model_floating_tensors(model, target_dtype)
    model.eval()
    ensure_latent_w_remap(model, local_id, persist=True)
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
    alt = Path("/root/exp-mmar-question-difficulty/deploy") / candidate.name
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
    return out


def load_qwen3_omni(args: SimpleNamespace):
    """Qwen3-Omni Thinking via plain vLLM thinker-only path."""
    from vllm import LLM

    spec = MODEL_SPECS["qwen3-omni"]
    local_id = resolve_model_dir(args.model_id, getattr(args, "local_model_dir", None))
    engine = _apply_engine_overrides(spec["engine"], args)
    llm = LLM(model=local_id, **engine)
    print(f"Qwen3-Omni vLLM thinker ready from {local_id} engine={engine}")
    return {
        "backend": "vllm",
        "llm": llm,
        "parse_fn": parse_think_tagged_output,
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
    samples: list[dict],
    *,
    prompt_fn: Callable[[dict], str],
    sampling_rate: int,
    args: SimpleNamespace,
    seeds: list[int],
    stop_token_ids: list[int] | None = None,
    repetition_penalty: float | None = None,
    max_audio_samples: int | None = None,
    sampling_overrides: dict[str, Any] | None = None,
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
                args,
                seed,
                stop_token_ids=stop_token_ids,
                repetition_penalty=repetition_penalty,
                overrides=sampling_overrides,
            )
        )
    return prompts, sampling


def generate_batch(
    label: str,
    handle: dict,
    samples: list[dict],
    args: SimpleNamespace,
    *,
    seeds: list[int] | None = None,
) -> list[dict]:
    """Generate one completion per sample (already expanded to shot rows)."""
    if not samples:
        return []
    if seeds is None:
        seeds = [int(args.seed) + i for i in range(len(samples))]
    if len(seeds) != len(samples):
        raise ValueError("seeds length must match samples length")

    backend = handle["backend"]
    parse_fn = _parse_fn_for(args, handle.get("parse_fn", parse_choice_output))

    if backend == "vllm":
        if label == "af-next-think":
            prompt_fn = lambda s: _af_next_prompt(s, args)  # noqa: E731
        elif label == "qwen3-omni":
            prompt_fn = lambda s: _qwen3_omni_prompt(s, args)  # noqa: E731
        else:
            prompt_fn = lambda s: _build_prompt(s, args)  # noqa: E731
        sampling_overrides = MODEL_SPECS.get(label, {}).get("sampling")
        prompts, sampling = _build_vllm_audio_inputs(
            samples,
            prompt_fn=prompt_fn,
            sampling_rate=16000,
            args=args,
            seeds=seeds,
            sampling_overrides=sampling_overrides,
        )
        outputs = handle["llm"].generate(prompts, sampling_params=sampling)
        return [
            _output_dict(_extract_text(out), sample.get("choices") or [], parse_fn=parse_fn)
            for sample, out in zip(samples, outputs)
        ]

    if backend == "hf_af_next":
        from audio_flamingo_runtime import generate_batch as af_generate_batch
        from audio_flamingo_runtime import seed_everything

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
                            getattr(args, "repetition_penalty", 1.2)
                        )
                    },
                )
            )
        return results

    if backend == "vllm_omni":
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
            samples,
            prompt_fn=prompt_fn,
            sampling_rate=int(handle["sampling_rate"]),
            args=args,
            seeds=seeds,
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
        messages = [_interactive_omni_messages(sample, args) for sample in samples]
        sampling = [_sampling_params_for_request(args, seed) for seed in seeds]
        outputs = handle["llm"].chat(messages, sampling_params=sampling)
        return [
            _output_dict(_extract_text(out), sample.get("choices") or [], parse_fn=parse_fn)
            for sample, out in zip(samples, outputs)
        ]

    if backend == "vllm_voxtral":
        return _generate_voxtral_batch(handle, samples, args, seeds, parse_fn=parse_fn)

    if backend == "hf_chat":
        return [
            _generate_interactive_omni_hf(handle, sample, args, seed)
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
) -> list[dict]:
    tokenizer = handle["tokenizer"]
    prompts = [_build_voxtral_request(tokenizer, sample, args) for sample in samples]
    overrides = MODEL_SPECS["voxtral-small-24b"].get("sampling")
    sampling = [
        _sampling_params_for_request(args, seed, overrides=overrides) for seed in seeds
    ]
    outputs = handle["llm"].generate(prompts, sampling_params=sampling)
    return [
        _output_dict(_extract_text(out), sample.get("choices") or [], parse_fn=parse_fn)
        for sample, out in zip(samples, outputs)
    ]


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
    handle: dict, sample: dict, args: SimpleNamespace, seed: int
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
    temperature = float(args.temperature)
    generation_config = {
        "max_new_tokens": int(getattr(args, "max_new_tokens", 512)),
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": float(getattr(args, "top_p", 1.0)),
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
    "voxtral-small-24b": load_voxtral,
}


def load_model(label: str, args: SimpleNamespace):
    if label not in _LOADERS:
        raise ValueError(f"No loader for model label {label!r}")
    return _LOADERS[label](args)


def generate_one(label: str, handle, sample: dict, args: SimpleNamespace) -> dict:
    """Single-sample convenience wrapper around ``generate_batch``."""
    return generate_batch(label, handle, [sample], args, seeds=[int(args.seed)])[0]
