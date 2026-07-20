"""Run Audio Flamingo Next Think on MMAR with MMAR-Rubrics grading on Modal.

Loads the local ``AudioFlamingoNextForConditionalGeneration`` nn.Module (not
HuggingFace ``MusicFlamingoForConditionalGeneration``) with hub/Volume weights.
The processor still comes from Transformers.

Uses the shared ``latent-reasoning`` Volume from ``seed_volume.py`` for data
and weights, and writes eval outputs to ``latent-reasoning-results``:

    /cache/data/mmar/          # MMAR audio + MMAR-meta.jsonl
    /cache/models/<repo_id>/   # AF-Next-Think weights
    /results/mmar/af-next-think/<run_id>/
      predictions.jsonl              # generations + CoT + answers
      predictions.evaluated.jsonl    # OpenAI rubric grades (pipelined)
      scores.json
      manifest.json

Local mirror (via ``download_results.py``):
``<repo>/outputs/mmar/af-next-think/<run_id>/``.

Model card: https://huggingface.co/nvidia/audio-flamingo-next-think-hf

Prereqs (one-time):

    uv run modal run seed_volume.py --datasets mmar --models af-next-think

Usage:

    uv run modal run run_audio_flamingo_next_mmar.py
    uv run modal run run_audio_flamingo_next_mmar.py --num-samples 8
    uv run modal run run_audio_flamingo_next_mmar.py --no-score --num-samples 4
    uv run modal run run_audio_flamingo_next_mmar.py --n-shots 5 --temperature 0.7 --num-samples 8
    uv run modal run --detach run_audio_flamingo_next_mmar.py --num-samples 200

Download results locally:

    uv run modal run download_results.py --remote-path mmar/af-next-think

AF-Next-Think may emit ``<think>...</think>`` reasoning traces. Prompting asks
for timestamp-grounded step-by-step reasoning before the final answer.
``max_new_tokens`` defaults to 4096 because reasoning traces can be long.

Requires Modal Secrets:
  - ``huggingface-secret`` with ``HF_TOKEN``
  - ``openai-secret`` with ``OPENAI_API_KEY`` (for rubric scoring; default on)
"""

from __future__ import annotations

from types import SimpleNamespace

import modal

from audio_flamingo_next import AudioFlamingoNextForConditionalGeneration
from audio_flamingo_runtime import (
    audio_tower_dtype,
    generate_batch,
    model_input_device,
    model_param_dtype,
    resolve_model_dir,
    torch_dtype_value,
)
from mmar_common import (
    AF_NEXT_THINK_SUFFIX,
    build_mmar_prompt,
    parse_think_tagged_output,
    run_mmar_evaluation,
)
from modal_cache import (
    DEFAULT_MMAR_DATA_ROOT,
    DEFAULT_MMAR_META,
    RESULTS_MOUNT,
    VOLUME_MOUNT,
    hf_secret,
    mmar_eval_image,
    openai_secret,
    results_volume,
    volume,
)

DEFAULT_MODEL_ID = "nvidia/audio-flamingo-next-think-hf"
DEFAULT_OUTPUT_DIR = RESULTS_MOUNT / "mmar" / "af-next-think"
DEFAULT_REPETITION_PENALTY = 1.2

image = mmar_eval_image()
app = modal.App("audio-flamingo-next-mmar", image=image)


def _resolve_device(device_map: str | None):
    import torch

    if device_map in (None, "auto", "cuda", "cuda:0"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_map == "cpu":
        return torch.device("cpu")
    if device_map == "mps":
        return torch.device("mps")
    # e.g. "cuda:1"
    return torch.device(device_map)


def load_audio_flamingo_next(args):
    import torch
    from transformers import AutoProcessor

    target_dtype = torch_dtype_value(torch, args.torch_dtype)
    if target_dtype == "auto":
        target_dtype = torch.bfloat16

    local_id = resolve_model_dir(args.model_id, args.local_model_dir)
    device = _resolve_device(getattr(args, "device_map", "auto"))
    if getattr(args, "attn_implementation", None):
        print(
            f"Note: local AudioFlamingoNext ignores attn_implementation="
            f"{args.attn_implementation!r} (eager only)."
        )

    processor = AutoProcessor.from_pretrained(local_id)
    model = AudioFlamingoNextForConditionalGeneration.from_pretrained(
        local_id,
        dtype=target_dtype,
        device=device,
        strict=True,
    )
    model.eval()
    print(
        f"Model ready: class={type(model).__name__} (local nn.Module), "
        f"param_dtype={model_param_dtype(model)}, "
        f"audio_tower_dtype={audio_tower_dtype(model)}, "
        f"device={model_input_device(model)}"
    )
    return model, processor


def generate_af_next_batch(model, processor, samples, args):
    return generate_batch(
        model,
        processor,
        samples,
        args,
        build_prompt=lambda sample: build_mmar_prompt(
            sample, think_suffix=AF_NEXT_THINK_SUFFIX
        ),
        parse_output=parse_think_tagged_output,
        generation_extra={"repetition_penalty": args.repetition_penalty},
    )


@app.function(
    gpu="L40S",
    timeout=6 * 60 * 60,
    volumes={
        VOLUME_MOUNT: volume,
        RESULTS_MOUNT: results_volume,
    },
    secrets=[hf_secret, openai_secret],
    memory=65536,
)
def run_mmar(
    model_id: str = DEFAULT_MODEL_ID,
    local_model_dir: str | None = None,
    meta: str = str(DEFAULT_MMAR_META),
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
    audio_dir: str | None = None,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    num_samples: int = -1,
    start: int = 0,
    batch_size: int = 1,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    attn_implementation: str | None = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    seed: int = 42,
    score: bool = True,
    score_only: bool = False,
    n_shots: int = 1,
    print_every: int = 10,
    run_id: str | None = None,
) -> dict:
    """Run AF-Next-Think inference with pipelined MMAR-Rubrics grading on Modal."""
    args = SimpleNamespace(
        model_id=model_id,
        local_model_dir=local_model_dir,
        meta=meta,
        data_root=data_root,
        audio_dir=audio_dir,
        output_dir=output_dir,
        num_samples=num_samples,
        start=start,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        device_map=device_map,
        seed=seed,
        score=score,
        score_only=score_only,
        n_shots=n_shots,
        print_every=print_every,
        run_id=run_id,
    )
    return run_mmar_evaluation(
        args=args,
        load_model=load_audio_flamingo_next,
        generate_batch_fn=generate_af_next_batch,
        model_label="af-next-think",
        manifest_extra={
            "repetition_penalty": repetition_penalty,
            "model_implementation": "audio_flamingo_next.AudioFlamingoNextForConditionalGeneration",
        },
    )


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL_ID,
    local_model_dir: str | None = None,
    meta: str = str(DEFAULT_MMAR_META),
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
    audio_dir: str | None = None,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    num_samples: int = -1,
    start: int = 0,
    batch_size: int = 1,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    attn_implementation: str | None = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    seed: int = 42,
    score: bool = True,
    score_only: bool = False,
    n_shots: int = 1,
    print_every: int = 10,
    run_id: str | None = None,
):
    """Launch AF-Next-Think MMAR eval on Modal.

    Args:
        model_id: Hugging Face repo id
            (default nvidia/audio-flamingo-next-think-hf).
        local_model_dir: Optional path under the models subpath; defaults to
            /cache/models/<model_id> when seeded.
        meta: Path to MMAR-meta.jsonl on the data subpath.
        data_root: MMAR root used to resolve ./audio paths.
        audio_dir: Optional override for wav directory.
        output_dir: Results Volume directory for run folders
            (default /results/mmar/af-next-think).
        num_samples: Number of items (-1 = all). Subsets are randomly
            sampled by index (seeded).
        start: Offset into the meta file before sampling.
        batch_size: Generation batch size.
        max_new_tokens: Generation length cap (default 4096 for long CoT).
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling parameter.
        repetition_penalty: Generation repetition penalty (model-card default 1.2).
        attn_implementation: Ignored for the local nn.Module (eager only).
        torch_dtype: auto / float16 / bfloat16 / float32.
        device_map: Device for the local module (auto/cuda/cpu/mps).
        seed: RNG seed.
        score: Pipeline OpenAI MMAR-Rubrics grading alongside inference
            (default True). Grades each batch while the next generates.
            Forced off when n_shots > 1.
        score_only: Skip inference; only score existing predictions.
        n_shots: Independent generation attempts per example (default 1).
            When > 1, rubric grading is disabled and each shot is scored with
            string match; example is correct if any shot succeeds.
        print_every: Progress print interval.
        run_id: Optional run folder name; default is a UTC timestamp.
    """
    # Use .spawn().get() (not .remote()) so `modal run --detach` keeps the job
    # alive after the local client disconnects.
    call = run_mmar.spawn(
        model_id=model_id,
        local_model_dir=local_model_dir,
        meta=meta,
        data_root=data_root,
        audio_dir=audio_dir,
        output_dir=output_dir,
        num_samples=num_samples,
        start=start,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        device_map=device_map,
        seed=seed,
        score=score,
        score_only=score_only,
        n_shots=n_shots,
        print_every=print_every,
        run_id=run_id,
    )
    print(f"Spawned run_mmar call_id={call.object_id}")
    result = call.get()
    print("Done:", result)
    if isinstance(result, dict) and result.get("volume_path"):
        print(
            "Download with:\n"
            f"  uv run modal run download_results.py "
            f"--remote-path {result['volume_path']}"
        )
