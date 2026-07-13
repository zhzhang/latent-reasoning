"""Run Audio Flamingo Next Think on MMAR with MMAR-Rubrics grading on Modal.

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

from audio_flamingo_runtime import (
    audio_tower_dtype,
    cast_model_floating_tensors,
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


def load_audio_flamingo_next(args):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoProcessor

    target_dtype = torch_dtype_value(torch, args.torch_dtype)
    # AF-Next Hub weights are MusicFlamingoForConditionalGeneration
    # (model_type musicflamingo). AutoModel loads the base MusicFlamingoModel,
    # which has no generate(); use the seq2seq auto class instead.
    # See: https://huggingface.co/nvidia/audio-flamingo-next-think-hf
    kwargs = {
        "device_map": args.device_map,
        "dtype": target_dtype,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    local_id = resolve_model_dir(args.model_id, args.local_model_dir)
    processor = AutoProcessor.from_pretrained(local_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_id, **kwargs)

    # Some AF-Next tensors initialized by the MusicFlamingo wrapper are not
    # loaded from the checkpoint and can remain float32 even when ``dtype`` is
    # bfloat16. The processor follows the model card and supplies audio
    # features in the model dtype, so mixed float32/bfloat16 modules fail in
    # the audio path. Normalize parameters and buffers after loading.
    uniform_dtype = (
        model_param_dtype(model) if target_dtype == "auto" else target_dtype
    )
    cast_model_floating_tensors(model, uniform_dtype)
    model.eval()
    print(
        f"Model ready: class={type(model).__name__}, "
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
        print_every=print_every,
        run_id=run_id,
    )
    return run_mmar_evaluation(
        args=args,
        load_model=load_audio_flamingo_next,
        generate_batch_fn=generate_af_next_batch,
        model_label="af-next-think",
        manifest_extra={"repetition_penalty": repetition_penalty},
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
        attn_implementation: Optional sdpa or flash_attention_2.
        torch_dtype: auto / float16 / bfloat16 / float32.
        device_map: Transformers device_map.
        seed: RNG seed.
        score: Pipeline OpenAI MMAR-Rubrics grading alongside inference
            (default True). Grades each batch while the next generates.
        score_only: Skip inference; only score existing predictions.
        print_every: Progress print interval.
        run_id: Optional run folder name; default is a UTC timestamp.
    """
    result = run_mmar.remote(
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
        print_every=print_every,
        run_id=run_id,
    )
    print("Done:", result)
    if isinstance(result, dict) and result.get("volume_path"):
        print(
            "Download with:\n"
            f"  uv run modal run download_results.py "
            f"--remote-path {result['volume_path']}"
        )
