"""Run Audio Flamingo 3 on MMAR with think-mode CoT on Modal.

Uses the shared ``latent-reasoning`` Volume from ``seed_volume.py`` for data
and weights, and writes eval outputs to ``latent-reasoning-results``:

    /cache/data/mmar/          # MMAR audio + MMAR-meta.jsonl
    /cache/models/<repo_id>/   # AF3 weights
    /results/mmar/af3/<run_id>/
      predictions.jsonl              # generations + CoT + answers
      predictions.evaluated.jsonl    # OpenAI rubric grades (pipelined)
      scores.json
      manifest.json

Local mirror (via ``download_results.py``): ``<repo>/outputs/mmar/af3/<run_id>/``.

Prereqs (one-time):

    uv run modal run seed_volume.py --datasets mmar --models af3

Usage:

    uv run modal run run_audio_flamingo3_mmar.py
    uv run modal run run_audio_flamingo3_mmar.py --num-samples 8
    uv run modal run run_audio_flamingo3_mmar.py --no-think --num-samples 4
    # (--no-think disables the AF-Think adapter)
    uv run modal run run_audio_flamingo3_mmar.py --n-shots 5 --temperature 0.7 --num-samples 8
    # (n_shots>1 disables rubric grading; accuracy = any-shot string match)
    uv run modal run --detach run_audio_flamingo3_mmar.py --num-samples 200

Download results locally:

    uv run modal run download_results.py
    uv run modal run download_results.py --list-only

Each published run includes:
  - predictions.jsonl            full generations + parsed CoT + answers
  - predictions.evaluated.jsonl  OpenAI MMAR-Rubrics grades
  - scores.json                  aggregate accuracy / rubric score
  - manifest.json

OpenAI grading runs concurrently with AF3 generation: each written prediction
batch is submitted to a background rubric grader so API traffic is spread over
the run (fewer burst rate-limit hits) and wall time overlaps GPU + API work.

Requires Modal Secrets:
  - ``huggingface-secret`` with ``HF_TOKEN``
  - ``openai-secret`` with ``OPENAI_API_KEY`` (for rubric scoring; default on)
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import modal

from audio_flamingo_runtime import (
    audio_tower_dtype,
    cast_floating_state_dict,
    cast_model_floating_tensors,
    generate_batch,
    model_input_device,
    model_param_dtype,
    resolve_model_dir,
    torch_dtype_value,
)
from mmar_common import (
    AF3_THINK_SUFFIX,
    build_mmar_prompt,
    parse_choice_output,
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

DEFAULT_MODEL_ID = "nvidia/audio-flamingo-3-hf"
DEFAULT_OUTPUT_DIR = RESULTS_MOUNT / "mmar" / "af3"
DEFAULT_LOCAL_MODEL_DIR = VOLUME_MOUNT / "models" / DEFAULT_MODEL_ID

image = mmar_eval_image()
app = modal.App("audio-flamingo3-mmar", image=image)


def load_audio_flamingo3(args):
    import torch
    from peft import PeftModel
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    target_dtype = torch_dtype_value(torch, args.torch_dtype)
    kwargs = {
        "device_map": args.device_map,
        "torch_dtype": target_dtype,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    local_id = resolve_model_dir(args.model_id, args.local_model_dir)
    processor = AutoProcessor.from_pretrained(local_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(local_id, **kwargs)

    if not args.no_think:
        non_lora_path = os.path.join(local_id, "think", "non_lora_trainables.bin")
        if not os.path.exists(non_lora_path):
            raise SystemExit(
                f"Think adapter weights not found at {non_lora_path}. "
                "Re-seed with seed_volume.py --models af3 (think/*.bin must be kept), "
                "or pass --no-think."
            )

        print("Loading AF-Think PEFT adapter ...")
        # non_lora weights are typically saved as float32; cast before load so we
        # do not silently mix Float and BFloat16 parameters inside the encoder.
        resolve_dtype = (
            model_param_dtype(model) if target_dtype == "auto" else target_dtype
        )
        non_lora_trainables = cast_floating_state_dict(
            torch.load(
                non_lora_path,
                map_location="cpu",
                weights_only=False,
            ),
            resolve_dtype,
        )
        model.load_state_dict(non_lora_trainables, strict=False)
        model = PeftModel.from_pretrained(model, local_id, subfolder="think")

    # Unify floating dtypes after adapter load. Think non_lora weights and some
    # embeddings/LayerNorms otherwise remain float32 while convs are bf16.
    unify_dtype = model_param_dtype(model) if target_dtype == "auto" else target_dtype
    cast_model_floating_tensors(model, unify_dtype)
    model.eval()
    print(
        f"Model ready: param_dtype={model_param_dtype(model)}, "
        f"audio_tower_dtype={audio_tower_dtype(model)}, "
        f"device={model_input_device(model)}"
    )
    return model, processor


def generate_af3_batch(model, processor, samples, args):
    use_think = not args.no_think
    return generate_batch(
        model,
        processor,
        samples,
        args,
        build_prompt=lambda sample: build_mmar_prompt(
            sample, think_suffix=AF3_THINK_SUFFIX if use_think else None
        ),
        parse_output=parse_choice_output,
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
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    attn_implementation: str | None = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    seed: int = 42,
    think: bool = True,
    score: bool = True,
    score_only: bool = False,
    n_shots: int = 1,
    print_every: int = 10,
    run_id: str | None = None,
) -> dict:
    """Run AF3 inference with pipelined MMAR-Rubrics OpenAI grading on Modal."""
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
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        device_map=device_map,
        seed=seed,
        no_think=not think,
        score=score,
        score_only=score_only,
        n_shots=n_shots,
        print_every=print_every,
        run_id=run_id,
    )
    return run_mmar_evaluation(
        args=args,
        load_model=load_audio_flamingo3,
        generate_batch_fn=generate_af3_batch,
        model_label="af3",
        manifest_extra={"think": think},
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
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    attn_implementation: str | None = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    seed: int = 42,
    think: bool = True,
    score: bool = True,
    score_only: bool = False,
    n_shots: int = 1,
    print_every: int = 10,
    run_id: str | None = None,
):
    """Launch AF3 MMAR eval on Modal.

    Args:
        model_id: Hugging Face repo id (default nvidia/audio-flamingo-3-hf).
        local_model_dir: Optional path under the models subpath; defaults to
            /cache/models/<model_id> when seeded.
        meta: Path to MMAR-meta.jsonl on the data subpath.
        data_root: MMAR root used to resolve ./audio paths.
        audio_dir: Optional override for wav directory.
        output_dir: Results Volume directory for run folders
            (default /results/mmar/af3).
        num_samples: Number of items (-1 = all). Subsets are randomly
            sampled by index (seeded).
        start: Offset into the meta file before sampling.
        batch_size: Generation batch size.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling parameter.
        attn_implementation: Optional sdpa or flash_attention_2.
        torch_dtype: auto / float16 / bfloat16 / float32.
        device_map: Transformers device_map.
        seed: RNG seed.
        think: Load AF-Think adapter and append the think suffix (default True).
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
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        device_map=device_map,
        seed=seed,
        think=think,
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
