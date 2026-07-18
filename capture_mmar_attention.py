"""Capture layer-wise attentions for one MMAR example (Modal).

Re-runs a single sample from an existing MMAR run with deterministic seeding
and ``attn_implementation=eager``, then writes:

    /results/mmar/<subdir>/<run_id>/attentions/<sample_id>.npz
    /results/mmar/<subdir>/<run_id>/attentions/<sample_id>.json

Usage:

    uv run modal run capture_mmar_attention.py \\
      --run-id 20260712T215605Z \\
      --sample-id BV1fu4y1a72a_00-00-01_00-00-18 \\
      --results-subdir mmar/af3

    uv run modal run capture_mmar_attention.py \\
      --run-id 20260713T025940Z \\
      --sample-id BV1fu4y1a72a_00-00-01_00-00-18 \\
      --results-subdir mmar/af-next-think
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import modal

from audio_flamingo_runtime import (
    generate_one_with_attentions,
    save_attention_artifact,
    seed_everything,
)
from mmar_common import (
    AF3_THINK_SUFFIX,
    AF_NEXT_THINK_SUFFIX,
    build_mmar_prompt,
    load_jsonl,
    parse_choice_output,
    parse_think_tagged_output,
    resolve_path,
    volume_relative_path,
)
from modal_cache import (
    DEFAULT_MMAR_DATA_ROOT,
    RESULTS_MOUNT,
    VOLUME_MOUNT,
    hf_secret,
    mmar_eval_image,
    results_volume,
    volume,
)

DEFAULT_REPETITION_PENALTY = 1.2

image = mmar_eval_image(
    "run_audio_flamingo3_mmar",
    "run_audio_flamingo_next_mmar",
)
app = modal.App("capture-mmar-attention", image=image)


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_prediction(run_dir: Path, sample_id: str) -> dict | None:
    for name in ("predictions.evaluated.jsonl", "predictions.jsonl"):
        path = run_dir / name
        if not path.is_file():
            continue
        for item in load_jsonl(path):
            if item.get("id") == sample_id:
                return item
    return None


def _infer_family(manifest: dict, results_subdir: str) -> str:
    label = (manifest.get("model_label") or "").lower()
    model_id = (manifest.get("model_id") or "").lower()
    sub = results_subdir.strip("/").lower()
    if "next" in label or "next" in model_id or "af-next" in sub:
        return "af-next-think"
    return "af3"


def _resolve_run_dir(results_subdir: str, run_id: str) -> Path:
    base = RESULTS_MOUNT / results_subdir.strip("/")
    run_dir = base / run_id
    if run_dir.is_dir() and (
        (run_dir / "manifest.json").is_file() or (run_dir / "predictions.jsonl").is_file()
    ):
        return run_dir
    # Flat layout fallback under /results/<run_id>
    alt = RESULTS_MOUNT / run_id
    if alt.is_dir() and (
        (alt / "manifest.json").is_file() or (alt / "predictions.jsonl").is_file()
    ):
        return alt
    raise FileNotFoundError(
        f"Run not found at {run_dir} (also checked {alt}). "
        "Download/sync the run onto the results volume first."
    )


@app.function(
    gpu="L40S",
    timeout=60 * 60,
    volumes={
        VOLUME_MOUNT: volume,
        RESULTS_MOUNT: results_volume,
    },
    secrets=[hf_secret],
    memory=65536,
)
def capture_attention(
    run_id: str,
    sample_id: str,
    results_subdir: str = "mmar/af3",
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
) -> dict:
    """Re-generate one MMAR sample and save layer-wise attentions."""
    results_volume.reload()
    volume.reload()

    run_dir = _resolve_run_dir(results_subdir, run_id)
    manifest = _load_json(run_dir / "manifest.json") or {}
    prediction = _find_prediction(run_dir, sample_id)
    if prediction is None:
        raise FileNotFoundError(
            f"Sample {sample_id!r} not found in predictions under {run_dir}"
        )

    family = _infer_family(manifest, results_subdir)
    seed = int(manifest.get("seed", 42))
    max_new_tokens = int(manifest.get("max_new_tokens", 1024 if family == "af3" else 4096))
    torch_dtype = str(manifest.get("torch_dtype", "bfloat16"))
    model_id = str(
        manifest.get("model_id")
        or (
            "nvidia/audio-flamingo-next-think-hf"
            if family == "af-next-think"
            else "nvidia/audio-flamingo-3-hf"
        )
    )
    think = bool(manifest.get("think", True))
    repetition_penalty = float(
        manifest.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
    )

    audio_path = resolve_path(data_root, prediction.get("audio_path"))
    if not Path(audio_path).is_file():
        raise FileNotFoundError(
            f"Audio not found at {audio_path}. "
            "Seed MMAR data: uv run modal run seed_volume.py --datasets mmar"
        )

    sample = {
        **prediction,
        "audio_path": audio_path,
    }

    seed_everything(seed)

    args = SimpleNamespace(
        model_id=model_id,
        local_model_dir=None,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map="auto",
        no_think=not think,
        repetition_penalty=repetition_penalty,
    )

    if family == "af-next-think":
        from run_audio_flamingo_next_mmar import load_audio_flamingo_next

        model, processor = load_audio_flamingo_next(args)
        build_prompt = lambda s: build_mmar_prompt(s, think_suffix=AF_NEXT_THINK_SUFFIX)
        parse_output = parse_think_tagged_output
        generation_extra = {"repetition_penalty": repetition_penalty}
    else:
        from run_audio_flamingo3_mmar import load_audio_flamingo3

        model, processor = load_audio_flamingo3(args)
        build_prompt = lambda s: build_mmar_prompt(
            s, think_suffix=AF3_THINK_SUFFIX if think else None
        )
        parse_output = parse_choice_output
        generation_extra = None

    print(
        f"Capturing attentions: run={run_id} sample={sample_id} "
        f"family={family} seed={seed} eager=True"
    )
    result = generate_one_with_attentions(
        model,
        processor,
        sample,
        args,
        build_prompt=build_prompt,
        parse_output=parse_output,
        generation_extra=generation_extra,
        stored_raw_tokens=prediction.get("raw_tokens"),
    )

    attentions = result.pop("attentions")
    meta = {
        "run_id": run_id,
        "results_subdir": results_subdir.strip("/"),
        "model_id": model_id,
        "model_family": family,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "torch_dtype": torch_dtype,
        "attn_implementation": "eager",
        "think": think if family == "af3" else True,
        "repetition_penalty": repetition_penalty if family == "af-next-think" else None,
        "prompt_len": result["prompt_len"],
        "generated_ids": result["generated_ids"],
        "estimated_bytes": result["estimated_bytes"],
        "token_match": result["token_match"],
        "mismatch_index": result["mismatch_index"],
        "generated_len": result["generated_len"],
        "stored_generated_len": result["stored_generated_len"],
        "thinking_prediction": result["thinking_prediction"],
        "answer_prediction": result["answer_prediction"],
    }
    saved = save_attention_artifact(run_dir, sample_id, attentions, meta)
    # Ensure handles are closed and sizes are visible before committing the
    # large npz; otherwise clients can see the path via listdir but hit
    # ``404 block not found`` when reading immediately after return.
    npz_path = Path(saved["npz_path"])
    json_path = npz_path.with_suffix(".json")
    for path in (npz_path, json_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Attention artifact missing or empty: {path}")
    results_volume.commit()
    # Re-open and touch-read a few bytes after commit to encourage block flush.
    with open(npz_path, "rb") as handle:
        handle.read(1)
    results_volume.commit()

    volume_path = volume_relative_path(run_dir, RESULTS_MOUNT)
    artifact_id = saved["artifact_id"]
    return {
        "status": "ok",
        "run_id": run_id,
        "sample_id": sample_id,
        "volume_path": volume_path,
        "attentions_remote": f"{volume_path}/attentions/{artifact_id}",
        "token_match": result["token_match"],
        "mismatch_index": result["mismatch_index"],
        "generated_len": result["generated_len"],
        "prompt_len": result["prompt_len"],
        "estimated_bytes": result["estimated_bytes"],
        "num_layers": saved["num_layers"],
        "num_heads": saved["num_heads"],
        "num_steps": saved["num_steps"],
        "npz_bytes": npz_path.stat().st_size,
        "json_bytes": json_path.stat().st_size,
    }


@app.local_entrypoint()
def main(
    run_id: str,
    sample_id: str,
    results_subdir: str = "mmar/af3",
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
):
    """Capture attentions for one MMAR sample on Modal.

    Args:
        run_id: Existing eval run folder name (UTC timestamp).
        sample_id: MMAR example id from predictions.jsonl.
        results_subdir: Volume subdir for the run (mmar/af3 or mmar/af-next-think).
        data_root: MMAR data root on the cache volume.
    """
    call = capture_attention.spawn(
        run_id=run_id,
        sample_id=sample_id,
        results_subdir=results_subdir,
        data_root=data_root,
    )
    print(f"Spawned capture_attention call_id={call.object_id}")
    result = call.get()
    print("CAPTURE_RESULT:" + json.dumps(result))
    print("Done:", json.dumps(result, indent=2))
    if isinstance(result, dict) and result.get("attentions_remote"):
        print(
            "Download with:\n"
            f"  uv run modal volume get latent-reasoning-results "
            f"{result['attentions_remote']}.npz ./outputs/ --force\n"
            f"  uv run modal volume get latent-reasoning-results "
            f"{result['attentions_remote']}.json ./outputs/ --force"
        )
