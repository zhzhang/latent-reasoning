"""Run Audio Flamingo 3 on MMAR with think-mode CoT on Modal.

Uses the shared ``latent-reasoning`` Volume from ``seed_volume.py`` for data
and weights, and writes eval outputs to ``latent-reasoning-results``:

    /cache/data/mmar/          # MMAR audio + MMAR-meta.jsonl
    /cache/models/<repo_id>/   # AF3 weights
    /results/mmar/audio_flamingo3_mmar/<run_id>/
      predictions.jsonl              # generations + CoT + answers
      predictions.evaluated.jsonl    # OpenAI rubric grades
      scores.json
      manifest.json

Prereqs (one-time):

    uv run modal run seed_volume.py --datasets mmar --models af3

Usage:

    uv run modal run run_audio_flamingo3_mmar.py
    uv run modal run run_audio_flamingo3_mmar.py --num-samples 8
    uv run modal run run_audio_flamingo3_mmar.py --no-think --num-samples 4
    # (--no-think disables the AF-Think adapter)
    OPENAI_API_KEY=... uv run modal run run_audio_flamingo3_mmar.py --num-samples 8
    # Rubric scoring is on by default; pass --no-score to skip OpenAI grading.

Download results locally:

    uv run modal run download_results.py
    uv run modal run download_results.py --list-only

Each published run includes:
  - predictions.jsonl            full generations + parsed CoT + answers
  - predictions.evaluated.jsonl  OpenAI MMAR-Rubrics grades
  - scores.json                  aggregate accuracy / rubric score
  - manifest.json

Requires Modal Secret ``huggingface-secret`` with ``HF_TOKEN``.
For rubric scoring (default), set ``OPENAI_API_KEY`` locally so it is injected
into the container.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import modal

# ---------------------------------------------------------------------------
# Modal app / volumes / image  (cache paths match seed_volume.py)
# ---------------------------------------------------------------------------

VOLUME_NAME = "latent-reasoning"
RESULTS_VOLUME_NAME = "latent-reasoning-results"
VOLUME_MOUNT = Path("/cache")
RESULTS_MOUNT = Path("/results")
DATA_ROOT = VOLUME_MOUNT / "data"
MODELS_ROOT = VOLUME_MOUNT / "models"

DEFAULT_MODEL_ID = "nvidia/audio-flamingo-3-hf"
DEFAULT_DATA_ROOT = DATA_ROOT / "mmar"
DEFAULT_META = DEFAULT_DATA_ROOT / "MMAR-meta.jsonl"
DEFAULT_OUTPUT_DIR = RESULTS_MOUNT / "mmar" / "audio_flamingo3_mmar"
DEFAULT_LOCAL_MODEL_DIR = MODELS_ROOT / DEFAULT_MODEL_ID

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
THINK_SUFFIX = "Please think and reason about the input audio before you respond."
ANSWER_MARKERS = (
    r"therefore[, ]+the answer is[:\s]*",
    r"the answer is[:\s]*",
    r"final answer[:\s]*",
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
# Injected when present locally; empty string is fine when not scoring.
openai_secret = modal.Secret.from_dict(
    {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")}
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "torch",
        "torchaudio",
        "transformers>=5.12.1",
        "accelerate>=1.14.0",
        "peft>=0.15.2",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "openai>=1.82.0",
        "tqdm>=4.67.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("evaluation_rubrics")
)

app = modal.App("audio-flamingo3-mmar", image=image)


# ---------------------------------------------------------------------------
# Shared helpers (same logic as the former local script)
# ---------------------------------------------------------------------------


def torch_dtype_value(torch_module, dtype_name):
    return {
        "auto": "auto",
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[dtype_name]


def generation_kwargs(args):
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def model_param_dtype(model):
    return next(model.parameters()).dtype


def prepare_model_inputs(inputs, model):
    """Move processor outputs to the model device/dtype.

    Audio features come back as float32; AF3 weights are often bfloat16/float16,
    so a plain ``.to(device)`` leaves a dtype mismatch in the audio encoder.
    """
    device = model_input_device(model)
    dtype = model_param_dtype(model)
    prepared = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            prepared[key] = value
            continue
        if getattr(value, "is_floating_point", lambda: False)():
            prepared[key] = value.to(device=device, dtype=dtype)
        else:
            prepared[key] = value.to(device=device)
    return prepared


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path, records, mode="a"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_path(data_root, value):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(data_root) / path).resolve())


def format_choices(choices):
    lines = []
    for index, choice in enumerate(choices):
        label = CHOICE_LABELS[index] if index < len(CHOICE_LABELS) else str(index)
        lines.append(f"({label}) {choice}")
    return "\n".join(lines)


def build_mmar_prompt(item, use_think):
    question = str(item["question"]).strip()
    choices_block = format_choices(item["choices"])
    prompt = (
        f"{question}\n"
        "Choose the correct option from the following options:\n"
        f"{choices_block}"
    )
    if use_think:
        prompt += f"\n{THINK_SUFFIX}"
    return prompt


def build_conversation(audio_path, prompt_text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio", "path": audio_path},
            ],
        }
    ]


def _normalize_for_match(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _match_choice_in_text(text, choices):
    matched = []
    for index, choice in enumerate(choices):
        label = CHOICE_LABELS[index] if index < len(CHOICE_LABELS) else str(index)
        patterns = (
            rf"\({label}\)\s*{re.escape(choice)}",
            rf"\b{re.escape(choice)}\b",
            rf"\({label}\)",
        )
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                matched.append((index, choice))
                break

    if not matched:
        return None

    # Prefer the last-mentioned choice in the output tail.
    return matched[-1][1]


def parse_af3_output(raw_text, choices):
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    search_regions = []
    if lines:
        search_regions.append(lines[-1])
        search_regions.append("\n".join(lines[-3:]))
    search_regions.append(text)

    answer_prediction = None
    for region in search_regions:
        answer_prediction = _match_choice_in_text(region, choices)
        if answer_prediction:
            break

    if not answer_prediction:
        for marker in ANSWER_MARKERS:
            match = re.search(marker + r"(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            candidate = match.group(1).strip().splitlines()[0].strip()
            answer_prediction = _match_choice_in_text(candidate, choices) or candidate
            break

    if not answer_prediction and lines:
        answer_prediction = _match_choice_in_text(lines[-1], choices) or lines[-1]

    if not answer_prediction:
        answer_prediction = text
        return "", answer_prediction

    answer_index = text.lower().rfind(answer_prediction.lower())
    if answer_index > 0:
        thinking_prediction = text[:answer_index].strip()
    elif len(lines) > 1:
        thinking_prediction = "\n".join(lines[:-1]).strip()
    else:
        thinking_prediction = ""

    return thinking_prediction, answer_prediction


def resolve_model_dir(model_id: str, local_model_dir: str | None) -> str:
    """Prefer a seeded Volume snapshot; otherwise download into the models subpath."""
    from huggingface_hub import snapshot_download

    if local_model_dir:
        path = Path(local_model_dir).expanduser()
        if path.is_dir() and any(path.iterdir()):
            return str(path.resolve())
        raise SystemExit(f"local_model_dir not found or empty: {path}")

    seeded = MODELS_ROOT / model_id
    if seeded.is_dir() and (
        (seeded / "config.json").exists()
        or any(seeded.glob("*.safetensors"))
        or any(seeded.rglob("*.safetensors"))
    ):
        print(f"Using seeded model snapshot at {seeded}")
        return str(seeded)

    print(f"Model not found at {seeded}; downloading {model_id} onto Volume ...")
    seeded.parent.mkdir(parents=True, exist_ok=True)
    # Keep think/*.bin (non_lora_trainables); skip redundant full pytorch dumps.
    snapshot_download(
        repo_id=model_id,
        local_dir=str(seeded),
        token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.pt", "*.gguf", "*.onnx", "*.h5", "pytorch_model*.bin"],
    )
    marker = seeded / ".seed_complete"
    marker.write_text("ok\n", encoding="utf-8")
    volume.commit()
    return str(seeded)


def load_audio_flamingo3(args):
    import torch
    from peft import PeftModel
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    kwargs = {
        "device_map": args.device_map,
        "torch_dtype": torch_dtype_value(torch, args.torch_dtype),
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
        non_lora_trainables = torch.load(
            non_lora_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(non_lora_trainables, strict=False)
        model = PeftModel.from_pretrained(model, local_id, subfolder="think")

    model.eval()
    return model, processor


def generate_batch(model, processor, samples, args):
    import torch

    use_think = not args.no_think
    conversations = []
    for sample in samples:
        prompt_text = build_mmar_prompt(sample, use_think=use_think)
        conversations.append(build_conversation(sample["audio_path"], prompt_text))

    inputs = prepare_model_inputs(
        processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ),
        model,
    )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs(args))

    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(
        outputs[:, prompt_len:],
        skip_special_tokens=True,
    )

    results = []
    for sample, raw_output in zip(samples, decoded):
        thinking_prediction, answer_prediction = parse_af3_output(
            raw_output,
            sample["choices"],
        )
        results.append(
            {
                # Full model generation (includes CoT + final answer text).
                "model_output": raw_output,
                # Parsed chain-of-thought / reasoning trace.
                "thinking_prediction": thinking_prediction,
                # Parsed final answer choice / text.
                "answer_prediction": answer_prediction,
            }
        )
    return results


def load_completed_ids(predictions_path):
    if not predictions_path.exists():
        return set()
    completed = set()
    for item in load_jsonl(predictions_path):
        record_id = item.get("id")
        if record_id:
            completed.add(record_id)
    return completed


def run_rubrics_scoring(predictions_path, meta_path, evaluated_path, *, required: bool = True):
    """Run OpenAI rubric grading; write ``predictions.evaluated.jsonl``."""
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key.strip():
        message = (
            "OPENAI_API_KEY is not set; cannot run MMAR-Rubrics grading.\n"
            "Export OPENAI_API_KEY locally before `modal run`, e.g.:\n"
            "  OPENAI_API_KEY=... uv run modal run run_audio_flamingo3_mmar.py --num-samples 8"
        )
        if required:
            raise SystemExit(message)
        print(message)
        return False

    if not predictions_path.exists() or predictions_path.stat().st_size == 0:
        message = f"No predictions to score at {predictions_path}"
        if required:
            raise SystemExit(message)
        print(message)
        return False

    rubrics_script = Path("/root/evaluation_rubrics.py")
    if not rubrics_script.exists():
        # add_local_python_source places modules on sys.path; resolve via import.
        import evaluation_rubrics as rubrics_mod

        rubrics_script = Path(rubrics_mod.__file__)

    cmd = [
        sys.executable,
        str(rubrics_script),
        "--input",
        str(predictions_path),
        "--meta",
        str(meta_path),
        "--output",
        str(evaluated_path),
    ]
    print("Running MMAR-Rubrics OpenAI evaluation ...")
    subprocess.run(cmd, check=True)
    return True


def summarize_evaluated(evaluated_path: Path) -> dict:
    """Aggregate accuracy / rubric score from an evaluated predictions file."""
    if not evaluated_path.exists():
        return {}

    records = load_jsonl(evaluated_path)
    if not records:
        return {"n": 0}

    n = len(records)
    n_correct = sum(1 for item in records if item.get("correct"))
    score_sum = sum(float(item.get("score") or 0.0) for item in records)
    by_modality: dict[str, dict] = {}
    by_category: dict[str, dict] = {}
    for item in records:
        for key, bucket in (
            (item.get("modality") or "unknown", by_modality),
            (item.get("category") or "unknown", by_category),
        ):
            stats = bucket.setdefault(key, {"n": 0, "correct": 0, "score_sum": 0.0})
            stats["n"] += 1
            stats["correct"] += int(bool(item.get("correct")))
            stats["score_sum"] += float(item.get("score") or 0.0)

    def finalize(bucket: dict) -> dict:
        out = {}
        for key, stats in bucket.items():
            total = stats["n"] or 1
            out[key] = {
                "n": stats["n"],
                "accuracy": stats["correct"] / total,
                "avg_score": stats["score_sum"] / total,
            }
        return out

    summary = {
        "n": n,
        "accuracy": n_correct / n,
        "avg_score": score_sum / n,
        "by_modality": finalize(by_modality),
        "by_category": finalize(by_category),
    }
    return summary


def count_wavs(audio_dir: Path) -> int:
    if not audio_dir.is_dir():
        return 0
    return sum(
        1 for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"
    )


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_output_dir(output_dir: str | Path, run_id: str | None = None) -> Path:
    """Place this run under ``output_dir/<run_id>/`` on the results Volume."""
    base = Path(output_dir).expanduser()
    rid = run_id or make_run_id()
    return (base / rid).resolve()


def write_manifest(run_dir: Path, payload: dict) -> Path:
    path = run_dir / "manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def volume_relative_path(path: Path, mount: Path) -> str:
    """Return a Volume-relative path even when Modal resolves mounts under /__modal."""
    path_r = path.resolve()
    mount_r = mount.resolve()
    try:
        return str(path_r.relative_to(mount_r))
    except ValueError:
        parts = path_r.parts
        for marker in ("mmar", "aha", "mcr"):
            if marker in parts:
                idx = parts.index(marker)
                return str(Path(*parts[idx:]))
        return str(path_r)


def publish_run_artifacts(run_dir: Path, manifest: dict | None = None) -> dict:
    """Commit the results Volume and summarize artifacts present in ``run_dir``."""
    if manifest is not None:
        write_manifest(run_dir, manifest)

    artifacts = {}
    for name in (
        "predictions.jsonl",
        "predictions.evaluated.jsonl",
        "scores.json",
        "manifest.json",
    ):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            artifacts[name] = str(path)

    results_volume.commit()
    # Also keep a stable "latest" copy for easy download defaults.
    latest_dir = run_dir.parent / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    results_volume.commit()

    volume_rel = volume_relative_path(run_dir, RESULTS_MOUNT)
    latest_rel = volume_relative_path(latest_dir, RESULTS_MOUNT)
    print(f"Published results to volume:{RESULTS_VOLUME_NAME}/{volume_rel}")
    print(f"Also updated volume:{RESULTS_VOLUME_NAME}/{latest_rel}")
    return {
        "run_dir": str(run_dir),
        "latest_dir": str(latest_dir),
        "volume_path": volume_rel,
        "latest_volume_path": latest_rel,
        "artifacts": artifacts,
    }


def finalize_scored_run(
    run_dir: Path,
    predictions_path: Path,
    evaluated_path: Path,
    meta_path: Path,
    base_manifest: dict,
    *,
    score: bool,
) -> dict:
    """Ensure rubric grading + summary scores are on disk, then publish."""
    scored = False
    if score:
        scored = run_rubrics_scoring(
            predictions_path,
            meta_path,
            evaluated_path,
            required=True,
        )
        summary = summarize_evaluated(evaluated_path)
        if summary:
            write_json(run_dir / "scores.json", summary)
            print(
                f"Rubric summary: n={summary.get('n')} "
                f"acc={summary.get('accuracy', 0):.3f} "
                f"avg_score={summary.get('avg_score', 0):.3f}"
            )

    manifest = {
        **base_manifest,
        "scored": scored,
        "predictions": str(predictions_path),
        "evaluated": str(evaluated_path) if evaluated_path.exists() else None,
        "scores": str(run_dir / "scores.json") if (run_dir / "scores.json").exists() else None,
        "fields": {
            "model_output": "full generation text",
            "thinking_prediction": "parsed CoT / reasoning trace",
            "answer_prediction": "parsed final answer",
            "score": "MMAR-Rubrics aggregate score (evaluated file)",
            "correct": "answer correctness from rubric grader",
            "raw_responses": "OpenAI grader raw responses",
            "rubric_results": "per-criterion rubric judgments",
        },
    }
    return publish_run_artifacts(run_dir, manifest)


# ---------------------------------------------------------------------------
# Remote Function
# ---------------------------------------------------------------------------


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
    meta: str = str(DEFAULT_META),
    data_root: str = str(DEFAULT_DATA_ROOT),
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
    print_every: int = 10,
    run_id: str | None = None,
) -> dict:
    """Run AF3 inference + MMAR-Rubrics OpenAI grading on Modal."""
    volume.reload()
    results_volume.reload()

    run_id = run_id or make_run_id()
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
        print_every=print_every,
        run_id=run_id,
    )
    random.seed(args.seed)

    data_root_path = Path(args.data_root).expanduser().resolve()
    meta_path = Path(args.meta).expanduser().resolve()
    run_dir = resolve_run_output_dir(args.output_dir, run_id=run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    evaluated_path = run_dir / "predictions.evaluated.jsonl"

    base_manifest = {
        "run_id": run_id,
        "benchmark": "mmar",
        "model_id": model_id,
        "think": think,
        "num_samples": num_samples,
        "start": start,
        "seed": seed,
        "torch_dtype": torch_dtype,
        "max_new_tokens": max_new_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if args.score_only:
        # Score the latest predictions under the parent output dir if this run is empty.
        if not predictions_path.exists():
            latest_preds = Path(args.output_dir).expanduser().resolve() / "latest" / "predictions.jsonl"
            if latest_preds.exists():
                run_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(latest_preds, predictions_path)
            else:
                raise SystemExit(
                    f"No predictions found at {predictions_path} "
                    f"(and no latest at {latest_preds})"
                )
        published = finalize_scored_run(
            run_dir,
            predictions_path,
            evaluated_path,
            meta_path,
            {**base_manifest, "status": "scored", "score_only": True},
            score=True,
        )
        return {"status": "scored", **published}

    if not meta_path.exists():
        raise SystemExit(
            f"MMAR metadata not found: {meta_path}\n"
            "Seed first: uv run modal run seed_volume.py --datasets mmar --models none"
        )

    meta_items = load_jsonl(meta_path)
    expected_wavs = len(meta_items)
    resolved_audio_dir = Path(args.audio_dir) if args.audio_dir else data_root_path / "audio"
    wav_count = count_wavs(resolved_audio_dir)
    if wav_count < expected_wavs:
        raise SystemExit(
            f"MMAR audio missing in {resolved_audio_dir} "
            f"({wav_count}/{expected_wavs} wav files).\n"
            "Seed first: uv run modal run seed_volume.py --datasets mmar --models none"
        )

    end = None if args.num_samples < 0 else args.start + args.num_samples
    selected_items = meta_items[args.start : end]
    completed_ids = load_completed_ids(predictions_path)
    pending_items = []
    for item in selected_items:
        audio_path = resolve_path(data_root_path, item["audio_path"])
        if not os.path.exists(audio_path):
            print(f"Skipping {item['id']}: missing audio at {audio_path}")
            continue
        if item["id"] in completed_ids:
            continue
        pending_items.append({**item, "audio_path": audio_path})

    if not pending_items:
        print("No pending MMAR items to evaluate.")
        if not predictions_path.exists() or predictions_path.stat().st_size == 0:
            raise SystemExit(
                f"No predictions available to score at {predictions_path}. "
                "Re-run inference first."
            )
        published = finalize_scored_run(
            run_dir,
            predictions_path,
            evaluated_path,
            meta_path,
            {**base_manifest, "status": "noop", "pending": 0},
            score=args.score,
        )
        return {"status": "noop", "pending": 0, **published}

    print(f"Evaluating {len(pending_items)} MMAR items with {args.model_id} ...")
    print(f"Writing run artifacts to {run_dir}")
    model, processor = load_audio_flamingo3(args)

    start_time = time.time()
    completed = 0
    failures = []
    for batch_start in range(0, len(pending_items), args.batch_size):
        batch = pending_items[batch_start : batch_start + args.batch_size]
        try:
            outputs = generate_batch(model, processor, batch, args)
        except (OSError, ValueError, RuntimeError) as exc:
            ids = ", ".join(item["id"] for item in batch)
            print(f"Skipping batch ({ids}): {exc}")
            failures.append({"ids": [item["id"] for item in batch], "error": str(exc)})
            continue

        records = []
        for item, output in zip(batch, outputs):
            # Keep the full MMAR meta row plus generation / CoT / answer fields.
            record = {
                **item,
                **output,
            }
            records.append(record)
            completed += 1
            if args.print_every > 0 and completed % args.print_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{completed}/{len(pending_items)}] "
                    f"id={item['id']} answer={output['answer_prediction']!r} "
                    f"({elapsed:.1f}s elapsed)"
                )

        write_jsonl(predictions_path, records, mode="a")
        # Persist incrementally so preemption / timeout does not lose progress.
        results_volume.commit()

    print(f"Wrote predictions to {predictions_path}")
    if completed == 0:
        raise SystemExit(
            f"All {len(pending_items)} generations failed; not publishing empty results.\n"
            f"First errors: {failures[:3]}"
        )

    elapsed_s = round(time.time() - start_time, 1)
    published = finalize_scored_run(
        run_dir,
        predictions_path,
        evaluated_path,
        meta_path,
        {
            **base_manifest,
            "status": "ok",
            "completed": completed,
            "failed_batches": len(failures),
            "elapsed_s": elapsed_s,
        },
        score=args.score,
    )
    return {
        "status": "ok",
        "completed": completed,
        "failed_batches": len(failures),
        "elapsed_s": elapsed_s,
        **published,
    }


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL_ID,
    local_model_dir: str | None = None,
    meta: str = str(DEFAULT_META),
    data_root: str = str(DEFAULT_DATA_ROOT),
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
            (default /results/mmar/audio_flamingo3_mmar).
        num_samples: Number of items (-1 = all).
        start: Offset into the meta file.
        batch_size: Generation batch size.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling parameter.
        attn_implementation: Optional sdpa or flash_attention_2.
        torch_dtype: auto / float16 / bfloat16 / float32.
        device_map: Transformers device_map.
        seed: RNG seed.
        think: Load AF-Think adapter and append the think suffix (default True).
        score: Run OpenAI MMAR-Rubrics grading after inference (default True).
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
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        device_map=device_map,
        seed=seed,
        think=think,
        score=score,
        score_only=score_only,
        print_every=print_every,
        run_id=run_id,
    )
    print("Done:", result)
    if isinstance(result, dict) and result.get("latest_volume_path"):
        print(
            "Download with:\n"
            f"  uv run modal run download_results.py "
            f"--remote-path {result['latest_volume_path']}"
        )
