"""Re-judge an existing freeform difficulty run with a local vLLM judge.

Only freeform runs are supported — MCQ / string-match runs exit with an
error. Adds (or re-grades) judges into ``shots[].judges`` without
re-running generation, then regenerates ``difficulty.jsonl`` / ``scores.json``.

Prereq: seed the judge weights on the data volume, e.g.::

    uv run modal run seed_volume.py --datasets none --models qwen2.5-3b
    uv run modal run --detach seed_volume.py --datasets none --models qwen3.6-35b-a3b-fp8

Usage::

    # Add a new text judge; keep the existing primary for difficulty ranking
    uv run modal run --detach rejudge_run.py \\
      --run-id 20260807T145000Z \\
      --judge-model-id qwen3.6-35b-a3b-fp8

    # Round-robin: each suite model grades every other model's first shot
    uv run modal run --detach rejudge_run.py \\
      --run-id 20260807T145000Z \\
      --round-robin \\
      --grade-prompt permissive,neutral \\
      --include-gold \\
      --first-shot-only
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal

from aggregate import aggregate_difficulty, discover_model_labels
from mmar_common import judge_label, write_json
from modal_cache import (
    RESULTS_MOUNT,
    VOLUME_MOUNT,
    hf_secret,
    results_volume,
    volume,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = RESULTS_MOUNT / "exp-mmar-question-difficulty"

app = modal.App("exp-mmar-rejudge")


def _cuda_base_image(python_version: str = "3.12") -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            add_python=python_version,
        )
        .entrypoint([])
        .apt_install("ffmpeg", "git")
    )


def _mount_sources(image: modal.Image) -> modal.Image:
    return image.add_local_python_source(
        "modal_cache",
        "mmar_common",
        "audio_flamingo_runtime",
        "aggregate",
        "grader",
        "mmar_models",
    )


_INPROC_VLLM_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}

# Same fused-MoE stand-in as run_experiment.py (Qwen3-Omni / Nemotron on A100).
_FUSED_MOE_CONFIG_CMD = (
    "D=/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs && "
    "SRC=\"$D/E=128,N=768,device_name=NVIDIA_H200.json\" && "
    "if [ ! -f \"$SRC\" ]; then echo \"fused_moe: no H200 config, skipping\"; "
    "else "
    "for name in NVIDIA_A100_80GB_PCIe NVIDIA_A100-SXM4-80GB; do "
    "DST=\"$D/E=128,N=768,device_name=$name.json\"; "
    "cp -n \"$SRC\" \"$DST\" 2>/dev/null || cp \"$SRC\" \"$DST\"; "
    "echo \"fused_moe: installed $name from H200\"; "
    "done; fi"
)

# 0.26+ recommended for Qwen3.6 gated-delta hybrid / FP8 checkpoints.
grader_image = _mount_sources(
    _cuda_base_image()
    .uv_pip_install(
        "vllm==0.26.0",
        "transformers>=5.5.3",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "numpy",
        "tqdm>=4.67.0",
        "accelerate>=1.14.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
)

# Suite judges load via mmar_models (same image/GPU as inference).
large_mm_image = _mount_sources(
    _cuda_base_image()
    .uv_pip_install(
        "vllm[audio]==0.26.0",
        "transformers>=5.5.3",
        "mistral-common[audio]",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "soxr",
        "av",
        "numpy",
        "tqdm>=4.67.0",
        "accelerate>=1.14.0",
        "torch",
        "torchaudio",
    )
    .run_commands(_FUSED_MOE_CONFIG_CMD)
    .env(_INPROC_VLLM_ENV)
)

cpu_image = _mount_sources(
    modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "numpy", "tqdm>=4.67.0"
    )
)


def _run_dir(output_dir: str, run_id: str) -> Path:
    return Path(output_dir).expanduser().resolve() / run_id


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"manifest.json not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid manifest.json at {path}: {exc}") from exc


def _assert_freeform_run(manifest: dict, run_id: str) -> str:
    """Return normalized mode or exit if this is an MCQ run."""
    mode = str(manifest.get("mode") or "").strip().lower()
    scoring = str(manifest.get("scoring") or "").lower()

    if mode in {"mc", "multiple_choice", "multiple-choice", "mcq", "choice"}:
        raise SystemExit(
            f"Run {run_id} is an MCQ run (mode={manifest.get('mode')!r}). "
            "rejudge_run.py only supports freeform runs."
        )
    if mode in {"freeform", "free_form", "free-form", "open"}:
        return "freeform"
    if "freeform" in scoring or "qwen_freeform" in scoring:
        return "freeform"
    if mode:
        raise SystemExit(
            f"Run {run_id} has unrecognized mode={manifest.get('mode')!r}; "
            "expected freeform. Refusing to re-judge."
        )
    # No mode stamp — fall back to scoring / graders list.
    judges = manifest.get("judges") or []
    if any(
        (isinstance(j, dict) and j.get("label") == "string-match")
        or j == "string-match"
        for j in judges
    ) and not any(
        (isinstance(j, dict) and j.get("label") not in {None, "string-match"})
        or (isinstance(j, str) and j != "string-match")
        for j in judges
    ):
        raise SystemExit(
            f"Run {run_id} looks like an MCQ / string-match run "
            f"(judges={judges!r}). rejudge_run.py only supports freeform runs."
        )
    if manifest.get("grader_model_id") or judges:
        return "freeform"
    raise SystemExit(
        f"Run {run_id} has no freeform mode stamp in manifest.json "
        f"(mode={manifest.get('mode')!r}, scoring={manifest.get('scoring')!r}). "
        "Refusing to re-judge; pass a freeform run_id."
    )


def _merge_judge_manifest(
    manifest: dict,
    *,
    model_id: str,
    judge_key: str,
    primary: str,
    make_primary: bool = False,
    prompt: str | None = None,
    include_gold: bool | None = None,
) -> dict:
    existing_primary = manifest.get("primary_judge")
    if not make_primary:
        primary = existing_primary or primary
    entries = list(manifest.get("judges") or [])
    by_label = {str(e.get("label")): dict(e) for e in entries if e.get("label")}
    entry = {
        "label": judge_key,
        "model_id": model_id,
        "primary": False,
    }
    if prompt is not None:
        entry["prompt"] = prompt
    if include_gold is not None:
        entry["include_gold"] = bool(include_gold)
    prev = by_label.get(judge_key) or {}
    prev.update(entry)
    by_label[judge_key] = prev
    if make_primary:
        primary = judge_key
    elif not primary:
        primary = existing_primary or judge_key
    ordered: list[dict] = []
    if primary in by_label:
        ordered.append(by_label[primary])
    for label, item in by_label.items():
        if label == primary:
            continue
        ordered.append(item)
    if not ordered:
        ordered = [by_label[judge_key]]
        primary = judge_key
    for item in ordered:
        item["primary"] = item.get("label") == primary
    manifest["judges"] = ordered
    if make_primary or not existing_primary:
        manifest["primary_judge"] = primary
        primary_entry = next((e for e in ordered if e.get("label") == primary), None)
        manifest["grader_model_id"] = (primary_entry or {}).get("model_id") or model_id
    manifest["scoring"] = manifest.get("scoring") or "qwen_freeform_judge"
    manifest["graded_at"] = datetime.now(timezone.utc).isoformat()
    return manifest


@app.function(
    image=cpu_image,
    timeout=10 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def prepare_rejudge(
    run_id: str,
    judge_model_id: str = "",
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    make_primary: bool = False,
    models: str = "all",
    round_robin: bool = False,
) -> dict:
    """Validate the run is freeform and resolve model labels / primary."""
    from grader import ROUND_ROBIN_SUITE, resolve_judge_model_id

    results_volume.reload()
    run_dir = _run_dir(output_dir, run_id)
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    manifest = _load_manifest(run_dir)
    mode = _assert_freeform_run(manifest, run_id)

    existing_primary = manifest.get("primary_judge")
    if not existing_primary:
        for entry in manifest.get("judges") or []:
            if isinstance(entry, dict) and entry.get("primary") and entry.get("label"):
                existing_primary = entry["label"]
                break
        if not existing_primary and manifest.get("grader_model_id"):
            existing_primary = judge_label(manifest["grader_model_id"])

    labels = discover_model_labels(run_dir, manifest=manifest)
    if models and models.strip().lower() != "all":
        requested = [part.strip() for part in models.split(",") if part.strip()]
        missing = [label for label in requested if label not in labels]
        if missing:
            raise SystemExit(
                f"Requested model(s) not found under {run_dir / 'models'}: {missing}. "
                f"Available: {labels}"
            )
        labels = requested
    if not labels:
        raise SystemExit(f"No model predictions found under {run_dir / 'models'}")

    if round_robin:
        suite = [label for label in ROUND_ROBIN_SUITE if label in labels]
        if models and models.strip().lower() != "all":
            suite = [label for label in labels if label in ROUND_ROBIN_SUITE]
        if len(suite) < 2:
            raise SystemExit(
                f"Round-robin needs at least two suite models in the run; found {suite}. "
                f"Available: {labels}"
            )
        return {
            "run_id": run_id,
            "mode": mode,
            "round_robin": True,
            "judge_model_id": None,
            "judge_label": None,
            "primary_judge": existing_primary,
            "make_primary": make_primary,
            "model_labels": suite,
            "existing_judges": [
                (e.get("label") if isinstance(e, dict) else e)
                for e in (manifest.get("judges") or [])
            ],
            "existing_primary": existing_primary,
        }

    if not judge_model_id or not str(judge_model_id).strip():
        raise SystemExit("--judge-model-id is required unless --round-robin is set")

    judge_model_id = resolve_judge_model_id(judge_model_id)
    judge_key = judge_label(judge_model_id)
    if not judge_key:
        raise SystemExit(f"Invalid judge_model_id: {judge_model_id!r}")

    if make_primary:
        primary = judge_key
    else:
        primary = existing_primary or judge_key

    return {
        "run_id": run_id,
        "mode": mode,
        "round_robin": False,
        "judge_model_id": judge_model_id,
        "judge_label": judge_key,
        "primary_judge": primary,
        "make_primary": make_primary,
        "model_labels": labels,
        "existing_judges": [
            (e.get("label") if isinstance(e, dict) else e)
            for e in (manifest.get("judges") or [])
        ],
        "existing_primary": existing_primary,
    }


@app.function(
    image=grader_image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=32768,
)
def grade_with_judge(
    run_id: str,
    judge_model_id: str,
    primary_judge: str,
    model_labels: list[str],
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    batch_size: int | None = None,
    force: bool = False,
    grade_prompt: str = "permissive",
    include_gold: bool = True,
    first_shot_only: bool = False,
    make_primary: bool = False,
) -> dict:
    """Grade with one dedicated text judge; merge into predictions + manifest."""
    from grader import (
        grade_predictions_file,
        load_grader,
        parse_grade_prompt_list,
        parse_shot_indices,
        resolve_grade_judge_key,
        resolve_judge_batch_size,
        resolve_judge_model_id,
    )

    volume.reload()
    results_volume.reload()

    run_dir = _run_dir(output_dir, run_id)
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    manifest = _load_manifest(run_dir)
    _assert_freeform_run(manifest, run_id)

    judge_model_id = resolve_judge_model_id(judge_model_id)
    prompts = parse_grade_prompt_list(grade_prompt)
    shot_indices = parse_shot_indices(first_shot_only)
    handle = load_grader(judge_model_id)
    per_model: dict[str, dict] = {}
    last_key = None
    last_prompt = prompts[-1] if prompts else "permissive"
    for prompt_name in prompts:
        key = resolve_grade_judge_key(
            handle, prompt=prompt_name, include_gold=include_gold
        )
        last_key = key
        effective_batch_size = resolve_judge_batch_size(judge_model_id, batch_size)
        for label in model_labels:
            predictions_path = run_dir / "models" / label / "predictions.jsonl"
            print(
                f"[rejudge] {label} with {key} "
                f"(primary={primary_judge}, batch_size={effective_batch_size}) "
                f"-> {predictions_path}"
            )
            per_model[f"{label}/{key}"] = grade_predictions_file(
                predictions_path,
                handle,
                judge_key=key,
                primary_judge=primary_judge,
                batch_size=effective_batch_size,
                force=force,
                prompt=prompt_name,
                include_gold=include_gold,
                shot_indices=shot_indices,
                make_primary=make_primary,
            )
            results_volume.commit()
            print(f"[rejudge] {label}:", per_model[f"{label}/{key}"])
        manifest = _merge_judge_manifest(
            manifest,
            model_id=judge_model_id,
            judge_key=key,
            primary=primary_judge,
            make_primary=make_primary,
            prompt=prompt_name,
            include_gold=include_gold,
        )
    write_json(run_dir / "manifest.json", manifest)
    results_volume.commit()
    return {
        "status": "ok",
        "run_id": run_id,
        "judge_model_id": judge_model_id,
        "judge_label": last_key,
        "primary_judge": manifest.get("primary_judge"),
        "by_model": per_model,
        "judges": manifest.get("judges"),
        "prompt": last_prompt,
        "include_gold": include_gold,
    }


def _grade_suite_judge(
    judge_label: str,
    *,
    run_id: str,
    model_labels: list[str],
    output_dir: str,
    grade_prompt: str,
    include_gold: bool,
    first_shot_only: bool,
    force: bool,
    batch_size: int | None,
) -> dict:
    from grader import (
        compose_judge_key,
        grade_predictions_file,
        load_grader,
        parse_grade_prompt_list,
        parse_shot_indices,
    )

    volume.reload()
    results_volume.reload()
    run_dir = _run_dir(output_dir, run_id)
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")
    _assert_freeform_run(_load_manifest(run_dir), run_id)

    prompts = parse_grade_prompt_list(grade_prompt)
    shot_indices = parse_shot_indices(first_shot_only)
    handle = load_grader(judge_label)
    model_id = handle.get("model_id") or judge_label
    per_model: dict[str, dict] = {}
    keys: list[str] = []
    gradees = [label for label in model_labels if label != judge_label]
    for gradee in gradees:
        predictions_path = run_dir / "models" / gradee / "predictions.jsonl"
        for prompt_name in prompts:
            key = compose_judge_key(
                judge_label, prompt=prompt_name, include_gold=include_gold
            )
            if key not in keys:
                keys.append(key)
            sidecar = (
                run_dir / "models" / gradee / "judge_partials" / f"{key}.jsonl"
            )
            print(
                f"[rejudge-rr] {judge_label} -> {gradee} key={key} "
                f"sidecar={sidecar}"
            )
            per_model[f"{gradee}/{key}"] = grade_predictions_file(
                predictions_path,
                handle,
                judge_key=key,
                batch_size=batch_size,
                force=force,
                prompt=prompt_name,
                include_gold=include_gold,
                shot_indices=shot_indices,
                sidecar_path=sidecar,
            )
            results_volume.commit()
            print(f"[rejudge-rr] {gradee}/{key}:", per_model[f"{gradee}/{key}"])
    return {
        "status": "ok",
        "judge_label": judge_label,
        "model_id": model_id,
        "judge_keys": keys,
        "gradees": gradees,
        "by_model": per_model,
        "include_gold": include_gold,
        "prompts": prompts,
    }


_SUITE_GRADE_KW = dict(
    image=large_mm_image,
    timeout=12 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
    secrets=[hf_secret],
    memory=65536,
)


@app.function(gpu="L40S", **_SUITE_GRADE_KW)
def grade_suite_l40s(
    judge_label: str,
    run_id: str,
    model_labels: list[str],
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    grade_prompt: str = "permissive",
    include_gold: bool = True,
    first_shot_only: bool = True,
    force: bool = True,
    batch_size: int | None = None,
) -> dict:
    return _grade_suite_judge(
        judge_label,
        run_id=run_id,
        model_labels=model_labels,
        output_dir=output_dir,
        grade_prompt=grade_prompt,
        include_gold=include_gold,
        first_shot_only=first_shot_only,
        force=force,
        batch_size=batch_size,
    )


@app.function(gpu="A100-80GB", **_SUITE_GRADE_KW)
def grade_suite_a100(
    judge_label: str,
    run_id: str,
    model_labels: list[str],
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    grade_prompt: str = "permissive",
    include_gold: bool = True,
    first_shot_only: bool = True,
    force: bool = True,
    batch_size: int | None = None,
) -> dict:
    return _grade_suite_judge(
        judge_label,
        run_id=run_id,
        model_labels=model_labels,
        output_dir=output_dir,
        grade_prompt=grade_prompt,
        include_gold=include_gold,
        first_shot_only=first_shot_only,
        force=force,
        batch_size=batch_size,
    )


@app.function(gpu="H100", **_SUITE_GRADE_KW)
def grade_suite_h100(
    judge_label: str,
    run_id: str,
    model_labels: list[str],
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    grade_prompt: str = "permissive",
    include_gold: bool = True,
    first_shot_only: bool = True,
    force: bool = True,
    batch_size: int | None = None,
) -> dict:
    return _grade_suite_judge(
        judge_label,
        run_id=run_id,
        model_labels=model_labels,
        output_dir=output_dir,
        grade_prompt=grade_prompt,
        include_gold=include_gold,
        first_shot_only=first_shot_only,
        force=force,
        batch_size=batch_size,
    )


_SUITE_GRADE_FNS = {
    "qwen2.5-omni-7b": grade_suite_l40s,
    "phi-4-multimodal": grade_suite_l40s,
    "gemma-4-e4b": grade_suite_l40s,
    "qwen3-omni-instruct": grade_suite_a100,
    "nemotron-3-nano-omni": grade_suite_h100,
}


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def merge_round_robin(
    run_id: str,
    model_labels: list[str],
    judge_entries: list[dict],
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    make_primary: bool = False,
    primary_judge: str | None = None,
) -> dict:
    """Fold judge sidecars into predictions.jsonl and append manifest entries."""
    from grader import apply_judge_partials

    results_volume.reload()
    run_dir = _run_dir(output_dir, run_id)
    manifest = _load_manifest(run_dir)
    _assert_freeform_run(manifest, run_id)

    by_model: dict[str, dict] = {}
    for label in model_labels:
        pred = run_dir / "models" / label / "predictions.jsonl"
        partials_dir = run_dir / "models" / label / "judge_partials"
        paths = sorted(partials_dir.glob("*.jsonl")) if partials_dir.is_dir() else []
        by_model[label] = apply_judge_partials(
            pred,
            paths,
            make_primary=make_primary,
            primary_judge=primary_judge,
        )
        print(f"[rejudge-rr] merged {label}:", by_model[label])

    for entry in judge_entries:
        manifest = _merge_judge_manifest(
            manifest,
            model_id=str(entry.get("model_id") or ""),
            judge_key=str(entry.get("judge_key") or ""),
            primary=primary_judge or manifest.get("primary_judge") or "",
            make_primary=make_primary,
            prompt=entry.get("prompt"),
            include_gold=entry.get("include_gold"),
        )
    write_json(run_dir / "manifest.json", manifest)
    results_volume.commit()
    return {
        "status": "ok",
        "run_id": run_id,
        "by_model": by_model,
        "judges": manifest.get("judges"),
        "primary_judge": manifest.get("primary_judge"),
    }


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def run_aggregate(run_id: str, output_dir: str = str(DEFAULT_OUTPUT_DIR)) -> dict:
    results_volume.reload()
    run_dir = _run_dir(output_dir, run_id)
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")
    result = aggregate_difficulty(run_dir)
    manifest = _load_manifest(run_dir)
    scores = result.get("scores") or {}
    if manifest.get("scoring"):
        scores["scoring"] = manifest["scoring"]
    if manifest.get("mode"):
        scores["mode"] = manifest["mode"]
    if manifest.get("grader_model_id"):
        scores["grader_model_id"] = manifest["grader_model_id"]
    if manifest.get("judges") is not None:
        scores["judges"] = manifest["judges"]
    if manifest.get("primary_judge"):
        scores["primary_judge"] = manifest["primary_judge"]
    write_json(run_dir / "scores.json", scores)
    result["scores"] = scores
    results_volume.commit()
    print("Aggregated:", scores)
    return result


@app.local_entrypoint()
def main(
    run_id: str,
    judge_model_id: str = "",
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    models: str = "all",
    make_primary: bool = False,
    force: bool = True,
    batch_size: int | None = None,
    skip_aggregate: bool = False,
    round_robin: bool = False,
    grade_prompt: str = "permissive",
    include_gold: bool = True,
    first_shot_only: bool = False,
):
    """Re-judge a freeform run with ``judge_model_id`` or suite round-robin.

    Re-running the same judge key replaces prior verdicts for that key
    (``force`` defaults to True).

    Args:
        run_id: Existing ``exp-mmar-question-difficulty/<run_id>`` folder.
        judge_model_id: Hugging Face id / alias for a dedicated text judge.
            Required unless ``round_robin``.
        output_dir: Results volume root (default experiment dir).
        models: Comma-separated test-model labels or ``all``. Under
            round-robin, defaults to the five-label suite intersected with
            the run.
        make_primary: If True, the new judge becomes primary (affects ranking).
            Default keeps the existing primary.
        force: Replace existing verdicts for this judge (default True).
            Pass ``--no-force`` to only fill missing shots.
        batch_size: Shots per grader generate() call (default: per-judge spec).
        skip_aggregate: Grade only; skip difficulty.jsonl / scores.json.
        round_robin: Each suite model grades every other model's traces.
        grade_prompt: ``permissive``, ``neutral``, or a comma list.
        include_gold: Insert the MCQ gold answer in the grade prompt (default
            True). Pass ``--no-include-gold`` to omit it.
        first_shot_only: Grade ``shot_index == 0`` only. Implied by round-robin.
    """
    if not run_id or not str(run_id).strip():
        raise SystemExit("--run-id is required")
    if round_robin:
        first_shot_only = True
    elif not judge_model_id or not str(judge_model_id).strip():
        raise SystemExit("--judge-model-id is required unless --round-robin is set")

    prep = prepare_rejudge.remote(
        run_id=run_id.strip(),
        judge_model_id=(judge_model_id or "").strip(),
        output_dir=output_dir,
        make_primary=make_primary,
        models=models,
        round_robin=round_robin,
    )

    if prep.get("round_robin"):
        from grader import compose_judge_key, parse_grade_prompt_list

        prompts = parse_grade_prompt_list(grade_prompt)
        suite = list(prep["model_labels"])
        print(
            f"Round-robin run_id={prep['run_id']} mode={prep['mode']} "
            f"judges={suite} prompts={prompts} include_gold={include_gold} "
            f"first_shot_only={first_shot_only} "
            f"primary={prep['primary_judge']} force={force}"
        )
        handles = []
        for judge_label in suite:
            fn = _SUITE_GRADE_FNS.get(judge_label)
            if fn is None:
                raise SystemExit(f"No GPU worker for suite judge {judge_label!r}")
            handles.append(
                (
                    judge_label,
                    fn.spawn(
                        judge_label,
                        run_id=prep["run_id"],
                        model_labels=suite,
                        output_dir=output_dir,
                        grade_prompt=",".join(prompts),
                        include_gold=include_gold,
                        first_shot_only=first_shot_only,
                        force=force,
                        batch_size=batch_size,
                    ),
                )
            )
        grade_results = []
        judge_entries: list[dict] = []
        for judge_label, handle in handles:
            result = handle.get()
            grade_results.append(result)
            print(f"Judge {judge_label}:", result)
            model_id = result.get("model_id") or judge_label
            for prompt_name in result.get("prompts") or prompts:
                judge_entries.append(
                    {
                        "judge_key": compose_judge_key(
                            judge_label,
                            prompt=prompt_name,
                            include_gold=include_gold,
                        ),
                        "model_id": model_id,
                        "prompt": prompt_name,
                        "include_gold": include_gold,
                    }
                )
        merge = merge_round_robin.remote(
            run_id=prep["run_id"],
            model_labels=suite,
            judge_entries=judge_entries,
            output_dir=output_dir,
            make_primary=make_primary,
            primary_judge=prep.get("primary_judge"),
        )
        print("Merged:", merge)
        agg = None
        if not skip_aggregate:
            agg = run_aggregate.remote(run_id=prep["run_id"], output_dir=output_dir)
            print("Aggregated:", agg)
        print(
            "Done. Download with:\n"
            f"  uv run modal run download_results.py "
            f"--remote-path exp-mmar-question-difficulty/{prep['run_id']}"
        )
        return {
            "prepare": prep,
            "grade": grade_results,
            "merge": merge,
            "aggregate": agg,
        }

    existing = {str(x) for x in (prep.get("existing_judges") or []) if x}
    replace = bool(force) or prep["judge_label"] in existing
    print(
        f"Rejudging run_id={prep['run_id']} mode={prep['mode']} "
        f"judge={prep['judge_label']} ({prep['judge_model_id']}) "
        f"primary={prep['primary_judge']} "
        f"(existing_primary={prep['existing_primary']}) "
        f"models={prep['model_labels']} "
        f"existing_judges={prep['existing_judges']} "
        f"prompt={grade_prompt} include_gold={include_gold} "
        f"first_shot_only={first_shot_only} "
        f"force={force} replace={replace}"
    )

    grade = grade_with_judge.remote(
        run_id=prep["run_id"],
        judge_model_id=prep["judge_model_id"],
        primary_judge=prep["primary_judge"],
        model_labels=prep["model_labels"],
        output_dir=output_dir,
        batch_size=batch_size,
        force=replace,
        grade_prompt=grade_prompt,
        include_gold=include_gold,
        first_shot_only=first_shot_only,
        make_primary=make_primary,
    )
    print("Graded:", grade)

    agg = None
    if not skip_aggregate:
        agg = run_aggregate.remote(run_id=prep["run_id"], output_dir=output_dir)
        print("Aggregated:", agg)

    print(
        "Done. Download with:\n"
        f"  uv run modal run download_results.py "
        f"--remote-path exp-mmar-question-difficulty/{prep['run_id']}"
    )
    return {"prepare": prep, "grade": grade, "aggregate": agg}
