"""Re-judge an existing freeform difficulty run with a new local vLLM judge.

Only freeform runs are supported — MCQ / string-match runs exit with an
error. Adds (or re-grades) one judge into ``shots[].judges`` without
re-running generation, then regenerates ``difficulty.jsonl`` / ``scores.json``.

Prereq: seed the judge weights on the data volume, e.g.::

    uv run modal run seed_volume.py --datasets none --models qwen2.5-3b
    uv run modal run --detach seed_volume.py --datasets none --models qwen3.6-27b-fp8

Usage::

    # Add a new judge; keep the existing primary for difficulty ranking
    uv run modal run --detach rejudge_run.py \\
      --run-id 20260807T145000Z \\
      --judge-model-id qwen3.6-27b-fp8

    # Add a judge and make it primary
    uv run modal run --detach rejudge_run.py \\
      --run-id 20260807T145000Z \\
      --judge-model-id Qwen/Qwen3.6-27B-FP8 \\
      --make-primary

    # Re-grade even if this judge already has verdicts
    uv run modal run rejudge_run.py \\
      --run-id 20260807T145000Z \\
      --judge-model-id Qwen/Qwen2.5-3B-Instruct \\
      --force
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
DEFAULT_BATCH_SIZE = 256

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
) -> dict:
    entries = list(manifest.get("judges") or [])
    by_label = {str(e.get("label")): dict(e) for e in entries if e.get("label")}
    by_label[judge_key] = {
        "label": judge_key,
        "model_id": model_id,
        "primary": judge_key == primary,
    }
    ordered: list[dict] = []
    if primary in by_label:
        ordered.append(by_label[primary])
    for label, entry in by_label.items():
        if label == primary:
            continue
        entry["primary"] = False
        ordered.append(entry)
    if not ordered:
        ordered = [by_label[judge_key]]
        primary = judge_key
    for entry in ordered:
        entry["primary"] = entry.get("label") == primary
    manifest["judges"] = ordered
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
    judge_model_id: str,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    make_primary: bool = False,
    models: str = "all",
) -> dict:
    """Validate the run is freeform and resolve model labels / primary."""
    from grader import resolve_judge_model_id

    results_volume.reload()
    run_dir = _run_dir(output_dir, run_id)
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    manifest = _load_manifest(run_dir)
    mode = _assert_freeform_run(manifest, run_id)

    judge_model_id = resolve_judge_model_id(judge_model_id)
    judge_key = judge_label(judge_model_id)
    if not judge_key:
        raise SystemExit(f"Invalid judge_model_id: {judge_model_id!r}")

    existing_primary = manifest.get("primary_judge")
    if not existing_primary:
        for entry in manifest.get("judges") or []:
            if isinstance(entry, dict) and entry.get("primary") and entry.get("label"):
                existing_primary = entry["label"]
                break
        if not existing_primary and manifest.get("grader_model_id"):
            existing_primary = judge_label(manifest["grader_model_id"])

    if make_primary:
        primary = judge_key
    else:
        primary = existing_primary or judge_key

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

    return {
        "run_id": run_id,
        "mode": mode,
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
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> dict:
    """Grade all shots with one new judge; merge into predictions + manifest."""
    from grader import grade_predictions_file, load_grader

    volume.reload()
    results_volume.reload()

    run_dir = _run_dir(output_dir, run_id)
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    from grader import resolve_judge_model_id

    # Re-check mode inside the grader container (defense in depth).
    manifest = _load_manifest(run_dir)
    _assert_freeform_run(manifest, run_id)

    judge_model_id = resolve_judge_model_id(judge_model_id)
    judge_key = judge_label(judge_model_id)
    handle = load_grader(judge_model_id)
    per_model: dict[str, dict] = {}
    for label in model_labels:
        predictions_path = run_dir / "models" / label / "predictions.jsonl"
        print(
            f"[rejudge] {label} with {judge_key} "
            f"(primary={primary_judge}) -> {predictions_path}"
        )
        per_model[label] = grade_predictions_file(
            predictions_path,
            handle,
            judge_key=judge_key,
            primary_judge=primary_judge,
            batch_size=batch_size,
            force=force,
        )
        results_volume.commit()
        print(f"[rejudge] {label}:", per_model[label])

    manifest = _merge_judge_manifest(
        manifest,
        model_id=judge_model_id,
        judge_key=judge_key,
        primary=primary_judge,
    )
    write_json(run_dir / "manifest.json", manifest)
    results_volume.commit()
    return {
        "status": "ok",
        "run_id": run_id,
        "judge_model_id": judge_model_id,
        "judge_label": judge_key,
        "primary_judge": primary_judge,
        "by_model": per_model,
        "judges": manifest.get("judges"),
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
    judge_model_id: str,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    models: str = "all",
    make_primary: bool = False,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    skip_aggregate: bool = False,
):
    """Re-judge a freeform run with ``judge_model_id``.

    Args:
        run_id: Existing ``exp-mmar-question-difficulty/<run_id>`` folder.
        judge_model_id: Hugging Face id for the new local vLLM judge.
        output_dir: Results volume root (default experiment dir).
        models: Comma-separated test-model labels or ``all``.
        make_primary: If True, the new judge becomes primary (affects ranking).
            Default keeps the existing primary and only adds this judge.
        force: Re-grade shots even when this judge already has a verdict.
        batch_size: Shots per grader generate() call.
        skip_aggregate: Grade only; skip difficulty.jsonl / scores.json.
    """
    if not run_id or not str(run_id).strip():
        raise SystemExit("--run-id is required")
    if not judge_model_id or not str(judge_model_id).strip():
        raise SystemExit("--judge-model-id is required")

    prep = prepare_rejudge.remote(
        run_id=run_id.strip(),
        judge_model_id=judge_model_id.strip(),
        output_dir=output_dir,
        make_primary=make_primary,
        models=models,
    )
    print(
        f"Rejudging run_id={prep['run_id']} mode={prep['mode']} "
        f"judge={prep['judge_label']} ({prep['judge_model_id']}) "
        f"primary={prep['primary_judge']} "
        f"(existing_primary={prep['existing_primary']}) "
        f"models={prep['model_labels']} "
        f"existing_judges={prep['existing_judges']} force={force}"
    )

    grade = grade_with_judge.remote(
        run_id=prep["run_id"],
        judge_model_id=prep["judge_model_id"],
        primary_judge=prep["primary_judge"],
        model_labels=prep["model_labels"],
        output_dir=output_dir,
        batch_size=batch_size,
        force=force,
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
