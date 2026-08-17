"""Judge the collated MMAR freeform 5-shot pack.

Always reads and writes ``outputs/mmar-freeform-5-shot``. Shots that already
have a verdict for the same judge key (model + prompt + gold/nongold) are
skipped. Regenerates ``difficulty.jsonl`` / ``scores.json`` after grading.

vLLM suite / dedicated judges start Modal from this script. API judges
(gpt-audio-mini, gemini-3.7-flash) run locally and never open an App.

    uv run modal run seed_volume.py --datasets none --models qwen2.5-3b
    uv run modal run --detach seed_volume.py --datasets none --models qwen3.6-35b-a3b-fp8

    # All suite judges; each grades every other pack model's 5 shots
    uv run run_judges.py

    # Only these judges
    uv run run_judges.py \\
      --judge-model-id qwen3-omni-instruct,phi-4-multimodal

    # Dedicated text judge (not in the suite)
    uv run run_judges.py \\
      --judge-model-id qwen3.6-35b-a3b-fp8

    # Audio judge, no gold (hears the clip)
    uv run run_judges.py \\
      --judge-model-id qwen3-omni-instruct \\
      --no-include-gold

Local API judges::

    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...

    uv run run_judges.py --judge-model-id gpt-audio-mini,gemini-3.7-flash
    uv run run_judges.py --judge-model-id api --no-include-gold

Mixed (API locally while Modal vLLM runs detached)::

    uv run run_judges.py \\
      --judge-model-id gpt-audio-mini,qwen3-omni-instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
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
PACK_NAME = "mmar-freeform-5-shot"
LOCAL_PACK_DIR = REPO_ROOT / "outputs" / PACK_NAME
REMOTE_PACK_DIR = RESULTS_MOUNT / PACK_NAME
LOCAL_PACK_MOUNT = Path("/local-pack")
INGEST_DIR_NAME = "_local_ingest"

app = modal.App("run-judges")


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
        "mmar_api",
        "mmar_groundedness",
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
if LOCAL_PACK_DIR.is_dir():
    cpu_image = cpu_image.add_local_dir(
        str(LOCAL_PACK_DIR), remote_path=str(LOCAL_PACK_MOUNT)
    )


def _pack_dir() -> Path:
    return REMOTE_PACK_DIR


def _shot_answer(shot: dict) -> str:
    return str(
        shot.get("answer_prediction") or shot.get("model_output") or ""
    ).strip().lower()


def _merge_record_judges(local: dict, prior: dict) -> dict:
    """Keep prior verdicts when the shot answer is unchanged."""
    prior_shots = {
        int(shot.get("shot_index", 0)): shot for shot in (prior.get("shots") or [])
    }
    merged_shots = []
    for shot in local.get("shots") or []:
        prev = prior_shots.get(int(shot.get("shot_index", 0)))
        prev_judges = (prev or {}).get("judges") or {}
        if prev is not None and prev_judges and _shot_answer(shot) == _shot_answer(prev):
            shot = dict(shot)
            judges = dict(shot.get("judges") or {})
            for key, entry in prev_judges.items():
                judges.setdefault(key, entry)
            shot["judges"] = judges
        merged_shots.append(shot)
    merged = dict(local)
    merged["shots"] = merged_shots
    prior_labels = [str(x) for x in (prior.get("judges") or []) if x]
    local_labels = [str(x) for x in (merged.get("judges") or []) if x]
    if prior_labels:
        seen = set(local_labels)
        merged["judges"] = local_labels + [x for x in prior_labels if x not in seen]
    if prior.get("primary_judge") and not merged.get("primary_judge"):
        merged["primary_judge"] = prior["primary_judge"]
    return merged


def _merge_predictions_jsonl(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        shutil.copy2(src, dest)
        return
    prior_by_id: dict[str, dict] = {}
    with dest.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            rid = str(record.get("id") or "")
            if rid:
                prior_by_id[rid] = record
    merged: list[dict] = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            prior = prior_by_id.get(str(record.get("id") or ""))
            merged.append(_merge_record_judges(record, prior) if prior else record)
    with dest.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _merge_manifest_file(src: Path, dest: Path) -> None:
    local = json.loads(src.read_text(encoding="utf-8"))
    if not dest.is_file():
        dest.write_text(json.dumps(local, indent=2) + "\n", encoding="utf-8")
        return
    prior = json.loads(dest.read_text(encoding="utf-8"))
    keep = (
        "judges",
        "primary_judge",
        "grader_model_id",
        "scoring",
        "graded_at",
    )
    merged = dict(local)
    local_judges = local.get("judges") or []
    prior_judges = prior.get("judges") or []
    if local_judges:
        by_label: dict[str, dict] = {}
        for entry in list(prior_judges) + list(local_judges):
            if isinstance(entry, dict) and entry.get("label"):
                by_label[str(entry["label"])] = dict(entry)
            elif isinstance(entry, str) and entry:
                by_label.setdefault(entry, {"label": entry})
        ordered: list[dict] = []
        seen: set[str] = set()
        for source in (local_judges, prior_judges):
            for entry in source:
                label = (
                    str(entry.get("label"))
                    if isinstance(entry, dict)
                    else str(entry)
                )
                if label and label not in seen and label in by_label:
                    ordered.append(by_label[label])
                    seen.add(label)
        if ordered:
            merged["judges"] = ordered
        if not merged.get("primary_judge"):
            merged["primary_judge"] = prior.get("primary_judge")
        if not merged.get("grader_model_id"):
            merged["grader_model_id"] = prior.get("grader_model_id")
        if not merged.get("scoring"):
            merged["scoring"] = prior.get("scoring")
        if not merged.get("graded_at"):
            merged["graded_at"] = prior.get("graded_at")
    else:
        for key in keep:
            if prior.get(key) not in (None, [], ""):
                merged[key] = prior[key]
    dest.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


def _merge_pack_from(src: Path, dest: Path) -> None:
    """Merge predictions + manifest from ``src`` into ``dest``, keeping dest verdicts."""
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in {"models", INGEST_DIR_NAME}:
            continue
        target = dest / item.name
        if item.name == "manifest.json" and item.is_file():
            _merge_manifest_file(item, target)
        elif item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
    local_models = src / "models"
    if local_models.is_dir():
        for model_dir in sorted(local_models.iterdir()):
            if not model_dir.is_dir():
                continue
            dest_model = dest / "models" / model_dir.name
            dest_model.mkdir(parents=True, exist_ok=True)
            for child in model_dir.iterdir():
                if child.name == "predictions.jsonl" and child.is_file():
                    _merge_predictions_jsonl(child, dest_model / child.name)
                elif child.is_file():
                    shutil.copy2(child, dest_model / child.name)
                elif child.is_dir():
                    shutil.copytree(
                        child, dest_model / child.name, dirs_exist_ok=True
                    )


def _sync_local_pack() -> Path:
    """Copy the local pack onto the results volume, keeping existing verdicts."""
    dest = _pack_dir()
    if LOCAL_PACK_MOUNT.is_dir():
        _merge_pack_from(LOCAL_PACK_MOUNT, dest)
        results_volume.commit()
        print(f"[run-judges] synced {LOCAL_PACK_MOUNT} -> {dest}")
    if not dest.is_dir() or not (dest / "manifest.json").is_file():
        raise SystemExit(
            f"Pack not found at {dest}. Expected {LOCAL_PACK_DIR} "
            "(run collate_mmar_freeform.py first)."
        )
    return dest


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"manifest.json not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid manifest.json at {path}: {exc}") from exc


def _assert_freeform_run(manifest: dict) -> str:
    """Return normalized mode or exit if this is an MCQ pack."""
    mode = str(manifest.get("mode") or "").strip().lower()
    scoring = str(manifest.get("scoring") or "").lower()

    if mode in {"mc", "multiple_choice", "multiple-choice", "mcq", "choice"}:
        raise SystemExit(
            f"Pack {PACK_NAME} is an MCQ run (mode={manifest.get('mode')!r}). "
            "run_judges.py only supports freeform packs."
        )
    if mode in {"freeform", "free_form", "free-form", "open"}:
        return "freeform"
    if "freeform" in scoring or "qwen_freeform" in scoring:
        return "freeform"
    if mode:
        raise SystemExit(
            f"Pack {PACK_NAME} has unrecognized mode={manifest.get('mode')!r}; "
            "expected freeform."
        )
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
            f"Pack {PACK_NAME} looks like an MCQ / string-match run "
            f"(judges={judges!r}). run_judges.py only supports freeform packs."
        )
    if manifest.get("grader_model_id") or judges:
        return "freeform"
    raise SystemExit(
        f"Pack {PACK_NAME} has no freeform mode stamp in manifest.json "
        f"(mode={manifest.get('mode')!r}, scoring={manifest.get('scoring')!r})."
    )


def _merge_judge_manifest(
    manifest: dict,
    *,
    model_id: str,
    judge_key: str,
    primary: str,
    make_primary: bool = False,
    update_primary: bool = True,
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
    if update_primary and (make_primary or not existing_primary):
        manifest["primary_judge"] = primary
        primary_entry = next((e for e in ordered if e.get("label") == primary), None)
        manifest["grader_model_id"] = (primary_entry or {}).get("model_id") or model_id
    manifest["scoring"] = manifest.get("scoring") or "qwen_freeform_judge"
    manifest["graded_at"] = datetime.now(timezone.utc).isoformat()
    return manifest


def _csv_parts(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _as_suite_judge(raw: str) -> str | None:
    from grader import ROUND_ROBIN_SUITE, resolve_judge_model_id, _suite_label_for

    for candidate in (raw.strip(), resolve_judge_model_id(raw)):
        if not candidate:
            continue
        if candidate in ROUND_ROBIN_SUITE:
            return candidate
        label = _suite_label_for(candidate)
        if label in ROUND_ROBIN_SUITE:
            return label
    return None


def _select_judges(
    judge_model_id: str,
) -> tuple[list[str], list[dict], list[str], str | None]:
    """Suite labels, dedicated text judges, API labels, and the first requested.

    Empty ``judge_model_id`` means the full vLLM suite (no API judges).
    """
    from grader import ROUND_ROBIN_SUITE, resolve_judge_model_id
    from mmar_api import expand_api_judge_token

    requested = _csv_parts(judge_model_id)
    if not requested:
        suite = list(ROUND_ROBIN_SUITE)
        return suite, [], [], suite[0] if suite else None

    suite: list[str] = []
    dedicated: list[dict] = []
    api: list[str] = []
    seen: set[str] = set()
    first: str | None = None

    def _note(label: str) -> None:
        nonlocal first
        if first is None:
            first = label

    for item in requested:
        api_labels = expand_api_judge_token(item)
        if api_labels is not None:
            for label in api_labels:
                if label not in seen:
                    api.append(label)
                    seen.add(label)
                    _note(label)
            continue
        suite_label = _as_suite_judge(item)
        if suite_label:
            if suite_label not in seen:
                suite.append(suite_label)
                seen.add(suite_label)
                _note(suite_label)
            continue
        model_id = resolve_judge_model_id(item)
        key = judge_label(model_id)
        if not key:
            raise SystemExit(f"Invalid --judge-model-id: {item!r}")
        if key not in seen:
            dedicated.append({"model_id": model_id, "label": key})
            seen.add(key)
            _note(key)
    if not suite and not dedicated and not api:
        raise SystemExit("No judges resolved from --judge-model-id")
    return suite, dedicated, api, first


@app.function(
    image=cpu_image,
    timeout=10 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def prepare_judges(
    judge_model_id: str = "",
    make_primary: bool = False,
    models: str = "all",
) -> dict:
    """Validate the pack is freeform and resolve gradees / judges."""
    volume.reload()
    results_volume.reload()
    pack_dir = _sync_local_pack()

    manifest = _load_manifest(pack_dir)
    mode = _assert_freeform_run(manifest)

    existing_primary = manifest.get("primary_judge")
    if not existing_primary:
        for entry in manifest.get("judges") or []:
            if isinstance(entry, dict) and entry.get("primary") and entry.get("label"):
                existing_primary = entry["label"]
                break
        if not existing_primary and manifest.get("grader_model_id"):
            existing_primary = judge_label(manifest["grader_model_id"])

    labels = discover_model_labels(pack_dir, manifest=manifest)
    local_models = LOCAL_PACK_MOUNT / "models"
    if local_models.is_dir():
        local_labels = {p.name for p in local_models.iterdir() if p.is_dir()}
        labels = [label for label in labels if label in local_labels]
    if models and models.strip().lower() != "all":
        requested = _csv_parts(models)
        missing = [label for label in requested if label not in labels]
        if missing:
            raise SystemExit(
                f"Requested model(s) not found under {pack_dir / 'models'}: {missing}. "
                f"Available: {labels}"
            )
        labels = requested
    if not labels:
        raise SystemExit(f"No model predictions found under {pack_dir / 'models'}")

    suite_judges, dedicated_judges, api_judges, first_new = _select_judges(
        judge_model_id
    )
    if make_primary:
        primary = first_new
    else:
        primary = existing_primary or first_new

    return {
        "pack": PACK_NAME,
        "mode": mode,
        "suite_judges": suite_judges,
        "dedicated_judges": dedicated_judges,
        "api_judges": api_judges,
        "first_new": first_new,
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
    judge_model_id: str,
    primary_judge: str,
    model_labels: list[str],
    batch_size: int | None = None,
    force: bool = False,
    grade_prompt: str = "permissive",
    include_gold: bool = True,
    make_primary: bool = False,
    n_questions: int | None = None,
) -> dict:
    """Grade with one dedicated text judge; merge into predictions + manifest."""
    from grader import (
        grade_predictions_file,
        load_grader,
        parse_grade_prompt_list,
        resolve_grade_judge_key,
        resolve_judge_batch_size,
        resolve_judge_model_id,
    )

    volume.reload()
    results_volume.reload()

    pack_dir = _pack_dir()
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")

    manifest = _load_manifest(pack_dir)
    _assert_freeform_run(manifest)

    judge_model_id = resolve_judge_model_id(judge_model_id)
    prompts = parse_grade_prompt_list(grade_prompt)
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
            predictions_path = pack_dir / "models" / label / "predictions.jsonl"
            print(
                f"[run-judges] {label} with {key} "
                f"(primary={primary_judge}, batch_size={effective_batch_size}"
                f"{f', n_questions={n_questions}' if n_questions is not None else ''}) "
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
                make_primary=make_primary,
                n_questions=n_questions,
            )
            results_volume.commit()
            print(f"[run-judges] {label}:", per_model[f"{label}/{key}"])
        manifest = _merge_judge_manifest(
            manifest,
            model_id=judge_model_id,
            judge_key=key,
            primary=primary_judge,
            make_primary=make_primary,
            prompt=prompt_name,
            include_gold=include_gold,
        )
    write_json(pack_dir / "manifest.json", manifest)
    results_volume.commit()
    return {
        "status": "ok",
        "pack": PACK_NAME,
        "judge_model_id": judge_model_id,
        "judge_label": last_key,
        "primary_judge": manifest.get("primary_judge"),
        "by_model": per_model,
        "judges": manifest.get("judges"),
        "prompt": last_prompt,
        "include_gold": include_gold,
        "n_questions": n_questions,
    }


def _grade_suite_judge(
    judge_label: str,
    *,
    model_labels: list[str],
    grade_prompt: str,
    include_gold: bool,
    force: bool,
    batch_size: int | None,
    n_questions: int | None = None,
) -> dict:
    from grader import (
        compose_judge_key,
        grade_predictions_file,
        load_grader,
        parse_grade_prompt_list,
    )

    volume.reload()
    results_volume.reload()
    pack_dir = _pack_dir()
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")
    _assert_freeform_run(_load_manifest(pack_dir))

    prompts = parse_grade_prompt_list(grade_prompt)
    handle = load_grader(judge_label)
    model_id = handle.get("model_id") or judge_label
    per_model: dict[str, dict] = {}
    keys: list[str] = []
    gradees = [label for label in model_labels if label != judge_label]
    for gradee in gradees:
        predictions_path = pack_dir / "models" / gradee / "predictions.jsonl"
        for prompt_name in prompts:
            key = compose_judge_key(
                judge_label, prompt=prompt_name, include_gold=include_gold
            )
            if key not in keys:
                keys.append(key)
            sidecar = (
                pack_dir / "models" / gradee / "judge_partials" / f"{key}.jsonl"
            )
            print(
                f"[run-judges-rr] {judge_label} -> {gradee} key={key} "
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
                sidecar_path=sidecar,
                n_questions=n_questions,
            )
            results_volume.commit()
            print(f"[run-judges-rr] {gradee}/{key}:", per_model[f"{gradee}/{key}"])
    return {
        "status": "ok",
        "judge_label": judge_label,
        "model_id": model_id,
        "judge_keys": keys,
        "gradees": gradees,
        "by_model": per_model,
        "include_gold": include_gold,
        "prompts": prompts,
        "n_questions": n_questions,
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
    model_labels: list[str],
    grade_prompt: str = "permissive",
    include_gold: bool = True,
    force: bool = False,
    batch_size: int | None = None,
    n_questions: int | None = None,
) -> dict:
    return _grade_suite_judge(
        judge_label,
        model_labels=model_labels,
        grade_prompt=grade_prompt,
        include_gold=include_gold,
        force=force,
        batch_size=batch_size,
        n_questions=n_questions,
    )


@app.function(gpu="A100-80GB", **_SUITE_GRADE_KW)
def grade_suite_a100(
    judge_label: str,
    model_labels: list[str],
    grade_prompt: str = "permissive",
    include_gold: bool = True,
    force: bool = False,
    batch_size: int | None = None,
    n_questions: int | None = None,
) -> dict:
    return _grade_suite_judge(
        judge_label,
        model_labels=model_labels,
        grade_prompt=grade_prompt,
        include_gold=include_gold,
        force=force,
        batch_size=batch_size,
        n_questions=n_questions,
    )


@app.function(gpu="H100", **_SUITE_GRADE_KW)
def grade_suite_h100(
    judge_label: str,
    model_labels: list[str],
    grade_prompt: str = "permissive",
    include_gold: bool = True,
    force: bool = False,
    batch_size: int | None = None,
    n_questions: int | None = None,
) -> dict:
    return _grade_suite_judge(
        judge_label,
        model_labels=model_labels,
        grade_prompt=grade_prompt,
        include_gold=include_gold,
        force=force,
        batch_size=batch_size,
        n_questions=n_questions,
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
    model_labels: list[str],
    judge_entries: list[dict],
    make_primary: bool = False,
    primary_judge: str | None = None,
) -> dict:
    """Fold judge sidecars into predictions.jsonl and append manifest entries."""
    from grader import apply_judge_partials

    results_volume.reload()
    pack_dir = _pack_dir()
    manifest = _load_manifest(pack_dir)
    _assert_freeform_run(manifest)

    by_model: dict[str, dict] = {}
    for label in model_labels:
        pred = pack_dir / "models" / label / "predictions.jsonl"
        partials_dir = pack_dir / "models" / label / "judge_partials"
        paths = sorted(partials_dir.glob("*.jsonl")) if partials_dir.is_dir() else []
        by_model[label] = apply_judge_partials(
            pred,
            paths,
            make_primary=make_primary,
            primary_judge=primary_judge,
        )
        print(f"[run-judges-rr] merged {label}:", by_model[label])

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
    write_json(pack_dir / "manifest.json", manifest)
    results_volume.commit()
    return {
        "status": "ok",
        "pack": PACK_NAME,
        "by_model": by_model,
        "judges": manifest.get("judges"),
        "primary_judge": manifest.get("primary_judge"),
    }


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def ingest_local_pack() -> dict:
    """Merge a local-entrypoint upload of API verdicts into the volume pack."""
    results_volume.reload()
    dest = _pack_dir()
    src = dest / INGEST_DIR_NAME
    if not src.is_dir():
        return {"status": "missing", "pack": PACK_NAME}
    _merge_pack_from(src, dest)
    shutil.rmtree(src, ignore_errors=True)
    results_volume.commit()
    print(f"[run-judges] ingested local pack -> {dest}")
    return {"status": "ok", "pack": PACK_NAME}


def _aggregate_pack(pack_dir: Path) -> dict:
    result = aggregate_difficulty(pack_dir)
    manifest = _load_manifest(pack_dir)
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
    write_json(pack_dir / "scores.json", scores)
    result["scores"] = scores
    print("Aggregated:", scores)
    return result


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def run_aggregate() -> dict:
    results_volume.reload()
    pack_dir = _pack_dir()
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")
    result = _aggregate_pack(pack_dir)
    results_volume.commit()
    return result


def _download_pack() -> None:
    from download_results import download_results

    saved = download_results(remote_path=PACK_NAME, local_dir=REPO_ROOT / "outputs")
    print(f"[run-judges] downloaded pack -> {saved}")


def _require_local_pack() -> Path:
    if not LOCAL_PACK_DIR.is_dir() or not (LOCAL_PACK_DIR / "manifest.json").is_file():
        raise SystemExit(
            f"Pack not found at {LOCAL_PACK_DIR}. "
            "Run collate_mmar_freeform.py first."
        )
    return LOCAL_PACK_DIR


def _local_model_labels(models: str) -> list[str]:
    pack_dir = _require_local_pack()
    manifest = _load_manifest(pack_dir)
    _assert_freeform_run(manifest)
    labels = discover_model_labels(pack_dir, manifest=manifest)
    if models and models.strip().lower() != "all":
        requested = _csv_parts(models)
        missing = [label for label in requested if label not in labels]
        if missing:
            raise SystemExit(
                f"Requested model(s) not found under {pack_dir / 'models'}: {missing}. "
                f"Available: {labels}"
            )
        labels = requested
    if not labels:
        raise SystemExit(f"No model predictions found under {pack_dir / 'models'}")
    return labels


def _upload_local_ingest(pack_dir: Path) -> None:
    """Upload current local predictions/manifest for ingest after GPU workers finish.

    ``add_local_dir`` is snapshotted when the first remote function runs, so API
    verdicts written later must be pushed onto the volume separately.
    """
    prefix = f"/{PACK_NAME}/{INGEST_DIR_NAME}"
    uploads: list[tuple[str, str]] = []
    manifest = pack_dir / "manifest.json"
    if manifest.is_file():
        uploads.append((str(manifest), f"{prefix}/manifest.json"))
    models = pack_dir / "models"
    if models.is_dir():
        for model_dir in sorted(models.iterdir()):
            pred = model_dir / "predictions.jsonl"
            if model_dir.is_dir() and pred.is_file():
                uploads.append(
                    (str(pred), f"{prefix}/models/{model_dir.name}/predictions.jsonl")
                )
    if not uploads:
        return
    with results_volume.batch_upload(force=True) as batch:
        for local_path, remote_path in uploads:
            batch.put_file(local_path, remote_path)
    print(f"[run-judges] uploaded {len(uploads)} files -> {prefix}")


def _merge_api_manifest(
    pack_dir: Path,
    api_results: list[dict],
    *,
    make_primary: bool,
    primary_judge: str | None,
    update_primary: bool,
) -> dict:
    manifest = _load_manifest(pack_dir)
    _assert_freeform_run(manifest)
    for index, result in enumerate(api_results):
        manifest = _merge_judge_manifest(
            manifest,
            model_id=str(result.get("model_id") or ""),
            judge_key=str(result.get("judge_key") or ""),
            primary=primary_judge or manifest.get("primary_judge") or "",
            make_primary=bool(make_primary) and index == 0,
            update_primary=update_primary and index == 0,
            prompt=result.get("prompt"),
            include_gold=result.get("include_gold"),
        )
    write_json(pack_dir / "manifest.json", manifest)
    return manifest


def _spawn_remote_judges(
    *,
    suite: list[str],
    dedicated: list[dict],
    gradees: list[str],
    prompts: list[str],
    grade_prompt: str,
    include_gold: bool,
    force: bool,
    batch_size: int | None,
    n_questions: int | None,
    primary_judge: str,
    dedicated_make_primary: bool,
) -> tuple[list[tuple[str, object]], list[tuple[dict, object]]]:
    suite_handles: list[tuple[str, object]] = []
    for label in suite:
        fn = _SUITE_GRADE_FNS.get(label)
        if fn is None:
            raise SystemExit(f"No GPU worker for suite judge {label!r}")
        print(f"[run-judges] spawning suite judge {label}")
        suite_handles.append(
            (
                label,
                fn.spawn(
                    label,
                    model_labels=gradees,
                    grade_prompt=",".join(prompts),
                    include_gold=include_gold,
                    force=force,
                    batch_size=batch_size,
                    n_questions=n_questions,
                ),
            )
        )

    dedicated_handles: list[tuple[dict, object]] = []
    for i, entry in enumerate(dedicated):
        this_primary = bool(dedicated_make_primary) and i == 0
        print(f"[run-judges] spawning dedicated judge {entry['label']}")
        dedicated_handles.append(
            (
                entry,
                grade_with_judge.spawn(
                    judge_model_id=entry["model_id"],
                    primary_judge=primary_judge,
                    model_labels=gradees,
                    batch_size=batch_size,
                    force=force,
                    grade_prompt=grade_prompt,
                    include_gold=include_gold,
                    make_primary=this_primary,
                    n_questions=n_questions,
                ),
            )
        )
    return suite_handles, dedicated_handles


def _collect_remote_judges(
    suite_handles: list[tuple[str, object]],
    dedicated_handles: list[tuple[dict, object]],
    prompts: list[str],
    include_gold: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    from grader import compose_judge_key

    grade_results: list[dict] = []
    judge_entries: list[dict] = []
    for label, handle in suite_handles:
        result = handle.get()
        grade_results.append(result)
        print(f"Judge {label}:", result)
        model_id = result.get("model_id") or label
        for prompt_name in result.get("prompts") or prompts:
            judge_entries.append(
                {
                    "judge_key": compose_judge_key(
                        label,
                        prompt=prompt_name,
                        include_gold=include_gold,
                    ),
                    "model_id": model_id,
                    "prompt": prompt_name,
                    "include_gold": include_gold,
                }
            )

    dedicated_grades: list[dict] = []
    for _entry, handle in dedicated_handles:
        grade = handle.get()
        print("Graded:", grade)
        dedicated_grades.append(grade)
    return grade_results, dedicated_grades, judge_entries


def _finish_remote_pack(
    *,
    suite: list[str],
    gradees: list[str],
    judge_entries: list[dict],
    suite_make_primary: bool,
    primary_judge: str | None,
    skip_aggregate: bool,
    ingest: bool,
) -> tuple[dict | None, dict | None, dict | None]:
    ingest_result = None
    if ingest:
        ingest_result = ingest_local_pack.remote()
        print("Ingested local API pack:", ingest_result)

    merge = None
    if suite:
        merge = merge_round_robin.remote(
            model_labels=gradees,
            judge_entries=judge_entries,
            make_primary=suite_make_primary,
            primary_judge=primary_judge,
        )
        print("Merged:", merge)

    agg = None
    if not skip_aggregate:
        agg = run_aggregate.remote()
        print("Aggregated:", agg)
    return ingest_result, merge, agg


@app.function(
    image=cpu_image,
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, RESULTS_MOUNT: results_volume},
)
def run_judges_pipeline(
    judge_model_id: str = "",
    models: str = "all",
    make_primary: bool = False,
    force: bool = False,
    batch_size: int | None = None,
    skip_aggregate: bool = False,
    grade_prompt: str = "permissive",
    include_gold: bool = True,
    n_questions: int | None = None,
    ingest_local: bool = False,
) -> dict:
    """Remote orchestrator so a detached App stays alive across GPU phases."""
    from grader import parse_grade_prompt_list

    if n_questions is not None and int(n_questions) < 0:
        n_questions = None
    prompts = parse_grade_prompt_list(grade_prompt)
    prep = prepare_judges.remote(
        judge_model_id=(judge_model_id or "").strip(),
        make_primary=make_primary,
        models=models,
    )
    suite = list(prep.get("suite_judges") or [])
    dedicated = list(prep.get("dedicated_judges") or [])
    gradees = list(prep["model_labels"])
    first_new = prep.get("first_new")
    print(
        f"[run-judges] pipeline prep primary={prep['primary_judge']} "
        f"(existing_primary={prep['existing_primary']}) "
        f"existing_judges={prep['existing_judges']}"
    )
    suite_handles, dedicated_handles = _spawn_remote_judges(
        suite=suite,
        dedicated=dedicated,
        gradees=gradees,
        prompts=prompts,
        grade_prompt=grade_prompt,
        include_gold=include_gold,
        force=force,
        batch_size=batch_size,
        n_questions=n_questions,
        primary_judge=prep["primary_judge"],
        dedicated_make_primary=bool(make_primary)
        and any(first_new == entry["label"] for entry in dedicated),
    )
    n_remote = len(suite_handles) + len(dedicated_handles)
    if n_remote:
        print(f"[run-judges] {n_remote} remote judge(s) running")
    grade_results, dedicated_grades, judge_entries = _collect_remote_judges(
        suite_handles, dedicated_handles, prompts, include_gold
    )
    _ingest, merge, agg = _finish_remote_pack(
        suite=suite,
        gradees=gradees,
        judge_entries=judge_entries,
        suite_make_primary=bool(make_primary) and first_new in suite,
        primary_judge=prep.get("primary_judge"),
        skip_aggregate=skip_aggregate,
        ingest=ingest_local,
    )
    return {
        "prepare": prep,
        "grade": grade_results,
        "dedicated": dedicated_grades,
        "merge": merge,
        "aggregate": agg,
    }


def _run_judges(
    *,
    judge_model_id: str = "",
    models: str = "all",
    make_primary: bool = False,
    force: bool = False,
    batch_size: int | None = None,
    skip_aggregate: bool = False,
    grade_prompt: str = "permissive",
    include_gold: bool = True,
    n_questions: int | None = None,
    qps: float = 4.0,
    max_workers: int = 8,
    timeout: float = 180.0,
    retries: int = 20,
    retry_interval: float = 1.0,
) -> dict:
    from grader import compose_judge_key, parse_grade_prompt_list, require_audio_nongold_judge
    from mmar_api import grade_pack_with_api_judges

    if n_questions is not None and int(n_questions) < 0:
        n_questions = None

    suite, dedicated, api, first_new = _select_judges(judge_model_id)
    if not include_gold:
        if dedicated:
            for entry in dedicated:
                require_audio_nongold_judge(entry["model_id"], include_gold=False)
        elif not suite and not api:
            raise SystemExit("--no-include-gold needs an audio-capable judge")

    needs_modal = bool(suite or dedicated)
    prompts = parse_grade_prompt_list(grade_prompt)
    pack_dir: Path | None = None
    gradees: list[str] = []
    existing_primary = None
    mode = "freeform"
    if api or not needs_modal:
        pack_dir = _require_local_pack()
        manifest = _load_manifest(pack_dir)
        mode = _assert_freeform_run(manifest)
        gradees = _local_model_labels(models)
        existing_primary = manifest.get("primary_judge")
    if make_primary:
        primary = first_new
    else:
        primary = existing_primary or first_new

    api_make_primary = bool(make_primary) and first_new in api

    print(
        f"[run-judges] pack={PACK_NAME} mode={mode} "
        f"suite_judges={suite} dedicated_judges={[d['label'] for d in dedicated]} "
        f"api_judges={api} gradees={gradees or '(modal)'} primary={primary} "
        f"(existing_primary={existing_primary}) "
        f"prompt={grade_prompt} include_gold={include_gold} "
        f"n_questions={n_questions} force={force}"
    )

    api_results: list[dict] = []

    def _run_api() -> None:
        nonlocal api_results
        if not api:
            return
        if pack_dir is None:
            raise SystemExit(
                "API judges require a local pack at outputs/mmar-freeform-5-shot"
            )
        first_api_key = compose_judge_key(
            api[0], prompt=prompts[0], include_gold=include_gold
        )
        set_primary = api_make_primary or (
            not existing_primary and not suite and not dedicated
        )
        api_results = asyncio.run(
            grade_pack_with_api_judges(
                pack_dir,
                labels=api,
                model_labels=gradees,
                prompts=prompts,
                include_gold=include_gold,
                force=force,
                n_questions=n_questions,
                make_primary=set_primary,
                primary_judge=first_api_key if set_primary else primary,
                qps=qps,
                max_workers=max_workers,
                timeout=timeout,
                retries=retries,
                retry_interval=retry_interval,
            )
        )
        first_key = str(api_results[0].get("judge_key") or first_api_key)
        _merge_api_manifest(
            pack_dir,
            api_results,
            make_primary=set_primary,
            primary_judge=first_key or primary,
            update_primary=set_primary,
        )

    if not needs_modal:
        _run_api()
        if pack_dir is None:
            raise SystemExit(f"Pack not found at {LOCAL_PACK_DIR}")
        agg = None if skip_aggregate else _aggregate_pack(pack_dir)
        return {
            "prepare": None,
            "grade": [],
            "dedicated": [],
            "api": api_results,
            "merge": None,
            "aggregate": agg,
        }

    run_judges_pipeline.spawn(
        judge_model_id=(judge_model_id or "").strip(),
        models=models,
        make_primary=make_primary,
        force=force,
        batch_size=batch_size,
        skip_aggregate=skip_aggregate,
        grade_prompt=grade_prompt,
        include_gold=include_gold,
        n_questions=n_questions,
        ingest_local=bool(api),
    )
    dashboard = app.get_dashboard_url()
    if dashboard:
        print(f"[run-judges] Modal GPU pipeline started (detached): {dashboard}")
    else:
        print("[run-judges] Modal GPU pipeline started (detached)")

    _run_api()
    if api and pack_dir is not None:
        _upload_local_ingest(pack_dir)
        ingest = ingest_local_pack.remote()
        print("Ingested local API pack:", ingest)
        if not skip_aggregate:
            agg = run_aggregate.remote()
            print("Aggregated:", agg)

    return {
        "detached": True,
        "dashboard": dashboard,
        "prepare": None,
        "grade": [],
        "dedicated": [],
        "api": api_results,
        "merge": None,
        "aggregate": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-model-id",
        default="",
        help="Comma-separated judges, or 'api' for both API judges",
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--make-primary", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Shots per vLLM grader generate() call (default: per-judge spec)",
    )
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--grade-prompt", default="permissive")
    parser.add_argument(
        "--include-gold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--n-questions", type=int, default=None)
    parser.add_argument("--qps", type=float, default=4.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-interval", type=float, default=1.0)
    return parser.parse_args()


def _run_judges_from_args(args: argparse.Namespace) -> dict:
    return _run_judges(
        judge_model_id=args.judge_model_id,
        models=args.models,
        make_primary=args.make_primary,
        force=args.force,
        batch_size=args.batch_size,
        skip_aggregate=args.skip_aggregate,
        grade_prompt=args.grade_prompt,
        include_gold=args.include_gold,
        n_questions=args.n_questions,
        qps=args.qps,
        max_workers=args.max_workers,
        timeout=args.timeout,
        retries=args.retries,
        retry_interval=args.retry_interval,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    suite, dedicated, _api, _first = _select_judges(args.judge_model_id)
    if suite or dedicated:
        with app.run(detach=True):
            _run_judges_from_args(args)
    else:
        _run_judges_from_args(args)


