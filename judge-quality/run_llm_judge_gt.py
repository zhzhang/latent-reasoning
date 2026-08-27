"""One-off: LLM-as-judge with gold, 5 shots, all 1000 MMAR questions.

Grades ``shot_index`` 0–4 for every test-taker model that has a freeform
generation on the full MMAR set. Judge is ``qwen3.6-35b-a3b-fp8`` with the
``neutral_with_gt_no_audio`` recipe. Each answer is sampled 3 times
(``SamplingParams(n=3)`` shared prefill) and majority-voted.

Writes ``llm-judge-gt/`` on the ``mmar-judging`` volume. Uses one H100 so
the FP8 checkpoint has native Hopper support. Engine knobs match
``grader.JUDGE_SPECS`` (CUDA graphs, prefix caching, language_model_only,
batch 512 concurrent sequences).

Generations come only from ``mmar-freeform-thinking`` (volume root, then
the local ``outputs/mmar-freeform-thinking`` download). Question ids come
from that pack's ``question_ids.json``. Other experiment dirs are ignored.

Resume is the default: existing per-shot verdicts for this judge are
kept, the H100 is not started when nothing is left, and only ungraded
generations are sent to the model. Pass ``--force`` to replace them.

Usage::

    uv run modal run --detach run_llm_judge_gt.py
    uv run modal run --detach run_llm_judge_gt.py --force

Claude Sonnet 5 (Anthropic Messages API, no Modal container)::

    uv run python judge-quality/run_llm_judge_gt_claude.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from aggregate import aggregate_difficulty, order_model_labels
from mmar_common import recompute_multi_judge_scores, write_json, write_jsonl
from modal_cache import (
    JUDGING_MOUNT,
    LOCAL_MMAR_FREEFORM_THINKING_MOUNT,
    MMAR_FREEFORM_THINKING_MOUNT,
    VOLUME_MOUNT,
    VLLM_WHEEL_INDEX,
    hf_secret,
    judging_volume,
    mmar_freeform_thinking_volume,
    volume,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_NAME = "llm-judge-gt"
REMOTE_PACK_DIR = JUDGING_MOUNT / PACK_NAME
LOCAL_FREEFORM_THINKING_DIR = _REPO_ROOT / "outputs" / "mmar-freeform-thinking"

JUDGE_MODEL_ID = "qwen3.6-35b-a3b-fp8"
GRADE_PROMPT = "neutral_with_gt_no_audio"
N_SHOTS = 5
SHOT_INDICES = tuple(range(N_SHOTS))
N_SAMPLES = 3
# Qwen3.6 thinking-mode default. T=0 would make the 3 shots identical.
JUDGE_TEMPERATURE = 1.0

DROP_RECORD_KEYS = (
    "judges",
    "grader",
    "grader_output",
    "per_judge",
    "primary_judge",
    "pending_grade",
    "scoring",
    "correct",
    "n_shot_correct",
    "shot_success_rate",
)
DROP_SHOT_KEYS = (
    "judges",
    "grader",
    "grader_output",
    "pending_grade",
    "correct",
)

app = modal.App("run-llm-judge-gt")


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


# Same dedicated-text-judge image as run_judges.py / tune_judge.py.
grader_image = _mount_sources(
    _cuda_base_image()
    .uv_pip_install(
        "vllm==0.28.0",
        "transformers>=5.5.3",
        "huggingface-hub>=0.30.0",
        "librosa>=0.11.0",
        "soundfile",
        "numpy",
        "tqdm>=4.67.0",
        "accelerate>=1.14.0",
        extra_index_url=VLLM_WHEEL_INDEX,
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
if LOCAL_FREEFORM_THINKING_DIR.is_dir():
    cpu_image = cpu_image.add_local_dir(
        str(LOCAL_FREEFORM_THINKING_DIR),
        remote_path=str(LOCAL_MMAR_FREEFORM_THINKING_MOUNT),
    )


def _shot_index(shot: dict[str, Any]) -> int:
    raw = shot.get("shot_index")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _compact_shot(shot: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in shot.items() if key not in DROP_SHOT_KEYS}
    out["shot_index"] = _shot_index(shot)
    out["correct"] = None
    out["pending_grade"] = True
    return out


def _shots_record(record: dict[str, Any], *, model: str) -> dict[str, Any] | None:
    raw_shots = list(record.get("shots") or [])
    raw_shots.sort(key=_shot_index)
    kept: list[dict[str, Any]] = []
    seen: set[int] = set()
    for shot in raw_shots:
        idx = _shot_index(shot)
        if idx not in SHOT_INDICES or idx in seen:
            continue
        seen.add(idx)
        kept.append(_compact_shot(shot))
    if not kept:
        return None
    first = next((shot for shot in kept if _shot_index(shot) == 0), kept[0])
    out = {key: value for key, value in record.items() if key not in DROP_RECORD_KEYS}
    out["id"] = str(record.get("id") or "")
    out["model"] = model
    out["n_shots"] = len(kept)
    out["shots"] = kept
    out["answer_prediction"] = first.get("answer_prediction")
    out["model_output"] = first.get("model_output")
    out["thinking_prediction"] = first.get("thinking_prediction")
    out["raw_tokens"] = first.get("raw_tokens")
    out["correct"] = None
    out["n_shot_correct"] = None
    out["shot_success_rate"] = None
    out["pending_grade"] = True
    return out


def _load_question_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        ids = payload.get("ids") or []
    elif isinstance(payload, list):
        ids = payload
    else:
        return []
    return [str(qid) for qid in ids if str(qid).strip()]


def _iter_shots(predictions_path: Path, *, model: str) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    if not predictions_path.is_file():
        return found
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            compact = _shots_record(record, model=model)
            if compact is None or not compact["id"]:
                continue
            found[compact["id"]] = compact
    return found


def _source_model_dirs() -> list[tuple[str, Path]]:
    """Only ``mmar-freeform-thinking`` (remote volume, then local download)."""
    return [
        ("mmar-freeform-thinking", MMAR_FREEFORM_THINKING_MOUNT / "models"),
        (
            "local-mmar-freeform-thinking",
            LOCAL_MMAR_FREEFORM_THINKING_MOUNT / "models",
        ),
    ]


def _question_id_candidates() -> list[tuple[str, Path]]:
    return [
        ("mmar-freeform-thinking", MMAR_FREEFORM_THINKING_MOUNT / "question_ids.json"),
        (
            "local-mmar-freeform-thinking",
            LOCAL_MMAR_FREEFORM_THINKING_MOUNT / "question_ids.json",
        ),
    ]


def _resolve_question_ids() -> tuple[list[str], str]:
    for tag, path in _question_id_candidates():
        ids = _load_question_ids(path)
        if ids:
            print(f"[llm-judge-gt] question ids: {len(ids)} from {tag} ({path})")
            return ids, tag
    raise SystemExit(
        "No question_ids.json found on mmar-freeform-thinking "
        f"(remote {MMAR_FREEFORM_THINKING_MOUNT} or local download)"
    )


def _merge_shot_judges(local: dict, prior: dict) -> dict:
    """Keep prior per-shot verdicts when that shot's answer is unchanged."""
    prior_shots = {
        _shot_index(shot): shot for shot in (prior.get("shots") or [])
    }
    merged_shots = []
    for shot in local.get("shots") or []:
        prev = prior_shots.get(_shot_index(shot))
        prev_judges = (prev or {}).get("judges") or {}
        pred = str(shot.get("answer_prediction") or "").strip().lower()
        prev_pred = str((prev or {}).get("answer_prediction") or "").strip().lower()
        if prev is not None and prev_judges and pred == prev_pred:
            shot = dict(shot)
            judges = dict(shot.get("judges") or {})
            for key, entry in prev_judges.items():
                judges.setdefault(key, entry)
            shot["judges"] = judges
        merged_shots.append(shot)
    merged = dict(local)
    merged["shots"] = merged_shots
    if prior.get("primary_judge") and not merged.get("primary_judge"):
        merged["primary_judge"] = prior["primary_judge"]
    prior_labels = [str(x) for x in (prior.get("judges") or []) if x]
    if prior_labels:
        merged["judges"] = list(prior_labels)
    return merged


def _has_shot_judges(record: dict[str, Any]) -> bool:
    if record.get("judges"):
        return True
    return any((shot.get("judges") or {}) for shot in (record.get("shots") or []))


def _write_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    prior_by_id: dict[str, dict] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                rid = str(record.get("id") or "")
                if rid:
                    prior_by_id[rid] = record
    merged = []
    for record in records:
        prior = prior_by_id.get(str(record.get("id") or ""))
        row = _merge_shot_judges(record, prior) if prior else record
        if _has_shot_judges(row):
            recompute_multi_judge_scores(row, row.get("primary_judge"))
        merged.append(row)
    write_jsonl(path, merged, mode="w")


def _stamp_manifest(
    manifest: dict,
    *,
    model_id: str,
    judge_key: str,
    prompt: str,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "label": judge_key,
        "model_id": model_id,
        "primary": True,
        "prompt": prompt,
        "include_gold": True,
        "n_samples": N_SAMPLES,
        "temperature": JUDGE_TEMPERATURE,
    }
    by_label: dict[str, dict] = {}
    for item in manifest.get("judges") or []:
        if isinstance(item, dict) and item.get("label"):
            by_label[str(item["label"])] = dict(item)
        elif isinstance(item, str) and item:
            by_label.setdefault(item, {"label": item})
    by_label[judge_key] = {**by_label.get(judge_key, {}), **entry}
    ordered = [by_label[judge_key]]
    for label, item in by_label.items():
        if label != judge_key:
            item["primary"] = False
            ordered.append(item)
    manifest["judges"] = ordered
    manifest["primary_judge"] = judge_key
    manifest["grader_model_id"] = model_id
    manifest["scoring"] = "qwen_freeform_judge"
    manifest["graded_at"] = now
    manifest["updated_at"] = now
    return manifest


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={
        MMAR_FREEFORM_THINKING_MOUNT: mmar_freeform_thinking_volume,
        JUDGING_MOUNT: judging_volume,
    },
)
def prepare_pack(models: str = "all") -> dict:
    """Copy 5-shot freeform answers onto ``mmar-judging/llm-judge-gt``."""
    mmar_freeform_thinking_volume.reload()
    judging_volume.reload()

    question_ids, ids_source = _resolve_question_ids()
    if not question_ids:
        raise SystemExit("No MMAR question ids found")
    wanted = set(question_ids)

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for tag, models_dir in _source_model_dirs():
        if not models_dir.is_dir():
            print(f"[llm-judge-gt] skip missing source {tag}: {models_dir}")
            continue
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            label = model_dir.name
            if label in by_model:
                continue
            pred = model_dir / "predictions.jsonl"
            rows = _iter_shots(pred, model=label)
            kept = {qid: rows[qid] for qid in question_ids if qid in rows}
            if not kept:
                print(f"[llm-judge-gt] {tag}/{label}: no shot overlap")
                continue
            by_model[label] = kept
            sources[label] = tag
            n_shot_rows = sum(len(row.get("shots") or []) for row in kept.values())
            print(
                f"[llm-judge-gt] {tag}/{label}: {len(kept)}/{len(question_ids)} "
                f"questions, {n_shot_rows} shots"
            )

    requested = [part.strip() for part in str(models or "all").split(",") if part.strip()]
    if requested and requested != ["all"]:
        missing = [label for label in requested if label not in by_model]
        if missing:
            raise SystemExit(
                f"Requested model(s) not found: {missing}. "
                f"Available: {order_model_labels(list(by_model))}"
            )
        by_model = {label: by_model[label] for label in requested}

    labels = order_model_labels(list(by_model))
    if not labels:
        raise SystemExit("No test-taker generations found")

    pack_dir = REMOTE_PACK_DIR
    pack_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    for label in labels:
        ordered = [by_model[label][qid] for qid in question_ids if qid in by_model[label]]
        _write_predictions(pack_dir / "models" / label / "predictions.jsonl", ordered)

    write_json(
        pack_dir / "question_ids.json",
        {
            "n": len(question_ids),
            "ids": question_ids,
            "n_shots": N_SHOTS,
            "source": ids_source,
        },
    )
    write_json(
        pack_dir / "manifest.json",
        {
            "name": PACK_NAME,
            "mode": "freeform",
            "n_shots": N_SHOTS,
            "n_questions": len(question_ids),
            "models": labels,
            "sources": sources,
            "judge_model_id": JUDGE_MODEL_ID,
            "grade_prompt": GRADE_PROMPT,
            "n_samples": N_SAMPLES,
            "temperature": JUDGE_TEMPERATURE,
            "created_at": now,
            "updated_at": now,
        },
    )
    judging_volume.commit()
    print(
        f"[llm-judge-gt] prepared models={labels} questions={len(question_ids)} "
        f"wanted={len(wanted)} -> {pack_dir}"
    )
    return {
        "pack": PACK_NAME,
        "model_labels": labels,
        "question_ids": question_ids,
        "sources": sources,
        "n_questions": len(question_ids),
    }


@app.function(
    image=grader_image,
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, JUDGING_MOUNT: judging_volume},
    secrets=[hf_secret],
    memory=32768,
)
def grade_pack(
    model_labels: list[str],
    question_ids: list[str],
    force: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Grade all 5 shots with 3-sample majority vote on one H100."""
    from grader import (
        compose_judge_key,
        grade_predictions_file,
        load_grader,
        remaining_grade_work,
        resolve_grade_judge_key,
        resolve_judge_batch_size,
        resolve_judge_model_id,
    )

    volume.reload()
    judging_volume.reload()

    pack_dir = REMOTE_PACK_DIR
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")

    judge_key = compose_judge_key(
        JUDGE_MODEL_ID, prompt=GRADE_PROMPT, include_gold=True
    )
    remaining = remaining_grade_work(
        pack_dir,
        model_labels,
        judge_key,
        question_ids=question_ids,
        shot_indices=SHOT_INDICES,
    )
    for label in model_labels:
        n_left = len(question_ids) if force else len(remaining.get(label) or [])
        print(
            f"[llm-judge-gt] {label}: {n_left}/{len(question_ids)} "
            f"questions still need {judge_key}"
        )
    if not force:
        done = [label for label in model_labels if label not in remaining]
        if done:
            print(f"[llm-judge-gt] already graded: {done}")
        model_labels = [label for label in model_labels if label in remaining]
        if not model_labels:
            print("[llm-judge-gt] skip load: nothing left to grade")
            return {
                "status": "skipped",
                "pack": PACK_NAME,
                "judge_model_id": JUDGE_MODEL_ID,
                "judge_key": judge_key,
                "by_model": {},
                "n_samples": N_SAMPLES,
            }

    judge_model_id = resolve_judge_model_id(JUDGE_MODEL_ID)
    handle = load_grader(judge_model_id)
    key = resolve_grade_judge_key(handle, prompt=GRADE_PROMPT, include_gold=True)
    seq_budget = resolve_judge_batch_size(judge_model_id, batch_size)
    # Keep ~512 concurrent sequences (tuned JUDGE_SPECS batch) after n=3 fork.
    prompt_batch = max(1, seq_budget // N_SAMPLES)
    print(
        f"[llm-judge-gt] judge={judge_model_id} key={key} "
        f"n_samples={N_SAMPLES} temperature={JUDGE_TEMPERATURE} "
        f"prompt_batch={prompt_batch} (seq_budget={seq_budget})"
    )

    per_model: dict[str, dict] = {}
    for label in model_labels:
        predictions_path = pack_dir / "models" / label / "predictions.jsonl"
        print(f"[llm-judge-gt] grading {label} -> {predictions_path}")
        per_model[label] = grade_predictions_file(
            predictions_path,
            handle,
            judge_key=key,
            primary_judge=key,
            batch_size=prompt_batch,
            force=force,
            prompt=GRADE_PROMPT,
            include_gold=True,
            shot_indices=SHOT_INDICES,
            make_primary=True,
            question_ids=question_ids,
            n_samples=N_SAMPLES,
            temperature=JUDGE_TEMPERATURE,
        )
        judging_volume.commit()
        print(f"[llm-judge-gt] {label}:", per_model[label])

    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _stamp_manifest(
        manifest,
        model_id=judge_model_id,
        judge_key=key,
        prompt=GRADE_PROMPT,
    )
    write_json(manifest_path, manifest)
    judging_volume.commit()
    return {
        "status": "ok",
        "pack": PACK_NAME,
        "judge_model_id": judge_model_id,
        "judge_key": key or compose_judge_key(
            handle.get("judge_label") or judge_model_id,
            prompt=GRADE_PROMPT,
            include_gold=True,
        ),
        "by_model": per_model,
        "n_samples": N_SAMPLES,
        "prompt_batch": prompt_batch,
    }


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={JUDGING_MOUNT: judging_volume},
)
def run_aggregate() -> dict:
    judging_volume.reload()
    pack_dir = REMOTE_PACK_DIR
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    labels = [str(x) for x in (manifest.get("models") or []) if x]
    result = aggregate_difficulty(pack_dir, model_labels=labels or None)
    scores = result.get("scores") or {}
    for key in (
        "scoring",
        "mode",
        "grader_model_id",
        "judges",
        "primary_judge",
        "n_samples",
        "grade_prompt",
    ):
        if manifest.get(key) is not None:
            scores[key] = manifest[key]
    write_json(pack_dir / "scores.json", scores)
    result["scores"] = scores
    judging_volume.commit()
    print("Aggregated:", scores)
    return result


@app.function(
    image=cpu_image,
    timeout=24 * 60 * 60,
    volumes={JUDGING_MOUNT: judging_volume},
)
def run_pipeline(
    models: str = "all",
    force: bool = False,
    batch_size: int | None = None,
    skip_aggregate: bool = False,
) -> dict:
    from grader import compose_judge_key, remaining_grade_work

    prep = prepare_pack.remote(models=models)
    print(
        f"[llm-judge-gt] prepared pack={prep.get('pack')} "
        f"models={prep.get('model_labels')} "
        f"n_questions={prep.get('n_questions')} "
        f"sources={prep.get('sources')}"
    )
    judging_volume.reload()
    judge_key = compose_judge_key(
        JUDGE_MODEL_ID, prompt=GRADE_PROMPT, include_gold=True
    )
    remaining = remaining_grade_work(
        REMOTE_PACK_DIR,
        list(prep["model_labels"]),
        judge_key,
        question_ids=list(prep["question_ids"]),
        shot_indices=SHOT_INDICES,
    )
    if not force:
        for label in prep["model_labels"]:
            n_left = len(remaining.get(label) or [])
            print(
                f"[llm-judge-gt] {label}: {n_left}/{prep['n_questions']} "
                f"ungraded before GPU"
            )
        if not remaining:
            print("[llm-judge-gt] skip GPU: every generation already graded")
            grade = {
                "status": "skipped",
                "pack": PACK_NAME,
                "judge_key": judge_key,
                "by_model": {},
                "n_samples": N_SAMPLES,
            }
        else:
            grade = grade_pack.remote(
                model_labels=list(remaining),
                question_ids=list(prep["question_ids"]),
                force=force,
                batch_size=batch_size,
            )
    else:
        grade = grade_pack.remote(
            model_labels=list(prep["model_labels"]),
            question_ids=list(prep["question_ids"]),
            force=force,
            batch_size=batch_size,
        )
    print("[llm-judge-gt] graded:", grade)
    agg = None if skip_aggregate else run_aggregate.remote()
    if agg is not None:
        print("[llm-judge-gt] aggregated:", agg)
    return {"prepare": prep, "grade": grade, "aggregate": agg}


@app.local_entrypoint()
def main(
    models: str = "all",
    force: bool = False,
    batch_size: int | None = None,
    skip_aggregate: bool = False,
) -> None:
    """Materialize 5-shot answers, grade with 3-sample majority, aggregate.

    Args:
        models: Comma-separated test-taker labels, or ``all``.
        force: Replace existing verdicts for this judge key. Without
            this flag, already-graded generations are skipped.
        batch_size: Concurrent sequence budget (default: per-judge spec 512).
            Prompt batch is ``batch_size // 3``.
        skip_aggregate: Skip difficulty.jsonl / scores.json.
    """
    handle = run_pipeline.spawn(
        models=models,
        force=force,
        batch_size=batch_size,
        skip_aggregate=skip_aggregate,
    )
    dashboard = app.get_dashboard_url()
    if dashboard:
        print(f"[llm-judge-gt] pipeline started (detached): {dashboard}")
    else:
        print("[llm-judge-gt] pipeline started (detached)")
    print(f"[llm-judge-gt] call id: {handle.object_id}")


if __name__ == "__main__":
    print(
        "Run via Modal:\n"
        "  uv run modal run --detach run_llm_judge_gt.py",
        flush=True,
    )
    raise SystemExit(2)
