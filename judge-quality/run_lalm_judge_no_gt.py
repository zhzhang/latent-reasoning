"""One-off: LALM-as-judge, no gold, first shot, all 1000 MMAR questions.

Grades ``shot_index == 0`` for every test-taker that has a freeform
generation on the full MMAR set. Each audio suite judge
(``grader.ROUND_ROBIN_SUITE``) uses the ``neutral_no_gt`` recipe: hears
the clip, does not see gold. Default is one judge completion per
answer (``--n-samples 1``). Pass ``--n-samples 3`` to restore
3-sample majority vote (``SamplingParams(n=...)`` where the backend
supports it).

Writes ``lalm-judge-no-gt/`` on the ``mmar-judging`` volume. Each suite
judge gets its own GPU container (``single_use_containers``), same
spawn pattern as ``run_experiment.py``: a short CPU orchestrator
prepares remaining work, launches pending judges in parallel, and
returns without waiting. Fold sidecars with ``--merge-only`` after
workers finish (or on a re-run with nothing left to grade).

Sources, later runs only add models that are not already present:

    mmar-freeform-thinking                          # run_experiment.py (volume root)
    outputs/mmar-freeform-thinking                  # local download of that volume
    exp-mmar-question-difficulty/20260807T145000Z   # legacy 5 models × 1000 q
    exp-mmar-question-difficulty/20260816T050944Z   # extra models × 784 q
    mmar-freeform                                   # collated pack (API, …)
    mmar-freeform-5-shot-thinking                   # API models (volume root)

Question ids come from ``mmar-freeform-thinking/question_ids.json``
(remote volume, then local download). The 1000-id list from the legacy
full MMAR run is used only if that file is missing. Models that only
exist on the 784-id packs are graded on the overlap.

Resume is the default: existing sidecar / predictions verdicts for a
suite judge are kept, that judge's GPU is not started when nothing is
left, and only ungraded examples are sent to the model. Pass
``--force`` to replace them.

Usage::

    uv run modal run --detach judge-quality/run_lalm_judge_no_gt.py
    uv run modal run --detach judge-quality/run_lalm_judge_no_gt.py --n-samples 3
    uv run modal run --detach judge-quality/run_lalm_judge_no_gt.py --force
    uv run modal run --detach judge-quality/run_lalm_judge_no_gt.py \\
      --judges qwen2.5-omni-7b,phi-4-multimodal
    uv run modal run judge-quality/run_lalm_judge_no_gt.py --merge-only
    uv run modal run judge-quality/run_lalm_judge_no_gt.py --aggregate-only
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import modal

from aggregate import aggregate_difficulty, order_model_labels
from grader import ROUND_ROBIN_SUITE
from mmar_common import write_json, write_jsonl
from modal_cache import (
    FREEFORM_THINKING_MOUNT,
    JUDGING_MOUNT,
    LOCAL_MMAR_FREEFORM_THINKING_MOUNT,
    MMAR_FREEFORM_THINKING_MOUNT,
    RESULTS_MOUNT,
    VOLUME_MOUNT,
    VLLM_WHEEL_INDEX,
    freeform_thinking_volume,
    hf_secret,
    judging_volume,
    mmar_freeform_thinking_volume,
    results_volume,
    volume,
)

PACK_NAME = "lalm-judge-no-gt"
REMOTE_PACK_DIR = JUDGING_MOUNT / PACK_NAME
DEFAULT_OUTPUT_DIR = RESULTS_MOUNT / "exp-mmar-question-difficulty"
FULL_MMAR_RUN_ID = "20260807T145000Z"
OPEN_ENDED_RUN_ID = "20260816T050944Z"
COLLATED_PACK = FREEFORM_THINKING_MOUNT
FREEFORM_PACK = RESULTS_MOUNT / "mmar-freeform"
LOCAL_FREEFORM_DIR = _REPO_ROOT / "outputs" / "mmar-freeform"
LOCAL_FREEFORM_MOUNT = Path("/local-mmar-freeform")
LOCAL_FREEFORM_THINKING_DIR = _REPO_ROOT / "outputs" / "mmar-freeform-thinking"

GRADE_PROMPT = "neutral_no_gt"
INCLUDE_GOLD = False
DEFAULT_N_SAMPLES = 1
# Shared with run_llm_judge_gt.py: T=0 would make n>1 votes identical.
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

app = modal.App("run-lalm-judge-no-gt")


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

# Same fused-MoE stand-in as run_experiment.py / run_judges.py.
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

# Suite LALM judges load via mmar_models (same image/GPU as run_judges.py).
large_mm_image = _mount_sources(
    _cuda_base_image()
    .uv_pip_install(
        "vllm[audio]==0.28.0",
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
        extra_index_url=VLLM_WHEEL_INDEX,
    )
    .run_commands(_FUSED_MOE_CONFIG_CMD)
    .env(_INPROC_VLLM_ENV)
)

cpu_image = _mount_sources(
    modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "numpy", "tqdm>=4.67.0"
    )
)
if LOCAL_FREEFORM_DIR.is_dir():
    cpu_image = cpu_image.add_local_dir(
        str(LOCAL_FREEFORM_DIR), remote_path=str(LOCAL_FREEFORM_MOUNT)
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
    out["shot_index"] = 0
    out["correct"] = None
    out["pending_grade"] = True
    return out


def _first_shot_record(record: dict[str, Any], *, model: str) -> dict[str, Any] | None:
    shots = list(record.get("shots") or [])
    shots.sort(key=_shot_index)
    chosen = next((shot for shot in shots if _shot_index(shot) == 0), None)
    if chosen is None and shots:
        chosen = shots[0]
    if chosen is None:
        return None
    shot = _compact_shot(chosen)
    out = {key: value for key, value in record.items() if key not in DROP_RECORD_KEYS}
    out["id"] = str(record.get("id") or "")
    out["model"] = model
    out["n_shots"] = 1
    out["shots"] = [shot]
    out["answer_prediction"] = shot.get("answer_prediction")
    out["model_output"] = shot.get("model_output")
    out["thinking_prediction"] = shot.get("thinking_prediction")
    out["raw_tokens"] = shot.get("raw_tokens")
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


def _iter_first_shots(predictions_path: Path, *, model: str) -> dict[str, dict[str, Any]]:
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
            compact = _first_shot_record(record, model=model)
            if compact is None or not compact["id"]:
                continue
            found[compact["id"]] = compact
    return found


def _source_model_dirs() -> list[tuple[str, Path]]:
    """(source_tag, models_dir) in overlay order: first source wins per model."""
    return [
        ("mmar-freeform-thinking", MMAR_FREEFORM_THINKING_MOUNT / "models"),
        (
            "local-mmar-freeform-thinking",
            LOCAL_MMAR_FREEFORM_THINKING_MOUNT / "models",
        ),
        (
            FULL_MMAR_RUN_ID,
            DEFAULT_OUTPUT_DIR / FULL_MMAR_RUN_ID / "models",
        ),
        (
            OPEN_ENDED_RUN_ID,
            DEFAULT_OUTPUT_DIR / OPEN_ENDED_RUN_ID / "models",
        ),
        ("mmar-freeform", FREEFORM_PACK / "models"),
        ("mmar-freeform-5-shot-thinking", COLLATED_PACK / "models"),
        ("local-mmar-freeform", LOCAL_FREEFORM_MOUNT / "models"),
    ]


def _question_id_candidates() -> list[tuple[str, Path]]:
    return [
        ("mmar-freeform-thinking", MMAR_FREEFORM_THINKING_MOUNT / "question_ids.json"),
        (
            "local-mmar-freeform-thinking",
            LOCAL_MMAR_FREEFORM_THINKING_MOUNT / "question_ids.json",
        ),
        (
            FULL_MMAR_RUN_ID,
            DEFAULT_OUTPUT_DIR / FULL_MMAR_RUN_ID / "question_ids.json",
        ),
    ]


def _resolve_question_ids() -> tuple[list[str], str]:
    for tag, path in _question_id_candidates():
        ids = _load_question_ids(path)
        if ids:
            print(f"[lalm-judge-no-gt] question ids: {len(ids)} from {tag} ({path})")
            return ids, tag
    raise SystemExit(
        "No question_ids.json found on mmar-freeform-thinking "
        f"(remote {MMAR_FREEFORM_THINKING_MOUNT} or local download) "
        f"or legacy run {FULL_MMAR_RUN_ID}"
    )


def _merge_shot_judges(local: dict, prior: dict) -> dict:
    """Keep prior verdicts when the first-shot answer is unchanged."""
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
        merged.append(_merge_shot_judges(record, prior) if prior else record)
    write_jsonl(path, merged, mode="w")


def _parse_judge_list(raw: str) -> list[str]:
    from grader import _suite_label_for

    requested = [part.strip() for part in str(raw or "all").split(",") if part.strip()]
    if not requested or requested == ["all"]:
        return list(ROUND_ROBIN_SUITE)
    out: list[str] = []
    seen: set[str] = set()
    for item in requested:
        label = _suite_label_for(item) or item
        if label not in ROUND_ROBIN_SUITE:
            raise SystemExit(
                f"Unknown LALM judge {item!r}. Expected one of {list(ROUND_ROBIN_SUITE)}"
            )
        if label not in seen:
            out.append(label)
            seen.add(label)
    if not out:
        raise SystemExit("No LALM judges resolved")
    return out


def _clamp_n_samples(raw: int | None) -> int:
    return max(1, int(raw if raw is not None else DEFAULT_N_SAMPLES))


def _stamp_manifest(
    manifest: dict,
    *,
    judge_entries: list[dict],
    primary: str | None,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    n_samples = _clamp_n_samples(n_samples)
    by_label: dict[str, dict] = {}
    for item in manifest.get("judges") or []:
        if isinstance(item, dict) and item.get("label"):
            by_label[str(item["label"])] = dict(item)
        elif isinstance(item, str) and item:
            by_label.setdefault(item, {"label": item})
    for entry in judge_entries:
        key = str(entry.get("label") or entry.get("judge_key") or "")
        if not key:
            continue
        stamped = {
            **by_label.get(key, {}),
            **entry,
            "label": key,
            "prompt": GRADE_PROMPT,
            "include_gold": INCLUDE_GOLD,
            "n_samples": n_samples,
            "temperature": JUDGE_TEMPERATURE,
            "primary": False,
        }
        by_label[key] = stamped
    if not primary:
        primary = next(iter(by_label), None)
    ordered: list[dict] = []
    if primary and primary in by_label:
        by_label[primary]["primary"] = True
        ordered.append(by_label[primary])
    for label, item in by_label.items():
        if label == primary:
            continue
        item["primary"] = False
        ordered.append(item)
    manifest["judges"] = ordered
    if primary:
        manifest["primary_judge"] = primary
        manifest["grader_model_id"] = (by_label.get(primary) or {}).get("model_id")
    manifest["scoring"] = "qwen_freeform_judge"
    manifest["grade_prompt"] = GRADE_PROMPT
    manifest["n_samples"] = n_samples
    manifest["temperature"] = JUDGE_TEMPERATURE
    manifest["graded_at"] = now
    manifest["updated_at"] = now
    return manifest


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={
        VOLUME_MOUNT: volume,
        RESULTS_MOUNT: results_volume,
        FREEFORM_THINKING_MOUNT: freeform_thinking_volume,
        MMAR_FREEFORM_THINKING_MOUNT: mmar_freeform_thinking_volume,
        JUDGING_MOUNT: judging_volume,
    },
)
def prepare_pack(models: str = "all") -> dict:
    """Copy first-shot freeform answers onto ``mmar-judging/lalm-judge-no-gt``."""

    results_volume.reload()
    freeform_thinking_volume.reload()
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
            print(f"[lalm-judge-no-gt] skip missing source {tag}: {models_dir}")
            continue
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            label = model_dir.name
            if label in by_model:
                continue
            pred = model_dir / "predictions.jsonl"
            rows = _iter_first_shots(pred, model=label)
            kept = {qid: rows[qid] for qid in question_ids if qid in rows}
            if not kept:
                print(f"[lalm-judge-no-gt] {tag}/{label}: no first-shot overlap")
                continue
            by_model[label] = kept
            sources[label] = tag
            print(
                f"[lalm-judge-no-gt] {tag}/{label}: {len(kept)}/{len(question_ids)} "
                f"first shots"
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
        raise SystemExit("No test-taker first-shot generations found")

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
            "n_shots": 1,
            "source": ids_source,
        },
    )
    manifest_path = pack_dir / "manifest.json"
    existing: dict = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    manifest = {
        "name": PACK_NAME,
        "mode": "freeform",
        "n_shots": 1,
        "n_questions": len(question_ids),
        "models": labels,
        "sources": sources,
        "grade_prompt": GRADE_PROMPT,
        "include_gold": INCLUDE_GOLD,
        "n_samples": DEFAULT_N_SAMPLES,
        "temperature": JUDGE_TEMPERATURE,
        "suite_judges": list(ROUND_ROBIN_SUITE),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
    }
    for key in ("scoring", "grader_model_id", "judges", "primary_judge", "graded_at"):
        if existing.get(key) is not None:
            manifest[key] = existing[key]
    write_json(manifest_path, manifest)
    judging_volume.commit()
    print(
        f"[lalm-judge-no-gt] prepared models={labels} questions={len(question_ids)} "
        f"wanted={len(wanted)} -> {pack_dir}"
    )
    return {
        "pack": PACK_NAME,
        "model_labels": labels,
        "question_ids": question_ids,
        "sources": sources,
        "n_questions": len(question_ids),
    }


def _prompt_batch_for(
    judge_label: str,
    batch_size: int | None,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> int:
    from mmar_models import MODEL_SPECS

    n_samples = _clamp_n_samples(n_samples)
    if batch_size is not None:
        seq_budget = int(batch_size)
    else:
        engine = (MODEL_SPECS.get(judge_label) or {}).get("engine") or {}
        seq_budget = int(engine.get("max_num_seqs") or 32)
    return max(1, seq_budget // n_samples)


def _grade_one_lalm(
    judge_label: str,
    *,
    model_labels: list[str],
    question_ids: list[str],
    force: bool,
    batch_size: int | None,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> dict:
    from grader import (
        compose_judge_key,
        grade_predictions_file,
        load_grader,
        remaining_grade_work,
        resolve_grade_judge_key,
    )
    from mmar_models import MODEL_SPECS

    n_samples = _clamp_n_samples(n_samples)
    volume.reload()
    judging_volume.reload()
    pack_dir = REMOTE_PACK_DIR
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")

    key = compose_judge_key(
        judge_label, prompt=GRADE_PROMPT, include_gold=INCLUDE_GOLD
    )
    remaining = remaining_grade_work(
        pack_dir,
        model_labels,
        key,
        question_ids=question_ids,
        shot_indices=(0,),
        sidecar=True,
    )
    for gradee in model_labels:
        n_left = len(question_ids) if force else len(remaining.get(gradee) or [])
        print(
            f"[lalm-judge-no-gt] {judge_label} -> {gradee}: "
            f"{n_left}/{len(question_ids)} first shots still need {key}"
        )
    if not force:
        done = [label for label in model_labels if label not in remaining]
        if done:
            print(f"[lalm-judge-no-gt] {judge_label} already graded: {done}")
        model_labels = [label for label in model_labels if label in remaining]
        if not model_labels:
            print(f"[lalm-judge-no-gt] skip load {judge_label}: nothing left")
            model_id = (MODEL_SPECS.get(judge_label) or {}).get("model_id") or judge_label
            return {
                "status": "skipped",
                "judge_label": judge_label,
                "model_id": model_id,
                "judge_key": key,
                "gradees": [],
                "by_model": {},
                "prompt": GRADE_PROMPT,
                "include_gold": INCLUDE_GOLD,
                "n_samples": n_samples,
            }

    handle = load_grader(judge_label)
    model_id = handle.get("model_id") or (
        (MODEL_SPECS.get(judge_label) or {}).get("model_id") or judge_label
    )
    key = resolve_grade_judge_key(
        handle, prompt=GRADE_PROMPT, include_gold=INCLUDE_GOLD
    )
    prompt_batch = _prompt_batch_for(judge_label, batch_size, n_samples=n_samples)
    print(
        f"[lalm-judge-no-gt] judge={judge_label} key={key} "
        f"n_samples={n_samples} temperature={JUDGE_TEMPERATURE} "
        f"prompt_batch={prompt_batch}"
    )

    per_model: dict[str, dict] = {}
    for gradee in model_labels:
        predictions_path = pack_dir / "models" / gradee / "predictions.jsonl"
        sidecar = pack_dir / "models" / gradee / "judge_partials" / f"{key}.jsonl"
        print(
            f"[lalm-judge-no-gt] {judge_label} -> {gradee} key={key} "
            f"sidecar={sidecar}"
        )
        per_model[gradee] = grade_predictions_file(
            predictions_path,
            handle,
            judge_key=key,
            batch_size=prompt_batch,
            force=force,
            prompt=GRADE_PROMPT,
            include_gold=INCLUDE_GOLD,
            shot_indices=(0,),
            sidecar_path=sidecar,
            question_ids=question_ids,
            n_samples=n_samples,
            temperature=JUDGE_TEMPERATURE,
        )
        judging_volume.commit()
        print(f"[lalm-judge-no-gt] {gradee}:", per_model[gradee])

    return {
        "status": "ok",
        "judge_label": judge_label,
        "model_id": model_id,
        "judge_key": key or compose_judge_key(
            judge_label, prompt=GRADE_PROMPT, include_gold=INCLUDE_GOLD
        ),
        "gradees": list(model_labels),
        "by_model": per_model,
        "prompt": GRADE_PROMPT,
        "include_gold": INCLUDE_GOLD,
        "n_samples": n_samples,
        "prompt_batch": prompt_batch,
    }


# ---------------------------------------------------------------------------
# Modal grade workers (one GPU function per suite judge)
# ---------------------------------------------------------------------------
# single_use_containers keeps a GPU from being reused after that judge returns.

_SUITE_GRADE_KW = dict(
    timeout=12 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, JUDGING_MOUNT: judging_volume},
    secrets=[hf_secret],
    memory=65536,
    single_use_containers=True,
)

# GPU for suite judges whose MODEL_SPECS entry is currently commented out.
_JUDGE_GPU_FALLBACK = {
    "qwen2.5-omni-7b": "L40S",
    "phi-4-multimodal": "L40S",
    "gemma-4-e4b": "L40S",
    "qwen3-omni-instruct": "H100",
    "nemotron-3-nano-omni": "H100",
}
_JUDGE_MODEL_ID_FALLBACK = {
    "qwen2.5-omni-7b": "Qwen/Qwen2.5-Omni-7B",
    "phi-4-multimodal": "microsoft/Phi-4-multimodal-instruct",
    "qwen3-omni-instruct": "marksverdhei/Qwen3-Omni-30B-A3B-FP8",
}


def _judge_gpu(label: str) -> str | None:
    from mmar_models import MODEL_SPECS

    gpu = (MODEL_SPECS.get(label) or {}).get("gpu") or _JUDGE_GPU_FALLBACK.get(label)
    return str(gpu) if gpu else None


def _judge_model_id(label: str) -> str:
    from mmar_models import MODEL_SPECS

    return (
        (MODEL_SPECS.get(label) or {}).get("model_id")
        or _JUDGE_MODEL_ID_FALLBACK.get(label)
        or label
    )


def _planned_judge_entry(label: str, n_samples: int = DEFAULT_N_SAMPLES) -> dict:
    from grader import compose_judge_key

    key = compose_judge_key(
        label, prompt=GRADE_PROMPT, include_gold=INCLUDE_GOLD
    )
    return {
        "label": key,
        "judge_key": key,
        "model_id": _judge_model_id(label),
        "prompt": GRADE_PROMPT,
        "include_gold": INCLUDE_GOLD,
        "n_samples": _clamp_n_samples(n_samples),
    }


def _grade_function(label: str, gpu: str):
    def run(
        model_labels: list[str],
        question_ids: list[str],
        force: bool = False,
        batch_size: int | None = None,
        n_samples: int = DEFAULT_N_SAMPLES,
    ) -> dict:
        return _grade_one_lalm(
            label,
            model_labels=model_labels,
            question_ids=question_ids,
            force=force,
            batch_size=batch_size,
            n_samples=n_samples,
        )

    # Modal rejects nested @app.function unless serialized=True (cloudpickle
    # from this process into the CUDA image). Give the worker a global
    # __qualname__ and bind it on the module so FILE load works. Dots must
    # go too: ``qwen2.5-omni-7b`` → ``grade_qwen2.5_omni_7b`` looks like a
    # class method (``is_method_fn``) and raises InvalidError at import.
    name = f"grade_{label.replace('-', '_').replace('.', '_')}"
    run.__name__ = name
    run.__qualname__ = name
    fn = app.function(
        image=large_mm_image,
        gpu=gpu,
        name=f"grade-{label}",
        **_SUITE_GRADE_KW,
    )(run)
    globals()[name] = fn
    return fn


_SUITE_GRADE_FNS = {}
_missing_grade = []
for _label in ROUND_ROBIN_SUITE:
    _gpu = _judge_gpu(_label)
    if not _gpu:
        _missing_grade.append(_label)
        continue
    _SUITE_GRADE_FNS[_label] = _grade_function(_label, _gpu)
if _missing_grade:
    raise RuntimeError(f"No GPU grade worker for judges: {_missing_grade}")


def _spawn_judge_grade(label: str, **kwargs):
    """Start one dedicated GPU container for ``label`` (does not wait)."""
    fn = _SUITE_GRADE_FNS.get(label)
    if fn is None:
        raise SystemExit(f"No GPU worker for suite judge {label!r}")
    call = fn.spawn(**kwargs)
    print(f"[lalm-judge-no-gt] Spawned {label} call_id={call.object_id}")
    return call


def _print_grade_workload(
    judge_labels: list[str],
    remaining_by_judge: dict[str, dict[str, list[str]]],
    gradees: list[str],
    n_questions: int,
    force: bool,
) -> None:
    """Log ungraded first-shot counts before any GPU worker is spawned."""
    n_spawn = 0
    print("Workload (first shot, sidecar resume):")
    for label in judge_labels:
        remaining = remaining_by_judge.get(label) or {}
        n_left = (
            n_questions * len(gradees)
            if force
            else sum(len(ids) for ids in remaining.values())
        )
        n_gradees = len(gradees) if force else len(remaining)
        skip = n_left == 0
        if not skip:
            n_spawn += 1
        action = "skip spawn" if skip else "spawn"
        print(
            f"  {label}: {n_left} ungraded first shots across "
            f"{n_gradees}/{len(gradees)} gradee(s) → {action}"
        )
    print(
        f"  {n_spawn} GPU container(s) to launch "
        f"across {len(judge_labels)} judge(s)"
    )


def _stamp_planned_judges(
    judge_labels: list[str],
    primary: str | None,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> None:
    judging_volume.reload()
    manifest_path = REMOTE_PACK_DIR / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [
        _planned_judge_entry(label, n_samples=n_samples) for label in judge_labels
    ]
    manifest = _stamp_manifest(
        manifest,
        judge_entries=entries,
        primary=primary,
        n_samples=n_samples,
    )
    write_json(manifest_path, manifest)
    judging_volume.commit()


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={JUDGING_MOUNT: judging_volume},
)
def merge_pack(
    model_labels: list[str],
    judge_entries: list[dict],
    primary_judge: str | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> dict:
    """Fold judge sidecars into predictions.jsonl and stamp the manifest."""
    from grader import apply_judge_partials

    judging_volume.reload()
    pack_dir = REMOTE_PACK_DIR
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")

    by_model: dict[str, dict] = {}
    for label in model_labels:
        pred = pack_dir / "models" / label / "predictions.jsonl"
        partials_dir = pack_dir / "models" / label / "judge_partials"
        paths = sorted(partials_dir.glob("*.jsonl")) if partials_dir.is_dir() else []
        by_model[label] = apply_judge_partials(
            pred,
            paths,
            make_primary=True,
            primary_judge=primary_judge,
        )
        print(f"[lalm-judge-no-gt] merged {label}:", by_model[label])

    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _stamp_manifest(
        manifest,
        judge_entries=judge_entries,
        primary=primary_judge,
        n_samples=n_samples,
    )
    write_json(manifest_path, manifest)
    judging_volume.commit()
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
    volumes={JUDGING_MOUNT: judging_volume},
)
def run_aggregate() -> dict:
    judging_volume.reload()
    pack_dir = REMOTE_PACK_DIR
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")
    result = aggregate_difficulty(pack_dir)
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    scores = result.get("scores") or {}
    for key in (
        "scoring",
        "mode",
        "grader_model_id",
        "judges",
        "primary_judge",
        "n_samples",
        "grade_prompt",
        "temperature",
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
    timeout=30 * 60,
    volumes={JUDGING_MOUNT: judging_volume},
)
def run_pipeline(
    models: str = "all",
    judges: str = "all",
    force: bool = False,
    batch_size: int | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    skip_aggregate: bool = False,
    merge_only: bool = False,
    aggregate_only: bool = False,
) -> dict:
    """Remote orchestrator: prepare workload, spawn GPU judges, return.

    Does not wait on grading. GPU FunctionCalls keep a ``--detach`` app
    alive; waiting here would pin a preemptible CPU container for hours
    and re-spawn workers if that container is redelivered.

    Workload is computed on CPU before any GPU container starts. Each
    pending suite judge is spawned on its own GPU container; a multi-judge
    run launches those containers in parallel. Fold sidecars with
    ``--merge-only`` after workers finish.
    """
    from grader import compose_judge_key, remaining_grade_work

    n_samples = _clamp_n_samples(n_samples)
    judge_labels = _parse_judge_list(judges)
    primary = None
    if judge_labels:
        primary = compose_judge_key(
            judge_labels[0], prompt=GRADE_PROMPT, include_gold=INCLUDE_GOLD
        )

    if aggregate_only:
        result = run_aggregate.remote()
        print("[lalm-judge-no-gt] Done (aggregate-only):", result)
        return {"aggregate_only": True, "aggregate": result}

    prep = prepare_pack.remote(models=models)
    print(
        f"[lalm-judge-no-gt] prepared pack={prep.get('pack')} "
        f"models={prep.get('model_labels')} "
        f"n_questions={prep.get('n_questions')} "
        f"sources={prep.get('sources')} "
        f"judges={judge_labels} "
        f"n_samples={n_samples} "
        f"gpu_containers=per-judge parallel_launch=True"
    )
    gradees = list(prep["model_labels"])
    question_ids = list(prep["question_ids"])
    judge_entries = [
        _planned_judge_entry(label, n_samples=n_samples) for label in judge_labels
    ]

    def _merge() -> dict:
        merge = merge_pack.remote(
            model_labels=gradees,
            judge_entries=judge_entries,
            primary_judge=primary,
            n_samples=n_samples,
        )
        print("[lalm-judge-no-gt] merged:", merge)
        return merge

    if merge_only:
        merge = _merge()
        agg = None if skip_aggregate else run_aggregate.remote()
        if agg is not None:
            print("[lalm-judge-no-gt] aggregated:", agg)
        return {
            "prepare": prep,
            "merge_only": True,
            "merge": merge,
            "aggregate": agg,
        }

    judging_volume.reload()
    remaining_by_judge: dict[str, dict[str, list[str]]] = {}
    pending_labels: list[str] = []
    skipped_labels: list[str] = []
    remaining_gradees: dict[str, list[str]] = {}
    for label in judge_labels:
        key = compose_judge_key(
            label, prompt=GRADE_PROMPT, include_gold=INCLUDE_GOLD
        )
        remaining = remaining_grade_work(
            REMOTE_PACK_DIR,
            gradees,
            key,
            question_ids=question_ids,
            shot_indices=(0,),
            sidecar=True,
        )
        remaining_by_judge[label] = remaining
        if force or remaining:
            pending_labels.append(label)
            remaining_gradees[label] = (
                list(gradees) if force else [g for g in gradees if g in remaining]
            )
        else:
            skipped_labels.append(label)
    _print_grade_workload(
        judge_labels,
        remaining_by_judge,
        gradees,
        n_questions=len(question_ids),
        force=force,
    )

    results: list[dict] = [
        {
            "status": "already_complete",
            "judge_label": label,
            "model_id": _judge_model_id(label),
            "judge_key": compose_judge_key(
                label, prompt=GRADE_PROMPT, include_gold=INCLUDE_GOLD
            ),
            "gradees": gradees,
        }
        for label in skipped_labels
    ]

    _stamp_planned_judges(judge_labels, primary, n_samples=n_samples)

    if pending_labels:
        print(
            f"[lalm-judge-no-gt] Launching {len(pending_labels)} dedicated "
            f"GPU container(s)"
            f"{' in parallel' if len(pending_labels) > 1 else ''}: "
            f"{pending_labels}"
        )
        for label in pending_labels:
            call = _spawn_judge_grade(
                label,
                model_labels=remaining_gradees[label],
                question_ids=question_ids,
                force=force,
                batch_size=batch_size,
                n_samples=n_samples,
            )
            results.append(
                {
                    "status": "spawned",
                    "judge_label": label,
                    "call_id": call.object_id,
                    "gradees": remaining_gradees[label],
                }
            )
        return {
            "prepare": prep,
            "grade": results,
            "pending_labels": pending_labels,
            "skipped_labels": skipped_labels,
        }

    print("[lalm-judge-no-gt] All requested judges already complete; merging.")
    merge = _merge()
    agg = None if skip_aggregate else run_aggregate.remote()
    if agg is not None:
        print("[lalm-judge-no-gt] aggregated:", agg)
    return {
        "prepare": prep,
        "grade": results,
        "pending_labels": pending_labels,
        "skipped_labels": skipped_labels,
        "merge": merge,
        "aggregate": agg,
    }


@app.local_entrypoint()
def main(
    models: str = "all",
    judges: str = "all",
    force: bool = False,
    batch_size: int | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    skip_aggregate: bool = False,
    merge_only: bool = False,
    aggregate_only: bool = False,
) -> None:
    """Materialize first shots, spawn one GPU per LALM judge, return.

    Args:
        models: Comma-separated test-taker labels, or ``all``.
        judges: Comma-separated suite LALM labels, or ``all``
            (``qwen2.5-omni-7b``, ``phi-4-multimodal``, ``gemma-4-e4b``,
            ``qwen3-omni-instruct``, ``nemotron-3-nano-omni``).
        force: Replace existing verdicts for these judge keys. Without
            this flag, already-graded first shots are skipped.
        batch_size: Concurrent sequence budget (default: the judge's
            ``max_num_seqs``). Prompt batch is ``batch_size // n_samples``.
        n_samples: Judge completions per answer (default 1). ``1`` is a
            single verdict; ``3`` restores majority vote.
        skip_aggregate: Skip difficulty.jsonl / scores.json when merging.
        merge_only: Fold existing sidecars into predictions.jsonl; do not
            spawn GPU workers.
        aggregate_only: Skip prepare and grading; only build
            difficulty.jsonl / scores.json from existing predictions.
    """
    # Remote prepare+spawn so ``--detach`` keeps GPU FunctionCalls after
    # this process exits. Do not wait on workers here.
    out = run_pipeline.spawn(
        models=models,
        judges=judges,
        force=force,
        batch_size=batch_size,
        n_samples=n_samples,
        skip_aggregate=skip_aggregate,
        merge_only=merge_only,
        aggregate_only=aggregate_only,
    ).get()

    pending = out.get("pending_labels") or []
    if pending:
        print(f"Spawned {len(pending)} GPU worker(s): {pending}")
        print(
            "Use ``modal run --detach``; without it this process exiting "
            "stops the ephemeral app and kills those workers. "
            "Watch progress in the Modal dashboard."
        )
        print(
            "After GPU workers finish, fold sidecars with:\n"
            "  uv run modal run judge-quality/run_lalm_judge_no_gt.py --merge-only"
        )
    print("Orchestrator:", out)


if __name__ == "__main__":
    print(
        "Run via Modal:\n"
        "  uv run modal run --detach judge-quality/run_lalm_judge_no_gt.py",
        flush=True,
    )
    raise SystemExit(2)
