"""Cascade step 1: audio → caption (thinking off) on Modal.

For every question with ≥3 human ratings in ``exports/labels.csv``, each
uncommented ``MODEL_SPECS`` model generates ``n_shots`` captions of the
clip (no MMAR question text). Thinking is forced off for all models.
Writes to the root of the ``mmar-descriptions`` Modal Volume.

Each model flattens remaining questions (and leftover shots) into one
``generate()`` / ``chat()`` so vLLM continuous-batches the run.

Layout on ``mmar-descriptions`` (volume root)::

    question_ids.json
    manifest.json
    models/<label>/predictions.jsonl

Usage::

    # Smoke
    uv run modal run --detach cascade/run_descriptions.py \\
      --models af-next-think --limit 5

    # Full (all uncommented models, all triple-labeled questions)
    uv run modal run --detach cascade/run_descriptions.py

    # Download
    uv run modal run download_results.py --volume-name mmar-descriptions
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

# Repo root on sys.path so ``modal run cascade/run_descriptions.py`` resolves
# sibling packages (mmar_models, modal_cache, …).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import modal

from mmar_common import (
    aggregate_n_shot_record,
    build_mmar_description_prompt,
    count_wavs,
    load_jsonl,
    resolve_path,
    write_json,
    write_jsonl,
)
from mmar_models import (
    ALL_MODEL_LABELS,
    MODEL_SPECS,
    backend_duplicates_shots,
    generate_batch,
    load_model,
    parse_model_list,
    resolve_sampling,
)
from modal_cache import (
    DEFAULT_MMAR_DATA_ROOT,
    DEFAULT_MMAR_META,
    MMAR_DESCRIPTIONS_MOUNT,
    VOLUME_MOUNT,
    hf_secret,
    mmar_descriptions_volume,
    volume,
)
from modal_images import eval_image, mount_local_sources

REPO_ROOT = _REPO_ROOT
EXPORTS_DIR = REPO_ROOT / "exports"
LABELS_CSV_NAME = "labels.csv"
DEFAULT_OUTPUT_DIR = MMAR_DESCRIPTIONS_MOUNT
DEFAULT_N_SHOTS = 5
DEFAULT_SEED = 42
# Match alt_test.MIN_HUMANS_PER_INSTANCE (avoid importing alt_test in GPU workers).
MIN_HUMANS_PER_INSTANCE = 3

app = modal.App("cascade-descriptions")

cpu_image = mount_local_sources(
    modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "numpy", "tqdm>=4.67.0"
    )
)


# ---------------------------------------------------------------------------
# Shared helpers (run inside Modal containers)
# ---------------------------------------------------------------------------


def _pack_dir(output_dir: str) -> Path:
    return Path(output_dir).expanduser().resolve()


def _shot_seed(seed: int, question_id: str, shot_index: int) -> int:
    digest = hashlib.md5(f"{seed}:{question_id}:{shot_index}".encode()).hexdigest()
    return seed + (int(digest[:8], 16) % 1_000_000)


def _n_generated_shots(record: dict | None) -> int:
    """How many shot generations are stored on this prediction record."""
    if not record:
        return 0
    shots = record.get("shots")
    if isinstance(shots, list):
        return len(shots)
    try:
        return max(0, int(record.get("n_shots") or 0))
    except (TypeError, ValueError):
        return 0


def _load_prediction_records(predictions_path: Path) -> dict[str, dict]:
    """Return ``{id: record}`` in file order, repairing a truncated last line."""
    if not predictions_path.exists():
        return {}

    records: dict[str, dict] = {}
    valid_items: list[dict] = []
    corrupt = False
    with open(predictions_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                corrupt = True
                print(f"Skipping corrupt predictions line in {predictions_path}")
                continue
            record_id = item.get("id")
            if record_id:
                records[str(record_id)] = item
            valid_items.append(item)

    if corrupt:
        tmp_path = predictions_path.with_suffix(".jsonl.tmp")
        write_jsonl(tmp_path, valid_items, mode="w")
        tmp_path.replace(predictions_path)
        print(
            f"Repaired predictions file -> {predictions_path} "
            f"({len(valid_items)} lines)"
        )

    return records


def _rewrite_predictions_file(
    predictions_path: Path,
    records: dict[str, dict],
    order: list[str],
) -> None:
    ordered: list[dict] = []
    seen: set[str] = set()
    for qid in order:
        rec = records.get(qid)
        if rec is not None:
            ordered.append(rec)
            seen.add(qid)
    for qid, rec in records.items():
        if qid not in seen:
            ordered.append(rec)
    tmp_path = predictions_path.with_suffix(".jsonl.tmp")
    write_jsonl(tmp_path, ordered, mode="w")
    tmp_path.replace(predictions_path)


def _merge_shot_record(
    item: dict,
    existing: dict | None,
    new_outputs: list[dict],
    *,
    start_index: int,
) -> dict:
    """Append newly generated shots onto an existing record, or build a new one."""
    new_record = aggregate_n_shot_record(item, new_outputs, pending_grade=True)
    if not existing or start_index <= 0:
        return new_record

    shots = list(existing.get("shots") or [])[:start_index]
    for offset, shot in enumerate(new_record.get("shots") or []):
        merged = dict(shot)
        merged["shot_index"] = start_index + offset
        shots.append(merged)
    record = {**existing, **item}
    record["shots"] = shots
    record["n_shots"] = len(shots)
    record["pending_grade"] = True
    return record


def _model_workload(
    pack_dir: Path,
    model_label: str,
    question_ids: list[str],
    n_shots: int,
) -> dict:
    """Shots still needed for ``model_label`` to reach ``n_shots`` per question."""
    predictions_path = pack_dir / "models" / model_label / "predictions.jsonl"
    records = _load_prediction_records(predictions_path)
    n_total = len(question_ids)
    n_complete = 0
    n_partial = 0
    n_missing_questions = 0
    n_have_shots = 0
    n_missing_shots = 0
    for qid in question_ids:
        n_have = _n_generated_shots(records.get(qid))
        capped = min(n_have, n_shots)
        n_have_shots += capped
        n_need = n_shots - capped
        n_missing_shots += n_need
        if n_need == 0:
            n_complete += 1
        elif n_have == 0:
            n_missing_questions += 1
        else:
            n_partial += 1
    return {
        "model_label": model_label,
        "n_questions": n_total,
        "n_complete": n_complete,
        "n_partial": n_partial,
        "n_missing_questions": n_missing_questions,
        "n_have_shots": n_have_shots,
        "n_missing_shots": n_missing_shots,
        "n_target_shots": n_total * n_shots,
        "n_shots": n_shots,
        "complete": n_missing_shots == 0,
        "predictions_path": str(predictions_path),
    }


def _parse_ratings_cell(raw: object) -> list[bool]:
    if isinstance(raw, list):
        values = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            values = json.loads(text)
        except json.JSONDecodeError:
            return []
    if not isinstance(values, list) or not values:
        return []
    out: list[bool] = []
    for item in values:
        if isinstance(item, bool):
            out.append(item)
        else:
            return []
    return out


def triple_labeled_question_ids(labels_path: Path) -> list[str]:
    """Unique question_ids with ≥3 boolean ratings (order of first appearance)."""
    ids: list[str] = []
    seen: set[str] = set()
    with labels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            qid = str(raw.get("question_id") or "").strip()
            if not qid or qid in seen:
                continue
            ratings = _parse_ratings_cell(raw.get("ratings"))
            if len(ratings) < MIN_HUMANS_PER_INSTANCE:
                continue
            seen.add(qid)
            ids.append(qid)
    return ids


def _write_question_ids(ids_path: Path, ids: list[str], *, seed: int) -> None:
    payload = {
        "seed": seed,
        "n": len(ids),
        "ids": ids,
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(ids_path, payload)
    mmar_descriptions_volume.commit()


def _merge_question_ids(
    pack_dir: Path,
    *,
    requested: list[str],
    meta_path: Path,
    data_root: Path,
    seed: int,
) -> list[str]:
    """Union requested ids with existing pack ids; keep only those with audio.

    Never shrinks ``question_ids.json``: a ``--limit`` smoke run adds its
    slice; a later full run merges the rest.
    """
    ids_path = pack_dir / "question_ids.json"
    audio_ok = set()
    for item in load_jsonl(meta_path):
        audio_path = resolve_path(data_root, item["audio_path"])
        if os.path.exists(audio_path):
            audio_ok.add(str(item["id"]))

    existing: list[str] = []
    if ids_path.exists():
        try:
            payload = json.loads(ids_path.read_text(encoding="utf-8"))
            existing = [str(x) for x in payload.get("ids", [])]
        except json.JSONDecodeError:
            existing = []

    merged = [qid for qid in existing if qid in audio_ok]
    for qid in requested:
        if qid in audio_ok and qid not in merged:
            merged.append(qid)
        elif qid not in audio_ok:
            print(f"Skipping {qid}: missing audio under {data_root}")

    if not merged:
        raise SystemExit(
            f"No requested questions with audio under {data_root} "
            f"(requested={len(requested)})"
        )

    if merged == existing and existing and ids_path.exists():
        print(f"Reusing {len(existing)} question ids from {ids_path}")
        return existing

    _write_question_ids(ids_path, merged, seed=seed)
    if existing and len(merged) > len(existing):
        print(f"Expanded question set {len(existing)} -> {len(merged)} -> {ids_path}")
    else:
        print(f"Wrote {len(merged)} question ids -> {ids_path}")
    return merged


def _load_selected_items(
    meta_path: Path,
    data_root: Path,
    question_ids: list[str],
) -> list[dict]:
    by_id = {str(item["id"]): item for item in load_jsonl(meta_path)}
    items: list[dict] = []
    for qid in question_ids:
        item = by_id.get(qid)
        if item is None:
            print(f"Missing meta for id={qid}")
            continue
        audio_path = resolve_path(data_root, item["audio_path"])
        if not os.path.exists(audio_path):
            print(f"Skipping {qid}: missing audio at {audio_path}")
            continue
        items.append({**item, "audio_path": audio_path})
    return items


def _run_model_eval(
    *,
    model_label: str,
    output_dir: str,
    meta: str,
    data_root: str,
    n_shots: int,
    seed: int,
    print_every: int,
    question_ids: list[str],
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
    model_id: str | None = None,
    tokenizer_id: str | None = None,
) -> dict:
    """Load one model and write n-shot description captions for ``question_ids``."""
    volume.reload()
    mmar_descriptions_volume.reload()

    spec = MODEL_SPECS[model_label]
    args = SimpleNamespace(
        model_id=model_id or spec["model_id"],
        tokenizer_id=tokenizer_id or spec.get("tokenizer_id"),
        local_model_dir=None,
        local_tokenizer_dir=None,
        meta=meta,
        data_root=data_root,
        output_dir=output_dir,
        n_shots=n_shots,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        greedy_non_thinking=False,
        seed=seed,
        torch_dtype="bfloat16",
        print_every=print_every,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        prompt_mode="description",
        enable_thinking=False,
    )
    sampling = resolve_sampling(model_label, args)
    args.temperature = float(sampling["temperature"])
    args.top_p = float(sampling.get("top_p", 1.0))
    args.max_new_tokens = int(sampling["max_tokens"])
    args.repetition_penalty = float(sampling.get("repetition_penalty", 1.0))
    args.sampling = sampling

    pack_dir = _pack_dir(output_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    model_dir = pack_dir / "models" / model_label
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = model_dir / "predictions.jsonl"

    meta_path = Path(meta).expanduser().resolve()
    data_root_path = Path(data_root).expanduser().resolve()
    audio_dir = data_root_path / "audio"
    if not meta_path.exists():
        raise SystemExit(
            f"MMAR metadata not found: {meta_path}\n"
            "Seed first: uv run modal run seed_volume.py --datasets mmar --models none"
        )
    wav_count = count_wavs(audio_dir)
    if wav_count < 100:
        raise SystemExit(
            f"MMAR audio missing/incomplete in {audio_dir} ({wav_count} wavs)"
        )

    if not question_ids:
        raise SystemExit(f"[{model_label}] empty question_ids")

    items = _load_selected_items(meta_path, data_root_path, question_ids)
    existing_records = _load_prediction_records(predictions_path)
    mmar_descriptions_volume.commit()
    pending_items: list[dict] = []
    pending_have: list[int] = []
    n_done = 0
    for item in items:
        n_have = _n_generated_shots(existing_records.get(str(item["id"])))
        if n_have >= n_shots:
            n_done += 1
            continue
        pending_items.append(item)
        pending_have.append(n_have)
    print(
        f"[{model_label}] backend={spec.get('backend')} mode=description "
        f"enable_thinking=False {len(items)} selected, {n_done} done "
        f"(>= {n_shots} shots), {len(pending_items)} pending "
        f"(n_shots={n_shots}, sampling={sampling})"
    )

    if not pending_items:
        return {
            "status": "already_complete",
            "model_label": model_label,
            "n_predictions": n_done,
            "predictions_path": str(predictions_path),
            "mode": "description",
        }

    handle = load_model(model_label, args)
    try:
        volume.commit()
    except Exception as exc:  # noqa: BLE001 — cache commit is best-effort
        print(f"[{model_label}] volume.commit after load failed: {exc}")
    active_backend = handle.get("backend", spec.get("backend"))
    # Flatten every remaining question×shot into one generate() so vLLM
    # continuous-batches the whole run. HF cannot fork SamplingParams(n>1);
    # those backends always expand to one row per shot. Plain vLLM on an
    # all-fresh pending set keeps n=n_shots so samples of one prompt share
    # prefill.
    duplicate_shots = backend_duplicates_shots(str(active_backend))

    start_time = time.time()
    n_pending = len(pending_items)
    shot_outputs_by_index: list[list[dict]] = [[] for _ in pending_items]
    all_fresh = all(n_have == 0 for n_have in pending_have)

    if duplicate_shots or not all_fresh:
        gen_samples: list[dict] = []
        seeds: list[int] = []
        owners: list[tuple[int, int]] = []
        for item_index, (item, n_have) in enumerate(zip(pending_items, pending_have)):
            for shot_index in range(n_have, n_shots):
                gen_samples.append(item)
                seeds.append(_shot_seed(seed, str(item["id"]), shot_index))
                owners.append((item_index, shot_index))
        n_completions = 1
        n_requests = len(gen_samples)
    else:
        gen_samples = list(pending_items)
        seeds = [_shot_seed(seed, str(item["id"]), 0) for item in pending_items]
        owners = [
            (item_index, shot_index)
            for item_index in range(n_pending)
            for shot_index in range(n_shots)
        ]
        n_completions = n_shots
        n_requests = len(gen_samples)

    n_missing = sum(n_shots - n_have for n_have in pending_have)
    print(
        f"[{model_label}] flattening questions={n_pending} "
        f"n_missing_shots={n_missing} into one generate "
        f"n_requests={n_requests} n_completions={n_completions}",
        flush=True,
    )
    try:
        outputs = generate_batch(
            model_label,
            handle,
            gen_samples,
            args,
            seeds=seeds,
            n_completions=n_completions,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[{model_label}] generate failed "
            f"n_questions={n_pending} "
            f"n_requests={n_requests} n_completions={n_completions}: {exc}"
        ) from exc
    if len(outputs) != len(owners):
        raise RuntimeError(
            f"[{model_label}] expected {len(owners)} shot outputs, got {len(outputs)}"
        )
    for (item_index, _shot_index), output in zip(owners, outputs):
        shot_outputs_by_index[item_index].append(output)

    for item, n_have, new_outputs in zip(
        pending_items, pending_have, shot_outputs_by_index
    ):
        qid = str(item["id"])
        existing_records[qid] = _merge_shot_record(
            item,
            existing_records.get(qid),
            new_outputs,
            start_index=n_have,
        )
    # Order by the pack's full id list when present so resumes stay stable.
    order_ids = list(question_ids)
    ids_path = pack_dir / "question_ids.json"
    if ids_path.exists():
        try:
            payload = json.loads(ids_path.read_text(encoding="utf-8"))
            packed = [str(x) for x in payload.get("ids", [])]
            if packed:
                order_ids = packed
        except json.JSONDecodeError:
            pass
    _rewrite_predictions_file(predictions_path, existing_records, order_ids)
    with open(predictions_path, "rb") as pred_file:
        os.fsync(pred_file.fileno())
    mmar_descriptions_volume.commit()

    written = n_pending
    elapsed = time.time() - start_time
    try:
        volume.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[{model_label}] volume.commit after generate failed: {exc}")
    if print_every > 0:
        for idx, item in enumerate(pending_items, start=1):
            if idx % print_every == 0 or idx == written:
                record = existing_records[str(item["id"])]
                print(
                    f"[{model_label}] {idx}/{written} "
                    f"id={record['id']} captioned ({elapsed:.0f}s)"
                )

    total = sum(
        1
        for qid in question_ids
        if _n_generated_shots(existing_records.get(qid)) >= n_shots
    )
    print(
        f"[{model_label}] done: updated {written} questions "
        f"({n_missing} shots), total={total}/{len(question_ids)} "
        f"with {n_shots} shots ({elapsed:.0f}s)"
    )
    return {
        "status": "ok",
        "model_label": model_label,
        "n_written": written,
        "n_predictions": total,
        "predictions_path": str(predictions_path),
        "backend": active_backend,
        "mode": "description",
    }


# ---------------------------------------------------------------------------
# Modal eval workers (one GPU function per model_label)
# ---------------------------------------------------------------------------

_PACK_VOLUMES = {
    VOLUME_MOUNT: volume,
    MMAR_DESCRIPTIONS_MOUNT: mmar_descriptions_volume,
}

_EVAL_KW = dict(
    timeout=12 * 60 * 60,
    volumes=_PACK_VOLUMES,
    secrets=[hf_secret],
    memory=65536,
    single_use_containers=True,
)


def _eval_function(label: str, image: modal.Image, gpu: str):
    def run(**kwargs) -> dict:
        return _run_model_eval(model_label=label, **kwargs)

    name = f"eval_{label.replace('-', '_').replace('.', '_')}"
    run.__name__ = name
    run.__qualname__ = name
    fn = app.function(image=image, gpu=gpu, name=f"desc-{label}", **_EVAL_KW)(run)
    globals()[name] = fn
    return fn


_EVAL_FNS = {}
_missing_eval = []
for _label in ALL_MODEL_LABELS:
    _gpu = MODEL_SPECS[_label].get("gpu")
    if not _gpu:
        _missing_eval.append(_label)
        continue
    _EVAL_FNS[_label] = _eval_function(_label, eval_image, str(_gpu))
if _missing_eval:
    raise RuntimeError(f"No GPU eval worker for models: {_missing_eval}")


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes=_PACK_VOLUMES,
)
def prepare_run(
    output_dir: str,
    model_labels: list[str],
    question_ids: list[str],
    n_shots: int,
    seed: int,
    meta: str,
    data_root: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
) -> dict:
    """Merge question ids into the pack and compute missing-shot workload.

    ``question_ids`` is the active set for this run (may be a ``--limit``
    smoke slice). The on-disk ``question_ids.json`` only grows.
    """
    volume.reload()
    mmar_descriptions_volume.reload()

    pack_dir = _pack_dir(output_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(meta).expanduser().resolve()
    data_root_path = Path(data_root).expanduser().resolve()

    packed_ids = _merge_question_ids(
        pack_dir,
        requested=question_ids,
        meta_path=meta_path,
        data_root=data_root_path,
        seed=seed,
    )
    # Active set for this run: requested ids that survived the audio filter,
    # preserving request order (limit smoke stays small).
    packed_set = set(packed_ids)
    active_ids = [qid for qid in question_ids if qid in packed_set]
    if not active_ids:
        raise SystemExit(
            "No active question ids after audio filter "
            f"(requested={len(question_ids)}, packed={len(packed_ids)})"
        )

    now = datetime.now(UTC).isoformat()
    manifest_path = pack_dir / "manifest.json"
    existing: dict = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    prior_models = [str(x) for x in (existing.get("models") or [])]
    disk_models: list[str] = []
    models_root = pack_dir / "models"
    if models_root.is_dir():
        disk_models = [
            child.name
            for child in sorted(models_root.iterdir())
            if child.is_dir()
            and (child / "predictions.jsonl").is_file()
            and child.name in MODEL_SPECS
        ]
    merged_models = list(dict.fromkeys([*prior_models, *disk_models, *model_labels]))
    is_resume = bool(existing.get("created_at")) or bool(disk_models)
    description_prompt = build_mmar_description_prompt()
    if is_resume and existing.get("description_prompt") != description_prompt:
        raise SystemExit(
            "description prompt changed; captions on this pack are from a "
            "different prompt. Wipe models/*/predictions.jsonl on the "
            f"mmar-descriptions volume (pack {pack_dir}) or pass a fresh "
            "output_dir before re-running."
        )

    # Workload for this run uses the active (possibly limited) id list.
    pack_workload = {
        label: _model_workload(pack_dir, label, active_ids, n_shots)
        for label in merged_models
    }
    requested_workload = {label: pack_workload[label] for label in model_labels}
    mmar_descriptions_volume.commit()

    override_ns = SimpleNamespace(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
    )
    model_sampling = {
        label: resolve_sampling(label, override_ns) for label in merged_models
    }

    manifest = {
        "experiment": "cascade-descriptions",
        "mode": "description",
        "description_prompt": description_prompt,
        "enable_thinking": False,
        "models": merged_models,
        "n_shots": n_shots,
        "seed": existing.get("seed", seed),
        "model_sampling": model_sampling,
        "sampling_overrides": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        "inference": "vllm",
        "n_questions": len(packed_ids),
        "n_active_questions": len(active_ids),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "resumed": is_resume,
        "workload": {
            label: {
                key: value for key, value in info.items() if key != "predictions_path"
            }
            for label, info in pack_workload.items()
        },
        "model_specs": {
            label: {
                "model_id": MODEL_SPECS[label]["model_id"],
                "backend": MODEL_SPECS[label].get("backend"),
                "gpu": MODEL_SPECS[label].get("gpu"),
                "sampling": MODEL_SPECS[label].get("sampling"),
                "native_thinking": bool(MODEL_SPECS[label].get("native_thinking")),
            }
            for label in ALL_MODEL_LABELS
        },
    }
    write_json(manifest_path, manifest)
    mmar_descriptions_volume.commit()
    return {
        "manifest": manifest,
        "workload": requested_workload,
        "question_ids": active_ids,
        "packed_question_ids": packed_ids,
        "resumed": is_resume,
        "mode": "description",
    }


def _spawn_model_eval(label: str, **common):
    """Start one dedicated GPU container for ``label`` (does not wait)."""
    fn = _EVAL_FNS.get(label)
    if fn is None:
        raise SystemExit(f"No GPU worker for model {label!r}")
    call = fn.spawn(**common)
    print(f"Spawned {label} call_id={call.object_id}")
    return call


def _print_workload(model_labels: list[str], workload: dict, n_shots: int) -> None:
    """Log missing-shot counts before any GPU worker is spawned."""
    n_missing_total = 0
    n_spawn = 0
    print(f"Workload (n_shots={n_shots}):")
    for label in model_labels:
        info = workload.get(label) or {}
        n_missing = int(info.get("n_missing_shots") or 0)
        n_have = int(info.get("n_have_shots") or 0)
        n_target = int(info.get("n_target_shots") or 0)
        n_complete = int(info.get("n_complete") or 0)
        n_partial = int(info.get("n_partial") or 0)
        n_missing_questions = int(info.get("n_missing_questions") or 0)
        n_missing_total += n_missing
        skip = n_missing == 0
        if not skip:
            n_spawn += 1
        action = "skip spawn" if skip else "spawn"
        print(
            f"  {label}: {n_have}/{n_target} shots "
            f"({n_complete} complete, {n_partial} partial, "
            f"{n_missing_questions} uncovered questions) "
            f"missing={n_missing} → {action}"
        )
    print(
        f"  total missing shots={n_missing_total} across "
        f"{len(model_labels)} model(s); "
        f"{n_spawn} GPU container(s) to launch"
    )


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes=_PACK_VOLUMES,
)
def run_pipeline(
    models: str = "all",
    question_ids: list[str] | None = None,
    n_shots: int = DEFAULT_N_SHOTS,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    seed: int = DEFAULT_SEED,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
    meta: str = str(DEFAULT_MMAR_META),
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    print_every: int = 5,
) -> dict:
    """Remote orchestrator: prepare workload, spawn GPU workers, return.

    Does not wait on inference. GPU FunctionCalls keep a ``--detach`` app
    alive; waiting here would pin a preemptible CPU container for hours.
    """
    model_labels = parse_model_list(models)
    ids = list(question_ids or [])
    if not ids:
        raise SystemExit(
            "run_pipeline requires question_ids from the local entrypoint "
            "(triple-labeled rows in exports/labels.csv)."
        )

    common = dict(
        output_dir=output_dir,
        meta=meta,
        data_root=data_root,
        n_shots=n_shots,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        print_every=print_every,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    print(
        f"Cascade descriptions pack={output_dir} mode=description "
        f"enable_thinking=False models={model_labels} n_shots={n_shots} "
        f"n_questions={len(ids)} "
        f"gpu_containers=per-model parallel_launch=True inference=vllm "
        f"sampling_overrides={{temperature={temperature}, top_p={top_p}, "
        f"max_new_tokens={max_new_tokens}}}"
    )

    prep = prepare_run.remote(
        output_dir=output_dir,
        model_labels=model_labels,
        question_ids=ids,
        n_shots=n_shots,
        seed=seed,
        meta=meta,
        data_root=data_root,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    active_ids = list(prep.get("question_ids") or ids)
    common["question_ids"] = active_ids

    workload = prep.get("workload") or {}
    pending_labels = [
        label
        for label in model_labels
        if int((workload.get(label) or {}).get("n_missing_shots") or 0) > 0
    ]
    skipped_labels = [label for label in model_labels if label not in pending_labels]
    _print_workload(model_labels, workload, n_shots)

    results: list[dict] = [
        {
            "status": "already_complete",
            "model_label": label,
            "n_predictions": (workload.get(label) or {}).get("n_complete"),
            "n_have_shots": (workload.get(label) or {}).get("n_have_shots"),
            "n_missing_shots": 0,
            "predictions_path": (workload.get(label) or {}).get("predictions_path"),
        }
        for label in skipped_labels
    ]

    if pending_labels:
        print(
            f"Launching {len(pending_labels)} dedicated GPU container(s)"
            f"{' in parallel' if len(pending_labels) > 1 else ''}: "
            f"{pending_labels}"
        )
        for label in pending_labels:
            call = _spawn_model_eval(label, **common)
            results.append(
                {
                    "status": "spawned",
                    "model_label": label,
                    "call_id": call.object_id,
                }
            )
        print(
            f"[cascade-descriptions] spawned {len(pending_labels)} GPU worker(s); "
            "CPU orchestrator returning"
        )
    else:
        print("All requested models already complete; skipping inference.")

    return {
        "mode": "description",
        "enable_thinking": False,
        "models": results,
        "pending_labels": pending_labels,
        "skipped_labels": skipped_labels,
        "n_questions": len(active_ids),
        "workload": {
            label: {
                "n_complete": (workload.get(label) or {}).get("n_complete"),
                "n_partial": (workload.get(label) or {}).get("n_partial"),
                "n_missing_questions": (workload.get(label) or {}).get(
                    "n_missing_questions"
                ),
                "n_have_shots": (workload.get(label) or {}).get("n_have_shots"),
                "n_missing_shots": (workload.get(label) or {}).get("n_missing_shots"),
            }
            for label in model_labels
        },
    }


@app.local_entrypoint()
def main(
    models: str = "all",
    n_shots: int = DEFAULT_N_SHOTS,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    seed: int = DEFAULT_SEED,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
    meta: str = str(DEFAULT_MMAR_META),
    data_root: str = str(DEFAULT_MMAR_DATA_ROOT),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    print_every: int = 5,
    limit: int = 0,
    labels_csv: str = "",
):
    """Launch cascade audio→caption generation (thinking off).

    Args:
        models: Comma-separated labels or ``all`` (uncommented MODEL_SPECS).
        n_shots: Independent temperature captions per question (default 5).
        temperature: Optional override of each model's sampling temperature.
        top_p: Optional override of each model's top_p.
        max_new_tokens: Optional override of each model's max_tokens.
        seed: RNG seed for per-question sample seeds.
        max_num_seqs: Optional vLLM override.
        gpu_memory_utilization: Optional vLLM GPU memory fraction override.
        meta: Path to MMAR-meta.jsonl on the data volume.
        data_root: MMAR root used to resolve audio paths.
        output_dir: Pack directory on the ``mmar-descriptions`` volume.
        print_every: Progress print interval per model.
        limit: If >0, only the first N triple-labeled question ids (smoke).
            Does not shrink ``question_ids.json``; a later unlimited run
            merges the rest.
        labels_csv: Optional path to labels.csv (default ``exports/labels.csv``).
    """
    labels_path = (
        Path(labels_csv).expanduser() if labels_csv else EXPORTS_DIR / LABELS_CSV_NAME
    )
    if not labels_path.is_file():
        raise SystemExit(
            f"labels.csv not found at {labels_path}. "
            f"Expected {EXPORTS_DIR / LABELS_CSV_NAME}."
        )
    question_ids = triple_labeled_question_ids(labels_path)
    if not question_ids:
        raise SystemExit(
            f"No questions with ≥{MIN_HUMANS_PER_INSTANCE} ratings in {labels_path}"
        )
    if limit and int(limit) > 0:
        question_ids = question_ids[: int(limit)]
        print(
            f"[cascade-descriptions] --limit {limit}: "
            f"using {len(question_ids)} triple-labeled question(s)"
        )
    else:
        print(
            f"[cascade-descriptions] {len(question_ids)} triple-labeled "
            f"question(s) from {labels_path}"
        )

    out = run_pipeline.spawn(
        models=models,
        question_ids=question_ids,
        n_shots=n_shots,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        meta=meta,
        data_root=data_root,
        output_dir=output_dir,
        print_every=print_every,
    ).get()

    pending = out.get("pending_labels") or []
    if pending:
        print(f"Spawned {len(pending)} GPU worker(s): {pending}")
        print(
            "Use ``modal run --detach``; without it this process exiting "
            "stops the ephemeral app and kills those workers. "
            "Watch progress in the Modal dashboard."
        )
    print("Orchestrator:", out)
    print(
        "Download with:\n"
        "  uv run modal run download_results.py --volume-name mmar-descriptions"
    )
    if pending or out.get("skipped_labels"):
        print(
            "To fill remaining captions later:\n"
            "  uv run modal run --detach cascade/run_descriptions.py"
        )
