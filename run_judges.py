"""Judge labeled MMAR generations from ``exports/``.

Reads ``exports/labels.csv`` and ``exports/generations.csv``, joins
question / gold / audio from MMAR-meta, and writes the judged pack to
``outputs/mmar-judging`` (Modal volume ``mmar-judging``). Grades
only questions present in ``labels.csv``. Default is both with-GT and
no-GT recipes (one GPU load per judge). Shots that already have a
verdict for the same judge key are skipped. Regenerates
``difficulty.jsonl`` / ``scores.json`` after grading, then writes
``judge_accuracy.json`` (Alt-Test Average Advantage Probability).

vLLM suite / dedicated judges start Modal from this script. API judges
(gemini-3.7-flash) run locally and never open an App. Batch API judges
(gpt-5.6-luna, claude-sonnet-5) also run locally: they grade ``with_gt``
on shots in ``exports/labels.csv`` that have all three reviewer ratings.

    uv run modal run seed_volume.py --datasets none --models qwen2.5-3b
    uv run modal run --detach seed_volume.py --datasets none --models qwen3.6-35b-a3b-fp8

    # All suite judges; both gold modes; labeled questions only
    uv run run_judges.py

    # Only these judges
    uv run run_judges.py \\
      --judge-model-id qwen3-omni-instruct,phi-4-multimodal

    # Dedicated text judge (not in the suite; with-GT only)
    uv run run_judges.py \\
      --judge-model-id qwen3.6-35b-a3b-fp8

    # Audio judge, no gold (hears the clip)
    uv run run_judges.py \\
      --judge-model-id qwen3-omni-instruct \\
      --no-include-gold

    # Neutral with-GT: audio + gold, Correct/Incorrect (suite judges)
    uv run run_judges.py --grade-prompt neutral_with_gt

    # Any JUDGE_FORMATS key (text-only variant; dedicated judges OK)
    uv run run_judges.py --grade-prompt neutral_with_gt_no_audio

    # Every format in JUDGE_FORMATS
    uv run run_judges.py --grade-prompt all

    # Recompute Alt-Test scores from existing local verdicts
    uv run run_judges.py --accuracy-only

Local API judges::

    export GEMINI_API_KEY=...

    uv run run_judges.py --judge-model-id gemini-3.7-flash
    uv run run_judges.py --judge-model-id api --no-include-gold

OpenAI / Anthropic Batch API (with_gt, triple-labeled shots)::

    export OPENAI_API_KEY=...
    uv run run_judges.py --judge-model-id gpt-5.6-luna

    export ANTHROPIC_API_KEY=...
    uv run run_judges.py --judge-model-id claude-sonnet-5
    uv run run_judges.py --judge-model-id sonnet --batch-id msgbatch_abc123

Mixed (API locally while Modal vLLM runs detached)::

    uv run run_judges.py \\
      --judge-model-id gemini-3.7-flash,qwen3-omni-instruct

    # Download remote verdicts and inspect vs exports/ labels
    uv run modal run download_judges.py
    uv run python view_judges.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import modal

from aggregate import aggregate_difficulty, discover_model_labels, order_model_labels
from alt_test import DEFAULT_EPSILON, MIN_HUMANS_PER_INSTANCE, score_binary_judge
from mmar_common import judge_label, load_jsonl, resolve_path, write_json, write_jsonl
from modal_cache import (
    DEFAULT_MMAR_DATA_ROOT,
    DEFAULT_MMAR_META,
    JUDGING_MOUNT,
    VOLUME_MOUNT,
    VLLM_WHEEL_INDEX,
    hf_secret,
    judging_volume,
    volume,
)

REPO_ROOT = Path(__file__).resolve().parent
PACK_NAME = "mmar-judging"
EXPORTS_DIR = REPO_ROOT / "exports"
LOCAL_PACK_DIR = REPO_ROOT / "outputs" / PACK_NAME
REMOTE_PACK_DIR = JUDGING_MOUNT
LOCAL_EXPORTS_MOUNT = Path("/local-exports")
INGEST_DIR_NAME = "_local_ingest"
LABELS_CSV_NAME = "labels.csv"
GENERATIONS_CSV_NAME = "generations.csv"
ACCURACY_JSON_NAME = "judge_accuracy.json"
LOCAL_MMAR_DATA_ROOT = REPO_ROOT / "data" / "mmar"
LOCAL_MMAR_META = LOCAL_MMAR_DATA_ROOT / "MMAR-meta.jsonl"

app = modal.App("run-judges")


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


def load_pack_label_rows(labels_path: Path) -> list[dict]:
    """Rows with a non-empty boolean ``ratings`` list."""
    rows: list[dict] = []
    with labels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            qid = str(raw.get("question_id") or "").strip()
            model = str(raw.get("model_label") or "").strip()
            ratings = _parse_ratings_cell(raw.get("ratings"))
            if not qid or not model or not ratings:
                continue
            try:
                shot_index = int(raw.get("shot_index", 0))
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "question_id": qid,
                    "model_label": model,
                    "shot_index": shot_index,
                    "ratings": ratings,
                }
            )
    return rows


def labeled_question_ids(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(str(row["question_id"]) for row in rows))


def triple_labeled_rows(rows: list[dict]) -> list[dict]:
    """Rows with at least three reviewer ratings (Alt-Test sample)."""
    return [
        row
        for row in rows
        if len(row.get("ratings") or []) >= MIN_HUMANS_PER_INSTANCE
    ]


def _labels_path(pack_dir: Path) -> Path:
    return pack_dir / LABELS_CSV_NAME


def _require_labeled_question_ids(pack_dir: Path) -> list[str]:
    path = _labels_path(pack_dir)
    if not path.is_file():
        raise SystemExit(
            f"labels.csv not found at {path}. "
            f"Expected {EXPORTS_DIR / LABELS_CSV_NAME}."
        )
    ids = labeled_question_ids(load_pack_label_rows(path))
    if not ids:
        raise SystemExit(f"No labeled questions in {path}")
    return ids


def _exports_dir() -> Path:
    mounted = LOCAL_EXPORTS_MOUNT / LABELS_CSV_NAME
    if mounted.is_file():
        return LOCAL_EXPORTS_MOUNT
    local = EXPORTS_DIR / LABELS_CSV_NAME
    if local.is_file():
        return EXPORTS_DIR
    raise SystemExit(
        f"Exports not found. Expected {EXPORTS_DIR / LABELS_CSV_NAME} "
        f"and {EXPORTS_DIR / GENERATIONS_CSV_NAME}."
    )


def _meta_path() -> Path:
    if DEFAULT_MMAR_META.is_file():
        return DEFAULT_MMAR_META
    if LOCAL_MMAR_META.is_file():
        return LOCAL_MMAR_META
    raise SystemExit(
        f"MMAR-meta.jsonl not found at {DEFAULT_MMAR_META} or {LOCAL_MMAR_META}. "
        "Seed the volume or download MMAR locally under data/mmar/."
    )


def _data_root() -> Path:
    if DEFAULT_MMAR_DATA_ROOT.is_dir():
        return DEFAULT_MMAR_DATA_ROOT
    return LOCAL_MMAR_DATA_ROOT


def _load_mmar_meta(meta_path: Path) -> dict[str, dict]:
    rows = load_jsonl(meta_path)
    by_id: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("id") or "").strip()
        if qid:
            by_id[qid] = row
    if not by_id:
        raise SystemExit(f"No MMAR-meta rows in {meta_path}")
    return by_id


def _load_generation_rows(path: Path, labeled_ids: set[str]) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"generations.csv not found at {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            qid = str(raw.get("question_id") or "").strip()
            if not qid or qid not in labeled_ids:
                continue
            model = str(raw.get("model_label") or "").strip()
            prediction = raw.get("answer_prediction")
            if prediction is None or not model:
                continue
            try:
                shot_index = int(raw.get("shot_index", 0))
            except (TypeError, ValueError):
                continue
            generation_id = str(raw.get("generation_id") or "").strip()
            rows.append(
                {
                    "question_id": qid,
                    "model_label": model,
                    "shot_index": shot_index,
                    "answer_prediction": str(prediction),
                    "generation_id": generation_id,
                }
            )
    if not rows:
        raise SystemExit(f"No generation rows for labeled questions in {path}")
    return rows


def _resolve_audio_path(raw: object, data_root: Path) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    resolved = Path(resolve_path(data_root, text))
    if resolved.is_file():
        return str(resolved)
    return text


def _question_record(meta: dict, *, qid: str, data_root: Path) -> dict:
    question = str(meta.get("question") or "").strip()
    answer = str(meta.get("answer") or "").strip()
    if not question or not answer:
        raise SystemExit(
            f"MMAR-meta row {qid!r} is missing question or answer"
        )
    record = dict(meta)
    record["id"] = qid
    record["question"] = question
    record["answer"] = answer
    record["audio_path"] = _resolve_audio_path(meta.get("audio_path"), data_root)
    record.pop("model_output", None)
    return record


def _records_from_exports(
    exports_dir: Path,
    meta_path: Path,
    data_root: Path,
) -> dict[str, list[dict]]:
    label_rows = load_pack_label_rows(exports_dir / LABELS_CSV_NAME)
    labeled_ids = set(labeled_question_ids(label_rows))
    if not labeled_ids:
        raise SystemExit(f"No labeled questions in {exports_dir / LABELS_CSV_NAME}")
    gen_rows = _load_generation_rows(exports_dir / GENERATIONS_CSV_NAME, labeled_ids)
    missing_gen = labeled_ids - {row["question_id"] for row in gen_rows}
    if missing_gen:
        sample = ", ".join(sorted(missing_gen)[:5])
        raise SystemExit(
            f"{len(missing_gen)} labeled question(s) missing from generations.csv "
            f"(e.g. {sample})"
        )
    meta_by_id = _load_mmar_meta(meta_path)
    missing_meta = sorted(qid for qid in labeled_ids if qid not in meta_by_id)
    if missing_meta:
        sample = ", ".join(missing_meta[:5])
        raise SystemExit(
            f"{len(missing_meta)} labeled question(s) missing from MMAR-meta "
            f"at {meta_path} (e.g. {sample})"
        )

    grouped: dict[str, dict[str, dict[int, dict]]] = {}
    for row in gen_rows:
        model = row["model_label"]
        qid = row["question_id"]
        grouped.setdefault(model, {}).setdefault(qid, {})[row["shot_index"]] = row

    by_model: dict[str, list[dict]] = {}
    for model, by_qid in grouped.items():
        records: list[dict] = []
        for qid in labeled_question_ids(label_rows):
            shots_by_index = by_qid.get(qid)
            if not shots_by_index:
                continue
            shots = []
            for shot_index in sorted(shots_by_index):
                src = shots_by_index[shot_index]
                shot = {
                    "shot_index": shot_index,
                    "answer_prediction": src["answer_prediction"],
                    "correct": None,
                    "pending_grade": True,
                }
                if src["generation_id"]:
                    shot["generation_id"] = src["generation_id"]
                shots.append(shot)
            record = _question_record(meta_by_id[qid], qid=qid, data_root=data_root)
            primary = shots[0] if shots else {}
            record.update(
                {
                    "model": model,
                    "n_shots": len(shots),
                    "shots": shots,
                    "answer_prediction": primary.get("answer_prediction"),
                    "correct": None,
                    "n_shot_correct": None,
                    "shot_success_rate": None,
                    "pending_grade": True,
                }
            )
            records.append(record)
        if records:
            by_model[model] = records
    if not by_model:
        raise SystemExit(
            "No model generations to grade after joining exports + MMAR-meta"
        )
    return by_model


def materialize_exports_pack(
    dest: Path,
    *,
    exports_dir: Path | None = None,
    meta_path: Path | None = None,
    data_root: Path | None = None,
) -> dict:
    """Build/merge a judging pack from exports CSVs + MMAR-meta.

    Writes ``models/<label>/predictions.jsonl`` without ``model_output`` so
    reuse keys stay on ``answer_prediction``. Merges into ``dest`` so existing
    ``judges`` survive when shot answers are unchanged.
    """
    exports_dir = exports_dir or _exports_dir()
    meta_path = meta_path or _meta_path()
    data_root = data_root or _data_root()
    by_model = _records_from_exports(exports_dir, meta_path, data_root)
    labels = order_model_labels(list(by_model))
    question_ids = labeled_question_ids(
        load_pack_label_rows(exports_dir / LABELS_CSV_NAME)
    )
    n_shots = max(
        (
            len(record.get("shots") or [])
            for recs in by_model.values()
            for record in recs
        ),
        default=0,
    )
    now = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="mmar-judging-") as tmp:
        src = Path(tmp)
        for label in labels:
            pred = src / "models" / label / "predictions.jsonl"
            write_jsonl(pred, by_model[label], mode="w")
        shutil.copy2(exports_dir / LABELS_CSV_NAME, src / LABELS_CSV_NAME)
        write_json(
            src / "question_ids.json",
            {
                "n": len(question_ids),
                "ids": question_ids,
                "n_shots": n_shots,
                "source": "exports",
            },
        )
        write_json(
            src / "manifest.json",
            {
                "name": PACK_NAME,
                "mode": "freeform",
                "n_shots": n_shots,
                "n_questions": len(question_ids),
                "models": labels,
                "source": "exports",
                "created_at": now,
                "updated_at": now,
            },
        )
        dest.mkdir(parents=True, exist_ok=True)
        _merge_pack_from(src, dest)
    print(
        f"[run-judges] materialized models={len(labels)} "
        f"questions={len(question_ids)} n_shots={n_shots} from {exports_dir} -> {dest}"
    )
    return {
        "models": labels,
        "question_ids": question_ids,
        "n_shots": n_shots,
        "dest": str(dest),
    }


def _judge_mode_bucket(judge_key: str, entry: dict | None) -> str | None:
    from grader import judge_mode_bucket

    return judge_mode_bucket(judge_key, entry)


def _load_predictions_by_id(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    if not path.is_file():
        return by_id
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            qid = str(record.get("id") or "").strip()
            if qid:
                by_id[qid] = record
    return by_id


def _shot_for_index(record: dict, shot_index: int) -> dict | None:
    for shot in record.get("shots") or []:
        try:
            if int(shot.get("shot_index", 0)) == shot_index:
                return shot
        except (TypeError, ValueError):
            continue
    return None


def _instance_id(row: dict) -> str:
    return f"{row['question_id']}\t{row['model_label']}\t{row['shot_index']}"


def _entry_correct(entry: object) -> bool | None:
    if isinstance(entry, dict) and entry.get("correct") is not None:
        return bool(entry.get("correct"))
    return None


def _sort_judge_keys(by_judge: dict) -> list[str]:
    def _key(name: str) -> tuple[float, str]:
        rho = (by_judge.get(name) or {}).get("advantage_prob")
        rank = -float(rho) if isinstance(rho, (int, float)) else 1.0
        return (rank, name)

    return sorted(by_judge, key=_key)


def report_judge_accuracy(
    pack_dir: Path, *, epsilon: float = DEFAULT_EPSILON
) -> dict:
    """Alt-Test scores vs human ratings (Average Advantage Probability)."""
    labels_path = _labels_path(pack_dir)
    if not labels_path.is_file():
        raise SystemExit(f"labels.csv not found at {labels_path}")
    rows = load_pack_label_rows(labels_path)
    if not rows:
        raise SystemExit(f"No labeled rows in {labels_path}")

    pred_cache: dict[str, dict[str, dict]] = {}
    samples: list[tuple[str, list[bool], dict[str, dict] | None]] = []
    judge_keys: set[str] = set()

    for row in rows:
        model = row["model_label"]
        if model not in pred_cache:
            pred_cache[model] = _load_predictions_by_id(
                pack_dir / "models" / model / "predictions.jsonl"
            )
        record = pred_cache[model].get(row["question_id"])
        shot = _shot_for_index(record, row["shot_index"]) if record else None
        judges = None
        if shot is not None and isinstance(shot.get("judges"), dict):
            judges = {
                str(key): dict(entry)
                for key, entry in shot["judges"].items()
                if key
            }
            judge_keys.update(judges)
        samples.append((_instance_id(row), list(row["ratings"]), judges))

    from grader import accuracy_mode_names, grade_prompt_names

    key_mode: dict[str, str] = {}
    for key in sorted(judge_keys):
        sample_entry = next(
            (
                (judges or {}).get(key)
                for _iid, _ratings, judges in samples
                if judges and key in judges
            ),
            None,
        )
        mode = _judge_mode_bucket(
            key, sample_entry if isinstance(sample_entry, dict) else None
        )
        if mode is None:
            continue
        key_mode[key] = mode

    stats: dict[str, dict[str, dict]] = {name: {} for name in grade_prompt_names()}
    for key, mode in key_mode.items():
        instances = [
            (instance_id, ratings, _entry_correct((judges or {}).get(key)))
            for instance_id, ratings, judges in samples
        ]
        stats.setdefault(mode, {})[key] = score_binary_judge(
            instances, epsilon=epsilon
        )

    payload = {
        "pack": PACK_NAME,
        "labels_path": str(labels_path),
        "n_label_rows": len(rows),
        "n_questions": len(labeled_question_ids(rows)),
        "epsilon": float(epsilon),
        "modes": accuracy_mode_names(stats),
    }
    for mode in payload["modes"]:
        payload[mode] = stats.get(mode) or {}
    dest = pack_dir / ACCURACY_JSON_NAME
    write_json(dest, payload)
    _print_judge_accuracy(payload)
    print(f"[run-judges] wrote {dest}")
    return payload


def _fmt_rate(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.3f}"


def _print_judge_accuracy(payload: dict) -> None:
    from grader import accuracy_mode_names, grade_mode_title

    epsilon = payload.get("epsilon")
    eps_s = _fmt_rate(epsilon)
    for mode in accuracy_mode_names(payload):
        title = grade_mode_title(mode)
        print(f"\n=== judge Alt-Test ({title}, ε={eps_s}) ===")
        print("llm = judge agreement with the other two labelers (leave-one-out ACC)")
        print("hum = excluded labeler's agreement with those same two")
        print(
            f"{'judge':<52} {'n':>6} {'miss':>6} {'<3':>6} "
            f"{'ρ':>8} {'llm':>8} {'hum':>8} {'ω':>8} {'pass':>5}"
        )
        by_judge = payload.get(mode) or {}
        if not by_judge:
            print("(no verdicts)")
            continue
        for key in _sort_judge_keys(by_judge):
            row = by_judge[key]
            passed = row.get("passed")
            if passed is True:
                pass_s = "yes"
            elif passed is False:
                pass_s = "no"
            else:
                pass_s = "—"
            print(
                f"{key:<52} {row.get('n', 0):>6} {row.get('n_missing', 0):>6} "
                f"{row.get('n_skipped_lt3', 0):>6} "
                f"{_fmt_rate(row.get('advantage_prob')):>8} "
                f"{_fmt_rate(row.get('loo_agree_judge')):>8} "
                f"{_fmt_rate(row.get('loo_agree_human')):>8} "
                f"{_fmt_rate(row.get('winning_rate')):>8} {pass_s:>5}"
            )
    print()


def _gold_mode_note(include_gold: bool | None) -> str:
    from grader import grade_prompt_name, parse_grade_prompt_list

    if include_gold is None:
        return "both"
    names = parse_grade_prompt_list(include_gold=include_gold)
    return names[0] if names else grade_prompt_name(include_gold)


def _run_mode_note(prompt: str | None, include_gold: bool | None) -> str:
    if prompt and str(prompt).strip():
        return str(prompt).strip()
    return _gold_mode_note(include_gold)


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
        "alt_test",
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

# 0.28+ recommended for Qwen3.6 gated-delta hybrid / FP8 checkpoints.
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

# Suite judges load via mmar_models (same image/GPU as inference).
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
        "numpy", "scipy", "tqdm>=4.67.0"
    )
)
if EXPORTS_DIR.is_dir():
    cpu_image = cpu_image.add_local_dir(
        str(EXPORTS_DIR), remote_path=str(LOCAL_EXPORTS_MOUNT)
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


def _bootstrap_pack() -> dict:
    """Materialize exports + MMAR-meta onto the judging volume, keeping verdicts."""
    dest = _pack_dir()
    built = materialize_exports_pack(dest)
    judging_volume.commit()
    if not dest.is_dir() or not (dest / "manifest.json").is_file():
        raise SystemExit(f"Failed to materialize pack at {dest}")
    return built


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
    volumes={VOLUME_MOUNT: volume, JUDGING_MOUNT: judging_volume},
)
def prepare_judges(
    judge_model_id: str = "",
    make_primary: bool = False,
    models: str = "all",
) -> dict:
    """Validate the pack is freeform and resolve gradees / judges."""
    volume.reload()
    judging_volume.reload()
    built = _bootstrap_pack()
    pack_dir = _pack_dir()

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

    labels = list(built.get("models") or discover_model_labels(pack_dir, manifest=manifest))
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

    question_ids = _require_labeled_question_ids(pack_dir)
    print(
        f"[run-judges] labeled questions={len(question_ids)} "
        f"from {_labels_path(pack_dir)}"
    )

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
        "question_ids": question_ids,
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
    volumes={VOLUME_MOUNT: volume, JUDGING_MOUNT: judging_volume},
    secrets=[hf_secret],
    memory=32768,
)
def grade_with_judge(
    judge_model_id: str,
    primary_judge: str,
    model_labels: list[str],
    batch_size: int | None = None,
    force: bool = False,
    include_gold: bool | None = None,
    prompt: str | None = None,
    make_primary: bool = False,
    n_questions: int | None = None,
    question_ids: list[str] | None = None,
) -> dict:
    """Grade with one dedicated text judge; merge into predictions + manifest."""
    from grader import (
        DEFAULT_GRADE_PROMPT,
        get_judge_format,
        grade_predictions_file,
        iter_grade_modes,
        judge_is_audio_model,
        load_grader,
        resolve_grade_judge_key,
        resolve_judge_batch_size,
        resolve_judge_model_id,
    )

    volume.reload()
    judging_volume.reload()

    pack_dir = _pack_dir()
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")

    manifest = _load_manifest(pack_dir)
    _assert_freeform_run(manifest)

    judge_model_id = resolve_judge_model_id(judge_model_id)
    handle = load_grader(judge_model_id)
    audio_ok = judge_is_audio_model(handle=handle)
    per_model: dict[str, dict] = {}
    modes: list[dict] = []
    last_key = None
    last_prompt = DEFAULT_GRADE_PROMPT
    last_gold: bool | None = include_gold
    for prompt_name, gold_flag in iter_grade_modes(
        prompt=prompt, include_gold=include_gold
    ):
        fmt = get_judge_format(prompt_name, include_gold=gold_flag)
        if fmt.audio_included and not audio_ok:
            print(
                f"[run-judges] skipping audio format {prompt_name} "
                f"for text judge {judge_model_id}"
            )
            continue
        key = resolve_grade_judge_key(
            handle, prompt=prompt_name, include_gold=gold_flag
        )
        last_key = key
        last_prompt = prompt_name
        last_gold = gold_flag
        modes.append(
            {"prompt": prompt_name, "include_gold": gold_flag, "judge_key": key}
        )
        effective_batch_size = resolve_judge_batch_size(judge_model_id, batch_size)
        for label in model_labels:
            predictions_path = pack_dir / "models" / label / "predictions.jsonl"
            print(
                f"[run-judges] {label} with {key} "
                f"(primary={primary_judge}, batch_size={effective_batch_size}"
                f"{f', n_questions={n_questions}' if n_questions is not None else ''}"
                f", n_labeled={len(question_ids) if question_ids is not None else 'all'}) "
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
                include_gold=gold_flag,
                make_primary=make_primary,
                n_questions=n_questions,
                question_ids=question_ids,
            )
            judging_volume.commit()
            print(f"[run-judges] {label}:", per_model[f"{label}/{key}"])
        manifest = _merge_judge_manifest(
            manifest,
            model_id=judge_model_id,
            judge_key=key,
            primary=primary_judge,
            make_primary=make_primary,
            prompt=prompt_name,
            include_gold=gold_flag,
        )
    write_json(pack_dir / "manifest.json", manifest)
    judging_volume.commit()
    return {
        "status": "ok",
        "pack": PACK_NAME,
        "judge_model_id": judge_model_id,
        "judge_label": last_key,
        "primary_judge": manifest.get("primary_judge"),
        "by_model": per_model,
        "judges": manifest.get("judges"),
        "prompt": last_prompt,
        "include_gold": last_gold if include_gold is not None else None,
        "modes": modes,
        "n_questions": n_questions,
        "n_labeled": len(question_ids) if question_ids is not None else None,
    }


def _grade_suite_judge(
    judge_label: str,
    *,
    model_labels: list[str],
    include_gold: bool | None,
    force: bool,
    batch_size: int | None,
    n_questions: int | None = None,
    question_ids: list[str] | None = None,
    prompt: str | None = None,
) -> dict:
    from grader import (
        compose_judge_key,
        grade_predictions_file,
        iter_grade_modes,
        load_grader,
    )

    volume.reload()
    judging_volume.reload()
    pack_dir = _pack_dir()
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")
    _assert_freeform_run(_load_manifest(pack_dir))

    handle = load_grader(judge_label)
    model_id = handle.get("model_id") or judge_label
    per_model: dict[str, dict] = {}
    keys: list[str] = []
    modes: list[dict] = []
    gradees = [label for label in model_labels if label != judge_label]
    for prompt_name, gold_flag in iter_grade_modes(
        prompt=prompt, include_gold=include_gold
    ):
        key = compose_judge_key(
            judge_label, prompt=prompt_name, include_gold=gold_flag
        )
        if key not in keys:
            keys.append(key)
            modes.append(
                {
                    "prompt": prompt_name,
                    "include_gold": gold_flag,
                    "judge_key": key,
                }
            )
        for gradee in gradees:
            predictions_path = pack_dir / "models" / gradee / "predictions.jsonl"
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
                include_gold=gold_flag,
                sidecar_path=sidecar,
                n_questions=n_questions,
                question_ids=question_ids,
            )
            judging_volume.commit()
            print(f"[run-judges-rr] {gradee}/{key}:", per_model[f"{gradee}/{key}"])
    return {
        "status": "ok",
        "judge_label": judge_label,
        "model_id": model_id,
        "judge_keys": keys,
        "gradees": gradees,
        "by_model": per_model,
        "include_gold": include_gold,
        "modes": modes,
        "prompts": [item["prompt"] for item in modes],
        "n_questions": n_questions,
        "n_labeled": len(question_ids) if question_ids is not None else None,
    }


_SUITE_GRADE_KW = dict(
    image=large_mm_image,
    timeout=12 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, JUDGING_MOUNT: judging_volume},
    secrets=[hf_secret],
    memory=65536,
)


@app.function(gpu="L40S", **_SUITE_GRADE_KW)
def grade_suite_l40s(
    judge_label: str,
    model_labels: list[str],
    include_gold: bool | None = None,
    prompt: str | None = None,
    force: bool = False,
    batch_size: int | None = None,
    n_questions: int | None = None,
    question_ids: list[str] | None = None,
) -> dict:
    return _grade_suite_judge(
        judge_label,
        model_labels=model_labels,
        include_gold=include_gold,
        prompt=prompt,
        force=force,
        batch_size=batch_size,
        n_questions=n_questions,
        question_ids=question_ids,
    )


@app.function(gpu="H100", **_SUITE_GRADE_KW)
def grade_suite_h100(
    judge_label: str,
    model_labels: list[str],
    include_gold: bool | None = None,
    prompt: str | None = None,
    force: bool = False,
    batch_size: int | None = None,
    n_questions: int | None = None,
    question_ids: list[str] | None = None,
) -> dict:
    return _grade_suite_judge(
        judge_label,
        model_labels=model_labels,
        include_gold=include_gold,
        prompt=prompt,
        force=force,
        batch_size=batch_size,
        n_questions=n_questions,
        question_ids=question_ids,
    )


_SUITE_GRADE_FNS = {
    "qwen2.5-omni-7b": grade_suite_l40s,
    "phi-4-multimodal": grade_suite_l40s,
    "gemma-4-e4b": grade_suite_l40s,
    "qwen3-omni-instruct": grade_suite_h100,
    "nemotron-3-nano-omni": grade_suite_h100,
}


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume, JUDGING_MOUNT: judging_volume},
)
def merge_round_robin(
    model_labels: list[str],
    judge_entries: list[dict],
    make_primary: bool = False,
    primary_judge: str | None = None,
) -> dict:
    """Fold judge sidecars into predictions.jsonl and append manifest entries."""
    from grader import apply_judge_partials

    judging_volume.reload()
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
    volumes={VOLUME_MOUNT: volume, JUDGING_MOUNT: judging_volume},
)
def ingest_local_pack() -> dict:
    """Merge a local-entrypoint upload of API verdicts into the volume pack."""
    judging_volume.reload()
    dest = _pack_dir()
    src = dest / INGEST_DIR_NAME
    if not src.is_dir():
        return {"status": "missing", "pack": PACK_NAME}
    _merge_pack_from(src, dest)
    shutil.rmtree(src, ignore_errors=True)
    judging_volume.commit()
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
    volumes={VOLUME_MOUNT: volume, JUDGING_MOUNT: judging_volume},
)
def run_aggregate() -> dict:
    judging_volume.reload()
    pack_dir = _pack_dir()
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")
    result = _aggregate_pack(pack_dir)
    judging_volume.commit()
    return result


@app.function(
    image=cpu_image,
    timeout=30 * 60,
    volumes={VOLUME_MOUNT: volume, JUDGING_MOUNT: judging_volume},
)
def run_accuracy(epsilon: float = DEFAULT_EPSILON) -> dict:
    judging_volume.reload()
    pack_dir = _pack_dir()
    if not pack_dir.is_dir():
        raise SystemExit(f"Pack not found: {pack_dir}")
    result = report_judge_accuracy(pack_dir, epsilon=epsilon)
    judging_volume.commit()
    return result


def _download_pack() -> None:
    from download_judges import download_judges

    saved = download_judges(local_dir=LOCAL_PACK_DIR)
    print(f"[run-judges] downloaded pack -> {saved}")


def _bootstrap_local_pack(*, need_audio: bool = False) -> Path:
    if need_audio:
        from view_mmar import DEFAULT_AUDIO_DIR, ensure_mmar_audio

        ensure_mmar_audio(DEFAULT_AUDIO_DIR)
    materialize_exports_pack(LOCAL_PACK_DIR)
    if not (LOCAL_PACK_DIR / "manifest.json").is_file():
        raise SystemExit(f"Failed to materialize pack at {LOCAL_PACK_DIR}")
    return LOCAL_PACK_DIR


def _require_local_pack(*, need_audio: bool = False) -> Path:
    return _bootstrap_local_pack(need_audio=need_audio)


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
    prefix = f"/{INGEST_DIR_NAME}"
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
    with judging_volume.batch_upload(force=True) as batch:
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
    include_gold: bool | None,
    force: bool,
    batch_size: int | None,
    n_questions: int | None,
    question_ids: list[str] | None,
    primary_judge: str,
    dedicated_make_primary: bool,
    prompt: str | None = None,
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
                    include_gold=include_gold,
                    prompt=prompt,
                    force=force,
                    batch_size=batch_size,
                    n_questions=n_questions,
                    question_ids=question_ids,
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
                    include_gold=include_gold,
                    prompt=prompt,
                    make_primary=this_primary,
                    n_questions=n_questions,
                    question_ids=question_ids,
                ),
            )
        )
    return suite_handles, dedicated_handles


def _collect_remote_judges(
    suite_handles: list[tuple[str, object]],
    dedicated_handles: list[tuple[dict, object]],
) -> tuple[list[dict], list[dict], list[dict]]:
    from grader import compose_judge_key, gold_mode_flags, grade_prompt_name

    grade_results: list[dict] = []
    judge_entries: list[dict] = []
    for label, handle in suite_handles:
        result = handle.get()
        grade_results.append(result)
        print(f"Judge {label}:", result)
        model_id = result.get("model_id") or label
        modes = list(result.get("modes") or [])
        if not modes:
            include_gold = result.get("include_gold")
            for gold_flag in gold_mode_flags(
                None if include_gold is None else bool(include_gold)
            ):
                prompt_name = grade_prompt_name(gold_flag)
                modes.append(
                    {
                        "prompt": prompt_name,
                        "include_gold": gold_flag,
                        "judge_key": compose_judge_key(
                            label, prompt=prompt_name, include_gold=gold_flag
                        ),
                    }
                )
        for mode in modes:
            judge_entries.append(
                {
                    "judge_key": mode.get("judge_key")
                    or compose_judge_key(
                        label,
                        prompt=mode.get("prompt"),
                        include_gold=bool(mode.get("include_gold")),
                    ),
                    "model_id": model_id,
                    "prompt": mode.get("prompt"),
                    "include_gold": mode.get("include_gold"),
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
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
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
    accuracy = run_accuracy.remote(epsilon=epsilon)
    print("Judge Alt-Test:", accuracy)
    return ingest_result, merge, agg, accuracy


@app.function(
    image=cpu_image,
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT: volume, JUDGING_MOUNT: judging_volume},
)
def run_judges_pipeline(
    judge_model_id: str = "",
    models: str = "all",
    make_primary: bool = False,
    force: bool = False,
    batch_size: int | None = None,
    skip_aggregate: bool = False,
    include_gold: bool | None = None,
    prompt: str | None = None,
    n_questions: int | None = None,
    ingest_local: bool = False,
    epsilon: float = DEFAULT_EPSILON,
) -> dict:
    """Remote orchestrator so a detached App stays alive across GPU phases."""
    from grader import parse_grade_prompt_list

    if n_questions is not None and int(n_questions) < 0:
        n_questions = None
    prompts = parse_grade_prompt_list(prompt, include_gold=include_gold)
    prep = prepare_judges.remote(
        judge_model_id=(judge_model_id or "").strip(),
        make_primary=make_primary,
        models=models,
    )
    suite = list(prep.get("suite_judges") or [])
    dedicated = list(prep.get("dedicated_judges") or [])
    gradees = list(prep["model_labels"])
    question_ids = list(prep.get("question_ids") or [])
    first_new = prep.get("first_new")
    print(
        f"[run-judges] pipeline prep primary={prep['primary_judge']} "
        f"(existing_primary={prep['existing_primary']}) "
        f"existing_judges={prep['existing_judges']} "
        f"n_labeled={len(question_ids)} "
        f"gold={_run_mode_note(prompt, include_gold)}"
    )
    suite_handles, dedicated_handles = _spawn_remote_judges(
        suite=suite,
        dedicated=dedicated,
        gradees=gradees,
        include_gold=include_gold,
        prompt=prompt,
        force=force,
        batch_size=batch_size,
        n_questions=n_questions,
        question_ids=question_ids,
        primary_judge=prep["primary_judge"],
        dedicated_make_primary=bool(make_primary)
        and any(first_new == entry["label"] for entry in dedicated),
    )
    n_remote = len(suite_handles) + len(dedicated_handles)
    if n_remote:
        print(f"[run-judges] {n_remote} remote judge(s) running")
    grade_results, dedicated_grades, judge_entries = _collect_remote_judges(
        suite_handles, dedicated_handles
    )
    _ingest, merge, agg, accuracy = _finish_remote_pack(
        suite=suite,
        gradees=gradees,
        judge_entries=judge_entries,
        suite_make_primary=bool(make_primary) and first_new in suite,
        primary_judge=prep.get("primary_judge"),
        skip_aggregate=skip_aggregate,
        ingest=ingest_local,
        epsilon=epsilon,
    )
    return {
        "prepare": prep,
        "grade": grade_results,
        "dedicated": dedicated_grades,
        "merge": merge,
        "aggregate": agg,
        "accuracy": accuracy,
        "prompts": prompts,
    }


def _run_judges(
    *,
    judge_model_id: str = "",
    models: str = "all",
    make_primary: bool = False,
    force: bool = False,
    batch_size: int | None = None,
    skip_aggregate: bool = False,
    include_gold: bool | None = None,
    prompt: str | None = None,
    n_questions: int | None = None,
    qps: float = 4.0,
    max_workers: int = 8,
    timeout: float = 180.0,
    retries: int = 20,
    retry_interval: float = 1.0,
    epsilon: float = DEFAULT_EPSILON,
    batch_id: str = "",
    batch_poll_interval: float = 30.0,
) -> dict:
    from grader import (
        JUDGE_FORMATS,
        compose_judge_key,
        iter_grade_modes,
        parse_grade_prompt_list,
        require_audio_nongold_judge,
        resolve_grade_allowed_ids,
    )
    from mmar_api import (
        BATCH_API_GRADE_PROMPT,
        grade_pack_with_api_judges,
        grade_pack_with_batch_api,
        split_api_judges,
    )

    if n_questions is not None and int(n_questions) < 0:
        n_questions = None

    suite, dedicated, api, first_new = _select_judges(judge_model_id)
    live_api, batch_api = split_api_judges(api)
    modes = iter_grade_modes(prompt=prompt, include_gold=include_gold)
    needs_audio = any(JUDGE_FORMATS[name].audio_included for name, _ in modes)
    if not live_api and not suite and not dedicated:
        needs_audio = False
    audio_only = bool(modes) and all(
        JUDGE_FORMATS[name].audio_included for name, _ in modes
    )
    if audio_only:
        if dedicated:
            for entry in dedicated:
                require_audio_nongold_judge(
                    entry["model_id"],
                    include_gold=False,
                    audio_required=True,
                )
        elif not suite and not live_api:
            raise SystemExit(
                "Audio-only formats need an audio-capable judge "
                f"(selected: {', '.join(name for name, _ in modes)})"
            )

    needs_modal = bool(suite or dedicated)
    prompts = parse_grade_prompt_list(prompt, include_gold=include_gold)
    pack_dir: Path | None = None
    gradees: list[str] = []
    question_ids: list[str] = []
    existing_primary = None
    mode = "freeform"
    local_api = bool(live_api or batch_api)
    if local_api or not needs_modal:
        pack_dir = _require_local_pack(need_audio=needs_audio and bool(live_api))
        manifest = _load_manifest(pack_dir)
        mode = _assert_freeform_run(manifest)
        gradees = _local_model_labels(models)
        question_ids = _require_labeled_question_ids(pack_dir)
        existing_primary = manifest.get("primary_judge")
    if make_primary:
        primary = first_new
    else:
        primary = existing_primary or first_new

    api_make_primary = bool(make_primary) and first_new in api

    print(
        f"[run-judges] pack={PACK_NAME} mode={mode} "
        f"suite_judges={suite} dedicated_judges={[d['label'] for d in dedicated]} "
        f"api_judges={live_api} batch_judges={batch_api} "
        f"gradees={gradees or '(modal)'} primary={primary} "
        f"(existing_primary={existing_primary}) "
        f"gold={_run_mode_note(prompt, include_gold)} prompts={prompts} "
        f"n_questions={n_questions} n_labeled={len(question_ids) or '(modal)'} "
        f"force={force}"
    )

    api_results: list[dict] = []

    def _run_api() -> None:
        nonlocal api_results
        if not live_api:
            return
        if pack_dir is None:
            raise SystemExit(
                "API judges require exports/ plus a local pack at outputs/mmar-judging"
            )
        labeled = question_ids or _require_labeled_question_ids(pack_dir)
        collected: list[dict] = []
        first_prompt, first_gold = modes[0]
        first_api_key = compose_judge_key(
            live_api[0], prompt=first_prompt, include_gold=first_gold
        )
        set_primary = api_make_primary or (
            not existing_primary and not suite and not dedicated and not batch_api
        )
        for prompt_name, gold_flag in modes:
            collected.extend(
                asyncio.run(
                    grade_pack_with_api_judges(
                        pack_dir,
                        labels=live_api,
                        model_labels=gradees,
                        prompts=[prompt_name],
                        include_gold=gold_flag,
                        force=force,
                        n_questions=n_questions,
                        question_ids=labeled,
                        make_primary=set_primary and (prompt_name, gold_flag) == modes[0],
                        primary_judge=first_api_key if set_primary else primary,
                        qps=qps,
                        max_workers=max_workers,
                        timeout=timeout,
                        retries=retries,
                        retry_interval=retry_interval,
                    )
                )
            )
        api_results.extend(collected)
        first_key = str(collected[0].get("judge_key") or first_api_key)
        _merge_api_manifest(
            pack_dir,
            collected,
            make_primary=set_primary,
            primary_judge=first_key or primary,
            update_primary=set_primary,
        )

    def _run_batch() -> None:
        nonlocal api_results
        if not batch_api:
            return
        if pack_dir is None:
            raise SystemExit(
                "Batch API judges require exports/ plus a local pack "
                "at outputs/mmar-judging"
            )
        rows = triple_labeled_rows(load_pack_label_rows(_labels_path(pack_dir)))
        if not rows:
            raise SystemExit(
                f"No {LABELS_CSV_NAME} rows with {MIN_HUMANS_PER_INSTANCE} "
                "reviewer ratings"
            )
        qids = labeled_question_ids(rows)
        allowed = resolve_grade_allowed_ids(
            qids, n_questions=n_questions
        )
        if allowed is not None:
            rows = [row for row in rows if row["question_id"] in allowed]
        if not rows:
            raise SystemExit(
                "No triple-labeled shots remain after --n-questions filtering"
            )
        if prompt and str(prompt).strip() not in {"", BATCH_API_GRADE_PROMPT}:
            print(
                f"[run-judges] Batch API judges use {BATCH_API_GRADE_PROMPT} "
                "only; ignoring other --grade-prompt values"
            )
        first_batch_key = compose_judge_key(
            batch_api[0], prompt=BATCH_API_GRADE_PROMPT, include_gold=True
        )
        set_primary = api_make_primary or (
            not existing_primary and not suite and not dedicated and not live_api
        )
        collected: list[dict] = []
        for index, label in enumerate(batch_api):
            this_primary = set_primary and index == 0
            result = grade_pack_with_batch_api(
                pack_dir,
                label=label,
                model_labels=gradees,
                labeled_rows=rows,
                prompt=BATCH_API_GRADE_PROMPT,
                force=force,
                make_primary=this_primary,
                primary_judge=first_batch_key if set_primary else primary,
                poll_interval=batch_poll_interval,
                batch_id=(batch_id or "").strip() or None,
            )
            collected.append(result)
            print("Batch graded:", result)
        api_results.extend(collected)
        first_key = str(collected[0].get("judge_key") or first_batch_key)
        _merge_api_manifest(
            pack_dir,
            collected,
            make_primary=set_primary,
            primary_judge=first_key or primary,
            update_primary=set_primary,
        )

    if not needs_modal:
        _run_api()
        _run_batch()
        if pack_dir is None:
            raise SystemExit(f"Pack not found at {LOCAL_PACK_DIR}")
        agg = None if skip_aggregate else _aggregate_pack(pack_dir)
        accuracy = report_judge_accuracy(pack_dir, epsilon=epsilon)
        return {
            "prepare": None,
            "grade": [],
            "dedicated": [],
            "api": api_results,
            "merge": None,
            "aggregate": agg,
            "accuracy": accuracy,
        }

    run_judges_pipeline.spawn(
        judge_model_id=(judge_model_id or "").strip(),
        models=models,
        make_primary=make_primary,
        force=force,
        batch_size=batch_size,
        skip_aggregate=skip_aggregate,
        include_gold=include_gold,
        prompt=prompt,
        n_questions=n_questions,
        ingest_local=local_api,
        epsilon=epsilon,
    )
    dashboard = app.get_dashboard_url()
    if dashboard:
        print(f"[run-judges] Modal GPU pipeline started (detached): {dashboard}")
    else:
        print("[run-judges] Modal GPU pipeline started (detached)")

    _run_api()
    _run_batch()
    if local_api and pack_dir is not None:
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
        help="Comma-separated judges, 'api' for live API judges, "
        "or 'batch' for OpenAI/Anthropic Batch API judges",
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
    parser.add_argument(
        "--include-gold",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Grade with-GT only (--include-gold) or no-GT only "
        "(--no-include-gold). Default: both. Ignored when --grade-prompt is set.",
    )
    from grader import grade_prompt_names

    names = ", ".join(grade_prompt_names()) or "(none)"
    parser.add_argument(
        "--grade-prompt",
        default="",
        help="Judge recipe from JUDGE_FORMATS "
        f"({names}), comma-separated, or 'all'. "
        "Overrides --include-gold. Default: both with_gt and free.",
    )
    parser.add_argument("--n-questions", type=int, default=None)
    parser.add_argument(
        "--accuracy-only",
        action="store_true",
        help="Recompute judge_accuracy.json (Alt-Test) from the local pack; "
        "do not grade.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Alt-Test cost-benefit penalty for winning rate (default: 0.15). "
        "Average advantage probability ρ does not use this.",
    )
    parser.add_argument("--qps", type=float, default=4.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-interval", type=float, default=1.0)
    parser.add_argument(
        "--batch-id",
        default="",
        help="Resume an existing OpenAI (`batch_…`) or Anthropic "
        "(`msgbatch_…`) Batch API job instead of submitting a new one",
    )
    parser.add_argument(
        "--batch-poll-interval",
        type=float,
        default=30.0,
        help="Seconds between Batch API status polls (default: 30)",
    )
    return parser.parse_args()


def _run_judges_from_args(args: argparse.Namespace) -> dict:
    if args.accuracy_only:
        pack_dir = _require_local_pack()
        return report_judge_accuracy(pack_dir, epsilon=args.epsilon)
    return _run_judges(
        judge_model_id=args.judge_model_id,
        models=args.models,
        make_primary=args.make_primary,
        force=args.force,
        batch_size=args.batch_size,
        skip_aggregate=args.skip_aggregate,
        include_gold=args.include_gold,
        prompt=(args.grade_prompt or "").strip() or None,
        n_questions=args.n_questions,
        qps=args.qps,
        max_workers=args.max_workers,
        timeout=args.timeout,
        retries=args.retries,
        retry_interval=args.retry_interval,
        epsilon=args.epsilon,
        batch_id=args.batch_id,
        batch_poll_interval=args.batch_poll_interval,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.accuracy_only:
        _run_judges_from_args(args)
    else:
        suite, dedicated, _api, _first = _select_judges(args.judge_model_id)
        if suite or dedicated:
            # Show image-build / mount progress; otherwise App.run() is silent.
            with modal.enable_output():
                with app.run(detach=True):
                    _run_judges_from_args(args)
        else:
            _run_judges_from_args(args)


