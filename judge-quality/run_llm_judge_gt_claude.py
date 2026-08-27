"""LLM-as-judge with gold, 5 shots, all 1000 MMAR questions — Claude batch.

Same setup as ``run_llm_judge_gt.py`` (``neutral_with_gt_no_audio``, every
``shot_index`` 0–4, 3-sample majority vote) but the judge is
``claude-sonnet-5`` via Anthropic's Message Batches API. Runs locally;
no Modal container is started.

Before grading, overlays test-taker generations into the local pack
from the same sources as ``run_llm_judge_gt.py`` (first source wins,
existing per-shot verdicts are kept when the answer is unchanged):

    outputs/mmar-freeform-thinking                  # run_experiment.py download
    outputs/exp-mmar-question-difficulty/<run_id>   # legacy downloads
    outputs/mmar-freeform-5-shot-thinking           # API pack download
    outputs/judge-quality/llm-judge-gt              # already in the pack

Question ids come from ``outputs/mmar-freeform-thinking/question_ids.json``
when that file exists.

Writes verdicts into the local ``llm-judge-gt`` pack (default:
``outputs/judge-quality/llm-judge-gt``). If that directory is missing,
the pack is pulled from the ``mmar-judging`` volume with ``modal volume
get`` (CLI only). Existing Qwen GT verdicts are kept; Claude is added as
a second judge and does not steal ``primary_judge`` unless you pass
``--make-primary``.

Resume is the default: already-graded shots for this judge key are
skipped. Pass ``--force`` to replace them. ``--batch-id`` resumes an
in-flight Anthropic batch.

Usage::

    export ANTHROPIC_API_KEY=...

    uv run python judge-quality/run_llm_judge_gt_claude.py
    uv run python judge-quality/run_llm_judge_gt_claude.py --force
    uv run python judge-quality/run_llm_judge_gt_claude.py --models qwen3-omni
    uv run python judge-quality/run_llm_judge_gt_claude.py \\
      --batch-id msgbatch_abc123
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aggregate import aggregate_difficulty, discover_model_labels, order_model_labels
from grader import compose_judge_key, remaining_grade_work
from mmar_api import grade_pack_with_batch_api
from mmar_common import recompute_multi_judge_scores, write_json, write_jsonl
from modal_cache import JUDGING_VOLUME_NAME

PACK_NAME = "llm-judge-gt"
DEFAULT_PACK_DIR = _REPO_ROOT / "outputs" / "judge-quality" / PACK_NAME
LOCAL_FREEFORM_THINKING_DIR = _REPO_ROOT / "outputs" / "mmar-freeform-thinking"
LOCAL_RESULTS_DIR = _REPO_ROOT / "outputs" / "exp-mmar-question-difficulty"
LOCAL_API_PACK_DIR = _REPO_ROOT / "outputs" / "mmar-freeform-5-shot-thinking"
FULL_MMAR_RUN_ID = "20260807T145000Z"
OPEN_ENDED_RUN_ID = "20260816T050944Z"
JUDGE_LABEL = "claude-sonnet-5"
GRADE_PROMPT = "neutral_with_gt_no_audio"
N_SHOTS = 5
SHOT_INDICES = tuple(range(N_SHOTS))
N_SAMPLES = 3
# Same as run_llm_judge_gt.py: T=0 would collapse the 3 votes.
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_question_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        ids = payload.get("ids") or []
    elif isinstance(payload, list):
        ids = payload
    else:
        return []
    return [str(qid) for qid in ids if str(qid).strip()]


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


def _source_model_dirs(pack_dir: Path) -> list[tuple[str, Path]]:
    """(source_tag, models_dir) in overlay order: first source wins per model."""
    return [
        ("mmar-freeform-thinking", LOCAL_FREEFORM_THINKING_DIR / "models"),
        (
            FULL_MMAR_RUN_ID,
            LOCAL_RESULTS_DIR / FULL_MMAR_RUN_ID / "models",
        ),
        (
            OPEN_ENDED_RUN_ID,
            LOCAL_RESULTS_DIR / OPEN_ENDED_RUN_ID / "models",
        ),
        ("mmar-freeform-5-shot-thinking", LOCAL_API_PACK_DIR / "models"),
        ("llm-judge-gt-pack", pack_dir / "models"),
    ]


def _question_id_candidates(pack_dir: Path) -> list[tuple[str, Path]]:
    return [
        ("mmar-freeform-thinking", LOCAL_FREEFORM_THINKING_DIR / "question_ids.json"),
        (
            FULL_MMAR_RUN_ID,
            LOCAL_RESULTS_DIR / FULL_MMAR_RUN_ID / "question_ids.json",
        ),
        ("llm-judge-gt-pack", pack_dir / "question_ids.json"),
    ]


def _resolve_question_ids(pack_dir: Path) -> tuple[list[str], str]:
    for tag, path in _question_id_candidates(pack_dir):
        ids = _load_question_ids(path)
        if ids:
            print(f"[llm-judge-gt-claude] question ids: {len(ids)} from {tag} ({path})")
            return ids, tag
    raise SystemExit(
        "No question_ids.json found. Download the experiment pack:\n"
        "  uv run modal run download_results.py\n"
        f"Expected {LOCAL_FREEFORM_THINKING_DIR / 'question_ids.json'}"
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


def _overlay_pack(pack_dir: Path, models: str = "all") -> dict[str, Any]:
    """Copy 5-shot freeform answers into the local llm-judge-gt pack."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    question_ids, ids_source = _resolve_question_ids(pack_dir)
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for tag, models_dir in _source_model_dirs(pack_dir):
        if not models_dir.is_dir():
            print(f"[llm-judge-gt-claude] skip missing source {tag}: {models_dir}")
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
                print(f"[llm-judge-gt-claude] {tag}/{label}: no shot overlap")
                continue
            by_model[label] = kept
            sources[label] = tag
            n_shot_rows = sum(len(row.get("shots") or []) for row in kept.values())
            print(
                f"[llm-judge-gt-claude] {tag}/{label}: {len(kept)}/{len(question_ids)} "
                f"questions, {n_shot_rows} shots"
            )

    requested = [part.strip() for part in str(models or "all").split(",") if part.strip()]
    write_model = dict(by_model)
    if requested and requested != ["all"]:
        missing = [label for label in requested if label not in write_model]
        if missing:
            raise SystemExit(
                f"Requested model(s) not found: {missing}. "
                f"Available: {order_model_labels(list(write_model))}"
            )
        write_model = {label: write_model[label] for label in requested}

    labels = order_model_labels(list(write_model))
    if not labels:
        raise SystemExit(
            "No test-taker generations found. Download the experiment pack:\n"
            "  uv run modal run download_results.py\n"
            f"Expected {LOCAL_FREEFORM_THINKING_DIR}"
        )

    now = datetime.now(timezone.utc).isoformat()
    for label in labels:
        ordered = [
            write_model[label][qid] for qid in question_ids if qid in write_model[label]
        ]
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
    manifest_path = pack_dir / "manifest.json"
    existing = _load_json(manifest_path)
    disk_labels = discover_model_labels(pack_dir, manifest=existing)
    manifest_models = order_model_labels(
        list(dict.fromkeys([*disk_labels, *labels]))
    )
    manifest = {
        "name": PACK_NAME,
        "mode": "freeform",
        "n_shots": N_SHOTS,
        "n_questions": len(question_ids),
        "models": manifest_models,
        "sources": {**(existing.get("sources") or {}), **sources},
        "grade_prompt": GRADE_PROMPT,
        "n_samples": N_SAMPLES,
        "temperature": JUDGE_TEMPERATURE,
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
    }
    for key in (
        "scoring",
        "grader_model_id",
        "judges",
        "primary_judge",
        "judge_model_id",
        "graded_at",
    ):
        if existing.get(key) is not None:
            manifest[key] = existing[key]
    write_json(manifest_path, manifest)
    print(
        f"[llm-judge-gt-claude] prepared models={manifest_models} "
        f"questions={len(question_ids)} overlay={labels} -> {pack_dir}"
    )
    return {
        "pack": PACK_NAME,
        "model_labels": labels if requested and requested != ["all"] else manifest_models,
        "question_ids": question_ids,
        "sources": sources,
        "n_questions": len(question_ids),
    }


def _download_pack(pack_dir: Path, *, force: bool = False) -> Path:
    dest = pack_dir.parent
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "modal",
        "volume",
        "get",
        JUDGING_VOLUME_NAME,
        PACK_NAME,
        str(dest),
    ]
    if force:
        cmd.append("--force")
    print(f"[llm-judge-gt-claude] downloading volume:{JUDGING_VOLUME_NAME}/{PACK_NAME} -> {pack_dir}")
    subprocess.run(cmd, check=True)
    if not pack_dir.is_dir():
        raise SystemExit(f"Download finished but pack is missing at {pack_dir}")
    return pack_dir


def _require_pack(pack_dir: Path, *, download: bool = True) -> Path:
    if pack_dir.is_dir() and (pack_dir / "manifest.json").is_file():
        return pack_dir
    has_local_generations = (LOCAL_FREEFORM_THINKING_DIR / "models").is_dir()
    if not download:
        if has_local_generations:
            return pack_dir
        raise SystemExit(
            f"Pack not found at {pack_dir}. Download with:\n"
            "  uv run modal run judge-quality/download_judge_quality.py "
            f"--pack {PACK_NAME}\n"
            "or generations with:\n"
            "  uv run modal run download_results.py"
        )
    try:
        return _download_pack(pack_dir, force=pack_dir.exists())
    except subprocess.CalledProcessError as exc:
        if has_local_generations:
            print(
                "[llm-judge-gt-claude] judging-volume pack missing; "
                f"will overlay {LOCAL_FREEFORM_THINKING_DIR}"
            )
            return pack_dir
        raise SystemExit(
            f"Pack not found at {pack_dir} and download failed "
            f"(exit {exc.returncode}). Prepare it with:\n"
            "  uv run modal run --detach judge-quality/run_llm_judge_gt.py\n"
            "then download:\n"
            "  uv run modal run judge-quality/download_judge_quality.py "
            f"--pack {PACK_NAME}\n"
            "or download generations:\n"
            "  uv run modal run download_results.py"
        ) from exc


def _model_labels(pack_dir: Path, models: str) -> list[str]:
    manifest = _load_json(pack_dir / "manifest.json")
    labels = discover_model_labels(pack_dir, manifest=manifest)
    requested = [part.strip() for part in str(models or "all").split(",") if part.strip()]
    if requested and requested != ["all"]:
        missing = [label for label in requested if label not in labels]
        if missing:
            raise SystemExit(
                f"Requested model(s) not found: {missing}. "
                f"Available: {order_model_labels(labels)}"
            )
        labels = requested
    labels = order_model_labels(labels)
    if not labels:
        raise SystemExit(f"No test-taker generations found under {pack_dir / 'models'}")
    return labels


def _stamp_manifest(
    manifest: dict[str, Any],
    *,
    model_id: str,
    judge_key: str,
    prompt: str,
    make_primary: bool,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    existing_primary = str(manifest.get("primary_judge") or "").strip()
    primary = judge_key if (make_primary or not existing_primary) else existing_primary
    entry = {
        "label": judge_key,
        "model_id": model_id,
        "primary": primary == judge_key,
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
    for label, item in by_label.items():
        item["primary"] = label == primary
    ordered = []
    if primary in by_label:
        ordered.append(by_label[primary])
    for label, item in by_label.items():
        if label != primary:
            ordered.append(item)
    manifest["judges"] = ordered
    if make_primary or not existing_primary:
        manifest["primary_judge"] = primary
        manifest["grader_model_id"] = model_id if primary == judge_key else manifest.get(
            "grader_model_id", model_id
        )
    manifest["scoring"] = manifest.get("scoring") or "qwen_freeform_judge"
    manifest["graded_at"] = now
    manifest["updated_at"] = now
    return manifest


def _run_aggregate(pack_dir: Path) -> dict[str, Any]:
    result = aggregate_difficulty(pack_dir)
    manifest = _load_json(pack_dir / "manifest.json")
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
    print("[llm-judge-gt-claude] aggregated:", scores)
    return result


def run_claude_judge(
    *,
    models: str = "all",
    force: bool = False,
    skip_aggregate: bool = False,
    make_primary: bool = False,
    pack_dir: Path | None = None,
    batch_id: str | None = None,
    poll_interval: float = 30.0,
    download: bool = True,
) -> dict[str, Any]:
    dest = Path(pack_dir or DEFAULT_PACK_DIR).expanduser().resolve()
    dest = _require_pack(dest, download=download)
    prep = _overlay_pack(dest, models)
    question_ids = list(prep["question_ids"])
    model_labels = _model_labels(dest, models)
    judge_key = compose_judge_key(
        JUDGE_LABEL, prompt=GRADE_PROMPT, include_gold=True
    )
    remaining = remaining_grade_work(
        dest,
        model_labels,
        judge_key,
        question_ids=question_ids,
        shot_indices=SHOT_INDICES,
    )
    for label in model_labels:
        n_left = len(question_ids) if force else len(remaining.get(label) or [])
        print(
            f"[llm-judge-gt-claude] {label}: {n_left}/{len(question_ids)} "
            f"questions still need {judge_key}"
        )

    grade: dict[str, Any]
    if not force:
        done = [label for label in model_labels if label not in remaining]
        if done:
            print(f"[llm-judge-gt-claude] already graded: {done}")
        model_labels = [label for label in model_labels if label in remaining]
        if not model_labels:
            print("[llm-judge-gt-claude] skip batch: nothing left to grade")
            grade = {
                "status": "skipped",
                "pack": PACK_NAME,
                "judge_label": JUDGE_LABEL,
                "judge_key": judge_key,
                "n_samples": N_SAMPLES,
            }
        else:
            grade = grade_pack_with_batch_api(
                dest,
                label=JUDGE_LABEL,
                model_labels=model_labels,
                prompt=GRADE_PROMPT,
                force=force,
                make_primary=make_primary,
                poll_interval=poll_interval,
                batch_id=batch_id,
                n_samples=N_SAMPLES,
                temperature=JUDGE_TEMPERATURE,
                shot_indices=SHOT_INDICES,
                question_ids=question_ids,
            )
    else:
        grade = grade_pack_with_batch_api(
            dest,
            label=JUDGE_LABEL,
            model_labels=model_labels,
            prompt=GRADE_PROMPT,
            force=force,
            make_primary=make_primary,
            poll_interval=poll_interval,
            batch_id=batch_id,
            n_samples=N_SAMPLES,
            temperature=JUDGE_TEMPERATURE,
            shot_indices=SHOT_INDICES,
            question_ids=question_ids,
        )

    print("[llm-judge-gt-claude] graded:", grade)
    manifest_path = dest / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest = _stamp_manifest(
        manifest,
        model_id=str(grade.get("model_id") or JUDGE_LABEL),
        judge_key=str(grade.get("judge_key") or judge_key),
        prompt=GRADE_PROMPT,
        make_primary=make_primary,
    )
    write_json(manifest_path, manifest)
    agg = None if skip_aggregate else _run_aggregate(dest)
    return {"grade": grade, "aggregate": agg, "pack_dir": str(dest)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated test-taker labels, or ``all``.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing verdicts for this judge key.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip difficulty.jsonl / scores.json.",
    )
    parser.add_argument(
        "--make-primary",
        action="store_true",
        help="Make Claude the pack primary judge (default: keep Qwen GT).",
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=DEFAULT_PACK_DIR,
        help=(
            f"Local llm-judge-gt pack (default: {DEFAULT_PACK_DIR}). "
            "Generations are overlaid from outputs/mmar-freeform-thinking "
            "before grading."
        ),
    )
    parser.add_argument(
        "--batch-id",
        default="",
        help="Resume an existing Anthropic batch (`msgbatch_…`) instead of submitting.",
    )
    parser.add_argument(
        "--batch-poll-interval",
        type=float,
        default=30.0,
        help="Seconds between Batch API status polls (default: 30).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not pull the pack from the Modal volume if it is missing locally.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    run_claude_judge(
        models=args.models,
        force=args.force,
        skip_aggregate=args.skip_aggregate,
        make_primary=args.make_primary,
        pack_dir=args.pack_dir,
        batch_id=(args.batch_id or "").strip() or None,
        poll_interval=args.batch_poll_interval,
        download=not args.no_download,
    )


if __name__ == "__main__":
    main()
