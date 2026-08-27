"""LLM-as-judge with gold, 5 shots, all 1000 MMAR questions — Claude batch.

Same setup as ``run_llm_judge_gt.py`` (``neutral_with_gt_no_audio``, every
``shot_index`` 0–4, 3-sample majority vote) but the judge is
``claude-sonnet-5`` via Anthropic's Message Batches API. Runs locally;
no Modal container is started.

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
from mmar_common import write_json
from modal_cache import JUDGING_VOLUME_NAME

PACK_NAME = "llm-judge-gt"
DEFAULT_PACK_DIR = _REPO_ROOT / "outputs" / "judge-quality" / PACK_NAME
JUDGE_LABEL = "claude-sonnet-5"
GRADE_PROMPT = "neutral_with_gt_no_audio"
N_SHOTS = 5
SHOT_INDICES = tuple(range(N_SHOTS))
N_SAMPLES = 3
# Same as run_llm_judge_gt.py: T=0 would collapse the 3 votes.
JUDGE_TEMPERATURE = 1.0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_question_ids(pack_dir: Path) -> list[str]:
    payload = _load_json(pack_dir / "question_ids.json")
    ids = payload.get("ids") if isinstance(payload, dict) else None
    if isinstance(ids, list) and ids:
        return [str(qid) for qid in ids if str(qid).strip()]
    return []


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
    if not download:
        raise SystemExit(
            f"Pack not found at {pack_dir}. Download with:\n"
            "  uv run modal run judge-quality/download_judge_quality.py "
            f"--pack {PACK_NAME}"
        )
    try:
        return _download_pack(pack_dir, force=pack_dir.exists())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Pack not found at {pack_dir} and download failed "
            f"(exit {exc.returncode}). Prepare it with:\n"
            "  uv run modal run --detach judge-quality/run_llm_judge_gt.py\n"
            "then download:\n"
            "  uv run modal run judge-quality/download_judge_quality.py "
            f"--pack {PACK_NAME}"
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
    question_ids = _load_question_ids(dest)
    if len(question_ids) != 1000:
        raise SystemExit(
            f"Expected 1000 MMAR ids in {dest / 'question_ids.json'}, "
            f"got {len(question_ids)}"
        )
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
        help=f"Local llm-judge-gt pack (default: {DEFAULT_PACK_DIR})",
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
