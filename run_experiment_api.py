"""Local MMAR freeform test-taker via OpenAI / Gemini audio **Batch** APIs.

Writes under ``outputs/exp-mmar-question-difficulty-api/<run_id>/`` so Modal
downloads into ``outputs/exp-mmar-question-difficulty/`` cannot overwrite them.

Both providers run at ~50% of interactive pricing with a 24h completion window.

Usage::

    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...

    uv run python run_experiment_api.py \\
      --models gpt-audio-mini,gemini-3.7-flash \\
      --mode freeform \\
      --n-shots 5 \\
      --question-ids-csv answer-variety/open_ended_question_ids.csv
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api_batch import (
    gemini_generate_request,
    gemini_inline_audio_part,
    gemini_text_part,
    gemini_user_contents,
    openai_batch_chat_text,
    openai_chat_request,
    run_gemini_generate_batch,
    run_openai_chat_batch,
)
from mmar_common import (
    aggregate_n_shot_record,
    build_mmar_freeform_prompt,
    load_completed_ids,
    load_jsonl,
    load_question_ids_csv,
    make_run_id,
    parse_freeform_output,
    resolve_path,
    write_json,
    write_jsonl,
)
from view_difficulty import DEFAULT_AUDIO_DIR, DEFAULT_DATA_DIR, ensure_mmar_audio

REPO_ROOT = Path(__file__).resolve().parent
API_EXPERIMENT = "exp-mmar-question-difficulty-api"
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs"
DEFAULT_META = DEFAULT_DATA_DIR / "MMAR-meta.jsonl"
DEFAULT_IDS_CSV = REPO_ROOT / "answer-variety" / "open_ended_question_ids.csv"
DEFAULT_N_SHOTS = 5
DEFAULT_SEED = 42
DEFAULT_MODE = "freeform"

API_SPECS: dict[str, dict[str, Any]] = {
    "gpt-audio-mini": {
        "model_id": "gpt-audio-mini",
        "backend": "openai",
        "sampling": {"temperature": 1.0, "max_tokens": 1024},
    },
    "gemini-3.7-flash": {
        "model_id": "gemini-3.7-flash",
        "backend": "gemini",
        "sampling": {"temperature": 1.0, "max_tokens": 1024},
    },
}

ALL_API_LABELS = tuple(API_SPECS.keys())
logger = logging.getLogger(__name__)


def parse_api_model_list(value: str) -> list[str]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw or any(item.lower() == "all" for item in raw):
        return list(ALL_API_LABELS)
    unknown = [item for item in raw if item not in API_SPECS]
    if unknown:
        raise SystemExit(
            f"Unknown API model label(s): {unknown}. "
            f"Choose from {list(ALL_API_LABELS)} or 'all'."
        )
    return raw


def _shot_seed(seed: int, question_id: str, shot_index: int) -> int:
    digest = hashlib.md5(f"{seed}:{question_id}:{shot_index}".encode()).hexdigest()
    return seed + (int(digest[:8], 16) % 1_000_000)


def _normalize_mode(mode: str | None) -> str:
    value = str(mode or DEFAULT_MODE).strip().lower()
    if value in {"freeform", "free_form", "free-form", "open"}:
        return "freeform"
    raise SystemExit(f"API runner only supports freeform mode, got {mode!r}")


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
            print(f"Skipping {qid}: not in MMAR meta")
            continue
        audio_path = resolve_path(data_root, item["audio_path"])
        if not Path(audio_path).is_file():
            print(f"Skipping {qid}: missing audio at {audio_path}")
            continue
        items.append({**item, "audio_path": audio_path})
    return items


def _output_from_text(raw: str) -> dict:
    thinking, answer = parse_freeform_output(raw)
    return {
        "model_output": raw,
        "raw_tokens": None,
        "thinking_prediction": thinking,
        "answer_prediction": answer,
    }


def _shot_custom_id(question_id: str, shot_index: int) -> str:
    return f"{question_id}__shot{shot_index}"


def _run_openai_batch_label(
    *,
    label: str,
    pending: list[dict],
    predictions_path: Path,
    work_dir: Path,
    args: argparse.Namespace,
    question_ids: list[str],
) -> dict:
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        raise SystemExit("Set OPENAI_API_KEY to call gpt-audio-mini.")
    spec = API_SPECS[label]
    temperature = float(
        args.temperature
        if args.temperature is not None
        else spec["sampling"]["temperature"]
    )
    max_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else spec["sampling"]["max_tokens"]
    )
    requests: list[dict[str, Any]] = []
    owners: dict[str, tuple[str, int]] = {}
    for item in pending:
        prompt = build_mmar_freeform_prompt(item)
        audio_b64 = base64.b64encode(Path(item["audio_path"]).read_bytes()).decode(
            "ascii"
        )
        for shot_index in range(args.n_shots):
            custom_id = _shot_custom_id(str(item["id"]), shot_index)
            owners[custom_id] = (str(item["id"]), shot_index)
            seed = _shot_seed(args.seed, str(item["id"]), shot_index)
            requests.append(
                openai_chat_request(
                    custom_id=custom_id,
                    model=spec["model_id"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    modalities=["text"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": audio_b64,
                                        "format": "wav",
                                    },
                                },
                            ],
                        }
                    ],
                )
            )
    print(f"[{label}] submitting OpenAI batch with {len(requests)} requests")
    results = run_openai_chat_batch(
        requests,
        work_dir=work_dir,
        display_name=f"{label}-generate",
        poll_interval_s=args.poll_interval,
    )
    by_qid: dict[str, dict[int, str]] = {}
    n_fail = 0
    for custom_id, (qid, shot_index) in owners.items():
        text = openai_batch_chat_text(results.get(custom_id) or {})
        if not text:
            n_fail += 1
            print(f"[{label}] missing/empty result for {custom_id}")
            continue
        by_qid.setdefault(qid, {})[shot_index] = text

    written = 0
    for item in pending:
        qid = str(item["id"])
        shot_map = by_qid.get(qid) or {}
        if len(shot_map) < args.n_shots:
            print(
                f"[{label}] skipping write for {qid}: "
                f"{len(shot_map)}/{args.n_shots} shots"
            )
            continue
        shot_outputs = [
            _output_from_text(shot_map[shot_index])
            for shot_index in range(args.n_shots)
        ]
        record = aggregate_n_shot_record(item, shot_outputs, pending_grade=True)
        write_jsonl(predictions_path, [record], mode="a")
        written += 1
        if written % args.print_every == 0 or written == len(pending):
            print(
                f"[{label}] {written}/{len(pending)} id={item['id']} "
                f"answer={record.get('answer_prediction')!r}"
            )
    total = len(load_completed_ids(predictions_path) & set(question_ids))
    return {
        "status": "ok",
        "model_label": label,
        "n_written": written,
        "n_fail": n_fail,
        "n_predictions": total,
        "predictions_path": str(predictions_path),
        "backend": "openai-batch",
        "mode": "freeform",
    }


def _run_gemini_batch_label(
    *,
    label: str,
    pending: list[dict],
    predictions_path: Path,
    work_dir: Path,
    args: argparse.Namespace,
    question_ids: list[str],
) -> dict:
    api_key = (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    ).strip()
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY to call Gemini.")
    spec = API_SPECS[label]
    temperature = float(
        args.temperature
        if args.temperature is not None
        else spec["sampling"]["temperature"]
    )
    max_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else spec["sampling"]["max_tokens"]
    )
    requests: list[dict[str, Any]] = []
    owners: dict[str, tuple[str, int]] = {}
    for item in pending:
        prompt = build_mmar_freeform_prompt(item)
        audio_bytes = Path(item["audio_path"]).read_bytes()
        audio_part = gemini_inline_audio_part(audio_bytes)
        text_part = gemini_text_part(prompt)
        for shot_index in range(args.n_shots):
            custom_id = _shot_custom_id(str(item["id"]), shot_index)
            owners[custom_id] = (str(item["id"]), shot_index)
            requests.append(
                gemini_generate_request(
                    key=custom_id,
                    contents=gemini_user_contents(audio_part, text_part),
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
    print(f"[{label}] submitting Gemini batch with {len(requests)} requests")
    texts = run_gemini_generate_batch(
        requests,
        model=spec["model_id"],
        work_dir=work_dir,
        display_name=f"{label}-generate",
        poll_interval_s=args.poll_interval,
    )
    by_qid: dict[str, dict[int, str]] = {}
    n_fail = 0
    for custom_id, (qid, shot_index) in owners.items():
        text = (texts.get(custom_id) or "").strip()
        if not text:
            n_fail += 1
            print(f"[{label}] missing/empty result for {custom_id}")
            continue
        by_qid.setdefault(qid, {})[shot_index] = text

    written = 0
    for item in pending:
        qid = str(item["id"])
        shot_map = by_qid.get(qid) or {}
        if len(shot_map) < args.n_shots:
            print(
                f"[{label}] skipping write for {qid}: "
                f"{len(shot_map)}/{args.n_shots} shots"
            )
            continue
        shot_outputs = [
            _output_from_text(shot_map[shot_index])
            for shot_index in range(args.n_shots)
        ]
        record = aggregate_n_shot_record(item, shot_outputs, pending_grade=True)
        write_jsonl(predictions_path, [record], mode="a")
        written += 1
        if written % args.print_every == 0 or written == len(pending):
            print(
                f"[{label}] {written}/{len(pending)} id={item['id']} "
                f"answer={record.get('answer_prediction')!r}"
            )
    total = len(load_completed_ids(predictions_path) & set(question_ids))
    return {
        "status": "ok",
        "model_label": label,
        "n_written": written,
        "n_fail": n_fail,
        "n_predictions": total,
        "predictions_path": str(predictions_path),
        "backend": "gemini-batch",
        "mode": "freeform",
    }


def _run_label(
    *,
    label: str,
    items: list[dict],
    run_dir: Path,
    args: argparse.Namespace,
    question_ids: list[str],
) -> dict:
    spec = API_SPECS[label]
    model_dir = run_dir / "models" / label
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = model_dir / "predictions.jsonl"
    work_dir = model_dir / "batch"
    completed = {str(x) for x in load_completed_ids(predictions_path)}
    pending = [item for item in items if str(item["id"]) not in completed]
    print(
        f"[{label}] model={spec['model_id']} backend={spec['backend']}-batch "
        f"{len(items)} selected, {len(completed)} done, {len(pending)} pending "
        f"(n_shots={args.n_shots})"
    )
    if not pending:
        return {
            "status": "already_complete",
            "model_label": label,
            "n_predictions": len(completed),
            "predictions_path": str(predictions_path),
        }
    if spec["backend"] == "openai":
        return _run_openai_batch_label(
            label=label,
            pending=pending,
            predictions_path=predictions_path,
            work_dir=work_dir,
            args=args,
            question_ids=question_ids,
        )
    if spec["backend"] == "gemini":
        return _run_gemini_batch_label(
            label=label,
            pending=pending,
            predictions_path=predictions_path,
            work_dir=work_dir,
            args=args,
            question_ids=question_ids,
        )
    raise SystemExit(f"Unknown backend for {label}")


def main_sync(args: argparse.Namespace) -> dict:
    prompt_mode = _normalize_mode(args.mode)
    labels = parse_api_model_list(args.models)
    run_id = args.run_id or make_run_id()
    results_dir = Path(args.results_dir).expanduser().resolve()
    run_dir = results_dir / API_EXPERIMENT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.meta).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    if not args.skip_audio_download:
        audio_dir = ensure_mmar_audio(audio_dir)

    csv_path = Path(args.question_ids_csv).expanduser().resolve()
    wanted = load_question_ids_csv(csv_path)
    items = _load_selected_items(meta_path, data_root, wanted)
    question_ids = [str(item["id"]) for item in items]
    write_json(
        run_dir / "question_ids.json",
        {
            "n": len(question_ids),
            "ids": question_ids,
            "question_ids_csv": str(csv_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    sampling = {label: dict(API_SPECS[label]["sampling"]) for label in labels}
    if args.temperature is not None:
        for spec in sampling.values():
            spec["temperature"] = float(args.temperature)
    if args.max_new_tokens is not None:
        for spec in sampling.values():
            spec["max_tokens"] = int(args.max_new_tokens)

    manifest_path = run_dir / "manifest.json"
    existing: dict = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    prior = [str(x) for x in (existing.get("models") or [])]
    merged = list(dict.fromkeys([*prior, *labels]))
    now = datetime.now(timezone.utc).isoformat()
    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "experiment": API_EXPERIMENT,
            "mode": prompt_mode,
            "models": merged,
            "n_shots": args.n_shots,
            "seed": args.seed,
            "model_sampling": sampling,
            "scoring": "pending_local_api",
            "inference": "api-batch",
            "n_questions": len(question_ids),
            "question_ids_csv": str(csv_path),
            "model_specs": {
                label: {
                    "model_id": API_SPECS[label]["model_id"],
                    "backend": API_SPECS[label]["backend"],
                    "sampling": API_SPECS[label]["sampling"],
                }
                for label in ALL_API_LABELS
            },
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
        },
    )

    results = []
    for label in labels:
        results.append(
            _run_label(
                label=label,
                items=items,
                run_dir=run_dir,
                args=args,
                question_ids=question_ids,
            )
        )
    print("Done:", {"run_id": run_id, "models": results})
    return {"run_id": run_id, "mode": prompt_mode, "models": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="gpt-audio-mini,gemini-3.7-flash",
        help="Comma-separated labels or 'all'",
    )
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--n-shots", type=int, default=DEFAULT_N_SHOTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--question-ids-csv", type=Path, default=DEFAULT_IDS_CSV)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between Batch API status polls",
    )
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--skip-audio-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    main_sync(args)


if __name__ == "__main__":
    main()
