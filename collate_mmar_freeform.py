"""Collate MMAR 5-shot freeform runs into one zip-ready pack.

Merges the full-MMAR vLLM run, the open-ended vLLM run, and any local API
runs. Keeps only the IDs in ``answer-variety/open_ended_question_ids.csv``
that have ``n_shots`` independent samples.

Writes::

    outputs/mmar-freeform-5-shot/
      manifest.json
      question_ids.json
      models/<label>/predictions.jsonl

Zip that folder when you are ready to upload.

Usage::

    uv run python collate_mmar_freeform.py
    uv run python collate_mmar_freeform.py --keep-judges
    uv run python collate_mmar_freeform.py \\
      --out /tmp/mmar-freeform-5-shot
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aggregate import MODEL_LABEL_ORDER
from mmar_common import load_question_ids_csv, write_json

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs"
DEFAULT_IDS_CSV = REPO_ROOT / "answer-variety" / "open_ended_question_ids.csv"
DEFAULT_OUT_DIR = DEFAULT_RESULTS_DIR / "mmar-freeform-5-shot"
DEFAULT_N_SHOTS = 5
VLLM_EXPERIMENT = "exp-mmar-question-difficulty"
API_EXPERIMENT = "exp-mmar-question-difficulty-api"
ALL_API_LABELS = ("gpt-audio-mini", "gemini-3.7-flash", "gpt-4o-mini")

# Later entries override earlier ones for the same model label.
DEFAULT_VLLM_RUN_IDS = (
    "20260807T145000Z",  # full MMAR (covers the 784 open-ended IDs)
    "20260816T050944Z",  # open-ended CSV run
)

LABEL_ORDER = MODEL_LABEL_ORDER + ALL_API_LABELS

DROP_RECORD_KEYS = (
    "judges",
    "grader",
    "grader_output",
    "per_judge",
    "primary_judge",
)
DROP_SHOT_KEYS = ("judges", "grader", "grader_output")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _shot_index(shot: dict[str, Any]) -> int:
    raw = shot.get("shot_index")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _compact_shot(shot: dict[str, Any], *, keep_judges: bool) -> dict[str, Any]:
    if keep_judges:
        return dict(shot)
    return {key: value for key, value in shot.items() if key not in DROP_SHOT_KEYS}


def compact_record(
    record: dict[str, Any],
    *,
    model: str,
    source_run_id: str,
    n_shots: int,
    keep_judges: bool,
) -> dict[str, Any] | None:
    """Return a 5-shot record, or None if this row is short."""
    shots = list(record.get("shots") or [])
    shots.sort(key=_shot_index)
    if len(shots) < n_shots:
        return None
    shots = [_compact_shot(shot, keep_judges=keep_judges) for shot in shots[:n_shots]]
    for index, shot in enumerate(shots):
        shot["shot_index"] = index

    if keep_judges:
        out = dict(record)
    else:
        out = {key: value for key, value in record.items() if key not in DROP_RECORD_KEYS}
    out["id"] = str(record.get("id") or "")
    out["model"] = model
    out["source_run_id"] = source_run_id
    out["n_shots"] = n_shots
    out["shots"] = shots
    primary = shots[0] if shots else {}
    out.setdefault("model_output", primary.get("model_output"))
    out.setdefault("thinking_prediction", primary.get("thinking_prediction"))
    out.setdefault("answer_prediction", primary.get("answer_prediction"))
    out.setdefault("raw_tokens", primary.get("raw_tokens"))
    return out


def iter_wanted_records(
    predictions_path: Path,
    wanted: set[str],
    *,
    model: str,
    source_run_id: str,
    n_shots: int,
    keep_judges: bool,
) -> dict[str, dict[str, Any]]:
    """Stream ``predictions.jsonl`` and keep complete open-ended rows."""
    found: dict[str, dict[str, Any]] = {}
    if not predictions_path.is_file():
        return found
    with open(predictions_path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            qid = str(record.get("id") or "")
            if not qid or qid not in wanted or qid in found:
                continue
            compact = compact_record(
                record,
                model=model,
                source_run_id=source_run_id,
                n_shots=n_shots,
                keep_judges=keep_judges,
            )
            if compact is None:
                continue
            found[qid] = compact
            if len(found) >= len(wanted):
                break
    return found


def discover_source_runs(results_dir: Path) -> list[tuple[str, Path]]:
    """Return ``(kind, run_dir)`` in override order (later wins)."""
    sources: list[tuple[str, Path]] = []
    for run_id in DEFAULT_VLLM_RUN_IDS:
        run_dir = results_dir / VLLM_EXPERIMENT / run_id
        if run_dir.is_dir():
            sources.append(("vllm", run_dir))
    api_root = results_dir / API_EXPERIMENT
    if api_root.is_dir():
        for run_dir in sorted(p for p in api_root.iterdir() if p.is_dir()):
            sources.append(("api", run_dir))
    return sources


def list_model_dirs(run_dir: Path) -> list[str]:
    models_root = run_dir / "models"
    if not models_root.is_dir():
        return []
    labels: list[str] = []
    for child in sorted(models_root.iterdir()):
        if child.is_dir() and (child / "predictions.jsonl").is_file():
            labels.append(child.name)
    return labels


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def collate(
    *,
    results_dir: Path,
    ids_csv: Path,
    out_dir: Path,
    n_shots: int,
    keep_judges: bool,
) -> dict[str, Any]:
    question_ids = load_question_ids_csv(ids_csv)
    wanted = set(question_ids)
    sources = discover_source_runs(results_dir)
    if not sources:
        raise SystemExit(f"No source runs under {results_dir}")

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    source_for: dict[str, str] = {}
    source_meta: list[dict[str, Any]] = []

    for kind, run_dir in sources:
        run_id = run_dir.name
        labels = list_model_dirs(run_dir)
        source_meta.append(
            {
                "kind": kind,
                "run_id": run_id,
                "path": str(run_dir),
                "models": labels,
            }
        )
        for label in labels:
            rows = iter_wanted_records(
                run_dir / "models" / label / "predictions.jsonl",
                wanted,
                model=label,
                source_run_id=run_id,
                n_shots=n_shots,
                keep_judges=keep_judges,
            )
            if not rows:
                continue
            by_model.setdefault(label, {}).update(rows)
            source_for[label] = run_id

    found = list(by_model)
    known = [label for label in LABEL_ORDER if label in by_model]
    rest = [label for label in found if label not in LABEL_ORDER]
    labels = known + rest

    out_dir.mkdir(parents=True, exist_ok=True)
    progress: dict[str, Any] = {}
    for label in labels:
        records = [
            by_model[label][qid] for qid in question_ids if qid in by_model[label]
        ]
        write_jsonl(out_dir / "models" / label / "predictions.jsonl", records)
        n_done = len(records)
        progress[label] = {
            "model_label": label,
            "n_done": n_done,
            "n_total": len(question_ids),
            "n_shots": n_shots,
            "complete": n_done >= len(question_ids),
            "source_run_id": source_for.get(label),
        }

    write_json(
        out_dir / "question_ids.json",
        {
            "n": len(question_ids),
            "ids": question_ids,
            "question_ids_csv": str(ids_csv),
            "n_shots": n_shots,
        },
    )
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "name": "mmar-freeform-5-shot",
        "mode": "freeform",
        "n_shots": n_shots,
        "n_questions": len(question_ids),
        "question_ids_csv": str(ids_csv),
        "models": labels,
        "sources": source_meta,
        "keep_judges": keep_judges,
        "progress": progress,
        "created_at": now,
        "updated_at": now,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def _print_summary(manifest: dict[str, Any], out_dir: Path) -> None:
    n_total = int(manifest["n_questions"])
    n_shots = int(manifest["n_shots"])
    print(f"Wrote {out_dir}")
    print(f"questions={n_total} n_shots={n_shots}")
    print()
    print(f"{'model':<24} {'done':>8} {'status':<12} source")
    for label, row in (manifest.get("progress") or {}).items():
        status = "complete" if row.get("complete") else "pending"
        print(
            f"{label:<24} {row.get('n_done'):>4}/{n_total:<4} {status:<12} "
            f"{row.get('source_run_id') or '—'}"
        )
    expected_pending = [
        label
        for label in ALL_API_LABELS
        if label != "gpt-4o-mini" and label not in (manifest.get("progress") or {})
    ]
    for label in expected_pending:
        print(f"{label:<24} {0:>4}/{n_total:<4} {'missing':<12} —")
    try:
        zip_target = out_dir.relative_to(REPO_ROOT)
    except ValueError:
        zip_target = out_dir
    print()
    print(f"Zip with:  zip -r mmar-freeform-5-shot.zip {zip_target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collate MMAR 5-shot freeform runs into one zip-ready pack."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Local outputs root (default: ./outputs)",
    )
    parser.add_argument(
        "--question-ids-csv",
        type=Path,
        default=DEFAULT_IDS_CSV,
        help="Open-ended ID list (default: answer-variety/open_ended_question_ids.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory to zip later (default: outputs/mmar-freeform-5-shot)",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=DEFAULT_N_SHOTS,
        help="Required shots per question (default: 5)",
    )
    parser.add_argument(
        "--keep-judges",
        action="store_true",
        help="Keep freeform judge payloads (dropped by default to shrink the pack)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_shots = max(1, int(args.n_shots))
    manifest = collate(
        results_dir=Path(args.results_dir).expanduser().resolve(),
        ids_csv=Path(args.question_ids_csv).expanduser().resolve(),
        out_dir=Path(args.out).expanduser().resolve(),
        n_shots=n_shots,
        keep_judges=bool(args.keep_judges),
    )
    _print_summary(manifest, Path(args.out).expanduser().resolve())


if __name__ == "__main__":
    main()
