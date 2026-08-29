"""Local viewer for MMAR judge outcomes vs human labels in ``exports/``.

Joins:

* ``exports/labels.csv`` — ``question_id``, ``generation_id``, ``model_label``,
  ``shot_index``, ``ratings`` (JSON list of bools; gold is majority when
  there are at least three ratings)
* ``exports/generations.csv`` — same keys plus ``answer_prediction``
* ``outputs/mmar-judging`` — ``shots[].judges[<key>]`` entries
  (``correct``, ``verdict``, ``output``, ``generation``, ``model_id``,
  ``prompt``, ``include_gold``) and optional ``judge_partials/*.jsonl``

The viewer requires a ``JUDGE_FORMATS`` key first (first key on load).
Accuracy, question sorting, and shot verdicts then render only for that
format. The list can sort by disagreement or by mean agreement with the
human majority label within the selected format.

Usage::

    uv run modal run download_judges.py
    uv run python view_judges.py
    uv run python view_judges.py --port 7862
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from aggregate import order_model_labels, is_dropped_model
from grader import (
    accuracy_mode_names,
    grade_mode_title,
    grade_mode_titles,
    grade_prompt_names,
    judge_mode_bucket,
    parse_judge_key,
)
from alt_test import (
    DEFAULT_EPSILON,
    llm_annotation_from_entry,
    score_binary_judge,
    scoring_gold,
)
from mmar_common import load_jsonl
from view_mmar import (
    CONFIG as MMAR_CONFIG,
    DEFAULT_AUDIO_DIR,
    DEFAULT_DATA_DIR,
    QUESTION_KEYS,
    _compact_judge_entry,
    ensure_mmar_audio,
    resolve_audio,
)

REPO_ROOT = Path(__file__).resolve().parent
EXPORTS_DIR = REPO_ROOT / "exports"
PACK_DIR = REPO_ROOT / "outputs" / "mmar-judging"
LABELS_CSV_NAME = "labels.csv"
GENERATIONS_CSV_NAME = "generations.csv"
LOCAL_MMAR_META = DEFAULT_DATA_DIR / "MMAR-meta.jsonl"

CONFIG: dict[str, Any] = {}


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


def load_label_rows(path: Path) -> list[dict[str, Any]]:
    """Rows with a non-empty boolean ``ratings`` list (exports/labels.csv)."""
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", newline="") as handle:
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
            extra = {
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "question_id",
                    "generation_id",
                    "model_label",
                    "shot_index",
                    "ratings",
                }
            }
            rows.append(
                {
                    "question_id": qid,
                    "generation_id": str(raw.get("generation_id") or "").strip(),
                    "model_label": model,
                    "shot_index": shot_index,
                    "ratings": ratings,
                    "gold": scoring_gold(ratings),
                    "extra": extra,
                }
            )
    return rows


def load_generation_rows(path: Path) -> list[dict[str, Any]]:
    """Rows from exports/generations.csv."""
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            qid = str(raw.get("question_id") or "").strip()
            model = str(raw.get("model_label") or "").strip()
            if not qid or not model:
                continue
            try:
                shot_index = int(raw.get("shot_index", 0))
            except (TypeError, ValueError):
                continue
            prediction = raw.get("answer_prediction")
            extra = {
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "question_id",
                    "generation_id",
                    "model_label",
                    "shot_index",
                    "answer_prediction",
                }
            }
            rows.append(
                {
                    "question_id": qid,
                    "generation_id": str(raw.get("generation_id") or "").strip(),
                    "model_label": model,
                    "shot_index": shot_index,
                    "answer_prediction": (
                        "" if prediction is None else str(prediction)
                    ),
                    "extra": extra,
                }
            )
    return rows


def _tuple_key(qid: str, model: str, shot_index: int) -> tuple[str, str, int]:
    return (qid, model, shot_index)


def compact_judge_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    out = _compact_judge_entry(entry) or {}
    for key, value in entry.items():
        if key not in out:
            out[key] = value
    if "verdict" not in out and out.get("correct") is not None:
        out["verdict"] = "pass" if out.get("correct") else "fail"
    return out


def _question_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in QUESTION_KEYS if key in record}


def _load_meta(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if not isinstance(row, dict):
            continue
        qid = str(row.get("id") or "").strip()
        if qid:
            by_id[qid] = row
    return by_id


def _shot_for_index(record: dict[str, Any] | None, shot_index: int) -> dict | None:
    if not record:
        return None
    for shot in record.get("shots") or []:
        try:
            if int(shot.get("shot_index", 0)) == shot_index:
                return shot
        except (TypeError, ValueError):
            continue
    return None


def overlay_sidecars(model_dir: Path, by_id: dict[str, dict[str, Any]]) -> None:
    """Fill missing ``shots[].judges`` keys from ``judge_partials/*.jsonl``."""
    partials = model_dir / "judge_partials"
    if not partials.is_dir():
        return
    for path in sorted(partials.glob("*.jsonl")):
        for row in load_jsonl(path):
            if not isinstance(row, dict):
                continue
            qid = str(row.get("id") or "").strip()
            record = by_id.get(qid)
            key = str(row.get("judge_key") or "")
            entry = row.get("entry")
            if record is None or not key or not isinstance(entry, dict):
                continue
            try:
                shot_index = int(row.get("shot_index", 0))
            except (TypeError, ValueError):
                continue
            shot = _shot_for_index(record, shot_index)
            if shot is None:
                continue
            judges = shot.setdefault("judges", {})
            if not isinstance(judges, dict):
                judges = {}
                shot["judges"] = judges
            if key not in judges:
                judges[key] = entry


def load_pack_predictions(pack_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    models_root = pack_dir / "models"
    if not models_root.is_dir():
        return predictions
    for child in sorted(models_root.iterdir()):
        if is_dropped_model(child.name):
            continue
        pred_path = child / "predictions.jsonl"
        if not child.is_dir() or not pred_path.is_file():
            continue
        by_id: dict[str, dict[str, Any]] = {}
        for record in load_jsonl(pred_path):
            if not isinstance(record, dict):
                continue
            qid = str(record.get("id") or "").strip()
            if qid:
                by_id[qid] = record
        overlay_sidecars(child, by_id)
        predictions[child.name] = by_id
    return predictions


def _collect_judge_keys(
    predictions: dict[str, dict[str, dict[str, Any]]],
    manifest: dict[str, Any],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(key: str) -> None:
        if key and key not in seen and not is_dropped_model(key):
            seen.add(key)
            ordered.append(key)

    for raw in manifest.get("judges") or []:
        if isinstance(raw, dict) and raw.get("label"):
            _add(str(raw["label"]))
        elif raw:
            _add(str(raw))
    for by_id in predictions.values():
        for record in by_id.values():
            for shot in record.get("shots") or []:
                for key in (shot.get("judges") or {}):
                    _add(str(key))
            for key in record.get("judges") or []:
                _add(str(key))
    return ordered


def compute_accuracy(
    samples: list[tuple[str, list[bool], dict[str, dict[str, Any]] | None]],
    judge_keys: list[str],
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, Any]:
    stats: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in grade_prompt_names()
    }
    key_mode: dict[str, str] = {}
    for key in judge_keys:
        sample_entry = next(
            (
                (judges or {}).get(key)
                for _iid, _ratings, judges in samples
                if judges and key in judges
            ),
            None,
        )
        mode = judge_mode_bucket(
            key, sample_entry if isinstance(sample_entry, dict) else None
        )
        if mode is None:
            continue
        key_mode[key] = mode

    for key, mode in key_mode.items():
        instances = []
        for instance_id, ratings, judges in samples:
            entry = (judges or {}).get(key) if judges else None
            instances.append((instance_id, ratings, llm_annotation_from_entry(entry)))
        scored = score_binary_judge(instances, epsilon=epsilon)
        parsed = parse_judge_key(key)
        stats.setdefault(mode, {})[key] = {
            **scored,
            "model": parsed["model"],
            "prompt": parsed["prompt"],
            "gold_tag": parsed["gold_tag"],
            "mode": mode,
        }

    payload: dict[str, Any] = {
        "n_label_rows": len(samples),
        "epsilon": float(epsilon),
        "modes": accuracy_mode_names(stats),
    }
    for mode in payload["modes"]:
        payload[mode] = stats.get(mode) or {}
    return payload


def _average_judge_agreement(
    per_judge: dict[str, dict[str, int]],
    judge_keys: list[str],
) -> dict[str, Any]:
    """Mean per-judge agreement with human majority on comparable shots."""
    rates: list[float] = []
    n = 0
    n_agree = 0
    for key in judge_keys:
        bucket = per_judge.get(key) or {}
        count = int(bucket.get("n") or 0)
        agree = int(bucket.get("n_agree") or 0)
        n += count
        n_agree += agree
        if count:
            rates.append(agree / count)
    return {
        "n": n,
        "n_agree": n_agree,
        "rate": (sum(rates) / len(rates)) if rates else None,
    }


@lru_cache(maxsize=4)
def load_bundle(
    pack_dir_s: str, exports_dir_s: str, epsilon: float = DEFAULT_EPSILON
) -> dict[str, Any]:
    pack_dir = Path(pack_dir_s)
    exports_dir = Path(exports_dir_s)
    labels_path = exports_dir / LABELS_CSV_NAME
    if not labels_path.is_file():
        labels_path = pack_dir / LABELS_CSV_NAME
    generations_path = exports_dir / GENERATIONS_CSV_NAME

    label_rows = [
        row
        for row in load_label_rows(labels_path)
        if not is_dropped_model(str(row.get("model_label") or ""))
    ]
    gen_rows = [
        row
        for row in load_generation_rows(generations_path)
        if not is_dropped_model(str(row.get("model_label") or ""))
    ]
    gens: dict[tuple[str, str, int], dict[str, Any]] = {}
    gens_by_id: dict[str, dict[str, Any]] = {}
    for row in gen_rows:
        gens[_tuple_key(row["question_id"], row["model_label"], row["shot_index"])] = (
            row
        )
        if row["generation_id"]:
            gens_by_id[row["generation_id"]] = row

    predictions = load_pack_predictions(pack_dir) if pack_dir.is_dir() else {}
    manifest: dict[str, Any] = {}
    if (pack_dir / "manifest.json").is_file():
        try:
            payload = json.loads(
                (pack_dir / "manifest.json").read_text(encoding="utf-8")
            )
            if isinstance(payload, dict):
                manifest = payload
        except json.JSONDecodeError:
            manifest = {}

    meta = _load_meta(LOCAL_MMAR_META)
    judge_keys = _collect_judge_keys(predictions, manifest)

    models_from_labels = [str(row["model_label"]) for row in label_rows]
    models_from_pack = list(predictions)
    model_labels = order_model_labels(models_from_labels + models_from_pack)

    judge_meta: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for key in judge_keys:
        sample_entry = None
        for by_id in predictions.values():
            for record in by_id.values():
                for shot in record.get("shots") or []:
                    entry = (shot.get("judges") or {}).get(key)
                    if isinstance(entry, dict):
                        sample_entry = entry
                        break
                if sample_entry is not None:
                    break
            if sample_entry is not None:
                break
        parsed = parse_judge_key(key)
        mode = judge_mode_bucket(key, sample_entry)
        judge_meta.append(
            {
                "label": key,
                "model": parsed["model"],
                "prompt": (
                    (sample_entry or {}).get("prompt")
                    if isinstance(sample_entry, dict)
                    else None
                )
                or parsed["prompt"],
                "include_gold": (
                    (sample_entry or {}).get("include_gold")
                    if isinstance(sample_entry, dict)
                    else None
                ),
                "model_id": (
                    (sample_entry or {}).get("model_id")
                    if isinstance(sample_entry, dict)
                    else None
                ),
                "mode": mode,
                "gold_tag": parsed["gold_tag"],
            }
        )
        seen_keys.add(key)

    question_ids: list[str] = []
    seen_q: set[str] = set()
    for row in label_rows:
        qid = row["question_id"]
        if qid not in seen_q:
            seen_q.add(qid)
            question_ids.append(qid)

    samples: list[tuple[str, list[bool], dict[str, dict[str, Any]] | None]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for qid in question_ids:
        sample_record = None
        for label in model_labels:
            rec = predictions.get(label, {}).get(qid)
            if rec is not None:
                sample_record = rec
                break
        meta_row = meta.get(qid) or {}
        fields = _question_fields(sample_record or meta_row)
        if not fields.get("id"):
            fields["id"] = qid
        if not fields.get("question") and meta_row.get("question"):
            fields["question"] = meta_row.get("question")
        if not fields.get("answer") and meta_row.get("answer"):
            fields["answer"] = meta_row.get("answer")
        if not fields.get("audio_path") and meta_row.get("audio_path"):
            fields["audio_path"] = meta_row.get("audio_path")
        by_id[qid] = {
            **fields,
            "id": qid,
            "models": {},
        }

    for row in label_rows:
        qid = row["question_id"]
        model = row["model_label"]
        shot_index = row["shot_index"]
        record = predictions.get(model, {}).get(qid)
        shot = _shot_for_index(record, shot_index)
        gen = gens.get(_tuple_key(qid, model, shot_index))
        if gen is None and row["generation_id"]:
            gen = gens_by_id.get(row["generation_id"])
        judges_raw = shot.get("judges") if isinstance(shot, dict) else None
        judges: dict[str, dict[str, Any]] = {}
        if isinstance(judges_raw, dict):
            for key, entry in judges_raw.items():
                compact = compact_judge_entry(entry)
                if compact is not None:
                    judges[str(key)] = compact
        prediction = ""
        if isinstance(shot, dict) and shot.get("answer_prediction") is not None:
            prediction = str(shot.get("answer_prediction") or "")
        elif gen is not None:
            prediction = str(gen.get("answer_prediction") or "")
        shot_payload = {
            "shot_index": shot_index,
            "generation_id": row["generation_id"]
            or (
                str(shot.get("generation_id") or "")
                if isinstance(shot, dict)
                else ""
            )
            or (gen.get("generation_id") if gen else ""),
            "answer_prediction": prediction,
            "ratings": row["ratings"],
            "gold": row["gold"],
            "judges": judges,
            "extra_label": row.get("extra") or {},
            "extra_generation": (gen or {}).get("extra") or {},
        }
        samples.append(
            (
                f"{qid}\t{model}\t{shot_index}",
                list(row["ratings"]),
                judges,
            )
        )
        q_row = by_id[qid]
        model_entry = q_row["models"].setdefault(
            model, {"model_label": model, "shots": []}
        )
        model_entry["shots"].append(shot_payload)

    for q_row in by_id.values():
        for model_entry in q_row["models"].values():
            model_entry["shots"].sort(key=lambda s: s["shot_index"])

    accuracy = compute_accuracy(
        samples, [j["label"] for j in judge_meta], epsilon=epsilon
    )

    mode_keys: dict[str, list[str]] = {name: [] for name in grade_prompt_names()}
    label_mode: dict[str, str] = {}
    for judge in judge_meta:
        mode = judge.get("mode")
        if mode:
            mode_keys.setdefault(str(mode), []).append(judge["label"])
            label_mode[str(judge["label"])] = str(mode)
    gt_keys = mode_keys.get("with_gt") or []

    questions: list[dict[str, Any]] = []
    for qid in question_ids:
        q_row = by_id[qid]
        n_labeled = 0
        n_human_pass = 0
        n_disagree_any = 0
        n_disagree_by_mode: dict[str, int] = {mode: 0 for mode in mode_keys}
        per_judge: dict[str, dict[str, int]] = {
            key: {"n": 0, "n_agree": 0, "n_missing": 0} for key in seen_keys
        }
        for model_entry in q_row["models"].values():
            for shot in model_entry["shots"]:
                n_labeled += 1
                gold = shot.get("gold")
                if gold:
                    n_human_pass += 1
                shot_disagree = False
                shot_disagree_by_mode = {mode: False for mode in mode_keys}
                for key in seen_keys:
                    entry = (shot.get("judges") or {}).get(key)
                    bucket = per_judge[key]
                    correct = None
                    if isinstance(entry, dict) and entry.get("correct") is not None:
                        correct = bool(entry.get("correct"))
                    if correct is None:
                        bucket["n_missing"] += 1
                        continue
                    if gold is None:
                        continue
                    bucket["n"] += 1
                    if correct == gold:
                        bucket["n_agree"] += 1
                    else:
                        shot_disagree = True
                        mode = label_mode.get(key)
                        if mode:
                            shot_disagree_by_mode[mode] = True
                if shot_disagree:
                    n_disagree_any += 1
                for mode, disagreed in shot_disagree_by_mode.items():
                    if disagreed:
                        n_disagree_by_mode[mode] = n_disagree_by_mode.get(mode, 0) + 1
        gt_agree = _average_judge_agreement(per_judge, gt_keys)
        agree_by_mode = {
            mode: _average_judge_agreement(per_judge, keys)
            for mode, keys in mode_keys.items()
        }
        questions.append(
            {
                "id": qid,
                "question": q_row.get("question") or "",
                "answer": q_row.get("answer") or "",
                "modality": q_row.get("modality") or "",
                "category": q_row.get("category") or "",
                "n_labeled": n_labeled,
                "n_human_pass": n_human_pass,
                "n_disagree_any": n_disagree_any,
                "n_disagree_by_mode": n_disagree_by_mode,
                "n_gt": gt_agree["n"],
                "n_gt_agree": gt_agree["n_agree"],
                "avg_gt_agree": gt_agree["rate"],
                "agree_by_mode": agree_by_mode,
                "human_pass_rate": (n_human_pass / n_labeled) if n_labeled else None,
                "per_judge": {
                    key: {
                        **bucket,
                        "accuracy": (
                            (bucket["n_agree"] / bucket["n"]) if bucket["n"] else None
                        ),
                    }
                    for key, bucket in per_judge.items()
                },
                "models": list(q_row["models"]),
            }
        )

    return {
        "pack_dir": str(pack_dir),
        "exports_dir": str(exports_dir),
        "labels_path": str(labels_path),
        "generations_path": str(generations_path),
        "manifest": manifest,
        "model_labels": model_labels,
        "judges": judge_meta,
        "accuracy": accuracy,
        "questions": questions,
        "by_id": by_id,
        "n_label_rows": len(label_rows),
        "n_questions": len(questions),
        "n_generations": len(gen_rows),
        "pack_present": pack_dir.is_dir() and bool(predictions),
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MMAR Judge Outcomes</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg: #e7eef3;
    --ink: #14202a;
    --muted: #5a6b78;
    --line: #b7c7d2;
    --card: #f7fafc;
    --accent: #1f5f8b;
    --good: #1f6b4a;
    --bad: #9b3a3a;
    --soft-good: #d7eee3;
    --soft-bad: #f3dede;
    --soft-warn: #f3e6cf;
    --shadow: 0 1px 0 rgba(20,32,42,0.04), 0 10px 28px rgba(20,32,42,0.07);
    --radius: 12px;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    font-family: "IBM Plex Sans", system-ui, sans-serif;
    color: var(--ink);
    background:
      linear-gradient(160deg, #dfeaf1 0%, transparent 42%),
      linear-gradient(345deg, #cfdde6 0%, transparent 36%),
      var(--bg);
  }
  header {
    position: sticky; top: 0; z-index: 20;
    backdrop-filter: blur(10px);
    background: color-mix(in srgb, var(--bg) 82%, transparent);
    border-bottom: 1px solid var(--line);
    padding: 1rem 1.25rem;
  }
  .header-inner {
    max-width: 1480px; margin: 0 auto;
    display: flex; flex-wrap: wrap; gap: 1rem; align-items: end;
    justify-content: space-between;
  }
  .brand h1 {
    font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
    font-weight: 600; font-size: 1.4rem;
    margin: 0 0 0.15rem; letter-spacing: -0.03em;
  }
  .brand p { margin: 0; color: var(--muted); font-size: 0.9rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: end; }
  label { font-size: 0.75rem; color: var(--muted); display: grid; gap: 0.25rem; }
  select, input[type="search"], button {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.45rem 0.65rem;
  }
  select, input[type="search"] { min-width: 11rem; }
  #mode { min-width: 14rem; font-weight: 500; }
  button { cursor: pointer; }
  button.active { background: #e2eef6; border-color: #8fb3c9; }
  #app-body[hidden] { display: none; }
  .toolbar {
    max-width: 1480px; margin: 0 auto; padding: 0.85rem 1.25rem 0;
    display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: end;
  }
  main {
    max-width: 1480px; margin: 0 auto; padding: 1.25rem;
    display: grid; grid-template-columns: 380px 1fr; gap: 1rem;
  }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  .panel {
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--radius); box-shadow: var(--shadow);
    min-height: 0;
  }
  .panel h2 {
    margin: 0; padding: 0.85rem 1rem;
    font-size: 0.85rem; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--line);
  }
  .stats {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    padding: 0.75rem 1rem; border-bottom: 1px solid var(--line);
    font-size: 0.85rem; color: var(--muted);
  }
  .stats strong { color: var(--ink); }
  .acc-table {
    width: 100%; border-collapse: collapse; font-size: 0.78rem;
  }
  .acc-table th, .acc-table td {
    text-align: left; padding: 0.35rem 0.55rem;
    border-bottom: 1px solid var(--line);
  }
  .acc-table th {
    font-size: 0.68rem; letter-spacing: 0.04em; text-transform: uppercase;
    color: var(--muted); font-weight: 600;
  }
  .acc-table tr { cursor: pointer; }
  .acc-table tr:hover { background: #eef5fa; }
  .acc-table tr.selected { background: #e2eef6; }
  .acc-table .mono { font-family: "IBM Plex Mono", monospace; }
  .acc-wrap { padding: 0.4rem 0.55rem 0.7rem; overflow: auto; max-height: 240px; }
  .mode-label {
    font-size: 0.68rem; letter-spacing: 0.04em; text-transform: uppercase;
    color: var(--muted); padding: 0.45rem 0.55rem 0.1rem;
  }
  #qlist {
    list-style: none; margin: 0; padding: 0;
    max-height: calc(100vh - 460px); overflow: auto;
  }
  #qlist li {
    border-bottom: 1px solid var(--line);
    padding: 0.75rem 1rem; cursor: pointer;
  }
  #qlist li:hover { background: #eef5fa; }
  #qlist li.active { background: #e2eef6; }
  .qid {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.75rem; color: var(--muted);
  }
  .rate {
    font-family: "IBM Plex Mono", monospace;
    font-weight: 500; font-size: 0.95rem;
  }
  .qtext {
    margin: 0.35rem 0 0; font-size: 0.86rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  #detail { padding: 1rem 1.15rem; max-height: calc(100vh - 160px); overflow: auto; }
  .muted { color: var(--muted); }
  .model-block {
    border: 1px solid var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; margin: 0.75rem 0; background: #fff;
  }
  .model-block h3 {
    margin: 0 0 0.5rem; font-size: 1rem;
    display: flex; gap: 0.6rem; align-items: baseline; flex-wrap: wrap;
  }
  .shot {
    border-top: 1px dashed var(--line);
    padding: 0.65rem 0;
  }
  .shot:first-of-type { border-top: none; }
  .shot-head {
    display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;
    font-family: "IBM Plex Mono", monospace; font-size: 0.8rem;
    margin-bottom: 0.35rem;
  }
  .pass { color: var(--good); background: var(--soft-good); padding: 0.1rem 0.4rem; border-radius: 999px; }
  .fail { color: var(--bad); background: var(--soft-bad); padding: 0.1rem 0.4rem; border-radius: 999px; }
  .agree { color: var(--good); background: var(--soft-good); padding: 0.1rem 0.4rem; border-radius: 999px; }
  .disagree { color: var(--bad); background: var(--soft-bad); padding: 0.1rem 0.4rem; border-radius: 999px; }
  .pending { color: #5a3a12; background: var(--soft-warn); padding: 0.1rem 0.4rem; border-radius: 999px; }
  pre {
    white-space: pre-wrap; word-break: break-word;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem; margin: 0.25rem 0 0;
    background: #f2f6f9; padding: 0.55rem 0.65rem; border-radius: 8px;
  }
  audio { width: 100%; margin-top: 0.5rem; }
  .audio-source {
    margin: 0.3rem 0 0;
    font-size: 0.75rem;
    color: var(--muted);
    word-break: break-all;
  }
  .audio-source a { color: var(--accent); }
  .mode-badge {
    display: inline-flex; align-items: center;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.04em; text-transform: uppercase;
    padding: 0.2rem 0.55rem; border-radius: 999px;
    border: 1px solid var(--line);
    color: #5a3a12; background: #f3e6cf; border-color: #d4b88a;
  }
  .brand-row {
    display: flex; flex-wrap: wrap; gap: 0.55rem; align-items: center;
  }
  details.accordion {
    border: 1px solid var(--line); border-radius: 10px;
    background: #fff; margin: 0.75rem 0; overflow: hidden;
  }
  details.accordion > summary {
    cursor: pointer; list-style: none;
    padding: 0.75rem 0.9rem;
    font-weight: 600; font-size: 0.92rem;
    display: flex; align-items: center; justify-content: space-between;
    gap: 0.75rem; user-select: none;
  }
  details.accordion > summary::-webkit-details-marker { display: none; }
  details.accordion > summary::after {
    content: "+";
    font-family: "IBM Plex Mono", monospace;
    color: var(--muted); font-weight: 500;
  }
  details.accordion[open] > summary {
    border-bottom: 1px solid var(--line);
    background: #f2f6f9;
  }
  details.accordion[open] > summary::after { content: "−"; }
  .accordion-body { padding: 0.75rem 0.9rem; }
  .judge-chips { display: flex; flex-wrap: wrap; gap: 0.3rem; margin: 0.25rem 0 0.15rem; }
  .chip {
    font-size: 0.7rem; font-family: "IBM Plex Mono", monospace;
    padding: 0.15rem 0.4rem; border-radius: 999px;
    background: #e8eef2; color: var(--muted);
  }
  .chip.selected { background: #e2eef6; color: var(--accent); }
  .kv {
    display: grid; grid-template-columns: max-content 1fr;
    gap: 0.15rem 0.75rem; font-size: 0.8rem; margin: 0.35rem 0;
  }
  .kv dt { color: var(--muted); font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; }
  .kv dd { margin: 0; }
  .answer-box {
    border: 1px solid var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; margin: 0.75rem 0; background: #fff;
  }
  .banner {
    font-size: 0.85rem; color: var(--muted);
    background: var(--soft-warn); border-radius: 8px;
    padding: 0.45rem 0.7rem; margin: 0.75rem 0;
  }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <div class="brand-row">
        <h1>MMAR Judge Outcomes</h1>
        <span class="mode-badge">vs human labels</span>
      </div>
      <p>Judge verdicts joined to <code>exports/labels.csv</code> and <code>exports/generations.csv</code></p>
      <p id="format-hint" class="muted" hidden>Select a judge format to open the rest of the viewer.</p>
    </div>
    <div class="controls">
      <label>Judge format
        <select id="mode" required></select>
      </label>
    </div>
  </div>
</header>
<div id="app-body" hidden>
  <div class="toolbar">
    <label>Search
      <input id="search" type="search" placeholder="id / question / answer" />
    </label>
    <label>Model
      <select id="model"><option value="">All</option></select>
    </label>
    <label>Match
      <select id="match">
        <option value="">All</option>
        <option value="disagree">Disagree</option>
        <option value="agree">Agree</option>
        <option value="missing">Missing verdict</option>
      </select>
    </label>
    <label>Human
      <select id="human">
        <option value="">All</option>
        <option value="pass">Pass</option>
        <option value="fail">Fail</option>
      </select>
    </label>
    <label>Sort
      <select id="sort">
        <option value="disagree">Disagree (high → low)</option>
        <option value="gt_agree">Judge-format agree (low → high)</option>
      </select>
    </label>
  </div>
  <main>
    <section class="panel">
      <h2>Alt-Test vs human</h2>
      <div class="stats" id="stats">Loading…</div>
      <div id="accuracy"></div>
      <h2>Questions</h2>
      <ul id="qlist"></ul>
    </section>
    <section class="panel">
      <h2>Detail</h2>
      <div id="detail"><p class="muted">Select a question.</p></div>
    </section>
  </main>
</div>
<script>
const state = {
  questions: [],
  modelLabels: [],
  judges: [],
  accuracy: {},
  modeTitles: {},
  modeOrder: [],
  nLabelRows: 0,
  nQuestions: 0,
  nGenerations: 0,
  packPresent: true,
  selectedId: null,
  selectedJudge: null,
};

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => {
    return ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c];
  });
}

function fmtRate(v) {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return (100 * n).toFixed(1) + "%";
}

function prettyJudge(key) {
  if (!key) return "judge";
  const parts = String(key).split("__");
  if (parts.length >= 3) {
    const gold = parts[parts.length - 1] === "nongold" ? "no gold" : parts[parts.length - 1];
    const prompt = parts[parts.length - 2];
    const label = parts.slice(0, -2).join("__");
    const title = (state.modeTitles || {})[prompt];
    return title ? `${label} · ${title}` : `${label} · ${prompt} · ${gold}`;
  }
  return String(key);
}

function shortLabel(label) {
  const map = {
    "af-next-think": "af-next",
    "music-flamingo": "mf",
    "mimo-audio-7b": "mimo",
    "interactive-omni-8b": "i-omni",
    "qwen3-omni": "qwen3",
    "qwen3-omni-instruct": "qwen3-i",
    "voxtral-small-24b": "voxtral",
    "phi-4-multimodal": "phi-4",
    "gemma-4-e4b": "gemma-e4b",
    "gemma-4-12b": "gemma-12b",
    "nemotron-3-nano-omni": "nemotron",
    "gemini-3.7-flash": "gemini",
    "gpt-4o-mini": "4o-mini",
  };
  return map[label] || label;
}

function shortJudge(key) {
  const parts = String(key || "").split("__");
  const model = parts.length >= 3 ? parts.slice(0, -2).join("__") : String(key || "");
  const prompt = parts.length >= 3 ? parts[parts.length - 2] : "";
  const promptShort = String(prompt || "").replaceAll("_", "-");
  const base = shortLabel(model);
  return promptShort ? `${base}/${promptShort}` : base;
}

function formatOrder() {
  const named = Array.isArray(state.modeOrder) ? state.modeOrder.filter(Boolean) : [];
  if (named.length) {
    const extra = accuracyModeOrder().filter(mode => !named.includes(mode));
    return named.concat(extra);
  }
  return accuracyModeOrder();
}

function selectedMode() {
  return String((document.getElementById("mode") || {}).value || "");
}

function formatTitle(mode) {
  if (!mode) return "";
  const title = (state.modeTitles || {})[mode] || mode;
  return String(title).split(" (")[0];
}

function syncAppVisibility() {
  const mode = selectedMode();
  const body = document.getElementById("app-body");
  if (body) body.hidden = !mode;
  const hint = document.getElementById("format-hint");
  if (hint) hint.hidden = !!mode;
}

function fillModeSelect() {
  const sel = document.getElementById("mode");
  if (!sel) return;
  const current = sel.value;
  const order = formatOrder();
  sel.innerHTML = order.map(mode => {
    const label = formatTitle(mode) || mode;
    return `<option value="${escapeHtml(mode)}">${escapeHtml(label)}</option>`;
  }).join("");
  const preferred = (current && order.includes(current)) ? current : (order[0] || "");
  sel.value = preferred;
  syncAppVisibility();
}

function accuracyModeOrder() {
  const acc = state.accuracy || {};
  const fallback = state.modeOrder || [];
  const named = Array.isArray(acc.modes) && acc.modes.length ? acc.modes : fallback;
  const extra = Object.keys(acc).filter(key => {
    if (named.includes(key)) return false;
    if (["n_label_rows", "epsilon", "modes"].includes(key)) return false;
    return acc[key] && typeof acc[key] === "object" && !Array.isArray(acc[key]);
  });
  return named.concat(extra);
}

function rowModeAgree(row) {
  const mode = selectedMode();
  const byMode = row.agree_by_mode || {};
  if (mode && byMode[mode] && typeof byMode[mode].rate === "number") {
    return byMode[mode].rate;
  }
  return null;
}

function rowModeDisagree(row) {
  const mode = selectedMode();
  const byMode = row.n_disagree_by_mode || {};
  if (mode && typeof byMode[mode] === "number") return byMode[mode];
  return 0;
}

function modeAgreeLabel() {
  const mode = selectedMode();
  if (!mode) return "judge-format agree";
  return `${formatTitle(mode) || mode} agree`;
}

function visibleJudges() {
  const mode = selectedMode();
  if (!mode) return [];
  return (state.judges || []).filter(j => j.mode === mode);
}

function questionMatch(row) {
  const judge = state.selectedJudge;
  const match = document.getElementById("match").value;
  const human = document.getElementById("human").value;
  const model = document.getElementById("model").value;
  if (model && !(row.models || []).includes(model)) return false;
  if (human === "pass" && !(row.n_human_pass > 0)) return false;
  if (human === "fail" && !(row.n_labeled - (row.n_human_pass || 0) > 0)) return false;
  if (!match) return true;
  if (!judge) {
    if (match === "disagree") return rowModeDisagree(row) > 0;
    if (match === "agree") return rowModeDisagree(row) === 0;
    if (match === "missing") {
      return visibleJudges().some(j => Number((row.per_judge || {})[j.label]?.n_missing || 0) > 0);
    }
    return true;
  }
  const stats = (row.per_judge || {})[judge] || {};
  const n = Number(stats.n || 0);
  const agree = Number(stats.n_agree || 0);
  const missing = Number(stats.n_missing || 0);
  if (match === "disagree") return n > 0 && agree < n;
  if (match === "agree") return n > 0 && agree === n;
  if (match === "missing") return missing > 0;
  return true;
}

function filteredQuestions() {
  const q = (document.getElementById("search").value || "").trim().toLowerCase();
  const items = state.questions.filter(row => {
    if (!questionMatch(row)) return false;
    if (!q) return true;
    return String(row.id).toLowerCase().includes(q)
      || String(row.question || "").toLowerCase().includes(q)
      || String(row.answer || "").toLowerCase().includes(q);
  });
  const judge = state.selectedJudge;
  const sort = (document.getElementById("sort") || {}).value || "disagree";
  return items.slice().sort((a, b) => {
    if (sort === "gt_agree") {
      const ar = rowModeAgree(a);
      const br = rowModeAgree(b);
      const aOk = typeof ar === "number" && Number.isFinite(ar);
      const bOk = typeof br === "number" && Number.isFinite(br);
      if (!aOk && !bOk) return String(a.id).localeCompare(String(b.id));
      if (!aOk) return 1;
      if (!bOk) return -1;
      if (ar !== br) return ar - br;
      return String(a.id).localeCompare(String(b.id));
    }
    const statsA = judge ? (a.per_judge || {})[judge] : null;
    const statsB = judge ? (b.per_judge || {})[judge] : null;
    const discA = judge
      ? (Number(statsA?.n || 0) - Number(statsA?.n_agree || 0))
      : rowModeDisagree(a);
    const discB = judge
      ? (Number(statsB?.n || 0) - Number(statsB?.n_agree || 0))
      : rowModeDisagree(b);
    if (discA !== discB) return discB - discA;
    return String(a.id).localeCompare(String(b.id));
  });
}

function renderStats() {
  const items = filteredQuestions();
  const parts = [
    `<span><strong>${items.length}</strong> shown</span>`,
    `<span>${state.nQuestions} questions</span>`,
    `<span>${state.nLabelRows} labeled shots</span>`,
    `<span>${state.nGenerations} generations</span>`,
    `<span>${visibleJudges().length} judges</span>`,
  ];
  document.getElementById("stats").innerHTML = parts.join(" · ");
}

function sortJudgeKeys(byJudge) {
  return Object.keys(byJudge || {}).sort((a, b) => {
    const rhoA = Number(byJudge[a]?.advantage_prob);
    const rhoB = Number(byJudge[b]?.advantage_prob);
    const aOk = Number.isFinite(rhoA);
    const bOk = Number.isFinite(rhoB);
    if (aOk && bOk && rhoA !== rhoB) return rhoB - rhoA;
    if (aOk !== bOk) return aOk ? -1 : 1;
    return String(a).localeCompare(String(b));
  });
}

function passLabel(passed) {
  if (passed === true) return "yes";
  if (passed === false) return "no";
  return "—";
}

function renderAccuracy() {
  const filter = selectedMode();
  if (!filter) {
    document.getElementById("accuracy").innerHTML = "";
    return;
  }
  if (state.selectedJudge) {
    const meta = (state.judges || []).find(j => j.label === state.selectedJudge);
    if (meta && meta.mode && meta.mode !== filter) {
      state.selectedJudge = null;
    }
  }
  const wrap = document.getElementById("accuracy");
  const acc = state.accuracy || {};
  const selected = state.selectedJudge;
  let html = "";
  const byJudge = acc[filter] || {};
  const keys = sortJudgeKeys(byJudge);
  if (keys.length) {
    const title = (state.modeTitles || {})[filter] || filter;
    html += `<div class="mode-label">${escapeHtml(title)}</div>`;
    html += `<div class="acc-wrap"><table class="acc-table"><thead>
      <tr><th>Judge</th><th>n</th><th>miss</th><th>ρ</th><th>ω</th><th>pass</th></tr>
    </thead><tbody>`;
    for (const key of keys) {
      const row = byJudge[key];
      const klass = key === selected ? "selected" : "";
      html += `<tr class="${klass}" data-judge="${escapeHtml(key)}">
        <td title="${escapeHtml(prettyJudge(key))}">${escapeHtml(shortJudge(key))}</td>
        <td class="mono">${row.n ?? 0}</td>
        <td class="mono">${row.n_missing ?? 0}</td>
        <td class="mono">${fmtRate(row.advantage_prob)}</td>
        <td class="mono">${fmtRate(row.winning_rate)}</td>
        <td class="mono">${passLabel(row.passed)}</td>
      </tr>`;
    }
    html += `</tbody></table></div>`;
  }
  if (!html) {
    html = `<p class="muted" style="padding:0.6rem 1rem">No judge verdicts for <code>${escapeHtml(filter)}</code> in the pack yet. Download with <code>uv run modal run download_judges.py</code>.</p>`;
  }
  wrap.innerHTML = html;
  wrap.querySelectorAll("tr[data-judge]").forEach(tr => {
    tr.addEventListener("click", () => {
      const key = tr.dataset.judge;
      state.selectedJudge = state.selectedJudge === key ? null : key;
      renderAccuracy();
      renderList();
    });
  });
}

function renderList() {
  renderStats();
  const list = document.getElementById("qlist");
  const items = filteredQuestions();
  const judge = state.selectedJudge;
  list.innerHTML = items.map(row => {
    const stats = judge ? (row.per_judge || {})[judge] : null;
    const acc = stats ? stats.accuracy : null;
    const disc = judge
      ? (Number(stats?.n || 0) - Number(stats?.n_agree || 0))
      : rowModeDisagree(row);
    const sort = (document.getElementById("sort") || {}).value || "disagree";
    const rateLabel = sort === "gt_agree"
      ? `${fmtRate(rowModeAgree(row))} ${modeAgreeLabel()}`
      : judge
        ? `${fmtRate(acc)} agree · ${disc} disagree`
        : `${fmtRate(row.human_pass_rate)} human pass · ${disc} any disagree`;
    const active = row.id === state.selectedId ? "active" : "";
    return `<li class="${active}" data-id="${escapeHtml(row.id)}">
      <div class="qid">${escapeHtml(row.id)}</div>
      <div class="rate">${rateLabel} · ${row.n_labeled} shots</div>
      <p class="qtext">${escapeHtml(row.question || "")}</p>
    </li>`;
  }).join("");
  list.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => selectQuestion(li.dataset.id));
  });
}

function goldChip(gold) {
  if (gold === null || gold === undefined) {
    return `<span class="pending">human tie</span>`;
  }
  return gold
    ? `<span class="pass">human pass</span>`
    : `<span class="fail">human fail</span>`;
}

function matchChip(entry, gold) {
  if (!entry || entry.correct === null || entry.correct === undefined) {
    return `<span class="pending">missing</span>`;
  }
  if (gold === null || gold === undefined) {
    const verdict = entry.verdict || (entry.correct ? "pass" : "fail");
    return `<span class="pending">no majority · ${escapeHtml(verdict)}</span>`;
  }
  const same = !!entry.correct === !!gold;
  const verdict = entry.verdict || (entry.correct ? "pass" : "fail");
  return same
    ? `<span class="agree">agree · ${escapeHtml(verdict)}</span>`
    : `<span class="disagree">disagree · ${escapeHtml(verdict)}</span>`;
}

function extraKv(obj) {
  const keys = Object.keys(obj || {}).filter(k => obj[k] !== "" && obj[k] != null);
  if (!keys.length) return "";
  return `<dl class="kv">${keys.map(k =>
    `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(typeof obj[k] === "string" ? obj[k] : JSON.stringify(obj[k]))}</dd>`
  ).join("")}</dl>`;
}

function judgeFields(entry) {
  const skip = new Set(["correct", "verdict", "output", "generation"]);
  const rest = {};
  for (const [k, v] of Object.entries(entry || {})) {
    if (!skip.has(k) && v !== null && v !== undefined && v !== "") rest[k] = v;
  }
  return extraKv(rest);
}

function shotJudgeBlock(shot, gold) {
  const judges = visibleJudges();
  const onShot = shot.judges || {};
  const keys = judges.map(j => j.label).filter(k => k in onShot);
  if (!keys.length) {
    return `<div class="judge-chips"><span class="chip">no ${escapeHtml(formatTitle(selectedMode()) || "format")} verdicts</span></div>`;
  }
  const chips = keys.map(key => {
    const entry = onShot[key] || {};
    const selected = key === state.selectedJudge ? " selected" : "";
    return `<span class="chip${selected}" title="${escapeHtml(prettyJudge(key))}">${escapeHtml(shortJudge(key))} ${matchChip(entry, gold)}</span>`;
  }).join("");
  const accordions = keys.map(key => {
    const entry = onShot[key] || {};
    const text = entry.generation || entry.output || "";
    if (!text && !Object.keys(entry).length) return "";
    return `<details class="accordion">
      <summary><span>${escapeHtml(prettyJudge(key))} ${matchChip(entry, gold)}</span></summary>
      <div class="accordion-body">
        ${entry.output && entry.output !== entry.generation ? `<p class="muted">parsed output: <code>${escapeHtml(entry.output)}</code></p>` : ""}
        ${judgeFields(entry)}
        ${text ? `<pre>${escapeHtml(text)}</pre>` : `<p class="muted">No generation stored.</p>`}
      </div>
    </details>`;
  }).join("");
  return `<div class="judge-chips">${chips}</div>${accordions}`;
}

async function selectQuestion(id) {
  state.selectedId = id;
  renderList();
  const detail = document.getElementById("detail");
  detail.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const data = await api("/api/question?id=" + encodeURIComponent(id));
    const q = data.question || {};
    const models = data.model_labels || state.modelLabels;
    const banner = state.packPresent ? "" :
      `<div class="banner">Judging pack not found locally. Labels and generations still render from exports/. Run <code>uv run modal run download_judges.py</code>.</div>`;
    const audio = data.audio_url
      ? `<audio controls preload="none" src="${escapeHtml(data.audio_url)}"></audio>`
      : `<p class="muted">Audio not found locally.</p>`;
    const source = q.url
      ? `<p class="audio-source"><a href="${escapeHtml(q.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(q.url)}</a></p>`
      : "";
    const gold = q.answer
      ? `<div class="answer-box"><strong>Gold answer</strong><pre>${escapeHtml(q.answer)}</pre></div>`
      : "";
    const blocks = models.map(label => {
      const entry = (q.models || {})[label];
      if (!entry) return "";
      const shots = (entry.shots || []).map(shot => {
        const human = goldChip(shot.gold);
        const ratings = (shot.ratings || []).map((v, i) =>
          `<span class="${v ? "pass" : "fail"}">r${i}:${v ? "pass" : "fail"}</span>`
        ).join(" ");
        const gid = shot.generation_id
          ? `<span class="chip">gen ${escapeHtml(String(shot.generation_id))}</span>`
          : "";
        return `<div class="shot">
          <div class="shot-head">
            <span>s${shot.shot_index}</span>
            ${human}
            ${ratings}
            ${gid}
          </div>
          <pre>${escapeHtml(shot.answer_prediction || "")}</pre>
          ${extraKv(shot.extra_label)}
          ${shotJudgeBlock(shot, shot.gold)}
        </div>`;
      }).join("");
      return `<div class="model-block">
        <h3>${escapeHtml(label)}</h3>
        ${shots || `<p class="muted">No labeled shots.</p>`}
      </div>`;
    }).join("");
    detail.innerHTML = `
      ${banner}
      <div class="qid">${escapeHtml(q.id || id)} · ${escapeHtml(q.modality || "")} · ${escapeHtml(q.category || "")}</div>
      <h3 style="margin:0.35rem 0 0.2rem;font-family:Space Grotesk,sans-serif">${escapeHtml(q.question || "")}</h3>
      ${audio}
      ${source}
      ${gold}
      ${blocks || `<p class="muted">No labeled generations for this question.</p>`}
    `;
  } catch (err) {
    detail.innerHTML = `<p class="muted">Failed to load question: ${escapeHtml(String(err))}</p>`;
  }
}

async function init() {
  const pack = await api("/api/pack");
  state.questions = pack.questions || [];
  state.modelLabels = pack.model_labels || [];
  state.judges = pack.judges || [];
  state.accuracy = pack.accuracy || {};
  if (pack.mode_titles && typeof pack.mode_titles === "object") {
    state.modeTitles = pack.mode_titles;
  }
  if (Array.isArray(pack.mode_order) && pack.mode_order.length) {
    state.modeOrder = pack.mode_order;
  }
  fillModeSelect();
  state.nLabelRows = pack.n_label_rows || 0;
  state.nQuestions = pack.n_questions || 0;
  state.nGenerations = pack.n_generations || 0;
  state.packPresent = !!pack.pack_present;
  const sel = document.getElementById("model");
  sel.innerHTML = `<option value="">All</option>` + state.modelLabels.map(m =>
    `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`
  ).join("");
  const onFilter = () => {
    renderAccuracy();
    renderList();
    const items = filteredQuestions();
    if (items.length && !items.some(row => row.id === state.selectedId)) {
      selectQuestion(items[0].id);
    }
  };
  ["search", "model", "match", "human", "sort"].forEach(id => {
    document.getElementById(id).addEventListener("input", onFilter);
    document.getElementById(id).addEventListener("change", onFilter);
  });
  document.getElementById("mode").addEventListener("change", () => {
    syncAppVisibility();
    if (!selectedMode()) return;
    state.selectedJudge = null;
    renderAccuracy();
    renderList();
    const items = filteredQuestions();
    if (state.selectedId && items.some(row => row.id === state.selectedId)) {
      selectQuestion(state.selectedId);
      return;
    }
    if (items.length) selectQuestion(items[0].id);
  });
  if (!selectedMode()) return;
  renderAccuracy();
  if (!state.questions.length) {
    document.getElementById("stats").textContent = "No labeled questions in exports/.";
    document.getElementById("detail").innerHTML = `<p class="muted">Need exports/labels.csv.</p>`;
    return;
  }
  const preferred = new URLSearchParams(location.search).get("id");
  const start = (preferred && state.questions.find(q => q.id === preferred))
    ? preferred
    : filteredQuestions()[0]?.id || state.questions[0].id;
  renderList();
  await selectQuestion(start);
}

init().catch(err => {
  const body = document.getElementById("app-body");
  if (body) body.hidden = false;
  const stats = document.getElementById("stats");
  if (stats) stats.textContent = String(err);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[view_judges] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _bundle(self) -> dict[str, Any]:
        return load_bundle(
            str(CONFIG["pack_dir"]),
            str(CONFIG["exports_dir"]),
            float(CONFIG.get("epsilon", DEFAULT_EPSILON)),
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/pack":
                bundle = self._bundle()
                self._send_json(
                    {
                        "questions": bundle["questions"],
                        "manifest": bundle["manifest"],
                        "model_labels": bundle["model_labels"],
                        "judges": bundle["judges"],
                        "accuracy": bundle["accuracy"],
                        "mode_titles": grade_mode_titles(),
                        "mode_order": list(grade_prompt_names()),
                        "n_label_rows": bundle["n_label_rows"],
                        "n_questions": bundle["n_questions"],
                        "n_generations": bundle["n_generations"],
                        "pack_present": bundle["pack_present"],
                        "labels_path": bundle["labels_path"],
                        "generations_path": bundle["generations_path"],
                    }
                )
                return

            if path == "/api/question":
                qid = (qs.get("id") or [""])[0]
                if not qid:
                    self._send_json({"error": "missing id"}, 400)
                    return
                bundle = self._bundle()
                row = bundle["by_id"].get(qid)
                if row is None:
                    self._send_json({"error": "question not found"}, 404)
                    return
                audio = resolve_audio(row.get("audio_path"))
                audio_url = f"/audio/{audio.name}" if audio is not None else None
                self._send_json(
                    {
                        "question": row,
                        "audio_url": audio_url,
                        "model_labels": [
                            label
                            for label in bundle["model_labels"]
                            if label in (row.get("models") or {})
                        ],
                        "judges": bundle["judges"],
                    }
                )
                return

            if path.startswith("/audio/"):
                name = unquote(path[len("/audio/") :])
                audio = Path(CONFIG["audio_dir"]) / name
                if not audio.is_file():
                    self.send_error(404, "audio not found")
                    return
                data = audio.read_bytes()
                ctype = mimetypes.guess_type(str(audio))[0] or "audio/wav"
                self._send(200, data, ctype)
                return

            self.send_error(404, "not found")
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, 404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=PACK_DIR,
        help="Downloaded judging pack (default: outputs/mmar-judging)",
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=EXPORTS_DIR,
        help="Directory with labels.csv and generations.csv",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Local MMAR wav directory",
    )
    parser.add_argument(
        "--skip-audio-download",
        action="store_true",
        help="Do not download MMAR wavs if the local audio cache is incomplete",
    )
    parser.add_argument(
        "--force-audio-download",
        action="store_true",
        help="Re-download the MMAR wav archive even if wavs are present",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Alt-Test cost-benefit penalty for winning rate (default: 0.15). "
        "Average advantage probability ρ does not use this.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    pack_dir = args.pack_dir.expanduser().resolve()
    exports_dir = args.exports_dir.expanduser().resolve()
    CONFIG["pack_dir"] = pack_dir
    CONFIG["exports_dir"] = exports_dir
    CONFIG["epsilon"] = float(args.epsilon)
    audio_dir = args.audio_dir.expanduser().resolve()
    if not args.skip_audio_download:
        try:
            audio_dir = ensure_mmar_audio(
                audio_dir, force=args.force_audio_download
            )
        except SystemExit as exc:
            print(f"Audio setup failed: {exc}", flush=True)
            print("Continuing without local audio; pass --skip-audio-download to silence.")
    CONFIG["audio_dir"] = audio_dir
    MMAR_CONFIG["audio_dir"] = audio_dir
    load_bundle.cache_clear()

    print(f"Pack:    {pack_dir}")
    print(f"Exports: {exports_dir}")
    print(f"Audio:   {audio_dir}")
    labels_path = exports_dir / LABELS_CSV_NAME
    if not labels_path.is_file() and not (pack_dir / LABELS_CSV_NAME).is_file():
        print(f"No {LABELS_CSV_NAME} at {labels_path} or {pack_dir / LABELS_CSV_NAME}.")
    if not pack_dir.is_dir():
        print("Pack directory not found. Run: uv run modal run download_judges.py")
    bundle = load_bundle(
        str(pack_dir), str(exports_dir), float(args.epsilon)
    )
    print(
        f"Loaded {bundle['n_questions']} questions, "
        f"{bundle['n_label_rows']} labeled shots, "
        f"{len(bundle['judges'])} judges, "
        f"{len(bundle['model_labels'])} models "
        f"(ε={float(args.epsilon):.2f})"
    )
    if not bundle["pack_present"]:
        print("No local judging predictions; viewer will show exports/ only.")
    for mode in accuracy_mode_names(bundle.get("accuracy") or {}):
        by_judge = (bundle["accuracy"] or {}).get(mode) or {}
        if not by_judge:
            continue
        title = grade_mode_title(mode)
        print(f"  {title}:")

        def _rho_key(name: str, table: dict = by_judge) -> tuple[float, str]:
            rho = (table.get(name) or {}).get("advantage_prob")
            rank = -float(rho) if isinstance(rho, (int, float)) else 1.0
            return (rank, name)

        for key in sorted(by_judge, key=_rho_key):
            row = by_judge[key]
            rho = row.get("advantage_prob")
            rho_s = f"{rho:.3f}" if isinstance(rho, (int, float)) else "—"
            llm_a = row.get("loo_agree_judge")
            llm_s = f"{llm_a:.3f}" if isinstance(llm_a, (int, float)) else "—"
            hum_a = row.get("loo_agree_human")
            hum_s = f"{hum_a:.3f}" if isinstance(hum_a, (int, float)) else "—"
            wr = row.get("winning_rate")
            wr_s = f"{wr:.3f}" if isinstance(wr, (int, float)) else "—"
            passed = row.get("passed")
            if passed is True:
                pass_s = "yes"
            elif passed is False:
                pass_s = "no"
            else:
                pass_s = "—"
            print(
                f"    {key:<52} n={row.get('n', 0):<5} "
                f"ρ={rho_s:<6} llm={llm_s:<6} hum={hum_s:<6} "
                f"ω={wr_s:<6} pass={pass_s}"
            )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
