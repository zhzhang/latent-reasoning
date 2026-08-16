"""Side-by-side viewer for MMAR-Rubrics and audio-groundedness judge outputs.

Compares Qwen (Modal), GPT-4o, and Claude judgments over the same
test-taker traces (qwen3-omni or af-next-think first shot). Each
test-taker is a separate dropdown entry. Reads:

    outputs/exp-mmar-rubrics/<run_id>/              # Modal download
    outputs/exp-mmar-rubrics-api/<run_id>/          # local API (never overwritten)
    outputs/exp-mmar-groundedness/<run_id>/         # Qwen3-Omni audio judge
    outputs/exp-mmar-groundedness-api/<run_id>/     # Gemini 3.1 Pro audio judge

Usage::

    uv run python view_mmar_rubrics.py
    uv run python view_mmar_rubrics.py --port 7862
    uv run python view_mmar_rubrics.py --run-id 20260807T144946Z::qwen3-omni
    uv run python view_mmar_rubrics.py --run-id 20260807T144946Z::af-next-think
    uv run python view_mmar_rubrics.py --run-id 20260807T144946Z::af-next-think::groundedness
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from mmar_common import load_jsonl
from mmar_groundedness import GROUNDEDNESS_API_EXPERIMENT, GROUNDEDNESS_EXPERIMENT
from mmar_rubrics import (
    DEFAULT_MODEL_LABEL,
    RUBRICS_API_EXPERIMENT,
    RUBRICS_EXPERIMENT,
    SOURCE_EXPERIMENT,
    discover_test_taker_labels,
    eval_run_key,
    first_shot_fields,
    parse_eval_run_key,
    parse_eval_run_parts,
)
import view_difficulty as vd
from view_difficulty import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_RESULTS_DIR,
    ensure_mmar_audio,
)

REPO_ROOT = Path(__file__).resolve().parent
CONFIG: dict[str, Any] = {}

# Preferred column order for judges in rubrics / groundedness experiments.
PREFERRED_JUDGES = (
    "qwen3-omni",
    "gemini-3.1-pro-preview",
    "qwen3.6-35b-a3b-fp8",
    "gpt-4o-2024-11-20",
    "claude-sonnet-4-5",
)


def _sync_vd_config() -> None:
    """Keep view_difficulty.CONFIG in sync for shared audio helpers."""
    vd.CONFIG["results_dir"] = CONFIG.get("results_dir")
    vd.CONFIG["audio_dir"] = CONFIG.get("audio_dir")


def resolve_audio(audio_path: str | None) -> Path | None:
    """Resolve an audio file path without requiring view_difficulty.CONFIG."""
    if not audio_path:
        return None
    path = Path(audio_path)
    if path.is_file():
        return path
    name = path.name
    audio_dir = Path(CONFIG.get("audio_dir") or DEFAULT_AUDIO_DIR)
    candidates = [
        audio_dir / name,
        REPO_ROOT / "data" / "mmar" / "audio" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def discover_rubric_runs(results_dir: Path) -> list[dict]:
    """Find (source_run_id, model_label, kind) evals under rubrics or groundedness.

    Each test-taker is a separate dropdown entry even when they share a
    source difficulty run (e.g. qwen3-omni vs af-next-think). Groundedness
    evals are listed separately from rubric scoring.
    """
    groups = [
        (
            "rubrics",
            [
                (RUBRICS_EXPERIMENT, results_dir / RUBRICS_EXPERIMENT),
                (RUBRICS_API_EXPERIMENT, results_dir / RUBRICS_API_EXPERIMENT),
            ],
        ),
        (
            "groundedness",
            [
                (GROUNDEDNESS_EXPERIMENT, results_dir / GROUNDEDNESS_EXPERIMENT),
                (GROUNDEDNESS_API_EXPERIMENT, results_dir / GROUNDEDNESS_API_EXPERIMENT),
            ],
        ),
    ]
    by_key: dict[tuple[str, str, str], dict] = {}
    for kind, roots in groups:
        for experiment, root in roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                source_run_id = child.name
                manifest = _read_json(child / "manifest.json")
                model_labels = discover_test_taker_labels(child, manifest)
                if not model_labels:
                    model_labels = [
                        str(manifest.get("model_label") or DEFAULT_MODEL_LABEL)
                    ]
                declared_judges: list[str] = []
                for judge in manifest.get("judges") or []:
                    if isinstance(judge, dict) and judge.get("label"):
                        label = str(judge["label"])
                        if label not in declared_judges:
                            declared_judges.append(label)
                judges_dir = child / "judges"
                if judges_dir.is_dir():
                    for jdir in judges_dir.iterdir():
                        if jdir.is_dir() and jdir.name not in declared_judges:
                            declared_judges.append(jdir.name)
                for model_label in model_labels:
                    key = (source_run_id, model_label, kind)
                    entry = by_key.setdefault(
                        key,
                        {
                            "run_id": eval_run_key(
                                source_run_id, model_label, kind
                            ),
                            "source_run_id": source_run_id,
                            "model_label": model_label,
                            "eval_kind": kind,
                            "roots": {},
                            "judges": [],
                        },
                    )
                    entry["roots"][experiment] = str(child)
                    for label in declared_judges:
                        path = _evaluated_path(child, label, model_label)
                        if path.is_file() and label not in entry["judges"]:
                            entry["judges"].append(label)
    runs = list(by_key.values())
    for run in runs:
        run["judges"] = _order_judges(run["judges"])
    runs.sort(
        key=lambda r: (
            r["source_run_id"],
            r["eval_kind"],
            r["model_label"],
        ),
        reverse=True,
    )
    return runs


def _order_judges(labels: list[str]) -> list[str]:
    preferred = [j for j in PREFERRED_JUDGES if j in labels]
    rest = sorted(j for j in labels if j not in preferred)
    return preferred + rest


def _evaluated_path(run_root: Path, judge_label: str, model_label: str) -> Path:
    return (
        run_root
        / "judges"
        / judge_label
        / "models"
        / model_label
        / "predictions.evaluated.jsonl"
    )


def _judge_file_signature(results_dir: Path, run_id: str) -> tuple:
    """MTimes of evaluated jsonl files so the bundle cache refreshes after download."""
    source_run_id, model_label, kind = parse_eval_run_parts(run_id)
    experiments = (
        (GROUNDEDNESS_EXPERIMENT, GROUNDEDNESS_API_EXPERIMENT)
        if kind == "groundedness"
        else (RUBRICS_EXPERIMENT, RUBRICS_API_EXPERIMENT)
    )
    sig: list[tuple[str, str, int, int]] = []
    for experiment in experiments:
        root = results_dir / experiment / source_run_id / "judges"
        if not root.is_dir():
            continue
        pattern = f"*/models/{model_label}/predictions.evaluated.jsonl"
        for path in sorted(root.glob(pattern)):
            try:
                st = path.stat()
            except OSError:
                continue
            sig.append((experiment, str(path.relative_to(root)), st.st_mtime_ns, st.st_size))
    return tuple(sig)


def _load_evaluated_by_id(path: Path) -> dict[str, dict]:
    """Load evaluated rows; if an id appears twice, prefer the fully graded row."""
    by_id: dict[str, dict] = {}
    for item in load_jsonl(path):
        record_id = item.get("id")
        if not record_id:
            continue
        record_id = str(record_id)
        prev = by_id.get(record_id)
        if prev is None:
            by_id[record_id] = item
            continue
        prev_raw = any(str(r or "").strip() for r in (prev.get("raw_responses") or []))
        new_raw = any(str(r or "").strip() for r in (item.get("raw_responses") or []))
        if new_raw and not prev_raw:
            by_id[record_id] = item
        elif new_raw == prev_raw:
            by_id[record_id] = item  # last write wins
    return by_id


def _record_verdict(rec: dict | None, *, eval_kind: str) -> str | None:
    if not rec:
        return None
    raw = rec.get("verdict")
    if raw in {"Yes", "No", "pass", "fail"}:
        return str(raw)
    if rec.get("correct") is True:
        return "Yes" if eval_kind == "groundedness" else "pass"
    if rec.get("correct") is False:
        return "No" if eval_kind == "groundedness" else "fail"
    return None


@lru_cache(maxsize=16)
def _load_run_bundle_cached(run_id: str, signature: tuple) -> dict[str, Any]:
    del signature  # only used for cache identity
    results_dir = Path(CONFIG["results_dir"])
    runs = {r["run_id"]: r for r in discover_rubric_runs(results_dir)}
    meta = runs.get(run_id)
    if not meta:
        raise FileNotFoundError(f"Rubrics run not found: {run_id}")

    source_run_id = meta.get("source_run_id") or parse_eval_run_key(run_id)[0]
    model_label = meta.get("model_label") or DEFAULT_MODEL_LABEL
    eval_kind = meta.get("eval_kind") or parse_eval_run_parts(run_id)[2]
    judges: dict[str, dict[str, dict]] = {}
    question_ids: list[str] = []
    manifests: dict[str, dict] = {}

    for experiment, root_str in (meta.get("roots") or {}).items():
        root = Path(root_str)
        manifests[experiment] = _read_json(root / "manifest.json")
        qids_payload = _read_json(root / "question_ids.json")
        if not question_ids and isinstance(qids_payload.get("ids"), list):
            question_ids = [str(x) for x in qids_payload["ids"]]
        judges_root = root / "judges"
        if not judges_root.is_dir():
            continue
        for jdir in judges_root.iterdir():
            if not jdir.is_dir():
                continue
            path = _evaluated_path(root, jdir.name, model_label)
            if not path.is_file():
                continue
            by_id = _load_evaluated_by_id(path)
            existing = judges.get(jdir.name) or {}
            for record_id, item in by_id.items():
                prev = existing.get(record_id)
                if prev is None:
                    existing[record_id] = item
                    continue
                prev_raw = any(str(r or "").strip() for r in (prev.get("raw_responses") or []))
                new_raw = any(str(r or "").strip() for r in (item.get("raw_responses") or []))
                if new_raw and not prev_raw:
                    existing[record_id] = item
                elif new_raw == prev_raw:
                    existing[record_id] = item
            judges[jdir.name] = existing

    if not question_ids:
        for by_id in judges.values():
            question_ids = list(by_id.keys())
            break

    source_preds: dict[str, dict] = {}
    source_dir = results_dir / SOURCE_EXPERIMENT / source_run_id
    source_pred_path = source_dir / "models" / model_label / "predictions.jsonl"
    if source_pred_path.is_file():
        source_preds = {
            str(item["id"]): item
            for item in load_jsonl(source_pred_path)
            if item.get("id")
        }

    questions = []
    for qid in question_ids:
        sample = None
        for by_id in judges.values():
            if qid in by_id:
                sample = by_id[qid]
                break
        source = source_preds.get(qid) or {}
        base = sample or source
        if not base:
            continue
        judge_scores = {}
        for jlabel, by_id in judges.items():
            rec = by_id.get(qid)
            if rec is None:
                continue
            judge_scores[jlabel] = {
                "score": rec.get("score"),
                "correct": rec.get("correct"),
                "verdict": _record_verdict(rec, eval_kind=eval_kind),
            }
        questions.append(
            {
                "id": qid,
                "question": base.get("question") or source.get("question") or "",
                "answer": base.get("answer") or source.get("answer") or "",
                "modality": base.get("modality") or source.get("modality"),
                "category": base.get("category") or source.get("category"),
                "correct_any": any(
                    bool(v.get("correct")) for v in judge_scores.values()
                ),
                "scores": judge_scores,
                "disagree": _judges_disagree(judge_scores),
            }
        )

    return {
        "run_id": run_id,
        "source_run_id": source_run_id,
        "model_label": model_label,
        "eval_kind": eval_kind,
        "judge_labels": _order_judges(list(judges.keys())),
        "manifests": manifests,
        "questions": questions,
        "judges": judges,
        "source_preds": source_preds,
        "meta": meta,
    }


def _resolve_run_id(run_id: str) -> str:
    """Map a dropdown / URL value onto a composite ``source::model`` key."""
    results_dir = Path(CONFIG["results_dir"])
    runs = discover_rubric_runs(results_dir)
    by_id = {r["run_id"]: r for r in runs}
    if run_id in by_id:
        return run_id
    matches = [r for r in runs if r["source_run_id"] == run_id]
    if not matches:
        source_run_id, model_label, kind = parse_eval_run_parts(run_id)
        matches = [
            r
            for r in runs
            if r["source_run_id"] == source_run_id and r["model_label"] == model_label
        ]
        kind_matches = [r for r in matches if r.get("eval_kind") == kind]
        if kind_matches:
            matches = kind_matches
    if len(matches) == 1:
        return matches[0]["run_id"]
    preferred = next(
        (
            r
            for r in matches
            if r["model_label"] == DEFAULT_MODEL_LABEL
            and r.get("eval_kind") != "groundedness"
        ),
        None,
    )
    if preferred:
        return preferred["run_id"]
    if matches:
        return matches[0]["run_id"]
    return run_id


def load_run_bundle(run_id: str) -> dict[str, Any]:
    results_dir = Path(CONFIG["results_dir"])
    resolved = _resolve_run_id(run_id)
    signature = _judge_file_signature(results_dir, resolved)
    return _load_run_bundle_cached(resolved, signature)


def _judges_disagree(judge_scores: dict[str, dict]) -> bool:
    scores = [
        None if v.get("score") is None else round(float(v["score"]), 6)
        for v in judge_scores.values()
    ]
    present = [s for s in scores if s is not None]
    if len(present) < 2:
        return False
    return len(set(present)) > 1


def get_question_detail(run_id: str, question_id: str) -> dict[str, Any]:
    bundle = load_run_bundle(run_id)
    judges = bundle["judges"]
    source = (bundle.get("source_preds") or {}).get(question_id) or {}

    sample = None
    for by_id in judges.values():
        if question_id in by_id:
            sample = by_id[question_id]
            break
    base = sample or source
    if not base:
        raise FileNotFoundError(f"Question not found: {question_id}")

    audio_path = base.get("audio_path") or source.get("audio_path")
    audio = resolve_audio(audio_path)

    per_judge = {}
    for label in bundle["judge_labels"]:
        rec = (judges.get(label) or {}).get(question_id)
        if rec is None:
            per_judge[label] = None
            continue
        per_judge[label] = {
            "score": rec.get("score"),
            "correct": rec.get("correct"),
            "verdict": _record_verdict(
                rec, eval_kind=bundle.get("eval_kind") or "rubrics"
            ),
            "rubric_results": rec.get("rubric_results") or [],
            "raw_responses": rec.get("raw_responses") or [],
        }

    # Canonical rubric criteria names from first available judge or meta.
    rubric = base.get("rubric") or source.get("rubric") or []
    criterion_names = [item.get("name") for item in rubric if item.get("name")]
    if not criterion_names:
        for payload in per_judge.values():
            if payload and payload.get("rubric_results"):
                criterion_names = [
                    r.get("name") for r in payload["rubric_results"] if r.get("name")
                ]
                break

    thinking_prediction = ""
    answer_prediction = ""
    shot_index = 0
    if sample is not None:
        thinking_prediction = sample.get("thinking_prediction") or ""
        answer_prediction = sample.get("answer_prediction") or ""
        shot_index = sample.get("shot_index") if sample.get("shot_index") is not None else 0
    if source:
        src_thinking, src_answer, src_shot = first_shot_fields(source)
        thinking_prediction = thinking_prediction or src_thinking
        answer_prediction = answer_prediction or src_answer
        if sample is None:
            shot_index = src_shot

    return {
        "id": question_id,
        "question": base.get("question") or "",
        "answer": base.get("answer") or "",
        "choices": base.get("choices") or source.get("choices") or [],
        "thinking": base.get("thinking") or "",
        "cue": base.get("cue") or [],
        "rubric": rubric,
        "criterion_names": criterion_names,
        "thinking_prediction": thinking_prediction,
        "answer_prediction": answer_prediction,
        "shot_index": shot_index,
        "modality": base.get("modality"),
        "category": base.get("category"),
        "sub_category": base.get("sub-category"),
        "audio_url": f"/audio/{audio.name}" if audio else None,
        "model_label": bundle["model_label"],
        "eval_kind": bundle.get("eval_kind") or "rubrics",
        "source_run_id": bundle.get("source_run_id"),
        "judge_labels": bundle["judge_labels"],
        "judges": per_judge,
        "disagree": _judges_disagree(
            {
                k: {"score": v.get("score"), "correct": v.get("correct")}
                for k, v in per_judge.items()
                if v
            }
        ),
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MMAR Rubrics Compare</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg: #eef2f5;
    --ink: #1a242d;
    --muted: #5c6b78;
    --card: #f7f9fb;
    --line: #cfd8e0;
    --accent: #2f6fed;
    --good: #1f7a4c;
    --bad: #b42318;
    --soft-good: #e5f5ec;
    --soft-bad: #fdecea;
    --radius: 14px;
    --shadow: 0 10px 30px rgba(26, 36, 45, 0.06);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; color: var(--ink); background: var(--bg);
    font-family: "IBM Plex Sans", sans-serif;
  }
  header {
    position: sticky; top: 0; z-index: 10;
    backdrop-filter: blur(10px);
    background: rgba(238, 242, 245, 0.92);
    border-bottom: 1px solid var(--line);
  }
  .header-inner {
    max-width: 1600px; margin: 0 auto; padding: 0.9rem 1.25rem;
    display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  }
  h1 {
    margin: 0; font-family: "Space Grotesk", sans-serif;
    font-size: 1.35rem; letter-spacing: -0.02em;
  }
  .brand p { margin: 0.2rem 0 0; color: var(--muted); font-size: 0.9rem; }
  .controls { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: end; }
  label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.75rem; color: var(--muted); }
  select, input, button {
    font: inherit; color: var(--ink);
    border: 1px solid var(--line); background: #fff;
    border-radius: 8px; padding: 0.45rem 0.65rem; min-width: 12rem;
  }
  #run { min-width: 22rem; max-width: 36rem; }
  button { cursor: pointer; min-width: auto; background: #e8eef2; }
  button.active { background: #d9e8f3; border-color: #a9c4d8; color: #1a4a6e; }
  main {
    max-width: 1600px; margin: 0 auto; padding: 1.25rem;
    display: grid; grid-template-columns: 340px 1fr; gap: 1rem;
  }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  .panel {
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--radius); box-shadow: var(--shadow);
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
  #qlist {
    list-style: none; margin: 0; padding: 0;
    max-height: calc(100vh - 240px); overflow: auto;
  }
  #qlist li {
    border-bottom: 1px solid var(--line);
    padding: 0.75rem 1rem; cursor: pointer;
  }
  #qlist li:hover { background: #eef5fa; }
  #qlist li.active { background: #e2eef6; }
  .qid { font-family: "IBM Plex Mono", monospace; font-size: 0.75rem; color: var(--muted); }
  .qtext {
    margin: 0.35rem 0 0; font-size: 0.86rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .score-row { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.35rem; }
  .chip {
    font-size: 0.68rem; font-family: "IBM Plex Mono", monospace;
    padding: 0.12rem 0.35rem; border-radius: 999px;
    background: #e8eef2; color: var(--muted);
  }
  .chip.disagree { background: #fff1e8; color: #9a3412; }
  #detail { padding: 1rem 1.15rem; max-height: calc(100vh - 160px); overflow: auto; }
  .muted { color: var(--muted); }
  .box {
    border: 1px solid var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; margin: 0.75rem 0; background: #fff;
  }
  .box h3 { margin: 0 0 0.45rem; font-size: 0.95rem; }
  pre {
    white-space: pre-wrap; word-break: break-word;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem; margin: 0.25rem 0 0;
    background: #f2f6f9; padding: 0.55rem 0.65rem; border-radius: 8px;
  }
  audio { width: 100%; margin-top: 0.5rem; }
  .judge-summary {
    display: flex; flex-wrap: wrap; gap: 0.45rem;
    margin: 0.75rem 0 0.25rem;
  }
  .judge-summary .chip {
    font-size: 0.78rem; padding: 0.25rem 0.55rem;
    background: #fff; border: 1px solid var(--line);
  }
  .rubric-list { display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.75rem; }
  .rubric-card {
    border: 1px solid var(--line); border-radius: 10px;
    background: #fff; overflow: hidden;
  }
  .rubric-def {
    padding: 0.75rem 0.9rem;
    background: #f2f6f9;
    border-bottom: 1px solid var(--line);
  }
  .rubric-def .name {
    margin: 0; font-size: 0.95rem; font-weight: 600;
  }
  .rubric-def .scoring {
    margin: 0.35rem 0 0; font-size: 0.88rem;
  }
  .rubric-def .note {
    margin: 0.3rem 0 0; font-size: 0.8rem; color: var(--muted);
  }
  .rubric-def .choices {
    margin-top: 0.35rem;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; color: var(--muted);
  }
  .judge-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0;
  }
  .judge-col {
    border-right: 1px solid var(--line);
    padding: 0.65rem 0.8rem 0.85rem;
    min-width: 0;
  }
  .judge-col:last-child { border-right: none; }
  .judge-col .jlabel {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; color: var(--muted);
    margin-bottom: 0.25rem;
  }
  .judge-col .jscore {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.85rem; font-weight: 500;
  }
  .judge-col .just {
    font-size: 0.84rem; margin-top: 0.35rem;
    color: var(--ink);
  }
  .judge-col .missing { color: var(--muted); font-size: 0.84rem; }
  .pass { color: var(--good); }
  .fail { color: var(--bad); }
  .diff-mark { color: #9a3412; font-size: 0.75rem; margin-left: 0.35rem; }
  .verdict-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.65rem;
    margin-top: 0.5rem;
  }
  .verdict-card {
    border: 1px solid var(--line);
    border-radius: 10px;
    background: #fff;
    padding: 0.7rem 0.8rem 0.85rem;
    min-width: 0;
  }
  .verdict-card.yes, .verdict-card.pass { border-color: #9dceb3; background: var(--soft-good); }
  .verdict-card.no, .verdict-card.fail { border-color: #f0b4ae; background: var(--soft-bad); }
  .verdict-card .vlabel {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; color: var(--muted);
  }
  .verdict-card .vvalue {
    font-family: "IBM Plex Mono", monospace;
    font-size: 1.05rem; font-weight: 500; margin-top: 0.15rem;
  }
  .verdict-card pre { margin-top: 0.45rem; max-height: 18rem; overflow: auto; }
  details.raw-judges { margin-top: 0.75rem; }
  details.raw-judges summary {
    cursor: pointer; color: var(--muted); font-size: 0.85rem;
  }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <h1>MMAR Rubrics Compare</h1>
      <p id="subtitle">Side-by-side judge scores for first-shot reasoning traces</p>
    </div>
    <div class="controls">
      <label>Run
        <select id="run"></select>
      </label>
      <label>Search
        <input id="search" type="search" placeholder="id / question text" />
      </label>
      <label>&nbsp;
        <button id="filter-disagree" type="button">Disagree only</button>
      </label>
      <label>&nbsp;
        <button id="filter-correct" type="button">Pass / grounded</button>
      </label>
    </div>
  </div>
</header>
<main>
  <section class="panel">
    <h2>Questions</h2>
    <div class="stats" id="stats">Loading…</div>
    <ul id="qlist"></ul>
  </section>
  <section class="panel">
    <h2>Detail</h2>
    <div id="detail"><p class="muted">Select a question.</p></div>
  </section>
</main>
<script>
const state = {
  runs: [],
  runId: "",
  evalKind: "rubrics",
  judgeLabels: [],
  questions: [],
  selectedId: null,
  filterDisagree: false,
  filterCorrect: false,
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

function fmtScore(v) {
  if (v === null || v === undefined) return "—";
  return (100 * Number(v)).toFixed(1) + "%";
}

function fmtChip(item, judgeLabel) {
  const s = (item.scores || {})[judgeLabel];
  if (!s) return `${shortJudge(judgeLabel)} —`;
  if (state.evalKind === "groundedness") {
    const verdict = s.verdict || (s.correct === true ? "Yes" : s.correct === false ? "No" : "—");
    return `${shortJudge(judgeLabel)} ${verdict}`;
  }
  return `${shortJudge(judgeLabel)} ${fmtScore(s.score)}`;
}

function judgeVerdict(j) {
  if (!j) return "—";
  if (j.verdict) return j.verdict;
  if (state.evalKind === "groundedness") {
    if (j.correct === true) return "Yes";
    if (j.correct === false) return "No";
    return "—";
  }
  if (j.correct === true) return "pass";
  if (j.correct === false) return "fail";
  return fmtScore(j.score);
}

function shortJudge(label) {
  if (label.startsWith("qwen3-omni")) return "Qwen3-Omni";
  if (label.startsWith("gemini")) return "Gemini";
  if (label.startsWith("qwen")) return "Qwen";
  if (label.startsWith("gpt")) return "GPT-4o";
  if (label.startsWith("claude")) return "Claude";
  return label;
}

async function loadRuns() {
  const data = await api("/api/runs");
  state.runs = data.runs || [];
  const sel = document.getElementById("run");
  sel.innerHTML = state.runs.map(r => {
    const source = r.source_run_id || r.run_id;
    const model = r.model_label || "";
    const kind = r.eval_kind === "groundedness" ? "groundedness" : "rubrics";
    const label = model ? `${source} · ${model} · ${kind}` : `${source} · ${kind}`;
    return `<option value="${escapeHtml(r.run_id)}">${escapeHtml(label)} · ${(r.judges||[]).length} judges</option>`;
  }).join("");
  if (!state.runs.length) {
    document.getElementById("stats").textContent = "No rubrics runs found.";
    return;
  }
  const preferred = new URLSearchParams(location.search).get("run") || state.runs[0].run_id;
  sel.value = preferred;
  await loadRun(preferred);
}

async function loadRun(runId) {
  state.runId = runId;
  const data = await api(`/api/run?run_id=${encodeURIComponent(runId)}`);
  state.judgeLabels = data.judge_labels || [];
  state.questions = data.questions || [];
  state.evalKind = data.eval_kind || "rubrics";
  const subtitle = document.getElementById("subtitle");
  if (subtitle) {
    const model = data.model_label || "";
    const source = data.source_run_id || "";
    const kind = state.evalKind === "groundedness"
      ? "audio groundedness (no question / options)"
      : "first-shot traces";
    subtitle.textContent = model
      ? `${model} ${kind}${source ? " · " + source : ""}`
      : "Side-by-side judge scores for first-shot reasoning traces";
  }
  const filterBtn = document.getElementById("filter-correct");
  if (filterBtn) {
    filterBtn.textContent = state.evalKind === "groundedness" ? "Grounded only" : "Pass / grounded";
  }
  renderList();
  if (state.questions.length) {
    selectQuestion(state.questions[0].id);
  } else {
    document.getElementById("detail").innerHTML = `<p class="muted">No questions in this run.</p>`;
  }
}

function filteredQuestions() {
  const q = (document.getElementById("search").value || "").trim().toLowerCase();
  return state.questions.filter(item => {
    if (state.filterDisagree && !item.disagree) return false;
    if (state.filterCorrect && !item.correct_any) return false;
    if (!q) return true;
    return (item.id || "").toLowerCase().includes(q)
      || (item.question || "").toLowerCase().includes(q);
  });
}

function renderList() {
  const items = filteredQuestions();
  document.getElementById("stats").innerHTML =
    `<span><strong>${items.length}</strong> shown</span>` +
    `<span>${state.questions.length} total</span>` +
    `<span>${state.judgeLabels.map(shortJudge).join(" · ")}</span>`;
  const ul = document.getElementById("qlist");
  ul.innerHTML = items.map(item => {
    const chips = state.judgeLabels.map(j => {
      return `<span class="chip">${escapeHtml(fmtChip(item, j))}</span>`;
    }).join("");
    return `<li data-id="${escapeHtml(item.id)}" class="${item.id === state.selectedId ? "active" : ""}">
      <div class="qid">${escapeHtml(item.id)}</div>
      <div class="qtext">${escapeHtml(item.question)}</div>
      <div class="score-row">${chips}${item.disagree ? '<span class="chip disagree">disagree</span>' : ""}</div>
    </li>`;
  }).join("");
  ul.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => selectQuestion(li.dataset.id));
  });
}

async function selectQuestion(id) {
  state.selectedId = id;
  renderList();
  try {
    const data = await api(`/api/question?run_id=${encodeURIComponent(state.runId)}&id=${encodeURIComponent(id)}`);
    const choices = (data.choices || []).map(c => `<li>${escapeHtml(c)}</li>`).join("");
    const labels = data.judge_labels || [];

    const summaryChips = labels.map(label => {
      const j = (data.judges || {})[label];
      if (!j) {
        return `<span class="chip">${escapeHtml(shortJudge(label))} —</span>`;
      }
      const verdict = judgeVerdict(j);
      const cls = (verdict === "Yes" || verdict === "pass" || j.correct) ? "pass" : "fail";
      const extra = state.evalKind === "groundedness" ? verdict : fmtScore(j.score);
      return `<span class="chip"><span class="${cls}">${escapeHtml(shortJudge(label))} ${escapeHtml(extra)}</span></span>`;
    }).join("");

    const verdictCards = labels.map(label => {
      const j = (data.judges || {})[label];
      if (!j) {
        return `<div class="verdict-card">
          <div class="vlabel">${escapeHtml(shortJudge(label))}</div>
          <div class="missing">Missing</div>
        </div>`;
      }
      const verdict = judgeVerdict(j);
      const cls = (verdict === "Yes" || verdict === "pass") ? "yes"
        : (verdict === "No" || verdict === "fail") ? "no" : "";
      const raw = (j.raw_responses && j.raw_responses[0]) || "";
      return `<div class="verdict-card ${cls}">
        <div class="vlabel">${escapeHtml(shortJudge(label))}</div>
        <div class="vvalue ${cls === "yes" ? "pass" : cls === "no" ? "fail" : ""}">${escapeHtml(verdict)}</div>
        ${raw ? `<pre>${escapeHtml(raw)}</pre>` : `<div class="missing">No judge write-up</div>`}
      </div>`;
    }).join("");

    const rubric = data.rubric || [];
    const findResult = (label, name, index) => {
      const j = (data.judges || {})[label];
      if (!j) return null;
      const results = j.rubric_results || [];
      const byName = results.find(r => r && r.name === name);
      if (byName) return byName;
      return results[index] || null;
    };

    const rubricCards = rubric.map((item, index) => {
      const name = item.name || `Criterion ${index + 1}`;
      const choiceStr = Array.isArray(item.choices) ? item.choices.join(", ") : "";
      const judgeCols = labels.map(label => {
        const j = (data.judges || {})[label];
        if (!j) {
          return `<div class="judge-col">
            <div class="jlabel">${escapeHtml(shortJudge(label))}</div>
            <div class="missing">Missing from local results (re-download exp-mmar-rubrics?)</div>
          </div>`;
        }
        const r = findResult(label, name, index);
        if (!r) {
          return `<div class="judge-col">
            <div class="jlabel">${escapeHtml(shortJudge(label))}</div>
            <div class="missing">Missing criterion</div>
          </div>`;
        }
        const passed = !!r.pass;
        const score = r.score !== undefined && r.score !== null ? r.score : (passed ? "pass" : "fail");
        const matchNote = j.correct
          ? ""
          : `<div class="missing" style="margin-top:0.25rem">MC answer string-match fail</div>`;
        return `<div class="judge-col">
          <div class="jlabel">${escapeHtml(shortJudge(label))}</div>
          <div class="jscore ${passed ? "pass" : "fail"}">${escapeHtml(String(score))}${passed ? " · pass" : " · fail"}</div>
          <div class="just">${escapeHtml(r.justification || "")}</div>
          ${matchNote}
        </div>`;
      }).join("");

      return `<div class="rubric-card">
        <div class="rubric-def">
          <div class="name">${escapeHtml(name)}</div>
          <div class="scoring">${escapeHtml(item.scoring_point || "")}</div>
          ${item.note ? `<div class="note">${escapeHtml(item.note)}</div>` : ""}
          ${choiceStr ? `<div class="choices">Score options: [${escapeHtml(choiceStr)}]</div>` : ""}
        </div>
        <div class="judge-grid">${judgeCols}</div>
      </div>`;
    }).join("") || `<p class="muted">No dataset rubric on this example.</p>`;

    const rawBlocks = labels.map(label => {
      const j = (data.judges || {})[label];
      const raw = (j && j.raw_responses && j.raw_responses[0]) || "";
      if (!raw) return "";
      return `<details style="margin-top:0.45rem">
        <summary class="muted">${escapeHtml(shortJudge(label))} raw response</summary>
        <pre>${escapeHtml(raw)}</pre>
      </details>`;
    }).join("");

    document.getElementById("detail").innerHTML = `
      <div class="qid">${escapeHtml(data.id)}</div>
      <h3 style="margin:0.35rem 0 0.6rem;font-family:Space Grotesk,sans-serif">${escapeHtml(data.question)}</h3>
      <div class="muted">${escapeHtml(data.modality || "")} · ${escapeHtml(data.category || "")}${data.disagree ? ' <span class="diff-mark">judges disagree</span>' : ""}</div>
      ${data.audio_url ? `<audio controls src="${escapeHtml(data.audio_url)}"></audio>` : ""}
      <div class="box">
        <h3>Thinking trace</h3>
        <p class="muted" style="margin:0 0 0.35rem;font-size:0.85rem">${escapeHtml(data.model_label || "model")} first-shot reasoning (judged without the question or options).</p>
        <pre>${escapeHtml(data.thinking_prediction || "")}</pre>
      </div>
      <div class="box">
        <h3>Verdict</h3>
        <p class="muted" style="margin:0 0 0.35rem;font-size:0.85rem">${state.evalKind === "groundedness"
          ? "Yes = every factual claim is supported by the audio. No = at least one claim is not."
          : "Per-judge overall decision for this example."}</p>
        <div class="judge-summary">${summaryChips}</div>
        <div class="verdict-grid">${verdictCards}</div>
      </div>
      <div class="box">
        <h3>Ground truth</h3>
        <div><strong>Answer:</strong> ${escapeHtml(data.answer)}</div>
        ${choices ? `<ul>${choices}</ul>` : ""}
        <details><summary>Ideal reasoning</summary><pre>${escapeHtml(data.thinking || "")}</pre></details>
      </div>
      <div class="box">
        <h3>${escapeHtml(data.model_label || "model")} first-shot prediction</h3>
        <div><strong>Answer:</strong> ${escapeHtml(data.answer_prediction || "")}</div>
      </div>
      ${state.evalKind === "groundedness" ? "" : `<div class="box">
        <h3>Rubric criteria</h3>
        <p class="muted" style="margin:0 0 0.35rem;font-size:0.85rem">Dataset scoring bullets with each judge's decision underneath.</p>
        <div class="rubric-list">${rubricCards}</div>
      </div>`}
      ${rawBlocks ? `<details class="raw-judges"><summary>Raw judge responses</summary>${rawBlocks}</details>` : ""}
    `;
  } catch (err) {
    document.getElementById("detail").innerHTML =
      `<p class="muted">Failed to load question: ${escapeHtml(String(err))}</p>`;
  }
}

function move(delta) {
  const items = filteredQuestions();
  if (!items.length) return;
  const idx = Math.max(0, items.findIndex(x => x.id === state.selectedId));
  const next = items[(idx + delta + items.length) % items.length];
  selectQuestion(next.id);
}

document.getElementById("run").addEventListener("change", e => loadRun(e.target.value));
document.getElementById("search").addEventListener("input", renderList);
document.getElementById("filter-disagree").addEventListener("click", e => {
  state.filterDisagree = !state.filterDisagree;
  e.target.classList.toggle("active", state.filterDisagree);
  renderList();
});
document.getElementById("filter-correct").addEventListener("click", e => {
  state.filterCorrect = !state.filterCorrect;
  e.target.classList.toggle("active", state.filterCorrect);
  renderList();
});
document.addEventListener("keydown", e => {
  if (e.target.matches("input, textarea, select")) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); move(-1); }
  if (e.key === "ArrowRight") { e.preventDefault(); move(1); }
});

loadRuns().catch(err => {
  document.getElementById("stats").textContent = String(err);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[view_mmar_rubrics] {self.address_string()} - {fmt % args}")

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, body: str, content_type: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        try:
            if path in {"/", "/index.html"}:
                self._send_text(HTML_PAGE, "text/html; charset=utf-8")
                return
            if path == "/api/runs":
                runs = discover_rubric_runs(Path(CONFIG["results_dir"]))
                self._send_json({"runs": runs})
                return
            if path == "/api/run":
                run_id = (qs.get("run_id") or [None])[0]
                if not run_id:
                    self._send_json({"error": "run_id required"}, 400)
                    return
                bundle = load_run_bundle(run_id)
                self._send_json(
                    {
                        "run_id": bundle["run_id"],
                        "source_run_id": bundle.get("source_run_id"),
                        "model_label": bundle["model_label"],
                        "eval_kind": bundle.get("eval_kind") or "rubrics",
                        "judge_labels": bundle["judge_labels"],
                        "questions": bundle["questions"],
                        "meta": bundle["meta"],
                    }
                )
                return
            if path == "/api/question":
                run_id = (qs.get("run_id") or [None])[0]
                qid = (qs.get("id") or [None])[0]
                if not run_id or not qid:
                    self._send_json({"error": "run_id and id required"}, 400)
                    return
                self._send_json(get_question_detail(run_id, qid))
                return
            if path.startswith("/audio/"):
                name = unquote(path[len("/audio/") :])
                audio_dir = Path(CONFIG["audio_dir"])
                candidate = (audio_dir / name).resolve()
                if not str(candidate).startswith(str(audio_dir.resolve())):
                    self._send_json({"error": "invalid audio path"}, 400)
                    return
                if not candidate.is_file():
                    self._send_json({"error": "audio not found"}, 404)
                    return
                self._send_file(candidate)
                return
            self._send_json({"error": "not found"}, 404)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, 404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=7862)
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    p.add_argument("--run-id", default=None)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    audio_dir = ensure_mmar_audio(Path(args.audio_dir).expanduser().resolve())
    CONFIG["results_dir"] = results_dir
    CONFIG["audio_dir"] = audio_dir
    _sync_vd_config()

    _load_run_bundle_cached.cache_clear()
    runs = discover_rubric_runs(results_dir)
    print(f"Found {len(runs)} rubrics run(s) under {results_dir}")
    for run in runs[:8]:
        print(
            f"  {run['run_id']}: kind={run.get('eval_kind')} model={run.get('model_label')} "
            f"judges={run.get('judges')}"
        )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    if args.run_id:
        url += f"?run={args.run_id}"
    print(f"Serving MMAR rubrics viewer at {url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
