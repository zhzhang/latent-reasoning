"""Local viewer for unparsed judge answers in the accuracy-only sample.

Same universe as ``uv run run_judges.py --accuracy-only``: labeled shots in
the judging pack's ``labels.csv`` joined to ``models/<label>/predictions.jsonl``.
Sidecars under ``judge_partials/`` are ignored. A stored judge entry counts as
a parse failure when ``verdict`` is not ``pass`` or ``fail``. Missing entries
are misses, not listed here.

Usage::

    uv run python view_parse_failures.py
    uv run python view_parse_failures.py --port 7865
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

from aggregate import order_model_labels
from alt_test import scoring_gold
from grader import (
    accuracy_mode_names,
    grade_mode_title,
    grade_mode_titles,
    grade_prompt_names,
    judge_mode_bucket,
    parse_judge_key,
)
from mmar_common import load_jsonl
from view_mmar import (
    CONFIG as MMAR_CONFIG,
    DEFAULT_AUDIO_DIR,
    DEFAULT_DATA_DIR,
    QUESTION_KEYS,
    ensure_mmar_audio,
    resolve_audio,
)

REPO_ROOT = Path(__file__).resolve().parent
PACK_DIR = REPO_ROOT / "outputs" / "mmar-judging"
LABELS_CSV_NAME = "labels.csv"
LOCAL_MMAR_META = DEFAULT_DATA_DIR / "MMAR-meta.jsonl"

CONFIG: dict[str, Any] = {}
TAIL_CHARS = 220


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


def load_pack_label_rows(labels_path: Path) -> list[dict[str, Any]]:
    """Rows with a non-empty boolean ``ratings`` list (same as run_judges)."""
    rows: list[dict[str, Any]] = []
    if not labels_path.is_file():
        return rows
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
                    "generation_id": str(raw.get("generation_id") or "").strip(),
                }
            )
    return rows


def _load_predictions_by_id(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return by_id
    for record in load_jsonl(path):
        if not isinstance(record, dict):
            continue
        qid = str(record.get("id") or "").strip()
        if qid:
            by_id[qid] = record
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


def entry_parsed(entry: object) -> bool:
    """True when the stored judge answer has a pass/fail verdict.

    Matches ``run_judges._entry_parsed``.
    """
    if not isinstance(entry, dict):
        return False
    return str(entry.get("verdict") or "").strip().lower() in {"pass", "fail"}


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


def _generation_text(entry: dict[str, Any]) -> str:
    text = str(entry.get("generation") or "")
    if text.strip():
        return text
    return str(entry.get("reasoning") or "")


def _tail(text: str, n: int = TAIL_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= n:
        return stripped
    return stripped[-n:]


def _compact_failure_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Keep stored fields; do not invent a verdict from ``correct``."""
    keep = (
        "correct",
        "verdict",
        "output",
        "generation",
        "reasoning",
        "model_id",
        "prompt",
        "include_gold",
        "n_samples",
        "samples",
    )
    out: dict[str, Any] = {}
    for key in keep:
        if key in entry:
            out[key] = entry.get(key)
    return out


@lru_cache(maxsize=4)
def load_bundle(pack_dir_s: str) -> dict[str, Any]:
    pack_dir = Path(pack_dir_s)
    labels_path = pack_dir / LABELS_CSV_NAME
    label_rows = load_pack_label_rows(labels_path)
    meta = _load_meta(LOCAL_MMAR_META)

    pred_cache: dict[str, dict[str, dict[str, Any]]] = {}
    samples: list[tuple[str, str, int, list[bool], dict[str, dict[str, Any]] | None]] = []
    judge_keys: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}

    for row in label_rows:
        model = row["model_label"]
        qid = row["question_id"]
        if model not in pred_cache:
            pred_cache[model] = _load_predictions_by_id(
                pack_dir / "models" / model / "predictions.jsonl"
            )
        record = pred_cache[model].get(qid)
        shot = _shot_for_index(record, row["shot_index"]) if record else None
        judges: dict[str, dict[str, Any]] | None = None
        if shot is not None and isinstance(shot.get("judges"), dict):
            judges = {
                str(key): dict(entry)
                for key, entry in shot["judges"].items()
                if key and isinstance(entry, dict)
            }
            judge_keys.update(judges)
        samples.append((qid, model, row["shot_index"], list(row["ratings"]), judges))

        if qid not in by_id:
            sample_record = record
            if sample_record is None:
                for recs in pred_cache.values():
                    if qid in recs:
                        sample_record = recs[qid]
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
            if not fields.get("url") and meta_row.get("url"):
                fields["url"] = meta_row.get("url")
            if not fields.get("modality") and meta_row.get("modality"):
                fields["modality"] = meta_row.get("modality")
            if not fields.get("category") and meta_row.get("category"):
                fields["category"] = meta_row.get("category")
            by_id[qid] = fields

    key_mode: dict[str, str] = {}
    judge_meta: list[dict[str, Any]] = []
    for key in sorted(judge_keys):
        sample_entry = next(
            (
                (judges or {}).get(key)
                for _qid, _model, _shot, _ratings, judges in samples
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
        parsed = parse_judge_key(key)
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

    stats: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in grade_prompt_names()
    }
    failures: list[dict[str, Any]] = []
    shots: dict[str, dict[str, Any]] = {}

    for qid, model, shot_index, ratings, judges in samples:
        record = pred_cache.get(model, {}).get(qid)
        shot = _shot_for_index(record, shot_index) if record else None
        prediction = ""
        if isinstance(shot, dict) and shot.get("answer_prediction") is not None:
            prediction = str(shot.get("answer_prediction") or "")
        gold = scoring_gold(ratings)
        shot_id = f"{qid}\t{model}\t{shot_index}"
        shots[shot_id] = {
            "question_id": qid,
            "model_label": model,
            "shot_index": shot_index,
            "answer_prediction": prediction,
            "ratings": ratings,
            "gold": gold,
        }
        for key, mode in key_mode.items():
            entry = (judges or {}).get(key)
            bucket = stats.setdefault(mode, {}).setdefault(
                key,
                {
                    "n_judge_answers": 0,
                    "n_parsed": 0,
                    "n_unparsed": 0,
                    "n_empty": 0,
                    "parse_rate": None,
                },
            )
            if not isinstance(entry, dict):
                continue
            bucket["n_judge_answers"] += 1
            if entry_parsed(entry):
                bucket["n_parsed"] += 1
                continue
            bucket["n_unparsed"] += 1
            generation = _generation_text(entry)
            empty = not generation.strip()
            if empty:
                bucket["n_empty"] += 1
            compact = _compact_failure_entry(entry)
            failures.append(
                {
                    "i": len(failures),
                    "question_id": qid,
                    "model_label": model,
                    "shot_index": shot_index,
                    "judge_key": key,
                    "mode": mode,
                    "gen_len": len(generation),
                    "empty": empty,
                    "tail": _tail(generation),
                    "output": compact.get("output"),
                    "verdict": compact.get("verdict"),
                    "correct": compact.get("correct"),
                    "prompt": compact.get("prompt"),
                    "model_id": compact.get("model_id"),
                    "gold": gold,
                    "generation": generation,
                    "reasoning": compact.get("reasoning"),
                    "samples": compact.get("samples"),
                    "shot_id": shot_id,
                }
            )

    for mode_table in stats.values():
        for row in mode_table.values():
            present = int(row["n_judge_answers"])
            row["parse_rate"] = (row["n_parsed"] / present) if present else None

    payload_stats: dict[str, Any] = {
        "n_label_rows": len(label_rows),
        "modes": accuracy_mode_names(stats),
    }
    for mode in payload_stats["modes"]:
        payload_stats[mode] = stats.get(mode) or {}

    model_labels = order_model_labels([str(row["model_label"]) for row in label_rows])
    question_ids = list(dict.fromkeys(row["question_id"] for row in label_rows))

    return {
        "pack_dir": str(pack_dir),
        "labels_path": str(labels_path),
        "model_labels": model_labels,
        "judges": judge_meta,
        "stats": payload_stats,
        "failures": failures,
        "shots": shots,
        "by_id": by_id,
        "n_label_rows": len(label_rows),
        "n_questions": len(question_ids),
        "n_unparsed": len(failures),
        "pack_present": pack_dir.is_dir() and bool(pred_cache),
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MMAR Judge Parse Failures</title>
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
  #mode { min-width: 16rem; font-weight: 500; }
  button { cursor: pointer; }
  .toolbar {
    max-width: 1480px; margin: 0 auto; padding: 0.85rem 1.25rem 0;
    display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: end;
  }
  main {
    max-width: 1480px; margin: 0 auto; padding: 1.25rem;
    display: grid; grid-template-columns: 420px 1fr; gap: 1rem;
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
  .acc-wrap { padding: 0.4rem 0.55rem 0.7rem; overflow: auto; max-height: 280px; }
  .mode-label {
    font-size: 0.68rem; letter-spacing: 0.04em; text-transform: uppercase;
    color: var(--muted); padding: 0.45rem 0.55rem 0.1rem;
  }
  #flist {
    list-style: none; margin: 0; padding: 0;
    max-height: calc(100vh - 520px); overflow: auto;
  }
  #flist li {
    border-bottom: 1px solid var(--line);
    padding: 0.7rem 1rem; cursor: pointer;
  }
  #flist li:hover { background: #eef5fa; }
  #flist li.active { background: #e2eef6; }
  .qid {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.75rem; color: var(--muted);
  }
  .tail {
    margin: 0.3rem 0 0;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.75rem;
    color: var(--ink);
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  #detail { padding: 1rem 1.15rem; max-height: calc(100vh - 160px); overflow: auto; }
  .muted { color: var(--muted); }
  .pass { color: var(--good); background: var(--soft-good); padding: 0.1rem 0.4rem; border-radius: 999px; }
  .fail { color: var(--bad); background: var(--soft-bad); padding: 0.1rem 0.4rem; border-radius: 999px; }
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
    border: 1px solid #d4b88a;
    color: #5a3a12; background: #f3e6cf;
  }
  .brand-row {
    display: flex; flex-wrap: wrap; gap: 0.55rem; align-items: center;
  }
  .chip {
    font-size: 0.7rem; font-family: "IBM Plex Mono", monospace;
    padding: 0.15rem 0.4rem; border-radius: 999px;
    background: #e8eef2; color: var(--muted);
  }
  .kv {
    display: grid; grid-template-columns: max-content 1fr;
    gap: 0.15rem 0.75rem; font-size: 0.8rem; margin: 0.35rem 0;
  }
  .kv dt { color: var(--muted); font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; }
  .kv dd { margin: 0; }
  .answer-box, .shot-box {
    border: 1px solid var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; margin: 0.75rem 0; background: #fff;
  }
  .shot-head {
    display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;
    font-family: "IBM Plex Mono", monospace; font-size: 0.8rem;
    margin-bottom: 0.35rem;
  }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <div class="brand-row">
        <h1>Judge parse failures</h1>
        <span class="mode-badge">accuracy-only sample</span>
      </div>
      <p>Stored judge replies whose <code>verdict</code> is not pass/fail, on shots scored by <code>run_judges.py --accuracy-only</code></p>
    </div>
    <div class="controls">
      <label>Judge format
        <select id="mode"></select>
      </label>
    </div>
  </div>
</header>
<div class="toolbar">
  <label>Search
    <input id="search" type="search" placeholder="id / model / tail" />
  </label>
  <label>Gradee
    <select id="model"><option value="">All</option></select>
  </label>
  <label>Kind
    <select id="kind">
      <option value="">All unparsed</option>
      <option value="empty">Empty generation</option>
      <option value="text">Non-empty generation</option>
    </select>
  </label>
</div>
<main>
  <section class="panel">
    <h2>Parse rate vs stored answers</h2>
    <div class="stats" id="stats">Loading…</div>
    <div id="accuracy"></div>
    <h2>Failures</h2>
    <ul id="flist"></ul>
  </section>
  <section class="panel">
    <h2>Detail</h2>
    <div id="detail"><p class="muted">Select a failure.</p></div>
  </section>
</main>
<script>
const state = {
  failures: [],
  modelLabels: [],
  judges: [],
  stats: {},
  modeTitles: {},
  modeOrder: [],
  nLabelRows: 0,
  nQuestions: 0,
  nUnparsed: 0,
  selectedI: null,
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
    "qwen2.5-omni-7b": "qwen2.5",
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

function selectedMode() {
  return String((document.getElementById("mode") || {}).value || "");
}

function formatTitle(mode) {
  if (!mode) return "";
  const title = (state.modeTitles || {})[mode] || mode;
  return String(title).split(" (")[0];
}

function formatOrder() {
  const named = Array.isArray(state.modeOrder) ? state.modeOrder.filter(Boolean) : [];
  const acc = state.stats || {};
  const extra = Object.keys(acc).filter(key => {
    if (named.includes(key)) return false;
    if (["n_label_rows", "epsilon", "modes"].includes(key)) return false;
    return acc[key] && typeof acc[key] === "object" && !Array.isArray(acc[key]);
  });
  const order = named.concat(extra);
  return order.filter(mode => {
    const table = acc[mode] || {};
    return Object.values(table).some(row => Number(row?.n_unparsed || 0) > 0);
  });
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
}

function visibleJudges() {
  const mode = selectedMode();
  if (!mode) return [];
  return (state.judges || []).filter(j => j.mode === mode);
}

function filteredFailures() {
  const mode = selectedMode();
  const judge = state.selectedJudge;
  const model = document.getElementById("model").value;
  const kind = document.getElementById("kind").value;
  const q = (document.getElementById("search").value || "").trim().toLowerCase();
  return (state.failures || []).filter(row => {
    if (mode && row.mode !== mode) return false;
    if (judge && row.judge_key !== judge) return false;
    if (model && row.model_label !== model) return false;
    if (kind === "empty" && !row.empty) return false;
    if (kind === "text" && row.empty) return false;
    if (!q) return true;
    return String(row.question_id).toLowerCase().includes(q)
      || String(row.model_label || "").toLowerCase().includes(q)
      || String(row.judge_key || "").toLowerCase().includes(q)
      || String(row.tail || "").toLowerCase().includes(q);
  });
}

function renderStats() {
  const items = filteredFailures();
  const mode = selectedMode();
  const table = (state.stats || {})[mode] || {};
  let present = 0;
  let unparsed = 0;
  const keys = state.selectedJudge
    ? [state.selectedJudge]
    : visibleJudges().map(j => j.label);
  for (const key of keys) {
    const row = table[key] || {};
    present += Number(row.n_judge_answers || 0);
    unparsed += Number(row.n_unparsed || 0);
  }
  const parts = [
    `<span><strong>${items.length}</strong> shown</span>`,
    `<span>${state.nUnparsed} unparsed in sample</span>`,
    `<span>${state.nLabelRows} labeled shots</span>`,
    `<span>${state.nQuestions} questions</span>`,
    `<span>${fmtRate(present ? unparsed / present : null)} unparsed of stored answers</span>`,
  ];
  document.getElementById("stats").innerHTML = parts.join(" · ");
}

function sortJudgeKeys(byJudge) {
  return Object.keys(byJudge || {}).sort((a, b) => {
    const ua = Number(byJudge[a]?.n_unparsed || 0);
    const ub = Number(byJudge[b]?.n_unparsed || 0);
    if (ua !== ub) return ub - ua;
    return String(a).localeCompare(String(b));
  });
}

function renderAccuracy() {
  const filter = selectedMode();
  const wrap = document.getElementById("accuracy");
  if (!filter) {
    wrap.innerHTML = "";
    return;
  }
  if (state.selectedJudge) {
    const meta = (state.judges || []).find(j => j.label === state.selectedJudge);
    if (meta && meta.mode && meta.mode !== filter) {
      state.selectedJudge = null;
    }
  }
  const byJudge = (state.stats || {})[filter] || {};
  const keys = sortJudgeKeys(byJudge).filter(key => Number((byJudge[key] || {}).n_unparsed || 0) > 0);
  const selected = state.selectedJudge;
  let html = "";
  if (keys.length) {
    const title = (state.modeTitles || {})[filter] || filter;
    html += `<div class="mode-label">${escapeHtml(title)}</div>`;
    html += `<div class="acc-wrap"><table class="acc-table"><thead>
      <tr><th>Judge</th><th>n</th><th>unp</th><th>empty</th><th>parse</th></tr>
    </thead><tbody>`;
    for (const key of keys) {
      const row = byJudge[key];
      const klass = key === selected ? "selected" : "";
      html += `<tr class="${klass}" data-judge="${escapeHtml(key)}">
        <td title="${escapeHtml(prettyJudge(key))}">${escapeHtml(shortJudge(key))}</td>
        <td class="mono">${row.n_judge_answers ?? 0}</td>
        <td class="mono">${row.n_unparsed ?? 0}</td>
        <td class="mono">${row.n_empty ?? 0}</td>
        <td class="mono">${fmtRate(row.parse_rate)}</td>
      </tr>`;
    }
    html += `</tbody></table></div>`;
  } else {
    html = `<p class="muted" style="padding:0.6rem 1rem">No unparsed answers for <code>${escapeHtml(filter)}</code>.</p>`;
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
  const list = document.getElementById("flist");
  const items = filteredFailures();
  list.innerHTML = items.map(row => {
    const active = row.i === state.selectedI ? "active" : "";
    const kind = row.empty
      ? `<span class="pending">empty</span>`
      : `<span class="pending">${row.gen_len} chars</span>`;
    return `<li class="${active}" data-i="${row.i}">
      <div class="qid">${escapeHtml(row.question_id)} · ${escapeHtml(shortLabel(row.model_label))} s${row.shot_index} · ${escapeHtml(shortJudge(row.judge_key))}</div>
      <div>${kind}</div>
      <p class="tail">${escapeHtml(row.tail || "(empty generation)")}</p>
    </li>`;
  }).join("");
  list.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => selectFailure(Number(li.dataset.i)));
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

async function selectFailure(i) {
  state.selectedI = i;
  renderList();
  const detail = document.getElementById("detail");
  detail.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const data = await api("/api/failure?i=" + encodeURIComponent(i));
    const fail = data.failure || {};
    const q = data.question || {};
    const shot = data.shot || {};
    const audio = data.audio_url
      ? `<audio controls preload="none" src="${escapeHtml(data.audio_url)}"></audio>`
      : `<p class="muted">Audio not found locally.</p>`;
    const source = q.url
      ? `<p class="audio-source"><a href="${escapeHtml(q.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(q.url)}</a></p>`
      : "";
    const gold = q.answer
      ? `<div class="answer-box"><strong>Gold answer</strong><pre>${escapeHtml(q.answer)}</pre></div>`
      : "";
    const ratings = (shot.ratings || []).map((v, i) =>
      `<span class="${v ? "pass" : "fail"}">r${i}:${v ? "pass" : "fail"}</span>`
    ).join(" ");
    const stored = `<dl class="kv">
      <dt>verdict</dt><dd>${escapeHtml(fail.verdict == null ? "null" : String(fail.verdict))}</dd>
      <dt>output</dt><dd>${escapeHtml(fail.output == null ? "null" : String(fail.output))}</dd>
      <dt>correct</dt><dd>${escapeHtml(String(fail.correct))}</dd>
      <dt>prompt</dt><dd>${escapeHtml(fail.prompt || "")}</dd>
      <dt>model_id</dt><dd>${escapeHtml(fail.model_id || "")}</dd>
      <dt>gen_len</dt><dd>${escapeHtml(String(fail.gen_len ?? 0))}</dd>
    </dl>`;
    const reasoning = fail.reasoning && fail.reasoning !== fail.generation
      ? `<div class="shot-box"><strong>reasoning</strong><pre>${escapeHtml(fail.reasoning)}</pre></div>`
      : "";
    detail.innerHTML = `
      <div class="qid">${escapeHtml(q.id || fail.question_id || "")} · ${escapeHtml(q.modality || "")} · ${escapeHtml(q.category || "")}</div>
      <h3 style="margin:0.35rem 0 0.2rem;font-family:Space Grotesk,sans-serif">${escapeHtml(q.question || "")}</h3>
      ${audio}
      ${source}
      ${gold}
      <div class="shot-box">
        <div class="shot-head">
          <span>${escapeHtml(shot.model_label || fail.model_label || "")}</span>
          <span>s${shot.shot_index ?? fail.shot_index}</span>
          ${goldChip(shot.gold)}
          ${ratings}
        </div>
        <strong>Gradee answer</strong>
        <pre>${escapeHtml(shot.answer_prediction || "")}</pre>
      </div>
      <div class="shot-box">
        <div class="shot-head">
          <span>${escapeHtml(prettyJudge(fail.judge_key))}</span>
          <span class="pending">unparsed</span>
        </div>
        ${stored}
        <strong>Judge generation</strong>
        ${fail.generation
          ? `<pre>${escapeHtml(fail.generation)}</pre>`
          : `<p class="muted">Empty generation.</p>`}
        ${reasoning}
      </div>
    `;
  } catch (err) {
    detail.innerHTML = `<p class="muted">Failed to load failure: ${escapeHtml(String(err))}</p>`;
  }
}

async function init() {
  const pack = await api("/api/pack");
  state.failures = pack.failures || [];
  state.modelLabels = pack.model_labels || [];
  state.judges = pack.judges || [];
  state.stats = pack.stats || {};
  if (pack.mode_titles && typeof pack.mode_titles === "object") {
    state.modeTitles = pack.mode_titles;
  }
  if (Array.isArray(pack.mode_order) && pack.mode_order.length) {
    state.modeOrder = pack.mode_order;
  }
  fillModeSelect();
  state.nLabelRows = pack.n_label_rows || 0;
  state.nQuestions = pack.n_questions || 0;
  state.nUnparsed = pack.n_unparsed || 0;
  const sel = document.getElementById("model");
  sel.innerHTML = `<option value="">All</option>` + state.modelLabels.map(m =>
    `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`
  ).join("");
  const onFilter = () => {
    renderAccuracy();
    renderList();
    const items = filteredFailures();
    if (items.length && !items.some(row => row.i === state.selectedI)) {
      selectFailure(items[0].i);
    }
  };
  ["search", "model", "kind"].forEach(id => {
    document.getElementById(id).addEventListener("input", onFilter);
    document.getElementById(id).addEventListener("change", onFilter);
  });
  document.getElementById("mode").addEventListener("change", () => {
    state.selectedJudge = null;
    renderAccuracy();
    renderList();
    const items = filteredFailures();
    if (state.selectedI != null && items.some(row => row.i === state.selectedI)) {
      selectFailure(state.selectedI);
      return;
    }
    if (items.length) selectFailure(items[0].i);
  });
  renderAccuracy();
  if (!state.failures.length) {
    document.getElementById("stats").textContent = "No unparsed judge answers in the accuracy-only sample.";
    document.getElementById("detail").innerHTML = `<p class="muted">Every stored judge answer in labels.csv parsed as pass/fail.</p>`;
    return;
  }
  const preferred = new URLSearchParams(location.search).get("i");
  const start = (preferred !== null && state.failures.some(row => String(row.i) === preferred))
    ? Number(preferred)
    : filteredFailures()[0]?.i;
  renderList();
  if (start != null) await selectFailure(start);
}

init().catch(err => {
  const stats = document.getElementById("stats");
  if (stats) stats.textContent = String(err);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[view_parse_failures] {self.address_string()} {fmt % args}")

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
        return load_bundle(str(CONFIG["pack_dir"]))

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
                failures = [
                    {
                        "i": row["i"],
                        "question_id": row["question_id"],
                        "model_label": row["model_label"],
                        "shot_index": row["shot_index"],
                        "judge_key": row["judge_key"],
                        "mode": row["mode"],
                        "gen_len": row["gen_len"],
                        "empty": row["empty"],
                        "tail": row["tail"],
                    }
                    for row in bundle["failures"]
                ]
                self._send_json(
                    {
                        "failures": failures,
                        "model_labels": bundle["model_labels"],
                        "judges": bundle["judges"],
                        "stats": bundle["stats"],
                        "mode_titles": grade_mode_titles(),
                        "mode_order": list(grade_prompt_names()),
                        "n_label_rows": bundle["n_label_rows"],
                        "n_questions": bundle["n_questions"],
                        "n_unparsed": bundle["n_unparsed"],
                        "pack_present": bundle["pack_present"],
                        "labels_path": bundle["labels_path"],
                    }
                )
                return

            if path == "/api/failure":
                raw = (qs.get("i") or [""])[0]
                try:
                    index = int(raw)
                except (TypeError, ValueError):
                    self._send_json({"error": "missing i"}, 400)
                    return
                bundle = self._bundle()
                failures = bundle["failures"]
                if index < 0 or index >= len(failures):
                    self._send_json({"error": "failure not found"}, 404)
                    return
                row = failures[index]
                qid = row["question_id"]
                question = bundle["by_id"].get(qid) or {"id": qid}
                audio = resolve_audio(question.get("audio_path"))
                audio_url = f"/audio/{audio.name}" if audio is not None else None
                shot = bundle["shots"].get(row["shot_id"]) or {}
                self._send_json(
                    {
                        "failure": {
                            "i": row["i"],
                            "question_id": row["question_id"],
                            "model_label": row["model_label"],
                            "shot_index": row["shot_index"],
                            "judge_key": row["judge_key"],
                            "mode": row["mode"],
                            "gen_len": row["gen_len"],
                            "empty": row["empty"],
                            "verdict": row["verdict"],
                            "output": row["output"],
                            "correct": row["correct"],
                            "prompt": row["prompt"],
                            "model_id": row["model_id"],
                            "generation": row["generation"],
                            "reasoning": row["reasoning"],
                            "samples": row["samples"],
                        },
                        "question": question,
                        "shot": shot,
                        "audio_url": audio_url,
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
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=PACK_DIR,
        help="Downloaded judging pack (default: outputs/mmar-judging)",
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
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    pack_dir = args.pack_dir.expanduser().resolve()
    CONFIG["pack_dir"] = pack_dir
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

    print(f"Pack:  {pack_dir}", flush=True)
    print(f"Audio: {audio_dir}", flush=True)
    labels_path = pack_dir / LABELS_CSV_NAME
    if not labels_path.is_file():
        print(f"No {LABELS_CSV_NAME} at {labels_path}.", flush=True)
    if not pack_dir.is_dir():
        print(f"Pack directory not found at {pack_dir}.", flush=True)
    bundle = load_bundle(str(pack_dir))
    print(
        f"Loaded {bundle['n_questions']} questions, "
        f"{bundle['n_label_rows']} labeled shots, "
        f"{bundle['n_unparsed']} unparsed judge answers, "
        f"{len(bundle['judges'])} accuracy-only judges",
        flush=True,
    )
    stats = bundle.get("stats") or {}
    for mode in accuracy_mode_names(stats):
        by_judge = stats.get(mode) or {}
        if not isinstance(by_judge, dict):
            continue
        rows = [
            (key, row)
            for key, row in by_judge.items()
            if isinstance(row, dict) and int(row.get("n_unparsed") or 0) > 0
        ]
        if not rows:
            continue
        print(f"  {grade_mode_title(mode)}:", flush=True)
        for key, row in sorted(
            rows, key=lambda item: (-int(item[1].get("n_unparsed") or 0), item[0])
        ):
            rate = row.get("parse_rate")
            rate_s = f"{rate:.3f}" if isinstance(rate, (int, float)) else "—"
            print(
                f"    {key:<52} unp={row.get('n_unparsed', 0):<5} "
                f"empty={row.get('n_empty', 0):<4} "
                f"n={row.get('n_judge_answers', 0):<5} parse={rate_s}",
                flush=True,
            )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
