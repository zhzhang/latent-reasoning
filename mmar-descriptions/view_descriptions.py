"""Local viewer for the MMAR caption + transcription pack.

Shows clips from ``outputs/mmar-descriptions`` with audio and each
model's n-shot descriptions. The MMAR question is reference-only; models
did not see it. Doom-loop detection runs automatically on load.

Usage::

    uv run python mmar-descriptions/view_descriptions.py
    uv run python mmar-descriptions/view_descriptions.py --port 7865
    uv run python mmar-descriptions/view_descriptions.py \\
      --pack-dir ./outputs/mmar-descriptions
    uv run python mmar-descriptions/view_descriptions.py --min-chars 200
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
MMAR_DESC_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MMAR_DESC_DIR) not in sys.path:
    sys.path.insert(0, str(MMAR_DESC_DIR))

from mmar_common import DESCRIPTION_PROMPT, load_jsonl  # noqa: E402
from detect_doom_loops import (  # noqa: E402
    DEFAULT_MIN_CHARS,
    doom_loop_key,
    scan_quality,
)
from view_mmar import (  # noqa: E402
    DEFAULT_AUDIO_DIR,
    MMAR_CATEGORIES,
    MMAR_MODALITIES,
    QUESTION_KEYS,
    CONFIG as MMAR_CONFIG,
    discover_model_labels,
    ensure_mmar_audio,
    load_json,
    order_model_labels,
    resolve_audio,
)

DEFAULT_PACK_DIR = REPO_ROOT / "outputs" / "mmar-descriptions"
DEFAULT_PORT = 7865
CONFIG: dict[str, Any] = {}


def _ordered_unique(values: set[str], canonical: tuple[str, ...] = ()) -> list[str]:
    extras = sorted(name for name in values if name and name not in canonical)
    return [name for name in canonical if name in values] + extras


def _shot_index(shot: dict[str, Any]) -> int:
    raw = shot.get("shot_index")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _shot_text(shot: dict[str, Any]) -> str:
    text = shot.get("answer_prediction") or shot.get("model_output") or ""
    return str(text).strip()


def _preview(text: str, limit: int = 140) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _question_fields(record: dict[str, Any]) -> dict[str, Any]:
    out = {key: record.get(key) for key in QUESTION_KEYS if key in record}
    if not out.get("sub-category") and record.get("sub_category"):
        out["sub-category"] = record.get("sub_category")
    return out


def _compact_shot(shot: dict[str, Any]) -> dict[str, Any]:
    text = _shot_text(shot)
    return {
        "shot_index": _shot_index(shot),
        "text": text,
        "thinking_prediction": shot.get("thinking_prediction") or "",
        "n_chars": len(text),
    }


@lru_cache(maxsize=2)
def load_pack(pack_dir_s: str) -> dict[str, Any]:
    pack_dir = Path(pack_dir_s)
    if not pack_dir.is_dir():
        raise FileNotFoundError(pack_dir)
    manifest = load_json(pack_dir / "manifest.json")
    ids_payload = load_json(pack_dir / "question_ids.json")
    model_labels = discover_model_labels(pack_dir, manifest)

    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for label in model_labels:
        rows = load_jsonl(pack_dir / "models" / label / "predictions.jsonl")
        by_id: dict[str, dict[str, Any]] = {}
        for record in rows:
            qid = str(record.get("id") or "")
            if not qid:
                continue
            shots = [_compact_shot(shot) for shot in (record.get("shots") or [])]
            shots.sort(key=lambda row: int(row["shot_index"]))
            if not shots:
                text = _shot_text(record)
                if text:
                    shots = [
                        {
                            "shot_index": 0,
                            "text": text,
                            "thinking_prediction": record.get("thinking_prediction")
                            or "",
                            "n_chars": len(text),
                        }
                    ]
            by_id[qid] = {
                **_question_fields(record),
                "model": label,
                "n_shots": len(shots) or record.get("n_shots") or 0,
                "shots": shots,
                "preview": _preview(shots[0]["text"] if shots else ""),
            }
        predictions[label] = by_id

    model_labels = [
        label
        for label in order_model_labels(model_labels)
        if any((record.get("shots") or []) for record in predictions.get(label, {}).values())
    ]
    predictions = {label: predictions[label] for label in model_labels}

    preferred_ids = [str(qid) for qid in (ids_payload.get("ids") or []) if qid]
    seen: set[str] = set(preferred_ids)
    all_ids = list(preferred_ids)
    for label in model_labels:
        for qid in predictions[label]:
            if qid not in seen:
                seen.add(qid)
                all_ids.append(qid)

    questions: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    modalities: set[str] = set()
    categories: set[str] = set()
    subcategories: set[str] = set()
    category_subs: dict[str, set[str]] = {}
    n_complete = 0
    for qid in all_ids:
        sample: dict[str, Any] | None = None
        per_model: dict[str, dict[str, Any]] = {}
        n_present = 0
        for label in model_labels:
            record = predictions[label].get(qid)
            if record is None:
                per_model[label] = {"present": False, "n_shots": 0}
                continue
            n_present += 1
            if sample is None:
                sample = record
            per_model[label] = {
                "present": True,
                "n_shots": record.get("n_shots") or len(record.get("shots") or []),
                "preview": record.get("preview") or "",
            }
        if sample is None:
            continue
        complete = n_present >= len(model_labels) and len(model_labels) > 0
        if complete:
            n_complete += 1
        modality = str(sample.get("modality") or "")
        category = str(sample.get("category") or "")
        subcat = str(sample.get("sub-category") or "")
        if modality:
            modalities.add(modality)
        if category:
            categories.add(category)
        if subcat:
            subcategories.add(subcat)
        if category and subcat:
            category_subs.setdefault(category, set()).add(subcat)
        row = {
            "id": qid,
            "question": sample.get("question") or "",
            "answer": sample.get("answer") or "",
            "choices": sample.get("choices") or [],
            "audio_path": sample.get("audio_path"),
            "url": sample.get("url") or "",
            "source": sample.get("source") or "",
            "modality": modality,
            "category": category,
            "sub-category": subcat,
            "language": sample.get("language") or "",
            "n_models": n_present,
            "n_models_total": len(model_labels),
            "complete": complete,
            "per_model": per_model,
        }
        questions.append(
            {
                "id": qid,
                "question": row["question"],
                "modality": modality,
                "category": category,
                "sub-category": subcat,
                "n_models": n_present,
                "n_models_total": len(model_labels),
                "complete": complete,
                "preview": next(
                    (
                        per_model[label].get("preview") or ""
                        for label in model_labels
                        if per_model[label].get("present")
                    ),
                    "",
                ),
            }
        )
        by_id[qid] = row

    coverage = []
    for label in model_labels:
        n_done = len(predictions[label])
        coverage.append(
            {
                "model": label,
                "n_done": n_done,
                "n_total": int(ids_payload.get("n") or len(questions)),
                "complete": n_done >= len(questions) if questions else False,
            }
        )

    return {
        "pack_dir": str(pack_dir),
        "manifest": manifest,
        "model_labels": model_labels,
        "predictions": predictions,
        "questions": questions,
        "by_id": by_id,
        "modalities": _ordered_unique(modalities, MMAR_MODALITIES),
        "categories": _ordered_unique(categories, MMAR_CATEGORIES),
        "subcategories": _ordered_unique(subcategories),
        "category_subcategories": {
            name: _ordered_unique(category_subs.get(name) or set())
            for name in _ordered_unique(categories, MMAR_CATEGORIES)
        },
        "n_shots": int(manifest.get("n_shots") or 6),
        "coverage": coverage,
        "n_complete": n_complete,
        "n_questions": len(questions),
        "prompt": DESCRIPTION_PROMPT,
        "enable_thinking": bool(manifest.get("enable_thinking"))
        if "enable_thinking" in manifest
        else False,
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MMAR Descriptions</title>
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
  .run-meta { margin-top: 0.35rem; font-size: 0.82rem; color: var(--muted); }
  .mode-badge {
    display: inline-flex; align-items: center;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.04em; text-transform: uppercase;
    padding: 0.2rem 0.55rem; border-radius: 999px;
    border: 1px solid #d4b88a; color: #5a3a12; background: #f3e6cf;
    vertical-align: middle;
  }
  .controls { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: end; }
  label { font-size: 0.75rem; color: var(--muted); display: grid; gap: 0.25rem; }
  select, input[type="search"], input[type="number"], button {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.45rem 0.65rem;
  }
  select, input[type="search"] { min-width: 10rem; }
  select.filter-select { max-width: 14rem; }
  button { cursor: pointer; }
  button.active { background: #e2eef6; border-color: #8fb3c9; }
  main {
    max-width: 1480px; margin: 0 auto; padding: 1.25rem;
    display: grid; grid-template-columns: 360px 1fr; gap: 1rem;
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
  #qlist {
    list-style: none; margin: 0; padding: 0;
    max-height: calc(100vh - 220px); overflow: auto;
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
  .chip {
    font-size: 0.7rem; font-family: "IBM Plex Mono", monospace;
    padding: 0.15rem 0.4rem; border-radius: 999px;
    background: #e8eef2; color: var(--muted);
  }
  .chip.missing { background: #f3e6cf; color: #5a3a12; }
  .chip.doom { background: #fde8e8; color: #8b1a1a; border: 1px solid #e8a0a0; }
  .chip.short { background: #fff3dc; color: #6b4a0a; border: 1px solid #e8c878; }
  .chip.all-bad { background: #efe8ff; color: #4a2d7a; border: 1px solid #c4a8e8; }
  .quality-summary {
    flex-basis: 100%; font-size: 0.85rem; color: var(--muted);
    padding-top: 0.15rem;
  }
  .quality-summary strong.doom { color: #8b1a1a; }
  .quality-summary strong.short { color: #6b4a0a; }
  .quality-summary strong.all-bad { color: #4a2d7a; }
  details.quality-report {
    flex-basis: 100%; font-size: 0.82rem; color: var(--muted);
  }
  details.quality-report table {
    width: 100%; border-collapse: collapse; margin-top: 0.35rem;
    font-family: "IBM Plex Mono", monospace; font-size: 0.75rem;
  }
  details.quality-report th, details.quality-report td {
    border-bottom: 1px solid var(--line); padding: 0.3rem 0.45rem; text-align: left;
  }
  details.quality-report tbody tr { cursor: pointer; }
  details.quality-report tbody tr:hover { background: #eef5fa; }
  .doom-summary {
    flex-basis: 100%; font-size: 0.85rem; color: var(--muted);
    padding-top: 0.15rem;
  }
  .doom-summary strong { color: #8b1a1a; }
  .shot.too-short {
    background: #fffbf2;
    border-left: 3px solid #c98;
    padding-left: 0.55rem;
    margin-left: -0.55rem;
  }
  .model-block.all-bad {
    border-color: #c4a8e8;
    background: #faf7ff;
  }
  .qtext {
    margin: 0.35rem 0 0; font-size: 0.86rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  #detail { padding: 1rem 1.15rem; max-height: calc(100vh - 160px); overflow: auto; }
  .muted { color: var(--muted); }
  audio { width: 100%; margin-top: 0.5rem; }
  .audio-source {
    margin: 0.3rem 0 0; font-size: 0.75rem; color: var(--muted);
    word-break: break-all;
  }
  .audio-source a { color: var(--accent); }
  .meta-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.5rem 0; }
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
  .ref-q { margin: 0 0 0.5rem; font-size: 0.95rem; }
  .gold { margin: 0; font-size: 0.88rem; color: var(--muted); }
  .gold strong { color: var(--ink); }
  pre {
    white-space: pre-wrap; word-break: break-word;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem; margin: 0;
    background: #f2f6f9; padding: 0.55rem 0.65rem; border-radius: 8px;
  }
  .model-grid {
    display: grid; gap: 0.85rem;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    margin-top: 0.85rem;
  }
  .model-block {
    border: 1px solid var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; background: #fff;
    min-width: 0;
  }
  .model-block h3 {
    margin: 0 0 0.5rem; font-size: 1rem;
    display: flex; gap: 0.5rem; align-items: baseline; flex-wrap: wrap;
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
  .shot.hidden { display: none; }
  .shot.doom-loop {
    background: #fff8f8;
    border-left: 3px solid #c44;
    padding-left: 0.55rem;
    margin-left: -0.55rem;
  }
  .doom-reasons {
    font-size: 0.75rem; color: #8b1a1a;
    margin-bottom: 0.35rem; line-height: 1.35;
  }
  .empty-model { color: var(--muted); font-size: 0.85rem; }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <h1>MMAR Descriptions <span class="mode-badge">thinking off</span></h1>
      <p>Caption and transcribe each clip. Models never saw the MMAR question.</p>
      <div class="run-meta" id="run-meta"></div>
    </div>
    <div class="controls">
      <label>Search
        <input id="search" type="search" placeholder="id, question, caption…" />
      </label>
      <label>Category
        <select id="category" class="filter-select"></select>
      </label>
      <label>Subcategory
        <select id="subcategory" class="filter-select"></select>
      </label>
      <label>Modality
        <select id="modality" class="filter-select"></select>
      </label>
      <label>Shot
        <select id="shot-filter">
          <option value="all">All shots</option>
        </select>
      </label>
      <label>Min chars
        <input id="min-chars" type="number" min="1" step="1" value="200" />
      </label>
      <button id="filter-incomplete" type="button">Incomplete only</button>
      <button id="filter-doom" type="button">Doom loops only</button>
      <button id="filter-short" type="button">Too short only</button>
      <button id="filter-all-bad" type="button">All bad only</button>
      <div class="quality-summary" id="quality-summary"></div>
      <details class="quality-report" id="all-bad-report">
        <summary>All-bad clip×model pairs</summary>
        <div id="all-bad-table"></div>
      </details>
    </div>
  </div>
</header>
<main>
  <section class="panel">
    <h2>Clips</h2>
    <div class="stats" id="stats"></div>
    <ul id="qlist"></ul>
  </section>
  <section class="panel" id="detail">
    <p class="muted">Loading pack…</p>
  </section>
</main>
<script>
const state = {
  questions: [],
  modelLabels: [],
  modalities: [],
  categories: [],
  subcategories: [],
  categorySubcategories: {},
  nShots: 3,
  prompt: "",
  selectedId: null,
  filterIncomplete: false,
  filterDoom: false,
  filterShort: false,
  filterAllBad: false,
  minChars: 200,
  shotFilter: "all",
  qualityReport: null,
  doomReport: null,
  doomHits: [],
  doomByQid: {},
  doomLookup: {},
  shortHits: [],
  shortByQid: {},
  shortLookup: {},
  fullyFiltered: [],
  fullyFilteredByQid: {},
  fullyFilteredLookup: {},
};

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function api(path) {
  const res = await fetch(path);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function fillSelect(el, values, allLabel) {
  el.innerHTML = `<option value="">${escapeHtml(allLabel)}</option>` +
    values.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
}

function fillSubcategorySelect() {
  const category = document.getElementById("category").value;
  const values = category
    ? (state.categorySubcategories[category] || [])
    : state.subcategories;
  fillSelect(document.getElementById("subcategory"), values, "All");
}

function fillShotFilter() {
  const el = document.getElementById("shot-filter");
  const opts = [`<option value="all">All shots</option>`];
  for (let i = 0; i < state.nShots; i++) {
    opts.push(`<option value="${i}">Shot ${i}</option>`);
  }
  el.innerHTML = opts.join("");
}

function doomKey(qid, model, shotIndex) {
  return `${qid}|${model}|${shotIndex}`;
}

function modelKey(qid, model) {
  return `${qid}|${model}`;
}

function applyQualityReport(quality) {
  state.qualityReport = quality;
  state.minChars = Number(quality.min_chars || state.minChars);
  const doom = quality.doom || {};
  state.doomReport = doom;
  state.doomHits = doom.hits || [];
  state.doomByQid = doom.by_qid || {};
  state.doomLookup = {};
  for (const hit of state.doomHits) {
    state.doomLookup[doomKey(hit.id, hit.model, hit.shot_index)] = hit;
  }
  state.shortHits = quality.short_hits || [];
  state.shortByQid = quality.short_by_qid || {};
  state.shortLookup = {};
  for (const hit of state.shortHits) {
    state.shortLookup[doomKey(hit.id, hit.model, hit.shot_index)] = hit;
  }
  state.fullyFiltered = quality.fully_filtered || [];
  state.fullyFilteredByQid = quality.fully_filtered_by_qid || {};
  state.fullyFilteredLookup = {};
  for (const row of state.fullyFiltered) {
    state.fullyFilteredLookup[modelKey(row.id, row.model)] = row;
  }
  renderQualitySummary();
  renderAllBadTable();
}

function renderQualitySummary() {
  const el = document.getElementById("quality-summary");
  if (!state.qualityReport) {
    el.textContent = "";
    return;
  }
  const doom = state.doomReport || {};
  const { n_hits, n_questions_with_hits } = doom;
  const {
    n_short_hits,
    n_questions_with_short,
    n_fully_filtered,
    n_clips_with_fully_filtered_model,
    min_chars,
    n_shots_scanned,
  } = state.qualityReport;
  const modelBits = Object.entries(state.qualityReport.fully_filtered_by_model || {})
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 4)
    .map(([label, count]) => `${label}: ${count}`)
    .join(" · ");
  el.innerHTML =
    `<span><strong class="doom">${n_hits || 0}</strong> doom loops in ${n_questions_with_hits || 0} clips · ` +
    `<strong class="short">${n_short_hits || 0}</strong> too short (&lt;${min_chars} chars) in ${n_questions_with_short || 0} clips · ` +
    `<strong class="all-bad">${n_fully_filtered || 0}</strong> all-bad clip×model pairs in ${n_clips_with_fully_filtered_model || 0} clips</span>` +
    `<span class="muted"> (${n_shots_scanned || 0} shots scanned)</span>` +
    (modelBits ? `<span class="muted"> · all-bad: ${escapeHtml(modelBits)}</span>` : "");
}

function renderAllBadTable() {
  const el = document.getElementById("all-bad-table");
  const rows = state.fullyFiltered || [];
  if (!rows.length) {
    el.innerHTML = "<p class=\"muted\">No clip×model pairs where every shot is too short or doom-looped.</p>";
    return;
  }
  el.innerHTML = `<table>
    <thead><tr><th>Clip</th><th>Model</th><th>Shots</th><th>Short</th><th>Doom</th></tr></thead>
    <tbody>${rows.map(row => `<tr data-id="${escapeHtml(row.id)}" data-model="${escapeHtml(row.model)}">
      <td>${escapeHtml(row.id)}</td>
      <td>${escapeHtml(row.model)}</td>
      <td>${row.n_shots}</td>
      <td>${row.n_short}</td>
      <td>${row.n_doom}</td>
    </tr>`).join("")}</tbody>
  </table>`;
  el.querySelectorAll("tbody tr").forEach(tr => {
    tr.addEventListener("click", () => {
      selectQuestion(tr.dataset.id);
      document.getElementById("all-bad-report").open = false;
    });
  });
}

async function reloadQuality() {
  const minChars = Math.max(1, Number(document.getElementById("min-chars").value) || state.minChars);
  document.getElementById("min-chars").value = String(minChars);
  const quality = await api(`/api/quality?min_chars=${encodeURIComponent(minChars)}`);
  applyQualityReport(quality);
  renderList();
  if (state.selectedId) await selectQuestion(state.selectedId);
}

function filteredQuestions() {
  const q = document.getElementById("search").value.trim().toLowerCase();
  const category = document.getElementById("category").value;
  const sub = document.getElementById("subcategory").value;
  const modality = document.getElementById("modality").value;
  return state.questions.filter(row => {
    if (state.filterIncomplete && row.complete) return false;
    if (state.filterDoom && !(state.doomByQid[row.id] || 0)) return false;
    if (state.filterShort && !(state.shortByQid[row.id] || 0)) return false;
    if (state.filterAllBad && !(state.fullyFilteredByQid[row.id] || []).length) return false;
    if (category && row.category !== category) return false;
    if (sub && row["sub-category"] !== sub) return false;
    if (modality && row.modality !== modality) return false;
    if (!q) return true;
    const hay = [row.id, row.question, row.preview, row.category, row.modality]
      .join(" ").toLowerCase();
    return hay.includes(q);
  });
}

function renderList() {
  const items = filteredQuestions();
  document.getElementById("stats").innerHTML =
    `<span><strong>${items.length}</strong> / ${state.questions.length} clips</span>`;
  document.getElementById("qlist").innerHTML = items.map(row => {
    const missing = row.n_models < row.n_models_total;
    const doomN = state.doomByQid[row.id] || 0;
    const shortN = state.shortByQid[row.id] || 0;
    const allBadModels = state.fullyFilteredByQid[row.id] || [];
    return `<li data-id="${escapeHtml(row.id)}" class="${row.id === state.selectedId ? "active" : ""}">
      <div class="qid">${escapeHtml(row.id)}</div>
      <p class="qtext">${escapeHtml(row.preview || row.question || "—")}</p>
      <div class="meta-row">
        ${row.modality ? `<span class="chip">${escapeHtml(row.modality)}</span>` : ""}
        ${row.category ? `<span class="chip">${escapeHtml(row.category)}</span>` : ""}
        ${doomN ? `<span class="chip doom">${doomN} doom</span>` : ""}
        ${shortN ? `<span class="chip short">${shortN} short</span>` : ""}
        ${allBadModels.length ? `<span class="chip all-bad">${allBadModels.length} all-bad</span>` : ""}
        <span class="chip ${missing ? "missing" : ""}">${row.n_models}/${row.n_models_total}</span>
      </div>
    </li>`;
  }).join("");
  document.querySelectorAll("#qlist li").forEach(el => {
    el.addEventListener("click", () => selectQuestion(el.dataset.id));
  });
}

function applyTaxonomyFilters() {
  renderList();
  const items = filteredQuestions();
  if (items.length && !items.some(row => row.id === state.selectedId)) {
    selectQuestion(items[0].id);
  }
}

function renderShots(pred, qid, model) {
  const shots = (pred && pred.shots) || [];
  if (!shots.length) return `<p class="empty-model">No description yet.</p>`;
  return shots.map(shot => {
    const hidden = state.shotFilter !== "all" && String(shot.shot_index) !== state.shotFilter;
    const doom = state.doomLookup[doomKey(qid, model, shot.shot_index)];
    const short = state.shortLookup[doomKey(qid, model, shot.shot_index)];
    const doomBadge = doom
      ? `<span class="chip doom">doom ${Number(doom.score || 0).toFixed(2)}</span>`
      : "";
    const shortBadge = short
      ? `<span class="chip short">&lt;${state.minChars} chars</span>`
      : "";
    const reasons = doom && (doom.reasons || []).length
      ? `<div class="doom-reasons">${escapeHtml((doom.reasons || []).join(", "))}` +
        `${doom.snippet ? ` · “${escapeHtml(doom.snippet)}”` : ""}</div>`
      : "";
    const shotClasses = [
      hidden ? "hidden" : "",
      doom ? "doom-loop" : "",
      short ? "too-short" : "",
    ].filter(Boolean).join(" ");
    return `<div class="shot ${shotClasses}">
      <div class="shot-head">
        <span>shot ${shot.shot_index}</span>
        ${doomBadge}
        ${shortBadge}
        <span class="muted">${shot.n_chars || 0} chars</span>
      </div>
      ${reasons}
      <pre>${escapeHtml(shot.text || "—")}</pre>
    </div>`;
  }).join("");
}

async function selectQuestion(id) {
  state.selectedId = id;
  const params = new URLSearchParams(location.search);
  params.set("id", id);
  history.replaceState(null, "", `${location.pathname}?${params}`);
  document.querySelectorAll("#qlist li").forEach(el => {
    el.classList.toggle("active", el.dataset.id === id);
  });
  const data = await api(`/api/question?id=${encodeURIComponent(id)}`);
  const q = data.question || {};
  const audio = data.audio_url
    ? `<audio controls preload="none" src="${escapeHtml(data.audio_url)}"></audio>`
    : `<p class="muted">Audio not found locally.</p>`;
  const source = q.url
    ? `<p class="audio-source"><a href="${escapeHtml(q.url)}" target="_blank" rel="noreferrer">${escapeHtml(q.url)}</a></p>`
    : "";
  const models = (data.model_labels || []).map(label => {
    const pred = (data.predictions || {})[label];
    const n = pred ? (pred.shots || []).length : 0;
    const modelDoomN = pred
      ? (pred.shots || []).filter(shot => state.doomLookup[doomKey(id, label, shot.shot_index)]).length
      : 0;
    const modelShortN = pred
      ? (pred.shots || []).filter(shot => state.shortLookup[doomKey(id, label, shot.shot_index)]).length
      : 0;
    const allBad = !!state.fullyFilteredLookup[modelKey(id, label)];
    return `<article class="model-block ${allBad ? "all-bad" : ""}">
      <h3>${escapeHtml(label)}
        <span class="chip">${n} shot${n === 1 ? "" : "s"}</span>
        ${allBad ? `<span class="chip all-bad">all bad</span>` : ""}
        ${modelDoomN ? `<span class="chip doom">${modelDoomN} doom</span>` : ""}
        ${modelShortN ? `<span class="chip short">${modelShortN} short</span>` : ""}
      </h3>
      ${pred ? renderShots(pred, id, label) : `<p class="empty-model">Missing.</p>`}
    </article>`;
  }).join("");
  document.getElementById("detail").innerHTML = `
    <div class="qid">${escapeHtml(q.id || id)}</div>
    <div class="meta-row">
      ${q.modality ? `<span class="chip">${escapeHtml(q.modality)}</span>` : ""}
      ${q.category ? `<span class="chip">${escapeHtml(q.category)}</span>` : ""}
      ${q["sub-category"] ? `<span class="chip">${escapeHtml(q["sub-category"])}</span>` : ""}
      ${q.language ? `<span class="chip">${escapeHtml(q.language)}</span>` : ""}
    </div>
    ${audio}
    ${source}
    <details class="accordion">
      <summary>Description prompt</summary>
      <div class="accordion-body"><pre>${escapeHtml(data.prompt || state.prompt)}</pre></div>
    </details>
    <details class="accordion">
      <summary>MMAR question (not shown to model)</summary>
      <div class="accordion-body">
        <p class="ref-q">${escapeHtml(q.question || "—")}</p>
        <p class="gold">Gold answer: <strong>${escapeHtml(q.answer || "—")}</strong></p>
      </div>
    </details>
    <div class="model-grid">${models}</div>
  `;
}

function stepQuestion(delta) {
  const items = filteredQuestions();
  if (!items.length) return;
  const idx = items.findIndex(row => row.id === state.selectedId);
  const next = items[(Math.max(idx, 0) + delta + items.length) % items.length];
  selectQuestion(next.id);
}

async function init() {
  const data = await api("/api/pack");
  state.modelLabels = data.model_labels || [];
  state.questions = data.questions || [];
  state.modalities = data.modalities || [];
  state.categories = data.categories || [];
  state.subcategories = data.subcategories || [];
  state.categorySubcategories = data.category_subcategories || {};
  state.nShots = data.n_shots || 6;
  state.prompt = data.prompt || "";
  state.minChars = Number(data.min_chars || 200);
  document.getElementById("min-chars").value = String(state.minChars);
  await reloadQuality();
  const meta = [];
  meta.push(`${state.nShots} shots`);
  meta.push(`${state.modelLabels.length} models`);
  meta.push(`${state.questions.length} clips`);
  if (data.n_complete != null) meta.push(`${data.n_complete} with every model`);
  if (data.enable_thinking === false) meta.push("thinking off");
  document.getElementById("run-meta").textContent = meta.join(" · ");
  fillSelect(document.getElementById("category"), state.categories, "All");
  fillSelect(document.getElementById("modality"), state.modalities, "All");
  fillSubcategorySelect();
  fillShotFilter();
  document.getElementById("search").addEventListener("input", renderList);
  document.getElementById("category").addEventListener("change", () => {
    fillSubcategorySelect();
    applyTaxonomyFilters();
  });
  document.getElementById("subcategory").addEventListener("change", applyTaxonomyFilters);
  document.getElementById("modality").addEventListener("change", applyTaxonomyFilters);
  document.getElementById("shot-filter").addEventListener("change", (ev) => {
    state.shotFilter = ev.target.value;
    if (state.selectedId) selectQuestion(state.selectedId);
  });
  document.getElementById("filter-incomplete").addEventListener("click", () => {
    state.filterIncomplete = !state.filterIncomplete;
    document.getElementById("filter-incomplete").classList.toggle("active", state.filterIncomplete);
    applyTaxonomyFilters();
  });
  document.getElementById("filter-doom").addEventListener("click", () => {
    state.filterDoom = !state.filterDoom;
    document.getElementById("filter-doom").classList.toggle("active", state.filterDoom);
    applyTaxonomyFilters();
  });
  document.getElementById("filter-short").addEventListener("click", () => {
    state.filterShort = !state.filterShort;
    document.getElementById("filter-short").classList.toggle("active", state.filterShort);
    applyTaxonomyFilters();
  });
  document.getElementById("filter-all-bad").addEventListener("click", () => {
    state.filterAllBad = !state.filterAllBad;
    document.getElementById("filter-all-bad").classList.toggle("active", state.filterAllBad);
    applyTaxonomyFilters();
  });
  document.getElementById("min-chars").addEventListener("change", () => {
    reloadQuality().catch(err => {
      document.getElementById("quality-summary").textContent = String(err);
    });
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.target && ["INPUT", "SELECT", "TEXTAREA"].includes(ev.target.tagName)) return;
    if (ev.key === "j" || ev.key === "ArrowDown") { ev.preventDefault(); stepQuestion(1); }
    if (ev.key === "k" || ev.key === "ArrowUp") { ev.preventDefault(); stepQuestion(-1); }
  });
  if (!state.questions.length) {
    document.getElementById("stats").textContent = "No clips in this pack.";
    document.getElementById("detail").innerHTML =
      `<p class="muted">Pack is empty. Download with<br><code>uv run modal run download_results.py --volume-name mmar-descriptions</code></p>`;
    return;
  }
  const preferred = new URLSearchParams(location.search).get("id");
  const start = (preferred && state.questions.find(q => q.id === preferred))
    ? preferred
    : state.questions[0].id;
  renderList();
  await selectQuestion(start);
}

init().catch(err => {
  document.getElementById("stats").textContent = String(err);
  document.getElementById("detail").innerHTML = `<p class="muted">${escapeHtml(String(err))}</p>`;
});
</script>
</body>
</html>
"""


def _current_pack() -> dict[str, Any]:
    return load_pack(str(CONFIG.get("pack_dir") or ""))


def _doom_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        doom_loop_key(hit["id"], hit["model"], int(hit["shot_index"])): hit
        for hit in report.get("hits") or []
    }


@lru_cache(maxsize=8)
def load_quality_report(pack_dir_s: str, min_chars: int) -> dict[str, Any]:
    bundle = load_pack(pack_dir_s)
    return scan_quality(
        bundle["predictions"],
        model_labels=bundle["model_labels"],
        min_chars=min_chars,
    )


def _parse_min_chars(qs: dict[str, list[str]]) -> int:
    raw = (qs.get("min_chars") or [""])[0].strip()
    if not raw:
        return int(CONFIG.get("min_chars") or DEFAULT_MIN_CHARS)
    try:
        return max(1, int(raw))
    except ValueError:
        return int(CONFIG.get("min_chars") or DEFAULT_MIN_CHARS)


def _current_quality(min_chars: int | None = None) -> dict[str, Any]:
    threshold = int(min_chars if min_chars is not None else CONFIG.get("min_chars") or DEFAULT_MIN_CHARS)
    return load_quality_report(str(CONFIG.get("pack_dir") or ""), threshold)


def _current_doom(min_chars: int | None = None) -> dict[str, Any]:
    quality = _current_quality(min_chars)
    report = dict(quality.get("doom") or {})
    report["lookup"] = _doom_lookup(report)
    report["min_chars"] = quality.get("min_chars")
    return report


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[view_descriptions] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/pack":
                bundle = _current_pack()
                self._send_json(
                    {
                        "questions": bundle["questions"],
                        "manifest": bundle["manifest"],
                        "model_labels": bundle["model_labels"],
                        "modalities": bundle["modalities"],
                        "categories": bundle["categories"],
                        "subcategories": bundle["subcategories"],
                        "category_subcategories": bundle["category_subcategories"],
                        "coverage": bundle["coverage"],
                        "n_shots": bundle["n_shots"],
                        "n_complete": bundle["n_complete"],
                        "n_questions": bundle["n_questions"],
                        "prompt": bundle["prompt"],
                        "enable_thinking": bundle["enable_thinking"],
                        "min_chars": int(CONFIG.get("min_chars") or DEFAULT_MIN_CHARS),
                    }
                )
                return

            if path == "/api/quality":
                min_chars = _parse_min_chars(qs)
                refresh = (qs.get("refresh") or [""])[0].lower() in ("1", "true", "yes")
                if refresh:
                    load_quality_report.cache_clear()
                self._send_json(_current_quality(min_chars))
                return

            if path == "/api/question":
                qid = (qs.get("id") or [""])[0]
                if not qid:
                    self._send_json({"error": "missing id"}, 400)
                    return
                bundle = _current_pack()
                row = bundle["by_id"].get(qid)
                if row is None:
                    self._send_json({"error": "question not found"}, 404)
                    return
                model_labels = bundle["model_labels"]
                preds = {
                    label: bundle["predictions"].get(label, {}).get(qid)
                    for label in model_labels
                }
                audio = resolve_audio(row.get("audio_path"))
                audio_url = f"/audio/{audio.name}" if audio is not None else None
                self._send_json(
                    {
                        "question": row,
                        "predictions": preds,
                        "audio_url": audio_url,
                        "model_labels": model_labels,
                        "n_shots": bundle["n_shots"],
                        "prompt": bundle["prompt"],
                    }
                )
                return

            if path == "/api/doom-loops":
                min_chars = _parse_min_chars(qs)
                refresh = (qs.get("refresh") or [""])[0].lower() in ("1", "true", "yes")
                if refresh:
                    load_quality_report.cache_clear()
                report = _current_doom(min_chars)
                payload = {key: value for key, value in report.items() if key != "lookup"}
                self._send_json(payload)
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
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=DEFAULT_PACK_DIR,
        help="Descriptions pack (default: outputs/mmar-descriptions)",
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
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Flag generations shorter than this many characters (default: 200)",
    )
    args = parser.parse_args()

    pack_dir = args.pack_dir.expanduser().resolve()
    CONFIG["pack_dir"] = pack_dir
    CONFIG["min_chars"] = max(1, int(args.min_chars))
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
    load_pack.cache_clear()
    load_quality_report.cache_clear()

    print(f"Pack:  {pack_dir}")
    print(f"Audio: {audio_dir}")
    print(f"Min chars threshold: {CONFIG['min_chars']}")
    if not pack_dir.is_dir():
        print("Pack directory not found. Download with:")
        print("  uv run modal run download_results.py --volume-name mmar-descriptions")
    else:
        bundle = load_pack(str(pack_dir))
        print(
            f"Loaded {bundle['n_questions']} clips, "
            f"{len(bundle['model_labels'])} models, "
            f"{bundle['n_shots']} shots"
        )
        for row in bundle["coverage"]:
            status = "complete" if row["complete"] else "partial"
            print(
                f"  {row['model']:<24} {row['n_done']:>4}/{row['n_total']:<4} {status}"
            )
        quality = load_quality_report(str(pack_dir), CONFIG["min_chars"])
        print(
            f"All-bad clip×model pairs: {quality['n_fully_filtered']} "
            f"({quality['n_clips_with_fully_filtered_model']} clips) "
            f"at min_chars={quality['min_chars']}"
        )
        for label, count in sorted(
            (quality.get("fully_filtered_by_model") or {}).items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"  {label:<24} {count} all-bad")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
