"""Local web UI for MMAR-Rubrics examples + AF3 generations / grades.

Reads:
  - MMAR-meta.jsonl (questions, GT reasoning, instance rubrics)
  - output/results/audio_flamingo3_mmar/<run>/predictions.jsonl
  - output/results/audio_flamingo3_mmar/<run>/predictions.evaluated.jsonl
  - output/results/audio_flamingo3_mmar/<run>/scores.json / manifest.json

Usage:

    uv run python view_mmar_results.py
    uv run python view_mmar_results.py --port 7860
    uv run python view_mmar_results.py --results-dir ./output/results/audio_flamingo3_mmar
    uv run python view_mmar_results.py --meta ./output/data/mmar/MMAR-meta.jsonl
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.error
import urllib.request
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "output" / "results" / "audio_flamingo3_mmar"
DEFAULT_META = ROOT / "output" / "data" / "mmar" / "MMAR-meta.jsonl"
DEFAULT_AUDIO_DIR = ROOT / "output" / "data" / "mmar" / "audio"
MMAR_META_URL = (
    "https://raw.githubusercontent.com/ddlBoJack/MMAR/main/MMAR-meta.jsonl"
)

# Populated in main() before the server starts.
CONFIG: dict[str, Any] = {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def ensure_meta(meta_path: Path) -> Path:
    """Download MMAR-meta.jsonl (with rubrics) if missing."""
    if meta_path.exists() and meta_path.stat().st_size > 0:
        return meta_path
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MMAR-meta.jsonl -> {meta_path}")
    try:
        urllib.request.urlretrieve(MMAR_META_URL, meta_path)
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Failed to download MMAR meta from {MMAR_META_URL}: {exc}\n"
            f"Pass --meta PATH to an existing MMAR-meta.jsonl."
        ) from exc
    return meta_path


@lru_cache(maxsize=1)
def meta_by_id() -> dict[str, dict]:
    path = Path(CONFIG["meta"])
    return {item["id"]: item for item in load_jsonl(path) if "id" in item}


def list_runs(results_dir: Path) -> list[dict]:
    if not results_dir.is_dir():
        return []
    runs = []
    for child in sorted(results_dir.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        manifest = load_json(child / "manifest.json") or {}
        scores = load_json(child / "scores.json")
        preds = child / "predictions.jsonl"
        evaluated = child / "predictions.evaluated.jsonl"
        n_preds = sum(1 for _ in open(preds, encoding="utf-8")) if preds.exists() else 0
        n_eval = (
            sum(1 for _ in open(evaluated, encoding="utf-8")) if evaluated.exists() else 0
        )
        runs.append(
            {
                "id": child.name,
                "path": str(child),
                "manifest": manifest,
                "scores": scores,
                "n_predictions": n_preds,
                "n_evaluated": n_eval,
                "has_predictions": n_preds > 0,
                "has_evaluated": n_eval > 0,
            }
        )
    return runs


def resolve_run_dir(results_dir: Path, run_id: str | None) -> Path | None:
    if run_id:
        path = results_dir / run_id
        return path if path.is_dir() else None
    latest = results_dir / "latest"
    if latest.is_dir():
        return latest
    runs = [p for p in results_dir.iterdir() if p.is_dir()] if results_dir.is_dir() else []
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.name, reverse=True)[0]


def index_by_id(path: Path) -> dict[str, dict]:
    return {item["id"]: item for item in load_jsonl(path) if "id" in item}


def merge_example(
    meta: dict,
    prediction: dict | None,
    evaluated: dict | None,
) -> dict:
    """Prefer evaluated > prediction > meta for overlapping fields."""
    out = dict(meta)
    if prediction:
        out.update(prediction)
    if evaluated:
        out.update(evaluated)
    out["has_generation"] = bool(
        (prediction or evaluated)
        and (
            (prediction or evaluated or {}).get("model_output")
            or (prediction or evaluated or {}).get("answer_prediction")
            or (prediction or evaluated or {}).get("thinking_prediction")
        )
    )
    out["has_grade"] = evaluated is not None and "score" in (evaluated or {})
    return out


def build_examples(run_dir: Path | None, *, only_generated: bool = False) -> list[dict]:
    meta = meta_by_id()
    predictions: dict[str, dict] = {}
    evaluated: dict[str, dict] = {}
    if run_dir is not None:
        predictions = index_by_id(run_dir / "predictions.jsonl")
        evaluated = index_by_id(run_dir / "predictions.evaluated.jsonl")

    ids: list[str] = []
    seen: set[str] = set()
    # Prefer run order (evaluated, then predictions), then remaining meta.
    for source in (evaluated, predictions):
        for item_id in source:
            if item_id not in seen:
                ids.append(item_id)
                seen.add(item_id)
    if not only_generated:
        for item_id in meta:
            if item_id not in seen:
                ids.append(item_id)
                seen.add(item_id)

    examples = []
    for item_id in ids:
        base = meta.get(item_id) or predictions.get(item_id) or evaluated.get(item_id) or {}
        if "id" not in base:
            continue
        example = merge_example(
            base if item_id in meta else {**base, **{k: v for k, v in (meta.get(item_id) or {}).items()}},
            predictions.get(item_id),
            evaluated.get(item_id),
        )
        # Ensure rubric / thinking from meta win when prediction overwrote with missing keys.
        if item_id in meta:
            for key in ("rubric", "thinking", "cue", "question", "choices", "answer"):
                if key in meta[item_id] and (
                    key not in example or example.get(key) in (None, "", [])
                ):
                    example[key] = meta[item_id][key]
                elif key in meta[item_id] and key in ("rubric", "thinking", "cue"):
                    # Always prefer full meta rubrics / GT CoT when present.
                    example[key] = meta[item_id][key]
        if only_generated and not example.get("has_generation"):
            continue
        examples.append(summarize_example(example))
    return examples


def summarize_example(item: dict) -> dict:
    """Strip bulky fields for the list API; keep enough for the card."""
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "choices": item.get("choices") or [],
        "modality": item.get("modality"),
        "category": item.get("category"),
        "sub_category": item.get("sub-category") or item.get("sub_category"),
        "language": item.get("language"),
        "url": item.get("url"),
        "audio_path": item.get("audio_path"),
        "has_generation": bool(item.get("has_generation")),
        "has_grade": bool(item.get("has_grade")),
        "correct": item.get("correct"),
        "score": item.get("score"),
        "answer_prediction": item.get("answer_prediction"),
        "n_rubric": len(item.get("rubric") or []),
        "n_cues": len(item.get("cue") or []),
    }


def full_example(item_id: str, run_dir: Path | None) -> dict | None:
    meta = meta_by_id()
    prediction = None
    evaluated = None
    if run_dir is not None:
        prediction = index_by_id(run_dir / "predictions.jsonl").get(item_id)
        evaluated = index_by_id(run_dir / "predictions.evaluated.jsonl").get(item_id)
    base = meta.get(item_id) or prediction or evaluated
    if not base:
        return None
    example = merge_example(dict(base), prediction, evaluated)
    if item_id in meta:
        for key in ("rubric", "thinking", "cue", "question", "choices", "answer", "url", "audio_path"):
            if key in meta[item_id]:
                example[key] = meta[item_id][key]
    example["local_audio"] = resolve_local_audio(example.get("audio_path"))
    return example


def resolve_local_audio(audio_path: str | None) -> str | None:
    if not audio_path:
        return None
    audio_dir = Path(CONFIG["audio_dir"])
    candidates = [
        Path(audio_path),
        audio_dir / Path(audio_path).name,
        ROOT / audio_path.lstrip("./"),
        audio_dir.parent / audio_path.lstrip("./"),
    ]
    for path in candidates:
        if path.is_file():
            return f"/audio/{path.name}"
    return None


def find_audio_file(name: str) -> Path | None:
    name = Path(name).name
    audio_dir = Path(CONFIG["audio_dir"])
    candidates = [
        audio_dir / name,
        ROOT / "output" / "data" / "mmar" / "audio" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MMAR Rubrics Viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg: #e8eef2;
    --bg-2: #d9e3ea;
    --ink: #14202a;
    --muted: #5a6b78;
    --line: #b7c7d2;
    --card: #f7fafc;
    --accent: #1f5f8b;
    --accent-soft: #d5e8f4;
    --good: #1f6b4a;
    --bad: #9b3a3a;
    --warn: #7a5b16;
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
    max-width: 1280px; margin: 0 auto;
    display: flex; flex-wrap: wrap; gap: 1rem; align-items: end;
    justify-content: space-between;
  }
  .brand h1 {
    font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
    font-weight: 600; font-size: 1.45rem;
    margin: 0 0 0.15rem; letter-spacing: -0.03em;
  }
  .brand p { margin: 0; color: var(--muted); font-size: 0.92rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; }
  label { font-size: 0.75rem; color: var(--muted); display: grid; gap: 0.25rem; }
  select, input[type="search"] {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.45rem 0.65rem; min-width: 10rem;
  }
  input[type="search"] { min-width: 14rem; }
  main {
    max-width: 1280px; margin: 0 auto; padding: 1.25rem;
    display: grid; grid-template-columns: 340px 1fr; gap: 1rem;
  }
  @media (max-width: 960px) {
    main { grid-template-columns: 1fr; }
  }
  .panel {
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--radius); box-shadow: var(--shadow);
  }
  .stats {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem;
    margin-bottom: 1rem;
  }
  @media (max-width: 720px) { .stats { grid-template-columns: repeat(2, 1fr); } }
  .stat {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 10px; padding: 0.85rem 1rem;
  }
  .stat .k { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .stat .v { font-family: "Space Grotesk", sans-serif; font-size: 1.4rem; margin-top: 0.15rem; }
  .list-panel { overflow: hidden; display: flex; flex-direction: column; max-height: calc(100vh - 8rem); }
  .list-head {
    padding: 0.85rem 1rem; border-bottom: 1px solid var(--line);
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 0.85rem; color: var(--muted);
  }
  .list { overflow: auto; padding: 0.4rem; }
  .item {
    width: 100%; text-align: left; border: 0; background: transparent;
    border-radius: 8px; padding: 0.75rem 0.8rem; cursor: pointer;
    font: inherit; color: inherit; display: grid; gap: 0.35rem;
  }
  .item:hover { background: var(--bg-2); }
  .item.active { background: var(--accent-soft); }
  .item .qid {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.72rem; color: var(--muted);
  }
  .item .q { font-size: 0.92rem; line-height: 1.35; }
  .tags { display: flex; flex-wrap: wrap; gap: 0.3rem; }
  .tag {
    font-size: 0.68rem; padding: 0.15rem 0.45rem; border-radius: 4px;
    background: var(--bg-2); color: var(--muted); border: 1px solid var(--line);
  }
  .tag.good { background: #dff0e8; color: var(--good); border-color: #a9d4c0; }
  .tag.bad { background: #f3e0e0; color: var(--bad); border-color: #dfb4b4; }
  .tag.neutral { background: #ebe4cf; color: var(--warn); border-color: #d4c69a; }
  .detail { padding: 1.15rem 1.25rem 1.5rem; min-height: 60vh; }
  .detail.empty {
    display: grid; place-items: center; color: var(--muted); min-height: 40vh;
  }
  .detail h2 {
    font-family: "Space Grotesk", sans-serif; font-weight: 600;
    font-size: 1.3rem; margin: 0 0 0.75rem; line-height: 1.25; letter-spacing: -0.02em;
  }
  .meta-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1rem; }
  .section { margin-top: 1.25rem; }
  .section h3 {
    margin: 0 0 0.55rem; font-size: 0.78rem; letter-spacing: 0.05em;
    text-transform: uppercase; color: var(--muted); font-weight: 600;
  }
  .choices { display: grid; gap: 0.4rem; }
  .choice {
    border: 1px solid var(--line); border-radius: 8px; padding: 0.55rem 0.75rem;
    background: var(--bg);
  }
  .choice.correct { border-color: #8fcbb4; background: #eef8f3; }
  .choice.pred { outline: 2px solid color-mix(in srgb, var(--accent) 45%, transparent); }
  .choice .label {
    font-family: "IBM Plex Mono", monospace; font-size: 0.75rem; color: var(--muted);
    margin-right: 0.4rem;
  }
  .box {
    border: 1px solid var(--line); border-radius: 10px; padding: 0.85rem 1rem;
    background: var(--bg); white-space: pre-wrap; line-height: 1.5; font-size: 0.95rem;
  }
  .box.mono { font-family: "IBM Plex Mono", monospace; font-size: 0.85rem; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem; }
  @media (max-width: 720px) { .grid-2 { grid-template-columns: 1fr; } }
  .rubric { display: grid; gap: 0.55rem; }
  .rubric-item {
    border: 1px solid var(--line); border-radius: 10px; padding: 0.75rem 0.9rem;
    background: var(--card);
  }
  .rubric-item.pass { border-color: #8fcbb4; background: #f3faf7; }
  .rubric-item.fail { border-color: #dfb4b4; background: #faf3f3; }
  .rubric-item .name { font-weight: 600; margin-bottom: 0.25rem; }
  .rubric-item .point { color: var(--muted); font-size: 0.9rem; }
  .score-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-family: "IBM Plex Mono", monospace; font-size: 0.8rem;
  }
  audio { width: 100%; margin-top: 0.5rem; }
  a { color: var(--accent); }
  .links { display: flex; flex-wrap: wrap; gap: 0.75rem; font-size: 0.9rem; }
</style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div class="brand">
        <h1>MMAR Rubrics Viewer</h1>
        <p>Examples, AF3 generations, and OpenAI rubric grades</p>
      </div>
      <div class="controls">
        <label>Run
          <select id="run"></select>
        </label>
        <label>Filter
          <select id="filter">
            <option value="all">All examples</option>
            <option value="generated">With generations</option>
            <option value="graded">Graded</option>
            <option value="correct">Correct</option>
            <option value="incorrect">Incorrect</option>
            <option value="ungenerated">No generation yet</option>
          </select>
        </label>
        <label>Modality
          <select id="modality"><option value="">All</option></select>
        </label>
        <label>Search
          <input id="search" type="search" placeholder="id, question, answer…" />
        </label>
      </div>
    </div>
  </header>
  <main>
    <div>
      <div class="stats" id="stats"></div>
      <div class="panel list-panel">
        <div class="list-head">
          <span id="list-count">0 examples</span>
          <span id="run-label"></span>
        </div>
        <div class="list" id="list"></div>
      </div>
    </div>
    <div class="panel detail empty" id="detail">Select an example</div>
  </main>
<script>
const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
let state = { runs: [], examples: [], selectedId: null, runId: null };

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function fmtScore(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(3);
}

function tag(text, cls="") {
  return `<span class="tag ${cls}">${escapeHtml(text)}</span>`;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  }[c]));
}

function renderStats(run) {
  const scores = run?.scores || {};
  const m = run?.manifest || {};
  const cards = [
    ["Examples", state.examples.length],
    ["Generated", state.examples.filter(e => e.has_generation).length],
    ["Accuracy", scores.accuracy != null ? (100*scores.accuracy).toFixed(1)+"%" : "—"],
    ["Avg rubric", scores.avg_score != null ? fmtScore(scores.avg_score) : "—"],
  ];
  document.getElementById("stats").innerHTML = cards.map(([k,v]) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`
  ).join("");
  document.getElementById("run-label").textContent =
    m.model_id ? m.model_id.split("/").pop() : (run?.id || "");
}

function filteredExamples() {
  const filter = document.getElementById("filter").value;
  const modality = document.getElementById("modality").value;
  const q = document.getElementById("search").value.trim().toLowerCase();
  return state.examples.filter(e => {
    if (modality && e.modality !== modality) return false;
    if (filter === "generated" && !e.has_generation) return false;
    if (filter === "graded" && !e.has_grade) return false;
    if (filter === "correct" && e.correct !== true) return false;
    if (filter === "incorrect" && !(e.has_grade && e.correct === false)) return false;
    if (filter === "ungenerated" && e.has_generation) return false;
    if (q) {
      const hay = [e.id, e.question, e.answer, e.answer_prediction, e.category, e.modality]
        .join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function renderList() {
  const items = filteredExamples();
  document.getElementById("list-count").textContent = `${items.length} examples`;
  const list = document.getElementById("list");
  list.innerHTML = items.map(e => {
    const tags = [];
    if (e.modality) tags.push(tag(e.modality));
    if (e.category) tags.push(tag(e.category));
    if (e.has_grade) {
      tags.push(tag(e.correct ? "correct" : "incorrect", e.correct ? "good" : "bad"));
      tags.push(tag(`score ${fmtScore(e.score)}`, "neutral"));
    } else if (e.has_generation) {
      tags.push(tag("generated", "neutral"));
    }
    return `<button class="item ${e.id===state.selectedId?"active":""}" data-id="${escapeHtml(e.id)}">
      <div class="qid">${escapeHtml(e.id)}</div>
      <div class="q">${escapeHtml(e.question || "")}</div>
      <div class="tags">${tags.join("")}</div>
    </button>`;
  }).join("") || `<div style="padding:1rem;color:var(--muted)">No examples match.</div>`;
  list.querySelectorAll(".item").forEach(btn => {
    btn.addEventListener("click", () => selectExample(btn.dataset.id));
  });
}

function choiceClass(choice, answer, pred) {
  const classes = ["choice"];
  if (choice === answer) classes.push("correct");
  if (pred && choice === pred) classes.push("pred");
  return classes.join(" ");
}

function renderRubric(example) {
  const rubric = example.rubric || [];
  const results = example.rubric_results || [];
  const byName = Object.fromEntries((results || []).map(r => [r.name, r]));
  if (!rubric.length) return `<div class="box">No rubric on this example.</div>`;
  return `<div class="rubric">${rubric.map(r => {
    const res = byName[r.name];
    let cls = "rubric-item";
    let badge = "";
    const passed = (typeof res?.pass === "boolean") ? res.pass
      : (typeof res?.passed === "boolean") ? res.passed : null;
    if (passed === true || passed === false) {
      cls += passed ? " pass" : " fail";
      badge = `<span class="score-pill">${passed ? "pass" : "fail"}</span>`;
    } else if (res && res.score != null) {
      badge = `<span class="score-pill">score ${escapeHtml(res.score)}</span>`;
    }
    return `<div class="${cls}">
      <div class="name">${escapeHtml(r.name)} ${badge}</div>
      <div class="point">${escapeHtml(r.scoring_point || "")}</div>
      ${r.note ? `<div class="point" style="margin-top:0.35rem">${escapeHtml(r.note)}</div>` : ""}
      ${res?.justification ? `<div class="point" style="margin-top:0.45rem"><em>${escapeHtml(res.justification)}</em></div>` : ""}
    </div>`;
  }).join("")}</div>`;
}

async function selectExample(id) {
  state.selectedId = id;
  renderList();
  const detail = document.getElementById("detail");
  detail.classList.remove("empty");
  detail.innerHTML = `<div style="color:var(--muted)">Loading…</div>`;
  const run = encodeURIComponent(state.runId || "");
  const example = await api(`/api/example?id=${encodeURIComponent(id)}&run=${run}`);
  const choices = example.choices || [];
  const cues = example.cue || [];
  detail.innerHTML = `
    <h2>${escapeHtml(example.question || "")}</h2>
    <div class="meta-row">
      ${example.modality ? tag(example.modality) : ""}
      ${example.category ? tag(example.category) : ""}
      ${example["sub-category"] ? tag(example["sub-category"]) : ""}
      ${example.has_grade ? tag(example.correct ? "correct" : "incorrect", example.correct ? "good" : "bad") : ""}
      ${example.has_grade ? tag("score " + fmtScore(example.score), "neutral") : ""}
      ${tag(example.id, "")}
    </div>
    <div class="links">
      ${example.url ? `<a href="${escapeHtml(example.url)}" target="_blank" rel="noreferrer">Source video</a>` : ""}
      ${example.local_audio ? `<a href="${escapeHtml(example.local_audio)}">Download audio</a>` : ""}
    </div>
    ${example.local_audio ? `<audio controls src="${escapeHtml(example.local_audio)}"></audio>` : ""}

    <div class="section">
      <h3>Choices</h3>
      <div class="choices">
        ${choices.map((c,i) => `
          <div class="${choiceClass(c, example.answer, example.answer_prediction)}">
            <span class="label">(${LABELS[i] || i})</span>${escapeHtml(c)}
            ${c === example.answer ? " · GT" : ""}
            ${example.answer_prediction && c === example.answer_prediction ? " · pred" : ""}
          </div>`).join("")}
      </div>
    </div>

    <div class="section grid-2">
      <div>
        <h3>Ground-truth reasoning</h3>
        <div class="box">${escapeHtml(example.thinking || "—")}</div>
      </div>
      <div>
        <h3>Model reasoning</h3>
        <div class="box">${escapeHtml(example.thinking_prediction || (example.has_generation ? "—" : "No generation in this run"))}</div>
      </div>
    </div>

    <div class="section grid-2">
      <div>
        <h3>Ground-truth answer</h3>
        <div class="box">${escapeHtml(example.answer || "—")}</div>
      </div>
      <div>
        <h3>Model answer</h3>
        <div class="box">${escapeHtml(example.answer_prediction || "—")}</div>
      </div>
    </div>

    ${cues.length ? `<div class="section"><h3>Reasoning cues</h3><div class="tags">${cues.map(c => tag(c)).join("")}</div></div>` : ""}

    <div class="section">
      <h3>Instance rubric${example.has_grade ? " + grades" : ""}</h3>
      ${renderRubric(example)}
    </div>

    ${example.model_output ? `<div class="section"><h3>Raw model output</h3><div class="box mono">${escapeHtml(example.model_output)}</div></div>` : ""}
  `;
}

function fillModalities() {
  const mods = [...new Set(state.examples.map(e => e.modality).filter(Boolean))].sort();
  const sel = document.getElementById("modality");
  const current = sel.value;
  sel.innerHTML = `<option value="">All</option>` + mods.map(m =>
    `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");
  sel.value = mods.includes(current) ? current : "";
}

async function loadRun(runId) {
  state.runId = runId;
  const q = runId ? `?run=${encodeURIComponent(runId)}` : "";
  const data = await api(`/api/examples${q}`);
  state.examples = data.examples;
  const run = state.runs.find(r => r.id === runId) || data.run || null;
  renderStats(run);
  fillModalities();
  renderList();
  if (state.examples.length) {
    const first = filteredExamples()[0] || state.examples[0];
    await selectExample(first.id);
  } else {
    document.getElementById("detail").classList.add("empty");
    document.getElementById("detail").textContent = "No examples found.";
  }
}

async function init() {
  const data = await api("/api/runs");
  state.runs = data.runs;
  const sel = document.getElementById("run");
  if (!state.runs.length) {
    sel.innerHTML = `<option value="">(no runs — showing meta only)</option>`;
  } else {
    sel.innerHTML = state.runs.map(r => {
      const label = `${r.id} · ${r.n_predictions} preds · ${r.n_evaluated} graded`;
      return `<option value="${escapeHtml(r.id)}">${escapeHtml(label)}</option>`;
    }).join("");
  }
  const preferred = data.default_run || state.runs[0]?.id || "";
  if (preferred) sel.value = preferred;
  sel.addEventListener("change", () => loadRun(sel.value || null));
  ["filter","modality","search"].forEach(id => {
    document.getElementById(id).addEventListener("input", renderList);
    document.getElementById(id).addEventListener("change", renderList);
  });
  await loadRun(sel.value || null);
}

init().catch(err => {
  document.getElementById("detail").textContent = String(err);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "MMARViewer/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[viewer] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        results_dir = Path(CONFIG["results_dir"])

        if path in ("/", "/index.html"):
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/runs":
            runs = list_runs(results_dir)
            default = None
            if (results_dir / "latest").is_dir():
                default = "latest"
            elif runs:
                default = runs[0]["id"]
            self._json(
                {
                    "runs": runs,
                    "default_run": default,
                    "meta": str(CONFIG["meta"]),
                    "n_meta": len(meta_by_id()),
                }
            )
            return

        if path == "/api/examples":
            run_id = (qs.get("run") or [None])[0] or None
            only = (qs.get("only_generated") or ["0"])[0] in ("1", "true", "yes")
            run_dir = resolve_run_dir(results_dir, run_id)
            examples = build_examples(run_dir, only_generated=only)
            run_info = None
            if run_dir is not None:
                run_info = next(
                    (r for r in list_runs(results_dir) if r["id"] == run_dir.name),
                    {
                        "id": run_dir.name,
                        "manifest": load_json(run_dir / "manifest.json"),
                        "scores": load_json(run_dir / "scores.json"),
                    },
                )
            self._json({"run": run_info, "examples": examples, "n": len(examples)})
            return

        if path == "/api/example":
            item_id = (qs.get("id") or [None])[0]
            if not item_id:
                self._json({"error": "missing id"}, 400)
                return
            run_id = (qs.get("run") or [None])[0] or None
            run_dir = resolve_run_dir(results_dir, run_id)
            example = full_example(item_id, run_dir)
            if example is None:
                self._json({"error": "not found"}, 404)
                return
            self._json(example)
            return

        if path.startswith("/audio/"):
            name = unquote(path[len("/audio/") :])
            audio = find_audio_file(name)
            if audio is None:
                self._json({"error": "audio not found"}, 404)
                return
            data = audio.read_bytes()
            mime = mimetypes.guess_type(str(audio))[0] or "audio/wav"
            self._send(200, data, mime)
            return

        self._json({"error": "not found"}, 404)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing AF3 MMAR run folders",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_META,
        help="Path to MMAR-meta.jsonl (with rubrics)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Optional local wav directory for playback",
    )
    parser.add_argument(
        "--skip-meta-download",
        action="store_true",
        help="Do not auto-download meta if missing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = args.meta.expanduser().resolve()
    if not args.skip_meta_download:
        ensure_meta(meta_path)
    elif not meta_path.exists():
        raise SystemExit(f"Meta not found: {meta_path}")

    CONFIG["results_dir"] = str(args.results_dir.expanduser().resolve())
    CONFIG["meta"] = str(meta_path)
    CONFIG["audio_dir"] = str(args.audio_dir.expanduser().resolve())

    # Warm caches / validate.
    n_meta = len(meta_by_id())
    runs = list_runs(Path(CONFIG["results_dir"]))
    print(f"Loaded {n_meta} MMAR examples from {meta_path}", flush=True)
    print(f"Found {len(runs)} run(s) under {CONFIG['results_dir']}", flush=True)
    for run in runs[:5]:
        print(
            f"  - {run['id']}: {run['n_predictions']} predictions, "
            f"{run['n_evaluated']} evaluated",
            flush=True,
        )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Open {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
