"""Local viewer for MMAR question-difficulty experiment results.

Browse fixed-sample questions sorted hardest-first by mean shot success
rate across models, and inspect each model's 10 sampled responses.

Reads from ``outputs/exp-mmar-question-difficulty/<run_id>/``:

    difficulty.jsonl
    scores.json
    manifest.json
    models/<label>/predictions.jsonl

Usage:

    uv run python exp-mmar-question-difficulty/view_difficulty.py
    uv run python exp-mmar-question-difficulty/view_difficulty.py --port 7861
    uv run python exp-mmar-question-difficulty/view_difficulty.py \\
      --results-dir ./outputs/exp-mmar-question-difficulty
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

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmar_common import (  # noqa: E402
    AF_NEXT_THINK_SUFFIX,
    build_mmar_freeform_prompt,
    build_mmar_prompt,
)

DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs" / "exp-mmar-question-difficulty"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "mmar"
DEFAULT_AUDIO_DIR = DEFAULT_DATA_DIR / "audio"

MODEL_LABELS = (
    "af-next-think",
    "mimo-audio-7b",
    "interactive-omni-8b",
    "qwen3-omni",
    "voxtral-small-24b",
)

CONFIG: dict[str, Any] = {}


def infer_run_mode(manifest: dict | None = None, scores: dict | None = None) -> str:
    """Return ``freeform`` or ``mc`` from manifest / scores stamps."""
    manifest = manifest or {}
    scores = scores or {}
    mode = str(manifest.get("mode") or scores.get("mode") or "").strip().lower()
    if mode in {"freeform", "free_form", "free-form", "open"}:
        return "freeform"
    if mode in {"mc", "multiple_choice", "multiple-choice", "mcq", "choice"}:
        return "mc"
    scoring = str(
        manifest.get("scoring") or scores.get("scoring") or ""
    ).lower()
    if "freeform" in scoring or "qwen_freeform" in scoring:
        return "freeform"
    return "mc"


def mode_label(mode: str) -> str:
    return "Freeform" if mode == "freeform" else "MCQ"


def build_base_prompt(item: dict, mode: str, *, think_suffix: str | None = None) -> str:
    if mode == "freeform":
        return build_mmar_freeform_prompt(item, think_suffix=think_suffix)
    return build_mmar_prompt(item, think_suffix=think_suffix)


def build_model_prompts(item: dict, mode: str) -> dict[str, str]:
    """Reconstruct the text prompts sent to each model for this question."""
    base = build_base_prompt(item, mode)
    af_base = build_base_prompt(item, mode, think_suffix=AF_NEXT_THINK_SUFFIX)
    if mode == "freeform":
        step_system = (
            "You are an expert in audio analysis. "
            "Listen carefully and answer the question accurately.\n"
            f"{base}"
        )
    else:
        step_system = (
            "You are an expert in audio analysis. "
            "Listen carefully and answer the multiple-choice question accurately.\n"
            f"{base}"
        )
    return {
        "shared": base,
        "af-next-think": (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"<sound>{af_base}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "mimo-audio-7b": (
            "<|im_start|>user\n"
            f"<|sosp|><|empty|><|eosp|>{base}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n"
        ),
        "interactive-omni-8b": (
            "[audio attached]\n"
            f"{base}"
        ),
        "qwen3-omni": (
            "<|im_start|>system\n"
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
            "Group, capable of perceiving auditory and visual inputs, as well as "
            "generating text and speech.<|im_end|>\n"
            "<|im_start|>user\n"
            f"<|audio_start|><|audio_pad|><|audio_end|>{base}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "voxtral-small-24b": (
            "[audio attached]\n"
            f"{base}"
        ),
        "step-audio-2-mini": (
            f"<|im_start|>system\n{step_system}<|im_end|>\n"
            "<|im_start|>user\n<audio_patch><|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    }


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


def discover_runs(results_dir: Path) -> list[dict]:
    runs: list[dict] = []
    if not results_dir.is_dir():
        return runs
    for path in sorted(results_dir.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        manifest = load_json(path / "manifest.json") or {}
        scores = load_json(path / "scores.json") or {}
        difficulty = path / "difficulty.jsonl"
        mode = infer_run_mode(manifest, scores)
        runs.append(
            {
                "id": path.name,
                "path": str(path),
                "has_difficulty": difficulty.exists(),
                "n_questions": scores.get("n_questions"),
                "avg_success_rate": scores.get("avg_success_rate"),
                "models": manifest.get("models") or list(MODEL_LABELS),
                "seed": manifest.get("seed"),
                "n_shots": manifest.get("n_shots"),
                "temperature": manifest.get("temperature"),
                "mode": mode,
                "mode_label": mode_label(mode),
                "scoring": manifest.get("scoring") or scores.get("scoring"),
                "grader_model_id": manifest.get("grader_model_id")
                or scores.get("grader_model_id"),
                "source_run_id": manifest.get("source_run_id")
                or scores.get("source_run_id"),
            }
        )
    return runs


def run_dir_for(run_id: str) -> Path:
    return Path(CONFIG["results_dir"]) / run_id


@lru_cache(maxsize=8)
def load_run_bundle(run_id: str) -> dict[str, Any]:
    run_dir = run_dir_for(run_id)
    difficulty = load_jsonl(run_dir / "difficulty.jsonl")
    # Already hardest-first from aggregate; keep order.
    by_id = {str(row["id"]): row for row in difficulty}
    predictions: dict[str, dict[str, dict]] = {}
    for label in MODEL_LABELS:
        preds = load_jsonl(run_dir / "models" / label / "predictions.jsonl")
        predictions[label] = {str(p["id"]): p for p in preds if p.get("id")}
    manifest = load_json(run_dir / "manifest.json") or {}
    scores = load_json(run_dir / "scores.json") or {}
    mode = infer_run_mode(manifest, scores)
    return {
        "difficulty": difficulty,
        "by_id": by_id,
        "predictions": predictions,
        "scores": scores,
        "manifest": manifest,
        "mode": mode,
        "mode_label": mode_label(mode),
    }


def resolve_audio(audio_path: str | None) -> Path | None:
    if not audio_path:
        return None
    path = Path(audio_path)
    if path.is_file():
        return path
    # Volume paths like /cache/data/mmar/audio/foo.wav or ./audio/foo.wav
    name = path.name
    candidates = [
        Path(CONFIG["audio_dir"]) / name,
        Path(CONFIG["audio_dir"]) / path.name,
        REPO_ROOT / "data" / "mmar" / "audio" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MMAR Question Difficulty</title>
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
    max-width: 1400px; margin: 0 auto;
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
  select, input[type="search"] {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.45rem 0.65rem; min-width: 12rem;
  }
  main {
    max-width: 1400px; margin: 0 auto; padding: 1.25rem;
    display: grid; grid-template-columns: 360px 1fr; gap: 1rem;
  }
  @media (max-width: 980px) {
    main { grid-template-columns: 1fr; }
  }
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
  .rate {
    font-family: "IBM Plex Mono", monospace;
    font-weight: 500; font-size: 0.95rem;
  }
  .mini-rates {
    display: flex; flex-wrap: wrap; gap: 0.35rem;
    margin-top: 0.35rem;
  }
  .chip {
    font-size: 0.7rem; font-family: "IBM Plex Mono", monospace;
    padding: 0.15rem 0.4rem; border-radius: 999px;
    background: #e8eef2; color: var(--muted);
  }
  .qtext {
    margin: 0.35rem 0 0; font-size: 0.86rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  #detail { padding: 1rem 1.15rem; max-height: calc(100vh - 160px); overflow: auto; }
  .muted { color: var(--muted); }
  .answer-box, .choice-box, .model-block {
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
    display: flex; gap: 0.5rem; align-items: center;
    font-family: "IBM Plex Mono", monospace; font-size: 0.8rem;
    margin-bottom: 0.35rem;
  }
  .pass { color: var(--good); background: var(--soft-good); padding: 0.1rem 0.4rem; border-radius: 999px; }
  .fail { color: var(--bad); background: var(--soft-bad); padding: 0.1rem 0.4rem; border-radius: 999px; }
  pre {
    white-space: pre-wrap; word-break: break-word;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem; margin: 0.25rem 0 0;
    background: #f2f6f9; padding: 0.55rem 0.65rem; border-radius: 8px;
  }
  audio { width: 100%; margin-top: 0.5rem; }
  .mode-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.04em; text-transform: uppercase;
    padding: 0.2rem 0.55rem; border-radius: 999px;
    border: 1px solid var(--line);
    vertical-align: middle;
  }
  .mode-badge.mc {
    color: #1a4a6e; background: #d9e8f3; border-color: #a9c4d8;
  }
  .mode-badge.freeform {
    color: #5a3a12; background: #f3e6cf; border-color: #d4b88a;
  }
  .brand-row {
    display: flex; flex-wrap: wrap; gap: 0.55rem; align-items: center;
  }
  .run-meta {
    margin-top: 0.35rem; font-size: 0.82rem; color: var(--muted);
    display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center;
  }
  .stats .mode-badge { margin-right: 0.15rem; }
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
  details.prompt-model {
    border: 1px solid var(--line); border-radius: 8px;
    margin: 0.5rem 0; background: #fbfcfd;
  }
  details.prompt-model > summary {
    cursor: pointer; padding: 0.55rem 0.7rem;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem; font-weight: 500;
    list-style: none; display: flex; justify-content: space-between;
  }
  details.prompt-model > summary::-webkit-details-marker { display: none; }
  details.prompt-model > summary::after {
    content: "▸"; color: var(--muted);
  }
  details.prompt-model[open] > summary::after { content: "▾"; }
  details.prompt-model pre {
    margin: 0; border-radius: 0 0 8px 8px;
    border-top: 1px solid var(--line);
  }
  .choice-box.hidden-from-model {
    border-style: dashed; background: #faf7f2;
  }
  .choice-box .note {
    font-size: 0.78rem; color: var(--muted); margin: 0.25rem 0 0.45rem;
  }
  .grader-note {
    font-size: 0.78rem; color: var(--muted);
    font-family: "IBM Plex Mono", monospace;
  }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <div class="brand-row">
        <h1>MMAR Question Difficulty</h1>
        <span id="mode-badge" class="mode-badge mc">MCQ</span>
      </div>
      <p id="brand-sub">Hardest-first by mean shot success rate across models</p>
      <div class="run-meta" id="run-meta"></div>
    </div>
    <div class="controls">
      <label>Run
        <select id="run"></select>
      </label>
      <label>Search
        <input id="search" type="search" placeholder="id / question text" />
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
const MODEL_LABELS = [
  "af-next-think",
  "mimo-audio-7b",
  "interactive-omni-8b",
  "qwen3-omni",
  "voxtral-small-24b",
];
const state = {
  runs: [],
  runId: "",
  mode: "mc",
  modeLabel: "MCQ",
  manifest: {},
  scores: {},
  questions: [],
  selectedId: null,
};

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function fmtRate(v) {
  if (v === null || v === undefined) return "—";
  return (100 * Number(v)).toFixed(0) + "%";
}

function modeBadgeHtml(mode, label) {
  const cls = mode === "freeform" ? "freeform" : "mc";
  return `<span class="mode-badge ${cls}">${escapeHtml(label || (mode === "freeform" ? "Freeform" : "MCQ"))}</span>`;
}

function setHeaderMode(mode, modeLabel, manifest, scores) {
  const badge = document.getElementById("mode-badge");
  badge.className = `mode-badge ${mode === "freeform" ? "freeform" : "mc"}`;
  badge.textContent = modeLabel || (mode === "freeform" ? "Freeform" : "MCQ");
  const sub = document.getElementById("brand-sub");
  if (mode === "freeform") {
    sub.textContent = "Freeform answers · graded by Qwen judge · hardest-first";
  } else {
    sub.textContent = "Multiple-choice · string-match scoring · hardest-first";
  }
  const metaBits = [];
  const scoring = manifest.scoring || scores.scoring;
  if (scoring) metaBits.push(`scoring: ${scoring}`);
  if (manifest.grader_model_id || scores.grader_model_id) {
    metaBits.push(`grader: ${manifest.grader_model_id || scores.grader_model_id}`);
  }
  if (manifest.source_run_id || scores.source_run_id) {
    metaBits.push(`source: ${manifest.source_run_id || scores.source_run_id}`);
  }
  if (manifest.n_shots || scores.n_shots) {
    metaBits.push(`${manifest.n_shots || "—"} shots`);
  }
  document.getElementById("run-meta").textContent = metaBits.join(" · ");
}

function renderStats(scores, mode, modeLabel) {
  const by = scores.by_model || {};
  const parts = [
    modeBadgeHtml(mode, modeLabel),
    `<span><strong>${scores.n_questions ?? "—"}</strong> questions</span>`,
    `<span>avg <strong>${fmtRate(scores.avg_success_rate)}</strong></span>`,
  ];
  for (const label of MODEL_LABELS) {
    const m = by[label] || {};
    parts.push(`<span>${label}: <strong>${fmtRate(m.avg_shot_success_rate)}</strong></span>`);
  }
  document.getElementById("stats").innerHTML = parts.join(" · ");
}

function renderList() {
  const q = document.getElementById("search").value.trim().toLowerCase();
  const list = document.getElementById("qlist");
  const items = state.questions.filter(row => {
    if (!q) return true;
    return String(row.id).toLowerCase().includes(q)
      || String(row.question || "").toLowerCase().includes(q);
  });
  list.innerHTML = items.map(row => {
    const chips = MODEL_LABELS.map(label => {
      const pm = (row.per_model || {})[label] || {};
      return `<span class="chip">${label.split("-")[0]} ${fmtRate(pm.shot_success_rate)}</span>`;
    }).join("");
    const active = row.id === state.selectedId ? "active" : "";
    return `<li class="${active}" data-id="${row.id}">
      <div class="qid">${row.id}</div>
      <div class="rate">${fmtRate(row.avg_success_rate)} avg</div>
      <div class="mini-rates">${chips}</div>
      <p class="qtext">${escapeHtml(row.question || "")}</p>
    </li>`;
  }).join("");
  list.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => selectQuestion(li.dataset.id));
  });
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderPromptAccordion(prompts, mode) {
  if (!prompts) return "";
  const shared = prompts.shared || "";
  const modelOrder = [
    "af-next-think",
    "mimo-audio-7b",
    "interactive-omni-8b",
    "qwen3-omni",
    "voxtral-small-24b",
    "step-audio-2-mini",
  ];
  const modelBlocks = modelOrder
    .filter(label => prompts[label])
    .map(label => `<details class="prompt-model">
      <summary>${escapeHtml(label)}</summary>
      <pre>${escapeHtml(prompts[label])}</pre>
    </details>`)
    .join("");
  const modeNote = mode === "freeform"
    ? "Question only — multiple-choice options were not shown to the model."
    : "Multiple-choice prompt including the four options.";
  return `<details class="accordion" open>
    <summary><span>Full prompt</span></summary>
    <div class="accordion-body">
      <p class="muted" style="margin:0 0 0.55rem">${escapeHtml(modeNote)}</p>
      <strong style="font-size:0.82rem;color:var(--muted)">Shared question text</strong>
      <pre>${escapeHtml(shared)}</pre>
      <strong style="display:block;margin-top:0.75rem;font-size:0.82rem;color:var(--muted)">Per-model chat wrappers</strong>
      ${modelBlocks}
    </div>
  </details>`;
}

async function selectQuestion(id) {
  state.selectedId = id;
  renderList();
  const detail = document.getElementById("detail");
  detail.innerHTML = `<p class="muted">Loading…</p>`;
  const data = await api(`/api/question?run=${encodeURIComponent(state.runId)}&id=${encodeURIComponent(id)}`);
  const row = data.difficulty;
  const mode = data.mode || state.mode;
  const modeLabel = data.mode_label || state.modeLabel;
  const choices = (row.choices || []).map((c, i) =>
    `<div>(${String.fromCharCode(65+i)}) ${escapeHtml(c)}</div>`
  ).join("");
  let modelsHtml = "";
  for (const label of MODEL_LABELS) {
    const pred = (data.predictions || {})[label];
    const pm = (row.per_model || {})[label] || {};
    if (!pred) {
      modelsHtml += `<div class="model-block"><h3>${label} <span class="muted">missing</span></h3></div>`;
      continue;
    }
    const shots = pred.shots || [];
    const shotsHtml = shots.map(shot => {
      const ok = shot.correct;
      const grader = shot.grader_output
        ? `<span class="grader-note">judge: ${escapeHtml(shot.grader_output)}</span>`
        : "";
      return `<div class="shot">
        <div class="shot-head">
          <span>shot ${shot.shot_index}</span>
          <span class="${ok ? "pass" : "fail"}">${ok ? "pass" : "fail"}</span>
          <span class="muted">parsed: ${escapeHtml(shot.answer_prediction || "")}</span>
          ${grader}
        </div>
        <pre>${escapeHtml(shot.model_output || "")}</pre>
      </div>`;
    }).join("");
    modelsHtml += `<div class="model-block">
      <h3>${label}
        <span class="chip">${fmtRate(pm.shot_success_rate)} (${pm.n_shot_correct ?? "—"}/${pm.n_shots ?? "—"})</span>
      </h3>
      ${shotsHtml || "<p class='muted'>No shots stored.</p>"}
    </div>`;
  }
  const audio = data.audio_url
    ? `<audio controls preload="none" src="${data.audio_url}"></audio>`
    : `<p class="muted">Audio not found locally.</p>`;
  const choicesClass = mode === "freeform" ? "choice-box hidden-from-model" : "choice-box";
  const choicesTitle = mode === "freeform"
    ? `<strong>Choices</strong><div class="note">Not shown to the model in freeform mode (gold answer reference only).</div>`
    : `<strong>Choices</strong>`;
  detail.innerHTML = `
    <div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap">
      <div class="qid">${escapeHtml(row.id)}</div>
      ${modeBadgeHtml(mode, modeLabel)}
    </div>
    <h3 style="margin:0.4rem 0 0.2rem">${escapeHtml(row.question || "")}</h3>
    <p class="muted">${escapeHtml(row.modality || "")} · ${escapeHtml(row.category || "")} · avg ${fmtRate(row.avg_success_rate)}</p>
    ${audio}
    ${renderPromptAccordion(data.prompts, mode)}
    <div class="${choicesClass}">${choicesTitle}${choices}</div>
    <div class="answer-box"><strong>Gold</strong><div>${escapeHtml(row.answer || "")}</div></div>
    ${modelsHtml}
  `;
}

async function loadRun(runId) {
  state.runId = runId;
  const data = await api(`/api/questions?run=${encodeURIComponent(runId)}`);
  state.questions = data.questions || [];
  state.mode = data.mode || "mc";
  state.modeLabel = data.mode_label || (state.mode === "freeform" ? "Freeform" : "MCQ");
  state.manifest = data.manifest || {};
  state.scores = data.scores || {};
  setHeaderMode(state.mode, state.modeLabel, state.manifest, state.scores);
  renderStats(state.scores, state.mode, state.modeLabel);
  state.selectedId = state.questions[0]?.id || null;
  renderList();
  if (state.selectedId) await selectQuestion(state.selectedId);
}

async function init() {
  const data = await api("/api/runs");
  state.runs = data.runs || [];
  const sel = document.getElementById("run");
  if (!state.runs.length) {
    sel.innerHTML = `<option value="">(no runs found)</option>`;
    document.getElementById("stats").textContent = "No runs under results dir.";
    return;
  }
  sel.innerHTML = state.runs.map(r => {
    const tag = r.mode_label || (r.mode === "freeform" ? "Freeform" : "MCQ");
    const avail = r.has_difficulty ? "" : " (no difficulty yet)";
    return `<option value="${r.id}">[${tag}] ${r.id}${avail}</option>`;
  }).join("");
  sel.addEventListener("change", () => loadRun(sel.value));
  document.getElementById("search").addEventListener("input", renderList);
  await loadRun(state.runs[0].id);
}

init().catch(err => {
  document.getElementById("stats").textContent = String(err);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[viewer] {self.address_string()} {fmt % args}")

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

        if path in ("/", "/index.html"):
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/runs":
            self._send_json({"runs": discover_runs(Path(CONFIG["results_dir"]))})
            return

        if path == "/api/questions":
            run_id = (qs.get("run") or [""])[0]
            if not run_id:
                self._send_json({"error": "missing run"}, 400)
                return
            try:
                bundle = load_run_bundle(run_id)
            except FileNotFoundError:
                self._send_json({"error": "run not found"}, 404)
                return
            self._send_json(
                {
                    "questions": bundle["difficulty"],
                    "scores": bundle["scores"],
                    "manifest": bundle["manifest"],
                    "mode": bundle["mode"],
                    "mode_label": bundle["mode_label"],
                }
            )
            return

        if path == "/api/question":
            run_id = (qs.get("run") or [""])[0]
            qid = (qs.get("id") or [""])[0]
            if not run_id or not qid:
                self._send_json({"error": "missing run or id"}, 400)
                return
            bundle = load_run_bundle(run_id)
            row = bundle["by_id"].get(qid)
            if row is None:
                self._send_json({"error": "question not found"}, 404)
                return
            preds = {
                label: bundle["predictions"].get(label, {}).get(qid)
                for label in MODEL_LABELS
            }
            audio = resolve_audio(row.get("audio_path"))
            audio_url = None
            if audio is not None:
                audio_url = f"/audio/{audio.name}"
            # Prefer a prediction record (has full item fields) when building prompts.
            sample = next(
                (preds[label] for label in MODEL_LABELS if preds.get(label)),
                row,
            )
            prompts = build_model_prompts(sample, bundle["mode"])
            self._send_json(
                {
                    "difficulty": row,
                    "predictions": preds,
                    "audio_url": audio_url,
                    "mode": bundle["mode"],
                    "mode_label": bundle["mode_label"],
                    "prompts": prompts,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing run folders",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Local MMAR wav directory",
    )
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    CONFIG["results_dir"] = args.results_dir.expanduser().resolve()
    CONFIG["audio_dir"] = args.audio_dir.expanduser().resolve()
    load_run_bundle.cache_clear()

    print(f"Results: {CONFIG['results_dir']}")
    print(f"Audio:   {CONFIG['audio_dir']}")
    runs = discover_runs(CONFIG["results_dir"])
    print(f"Found {len(runs)} run(s)")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
