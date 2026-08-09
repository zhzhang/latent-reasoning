"""Compare MCQ vs freeform MMAR difficulty runs side by side.

Pairs one MC and one freeform run that share the same question set,
computes per-question average-correctness deltas, and lets you browse
sorted by that gap. Selecting a question shows both modes' per-model
shots and judges.

Usage:

    uv run python view_mode_compare.py
    uv run python view_mode_compare.py --port 7861
    uv run python view_mode_compare.py --mc-run 20260807T144946Z --ff-run 20260807T145000Z
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

import view_difficulty as vd
from view_difficulty import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_RESULTS_DIR,
    build_model_prompts,
    discover_runs,
    ensure_mmar_audio,
    load_run_bundle,
    resolve_audio,
    resolve_run_roots,
)

REPO_ROOT = Path(__file__).resolve().parent
CONFIG: dict[str, Any] = {}


def _sync_vd_config() -> None:
    """Keep view_difficulty.CONFIG in sync so its loaders resolve paths."""
    vd.CONFIG["results_dir"] = CONFIG["results_dir"]
    vd.CONFIG["audio_dir"] = CONFIG["audio_dir"]


def discover_pairs(results_dir: Path) -> list[dict]:
    """Find (mc, freeform) run pairs that share question ids when possible."""
    runs = discover_runs(results_dir)
    by_id = {r["id"]: r for r in runs}
    mc_runs = [r for r in runs if r.get("mode") == "mc"]
    ff_runs = [r for r in runs if r.get("mode") == "freeform"]
    pairs: list[dict] = []
    used_mc: set[str] = set()
    used_ff: set[str] = set()

    # Prefer freeform.source_run_id → MC when present.
    for ff in ff_runs:
        src = ff.get("source_run_id")
        if not src or src not in by_id:
            continue
        mc = by_id[src]
        if mc.get("mode") != "mc":
            continue
        pairs.append(_pair_meta(mc, ff, how="source_run_id"))
        used_mc.add(mc["id"])
        used_ff.add(ff["id"])

    # Remaining: if exactly one unused MC and one unused FF, pair them.
    leftover_mc = [r for r in mc_runs if r["id"] not in used_mc]
    leftover_ff = [r for r in ff_runs if r["id"] not in used_ff]
    if len(leftover_mc) == 1 and len(leftover_ff) == 1:
        pairs.append(_pair_meta(leftover_mc[0], leftover_ff[0], how="sole_remaining"))
        used_mc.add(leftover_mc[0]["id"])
        used_ff.add(leftover_ff[0]["id"])
        leftover_mc, leftover_ff = [], []

    # Otherwise expose every MC×FF combination among leftovers (small N).
    for mc in leftover_mc:
        for ff in leftover_ff:
            pairs.append(_pair_meta(mc, ff, how="cartesian"))

    # If nothing matched but we still have both modes, cartesian all.
    if not pairs and mc_runs and ff_runs:
        for mc in mc_runs:
            for ff in ff_runs:
                pairs.append(_pair_meta(mc, ff, how="cartesian"))

    pairs.sort(key=lambda p: (p["mc_run"], p["ff_run"]), reverse=True)
    return pairs


def _pair_meta(mc: dict, ff: dict, *, how: str) -> dict:
    return {
        "id": f"{mc['id']}__{ff['id']}",
        "mc_run": mc["id"],
        "ff_run": ff["id"],
        "how": how,
        "mc": mc,
        "ff": ff,
        "models": sorted(set(mc.get("models") or []) | set(ff.get("models") or [])),
        "mc_avg": mc.get("avg_success_rate"),
        "ff_avg": ff.get("avg_success_rate"),
        "n_shots_mc": mc.get("n_shots"),
        "n_shots_ff": ff.get("n_shots"),
    }


@lru_cache(maxsize=8)
def load_pair_bundle(mc_run: str, ff_run: str) -> dict[str, Any]:
    _sync_vd_config()
    mc_bundle = load_run_bundle(mc_run)
    ff_bundle = load_run_bundle(ff_run)
    if mc_bundle["mode"] != "mc":
        raise ValueError(f"{mc_run} is mode={mc_bundle['mode']}, expected mc")
    if ff_bundle["mode"] != "freeform":
        raise ValueError(f"{ff_run} is mode={ff_bundle['mode']}, expected freeform")

    mc_by = mc_bundle["by_id"]
    ff_by = ff_bundle["by_id"]
    shared_ids = sorted(set(mc_by) & set(ff_by))
    model_labels = sorted(
        set(mc_bundle["model_labels"]) | set(ff_bundle["model_labels"])
    )

    questions: list[dict] = []
    for qid in shared_ids:
        mc_row = mc_by[qid]
        ff_row = ff_by[qid]
        mc_rate = mc_row.get("avg_success_rate")
        ff_rate = ff_row.get("avg_success_rate")
        delta = None
        if mc_rate is not None and ff_rate is not None:
            delta = float(mc_rate) - float(ff_rate)
        per_model: dict[str, dict] = {}
        for label in model_labels:
            mc_pm = (mc_row.get("per_model") or {}).get(label) or {}
            ff_pm = (ff_row.get("per_model") or {}).get(label) or {}
            mc_sr = mc_pm.get("shot_success_rate")
            ff_sr = ff_pm.get("shot_success_rate")
            pm_delta = None
            if mc_sr is not None and ff_sr is not None:
                pm_delta = float(mc_sr) - float(ff_sr)
            per_model[label] = {
                "mc_shot_success_rate": mc_sr,
                "ff_shot_success_rate": ff_sr,
                "delta": pm_delta,
                "mc_n_shot_correct": mc_pm.get("n_shot_correct"),
                "ff_n_shot_correct": ff_pm.get("n_shot_correct"),
                "mc_n_shots": mc_pm.get("n_shots"),
                "ff_n_shots": ff_pm.get("n_shots"),
            }
        questions.append(
            {
                "id": qid,
                "question": mc_row.get("question") or ff_row.get("question"),
                "answer": mc_row.get("answer") or ff_row.get("answer"),
                "choices": mc_row.get("choices") or ff_row.get("choices") or [],
                "modality": mc_row.get("modality") or ff_row.get("modality"),
                "category": mc_row.get("category") or ff_row.get("category"),
                "audio_path": mc_row.get("audio_path") or ff_row.get("audio_path"),
                "mc_avg_success_rate": mc_rate,
                "ff_avg_success_rate": ff_rate,
                "delta": delta,  # MC − freeform
                "abs_delta": abs(delta) if delta is not None else None,
                "per_model": per_model,
            }
        )

    mc_avg = mc_bundle["scores"].get("avg_success_rate")
    ff_avg = ff_bundle["scores"].get("avg_success_rate")
    overall_delta = None
    if mc_avg is not None and ff_avg is not None:
        overall_delta = float(mc_avg) - float(ff_avg)

    return {
        "mc_run": mc_run,
        "ff_run": ff_run,
        "mc_bundle": mc_bundle,
        "ff_bundle": ff_bundle,
        "questions": questions,
        "by_id": {q["id"]: q for q in questions},
        "model_labels": model_labels,
        "overall": {
            "n_questions": len(questions),
            "mc_only": len(set(mc_by) - set(ff_by)),
            "ff_only": len(set(ff_by) - set(mc_by)),
            "mc_avg_success_rate": mc_avg,
            "ff_avg_success_rate": ff_avg,
            "delta": overall_delta,
            "mc_by_model": (mc_bundle["scores"].get("by_model") or {}),
            "ff_by_model": (ff_bundle["scores"].get("by_model") or {}),
        },
    }


def sort_questions(questions: list[dict], sort: str) -> list[dict]:
    """Sort paired question rows. Default: largest MC−FF gap first."""
    rows = list(questions)

    def _key_num(row: dict, field: str, *, missing: float) -> float:
        v = row.get(field)
        return missing if v is None else float(v)

    if sort == "delta_asc":
        rows.sort(key=lambda r: _key_num(r, "delta", missing=0.0))
    elif sort == "abs_delta_desc":
        rows.sort(key=lambda r: -_key_num(r, "abs_delta", missing=-1.0))
    elif sort == "mc_asc":
        rows.sort(key=lambda r: _key_num(r, "mc_avg_success_rate", missing=2.0))
    elif sort == "mc_desc":
        rows.sort(key=lambda r: -_key_num(r, "mc_avg_success_rate", missing=-1.0))
    elif sort == "ff_asc":
        rows.sort(key=lambda r: _key_num(r, "ff_avg_success_rate", missing=2.0))
    elif sort == "ff_desc":
        rows.sort(key=lambda r: -_key_num(r, "ff_avg_success_rate", missing=-1.0))
    else:  # delta_desc
        rows.sort(key=lambda r: -_key_num(r, "delta", missing=-2.0))
    return rows


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MMAR Mode Compare</title>
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
    --mc: #1a4a6e;
    --mc-bg: #d9e8f3;
    --ff: #5a3a12;
    --ff-bg: #f3e6cf;
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
    font-weight: 600; font-size: 1.35rem;
    margin: 0 0 0.15rem; letter-spacing: -0.03em;
  }
  .brand p { margin: 0; color: var(--muted); font-size: 0.88rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: end; }
  label { font-size: 0.75rem; color: var(--muted); display: grid; gap: 0.25rem; }
  select, input[type="search"] {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.45rem 0.65rem; min-width: 11rem;
  }
  main {
    max-width: 1480px; margin: 0 auto; padding: 1.25rem;
    display: grid; grid-template-columns: 380px 1fr; gap: 1rem;
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
    font-size: 0.82rem; color: var(--muted);
  }
  .stats strong { color: var(--ink); }
  #qlist {
    list-style: none; margin: 0; padding: 0;
    max-height: calc(100vh - 240px); overflow: auto;
  }
  #qlist li {
    border-bottom: 1px solid var(--line);
    padding: 0.7rem 1rem; cursor: pointer;
  }
  #qlist li:hover { background: #eef5fa; }
  #qlist li.active { background: #e2eef6; }
  .qid {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; color: var(--muted);
  }
  .delta-row {
    display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: baseline;
    margin-top: 0.2rem;
  }
  .rate {
    font-family: "IBM Plex Mono", monospace;
    font-weight: 500; font-size: 0.92rem;
  }
  .delta-pos { color: var(--mc); }
  .delta-neg { color: var(--ff); }
  .delta-zero { color: var(--muted); }
  .mini-rates {
    display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.3rem;
  }
  .chip {
    font-size: 0.68rem; font-family: "IBM Plex Mono", monospace;
    padding: 0.12rem 0.35rem; border-radius: 999px;
    background: #e8eef2; color: var(--muted);
  }
  .chip.mc { background: var(--mc-bg); color: var(--mc); }
  .chip.ff { background: var(--ff-bg); color: var(--ff); }
  .qtext {
    margin: 0.3rem 0 0; font-size: 0.84rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  #detail { padding: 1rem 1.15rem; max-height: calc(100vh - 160px); overflow: auto; }
  .muted { color: var(--muted); }
  .mode-badge {
    display: inline-flex; align-items: center;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.04em; text-transform: uppercase;
    padding: 0.18rem 0.5rem; border-radius: 999px;
    border: 1px solid var(--line);
  }
  .mode-badge.mc { color: var(--mc); background: var(--mc-bg); border-color: #a9c4d8; }
  .mode-badge.freeform { color: var(--ff); background: var(--ff-bg); border-color: #d4b88a; }
  .answer-box, .choice-box, .model-block, .mode-col {
    border: 1px solid var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; margin: 0.75rem 0; background: #fff;
  }
  .mode-pair {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;
    margin: 0.75rem 0;
  }
  @media (max-width: 900px) { .mode-pair { grid-template-columns: 1fr; } }
  .mode-col h4 {
    margin: 0 0 0.55rem; font-size: 0.92rem;
    display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;
  }
  .mode-col.mc { background: #f5f9fc; }
  .mode-col.ff { background: #fbf8f2; }
  .model-block h3 {
    margin: 0 0 0.5rem; font-size: 1rem;
    display: flex; gap: 0.55rem; align-items: baseline; flex-wrap: wrap;
  }
  .shot {
    border-top: 1px dashed var(--line);
    padding: 0.55rem 0;
  }
  .shot:first-of-type { border-top: none; }
  .shot-head {
    display: flex; gap: 0.45rem; align-items: center; flex-wrap: wrap;
    font-family: "IBM Plex Mono", monospace; font-size: 0.78rem;
    margin-bottom: 0.3rem;
  }
  .pass { color: var(--good); background: var(--soft-good); padding: 0.08rem 0.35rem; border-radius: 999px; }
  .fail { color: var(--bad); background: var(--soft-bad); padding: 0.08rem 0.35rem; border-radius: 999px; }
  pre {
    white-space: pre-wrap; word-break: break-word;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.78rem; margin: 0.2rem 0 0;
    background: #f2f6f9; padding: 0.5rem 0.6rem; border-radius: 8px;
  }
  audio { width: 100%; margin-top: 0.5rem; }
  details.accordion {
    border: 1px solid var(--line); border-radius: 10px;
    background: #fff; margin: 0.65rem 0; overflow: hidden;
  }
  details.accordion > summary {
    cursor: pointer; list-style: none;
    padding: 0.7rem 0.85rem;
    font-weight: 600; font-size: 0.9rem;
    display: flex; align-items: center; justify-content: space-between;
    gap: 0.75rem; user-select: none;
  }
  details.accordion > summary::-webkit-details-marker { display: none; }
  details.accordion > summary::after {
    content: "+";
    font-family: "IBM Plex Mono", monospace; color: var(--muted);
  }
  details.accordion[open] > summary {
    border-bottom: 1px solid var(--line); background: #f2f6f9;
  }
  details.accordion[open] > summary::after { content: "−"; }
  .accordion-body { padding: 0.7rem 0.85rem; }
  .judge-pills { display: flex; flex-wrap: wrap; gap: 0.25rem; align-items: center; }
  .judge-pill {
    font-size: 0.66rem; font-family: "IBM Plex Mono", monospace;
    padding: 0.06rem 0.3rem; border-radius: 999px;
  }
  .judge-pill.pass { color: var(--good); background: var(--soft-good); }
  .judge-pill.fail { color: var(--bad); background: var(--soft-bad); }
  .judge-pill.pending { color: var(--muted); background: #e8eef2; }
  .judge-gens { margin-top: 0.35rem; display: flex; flex-direction: column; gap: 0.3rem; }
  .judge-gen details.accordion { margin: 0; }
  .judge-gen details.accordion > summary { font-size: 0.76rem; padding: 0.35rem 0.55rem; }
  .judge-gen .accordion-body { padding: 0.45rem 0.55rem 0.6rem; }
  .summary-bar {
    display: flex; flex-wrap: wrap; gap: 0.55rem; align-items: center;
    margin: 0.5rem 0 0.25rem;
  }
  .delta-big {
    font-family: "IBM Plex Mono", monospace;
    font-size: 1.05rem; font-weight: 500;
  }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <h1>MMAR Mode Compare</h1>
      <p>Δ = MCQ avg − freeform avg · positive means choices helped</p>
      <div class="muted" id="pair-meta" style="margin-top:0.3rem;font-size:0.82rem"></div>
    </div>
    <div class="controls">
      <label>Pair
        <select id="pair"></select>
      </label>
      <label>Sort
        <select id="sort">
          <option value="delta_desc">Δ MC−FF (high → low)</option>
          <option value="delta_asc">Δ MC−FF (low → high)</option>
          <option value="abs_delta_desc">|Δ| largest</option>
          <option value="mc_asc">MC hardest-first</option>
          <option value="mc_desc">MC easiest-first</option>
          <option value="ff_asc">Freeform hardest-first</option>
          <option value="ff_desc">Freeform easiest-first</option>
        </select>
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
const state = {
  pairs: [],
  pairId: "",
  mcRun: "",
  ffRun: "",
  sort: "delta_desc",
  modelLabels: [],
  overall: {},
  questions: [],
  selectedId: null,
  mcJudges: [],
  ffJudges: [],
  mcPrimary: null,
  ffPrimary: null,
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

function fmtDelta(v) {
  if (v === null || v === undefined) return "—";
  const n = 100 * Number(v);
  const sign = n > 0 ? "+" : "";
  return sign + n.toFixed(0) + "pp";
}

function deltaClass(v) {
  if (v === null || v === undefined || Math.abs(v) < 1e-9) return "delta-zero";
  return v > 0 ? "delta-pos" : "delta-neg";
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderStats() {
  const o = state.overall || {};
  const parts = [
    `<span><strong>${o.n_questions ?? "—"}</strong> shared</span>`,
    `<span class="mode-badge mc">MCQ ${fmtRate(o.mc_avg_success_rate)}</span>`,
    `<span class="mode-badge freeform">FF ${fmtRate(o.ff_avg_success_rate)}</span>`,
    `<span>Δ <strong class="${deltaClass(o.delta)}">${fmtDelta(o.delta)}</strong></span>`,
  ];
  for (const label of state.modelLabels) {
    const mc = (o.mc_by_model || {})[label] || {};
    const ff = (o.ff_by_model || {})[label] || {};
    const d = (mc.avg_shot_success_rate != null && ff.avg_shot_success_rate != null)
      ? Number(mc.avg_shot_success_rate) - Number(ff.avg_shot_success_rate)
      : null;
    parts.push(
      `<span>${escapeHtml(label.split("-")[0])}: ` +
      `<span class="chip mc">${fmtRate(mc.avg_shot_success_rate)}</span> ` +
      `<span class="chip ff">${fmtRate(ff.avg_shot_success_rate)}</span> ` +
      `<span class="${deltaClass(d)}">${fmtDelta(d)}</span></span>`
    );
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
    const chips = state.modelLabels.map(label => {
      const pm = (row.per_model || {})[label] || {};
      return `<span class="chip" title="${escapeHtml(label)}">${label.split("-")[0]} ${fmtDelta(pm.delta)}</span>`;
    }).join("");
    const active = row.id === state.selectedId ? "active" : "";
    return `<li class="${active}" data-id="${escapeHtml(row.id)}">
      <div class="qid">${escapeHtml(row.id)}</div>
      <div class="delta-row">
        <span class="rate ${deltaClass(row.delta)}">${fmtDelta(row.delta)}</span>
        <span class="chip mc">MC ${fmtRate(row.mc_avg_success_rate)}</span>
        <span class="chip ff">FF ${fmtRate(row.ff_avg_success_rate)}</span>
      </div>
      <div class="mini-rates">${chips}</div>
      <p class="qtext">${escapeHtml(row.question || "")}</p>
    </li>`;
  }).join("");
  list.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => selectQuestion(li.dataset.id));
  });
}

function shotJudgeEntry(shot, judgeLabel) {
  const judges = shot.judges || {};
  if (judges[judgeLabel]) return judges[judgeLabel];
  if (shot.grader && String(shot.grader).toLowerCase().endsWith(judgeLabel)) {
    return {
      correct: shot.correct,
      output: shot.grader_output,
      generation: shot.grader_output,
      model_id: shot.grader,
    };
  }
  if (judgeLabel === "string-match" && shot.correct !== null && shot.correct !== undefined) {
    return { correct: shot.correct, output: shot.correct ? "MATCH" : "NO_MATCH", generation: "" };
  }
  return null;
}

function judgeGenerationText(entry) {
  if (!entry) return "";
  if (entry.generation != null && String(entry.generation).length) return String(entry.generation);
  if (entry.output != null && String(entry.output).length) return String(entry.output);
  return "";
}

function judgeVerdictLabel(entry) {
  if (!entry || entry.correct === null || entry.correct === undefined) return "pending";
  if (entry.verdict === "pass" || entry.verdict === "fail") return entry.verdict;
  if (entry.output) return String(entry.output);
  return entry.correct ? "Pass" : "Fail";
}

function renderJudgePills(shot, judges) {
  if (!judges.length) {
    const pending = shot.pending_grade || shot.correct === null || shot.correct === undefined;
    if (pending) return `<span class="chip">pending</span>`;
    return `<span class="${shot.correct ? "pass" : "fail"}">${shot.correct ? "pass" : "fail"}</span>`;
  }
  return `<span class="judge-pills">${judges.map(j => {
    const entry = shotJudgeEntry(shot, j.label);
    if (!entry || entry.correct === null || entry.correct === undefined) {
      return `<span class="judge-pill pending">${escapeHtml(j.label)}?</span>`;
    }
    const ok = !!entry.correct;
    return `<span class="judge-pill ${ok ? "pass" : "fail"}">${escapeHtml(j.label)} ${ok ? "✓" : "✗"}</span>`;
  }).join("")}</span>`;
}

function renderJudgeGenerations(shot, judges) {
  const blocks = [];
  const seen = new Set();
  for (const j of judges || []) {
    const entry = shotJudgeEntry(shot, j.label);
    const gen = judgeGenerationText(entry);
    if (!gen) continue;
    seen.add(j.label);
    blocks.push(`<div class="judge-gen">
      <details class="accordion">
        <summary><span>Judge ${escapeHtml(j.label)} · ${escapeHtml(judgeVerdictLabel(entry))}</span></summary>
        <div class="accordion-body"><pre>${escapeHtml(gen)}</pre></div>
      </details>
    </div>`);
  }
  for (const [label, entry] of Object.entries(shot.judges || {})) {
    if (seen.has(label)) continue;
    const gen = judgeGenerationText(entry);
    if (!gen) continue;
    blocks.push(`<div class="judge-gen">
      <details class="accordion">
        <summary><span>Judge ${escapeHtml(label)} · ${escapeHtml(judgeVerdictLabel(entry))}</span></summary>
        <div class="accordion-body"><pre>${escapeHtml(gen)}</pre></div>
      </details>
    </div>`);
  }
  return blocks.length ? `<div class="judge-gens">${blocks.join("")}</div>` : "";
}

function renderModeColumn(mode, modeLabel, pred, pm, judges, badgeClass) {
  const rate = mode === "mc" ? pm.mc_shot_success_rate : pm.ff_shot_success_rate;
  const nOk = mode === "mc" ? pm.mc_n_shot_correct : pm.ff_n_shot_correct;
  const nShots = mode === "mc" ? pm.mc_n_shots : pm.ff_n_shots;
  if (!pred) {
    return `<div class="mode-col ${badgeClass}">
      <h4><span class="mode-badge ${badgeClass}">${escapeHtml(modeLabel)}</span>
        <span class="muted">missing</span></h4>
    </div>`;
  }
  const shots = pred.shots || [];
  const shotsHtml = shots.map(shot => {
    return `<div class="shot">
      <div class="shot-head">
        <span>shot ${shot.shot_index}</span>
        ${renderJudgePills(shot, judges)}
        <span class="muted">parsed: ${escapeHtml(shot.answer_prediction || "")}</span>
      </div>
      <pre>${escapeHtml(shot.model_output || "")}</pre>
      ${renderJudgeGenerations(shot, judges)}
    </div>`;
  }).join("");
  return `<div class="mode-col ${badgeClass}">
    <h4>
      <span class="mode-badge ${badgeClass}">${escapeHtml(modeLabel)}</span>
      <span class="chip">${fmtRate(rate)} (${nOk ?? "—"}/${nShots ?? "—"})</span>
    </h4>
    ${shotsHtml || "<p class='muted'>No shots stored.</p>"}
  </div>`;
}

function renderPromptAccordion(promptsMc, promptsFf) {
  const sharedMc = (promptsMc && promptsMc.shared) || "";
  const sharedFf = (promptsFf && promptsFf.shared) || "";
  return `<details class="accordion">
    <summary><span>Prompts</span></summary>
    <div class="accordion-body">
      <div class="mode-pair">
        <div>
          <strong class="muted" style="font-size:0.8rem">MCQ shared</strong>
          <pre>${escapeHtml(sharedMc)}</pre>
        </div>
        <div>
          <strong class="muted" style="font-size:0.8rem">Freeform shared</strong>
          <pre>${escapeHtml(sharedFf)}</pre>
        </div>
      </div>
    </div>
  </details>`;
}

async function selectQuestion(id) {
  state.selectedId = id;
  renderList();
  const detail = document.getElementById("detail");
  detail.innerHTML = `<p class="muted">Loading…</p>`;
  const data = await api(
    `/api/question?mc=${encodeURIComponent(state.mcRun)}` +
    `&ff=${encodeURIComponent(state.ffRun)}&id=${encodeURIComponent(id)}`
  );
  const row = data.question;
  const choices = (row.choices || []).map((c, i) =>
    `<div>(${String.fromCharCode(65+i)}) ${escapeHtml(c)}</div>`
  ).join("");
  const audio = data.audio_url
    ? `<audio controls preload="none" src="${data.audio_url}"></audio>`
    : `<p class="muted">Audio not found locally.</p>`;

  let modelsHtml = "";
  for (const label of state.modelLabels) {
    const pm = (row.per_model || {})[label] || {};
    const mcPred = (data.mc_predictions || {})[label];
    const ffPred = (data.ff_predictions || {})[label];
    modelsHtml += `<div class="model-block">
      <h3>${escapeHtml(label)}
        <span class="chip mc">MC ${fmtRate(pm.mc_shot_success_rate)}</span>
        <span class="chip ff">FF ${fmtRate(pm.ff_shot_success_rate)}</span>
        <span class="rate ${deltaClass(pm.delta)}">${fmtDelta(pm.delta)}</span>
      </h3>
      <div class="mode-pair">
        ${renderModeColumn("mc", "MCQ", mcPred, pm, state.mcJudges, "mc")}
        ${renderModeColumn("ff", "Freeform", ffPred, pm, state.ffJudges, "ff")}
      </div>
    </div>`;
  }

  detail.innerHTML = `
    <div class="qid">${escapeHtml(row.id)}</div>
    <h3 style="margin:0.35rem 0 0.15rem">${escapeHtml(row.question || "")}</h3>
    <p class="muted">${escapeHtml(row.modality || "")} · ${escapeHtml(row.category || "")}</p>
    <div class="summary-bar">
      <span class="mode-badge mc">MCQ ${fmtRate(row.mc_avg_success_rate)}</span>
      <span class="mode-badge freeform">FF ${fmtRate(row.ff_avg_success_rate)}</span>
      <span class="delta-big ${deltaClass(row.delta)}">${fmtDelta(row.delta)}</span>
      <span class="muted">MC − freeform</span>
    </div>
    ${audio}
    ${renderPromptAccordion(data.mc_prompts, data.ff_prompts)}
    <div class="choice-box"><strong>Choices</strong>
      <div class="muted" style="font-size:0.78rem;margin:0.2rem 0 0.4rem">Shown in MCQ; gold reference only for freeform.</div>
      ${choices}
    </div>
    <div class="answer-box"><strong>Gold</strong><div>${escapeHtml(row.answer || "")}</div></div>
    ${modelsHtml}
  `;
}

async function loadPair(pairId) {
  const pair = state.pairs.find(p => p.id === pairId);
  if (!pair) return;
  state.pairId = pairId;
  state.mcRun = pair.mc_run;
  state.ffRun = pair.ff_run;
  const data = await api(
    `/api/questions?mc=${encodeURIComponent(state.mcRun)}` +
    `&ff=${encodeURIComponent(state.ffRun)}&sort=${encodeURIComponent(state.sort)}`
  );
  state.questions = data.questions || [];
  state.modelLabels = data.model_labels || [];
  state.overall = data.overall || {};
  state.mcJudges = data.mc_judges || [];
  state.ffJudges = data.ff_judges || [];
  state.mcPrimary = data.mc_primary_judge;
  state.ffPrimary = data.ff_primary_judge;
  document.getElementById("pair-meta").textContent =
    `MCQ ${state.mcRun}  ·  Freeform ${state.ffRun}` +
    (pair.how ? `  ·  paired via ${pair.how}` : "");
  renderStats();
  state.selectedId = state.questions[0]?.id || null;
  renderList();
  if (state.selectedId) await selectQuestion(state.selectedId);
}

async function init() {
  const data = await api("/api/pairs");
  state.pairs = data.pairs || [];
  const sel = document.getElementById("pair");
  if (!state.pairs.length) {
    sel.innerHTML = `<option value="">(no MC+freeform pairs found)</option>`;
    document.getElementById("stats").textContent = "Need one MC and one freeform run under results dir.";
    return;
  }
  sel.innerHTML = state.pairs.map(p => {
    const d = (p.mc_avg != null && p.ff_avg != null)
      ? (Number(p.mc_avg) - Number(p.ff_avg))
      : null;
    return `<option value="${escapeHtml(p.id)}">${escapeHtml(p.mc_run)} ↔ ${escapeHtml(p.ff_run)} (Δ ${fmtDelta(d)})</option>`;
  }).join("");
  sel.addEventListener("change", () => loadPair(sel.value));
  document.getElementById("sort").addEventListener("change", async (e) => {
    state.sort = e.target.value;
    await loadPair(state.pairId);
  });
  document.getElementById("search").addEventListener("input", renderList);
  const prefer = data.default_pair_id || state.pairs[0].id;
  sel.value = prefer;
  await loadPair(prefer);
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
        print(f"[compare] {self.address_string()} {fmt % args}")

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
        _sync_vd_config()

        if path in ("/", "/index.html"):
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/pairs":
            pairs = discover_pairs(Path(CONFIG["results_dir"]))
            default_pair_id = None
            mc_pref = CONFIG.get("mc_run")
            ff_pref = CONFIG.get("ff_run")
            if mc_pref and ff_pref:
                want = f"{mc_pref}__{ff_pref}"
                if any(p["id"] == want for p in pairs):
                    default_pair_id = want
            if default_pair_id is None and pairs:
                default_pair_id = pairs[0]["id"]
            # Strip nested run blobs for a lighter payload (keep summary fields).
            light = []
            for p in pairs:
                light.append(
                    {
                        "id": p["id"],
                        "mc_run": p["mc_run"],
                        "ff_run": p["ff_run"],
                        "how": p["how"],
                        "models": p["models"],
                        "mc_avg": p["mc_avg"],
                        "ff_avg": p["ff_avg"],
                        "n_shots_mc": p["n_shots_mc"],
                        "n_shots_ff": p["n_shots_ff"],
                    }
                )
            self._send_json({"pairs": light, "default_pair_id": default_pair_id})
            return

        if path == "/api/questions":
            mc_run = (qs.get("mc") or [""])[0]
            ff_run = (qs.get("ff") or [""])[0]
            sort = (qs.get("sort") or ["delta_desc"])[0]
            if not mc_run or not ff_run:
                self._send_json({"error": "missing mc or ff"}, 400)
                return
            try:
                bundle = load_pair_bundle(mc_run, ff_run)
            except FileNotFoundError:
                self._send_json({"error": "run not found"}, 404)
                return
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            questions = sort_questions(bundle["questions"], sort)
            self._send_json(
                {
                    "questions": questions,
                    "overall": bundle["overall"],
                    "model_labels": bundle["model_labels"],
                    "mc_run": mc_run,
                    "ff_run": ff_run,
                    "sort": sort,
                    "mc_judges": bundle["mc_bundle"]["judges"],
                    "ff_judges": bundle["ff_bundle"]["judges"],
                    "mc_primary_judge": bundle["mc_bundle"]["primary_judge"],
                    "ff_primary_judge": bundle["ff_bundle"]["primary_judge"],
                    "mc_scores": bundle["mc_bundle"]["scores"],
                    "ff_scores": bundle["ff_bundle"]["scores"],
                }
            )
            return

        if path == "/api/question":
            mc_run = (qs.get("mc") or [""])[0]
            ff_run = (qs.get("ff") or [""])[0]
            qid = (qs.get("id") or [""])[0]
            if not mc_run or not ff_run or not qid:
                self._send_json({"error": "missing mc, ff, or id"}, 400)
                return
            try:
                bundle = load_pair_bundle(mc_run, ff_run)
            except FileNotFoundError:
                self._send_json({"error": "run not found"}, 404)
                return
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            row = bundle["by_id"].get(qid)
            if row is None:
                self._send_json({"error": "question not found"}, 404)
                return
            mc_b = bundle["mc_bundle"]
            ff_b = bundle["ff_bundle"]
            labels = bundle["model_labels"]
            mc_preds = {
                label: mc_b["predictions"].get(label, {}).get(qid)
                for label in labels
            }
            ff_preds = {
                label: ff_b["predictions"].get(label, {}).get(qid)
                for label in labels
            }
            audio = resolve_audio(row.get("audio_path"))
            audio_url = f"/audio/{audio.name}" if audio is not None else None
            mc_sample = next(
                (mc_preds[label] for label in labels if mc_preds.get(label)),
                mc_b["by_id"].get(qid) or row,
            )
            ff_sample = next(
                (ff_preds[label] for label in labels if ff_preds.get(label)),
                ff_b["by_id"].get(qid) or row,
            )
            self._send_json(
                {
                    "question": row,
                    "mc_predictions": mc_preds,
                    "ff_predictions": ff_preds,
                    "audio_url": audio_url,
                    "mc_prompts": build_model_prompts(mc_sample, "mc"),
                    "ff_prompts": build_model_prompts(ff_sample, "freeform"),
                    "mc_judges": mc_b["judges"],
                    "ff_judges": ff_b["judges"],
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Local outputs directory (discovers exp-mmar-question-difficulty/)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Local MMAR wav directory",
    )
    parser.add_argument(
        "--mc-run",
        default=None,
        help="Prefer this MC run id when selecting the default pair",
    )
    parser.add_argument(
        "--ff-run",
        default=None,
        help="Prefer this freeform run id when selecting the default pair",
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
    args = parser.parse_args()

    CONFIG["results_dir"] = args.results_dir.expanduser().resolve()
    CONFIG["mc_run"] = args.mc_run
    CONFIG["ff_run"] = args.ff_run
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
    _sync_vd_config()
    load_pair_bundle.cache_clear()
    load_run_bundle.cache_clear()

    print(f"Results: {CONFIG['results_dir']}")
    for root in resolve_run_roots(CONFIG["results_dir"]):
        print(f"  run root: {root}")
    print(f"Audio:   {CONFIG['audio_dir']}")
    pairs = discover_pairs(CONFIG["results_dir"])
    print(f"Found {len(pairs)} pair(s)")
    for pair in pairs:
        mc_avg = pair.get("mc_avg")
        ff_avg = pair.get("ff_avg")
        delta = None
        if mc_avg is not None and ff_avg is not None:
            delta = float(mc_avg) - float(ff_avg)
        delta_s = f"{100 * delta:+.1f}pp" if delta is not None else "—"
        print(
            f"  {pair['mc_run']} (MC) ↔ {pair['ff_run']} (FF): "
            f"Δ {delta_s} via {pair['how']}"
        )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
