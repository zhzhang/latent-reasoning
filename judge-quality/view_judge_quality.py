"""Browse LALM-no-GT first-shot verdicts against the LLM-GT pack.

Loads ``outputs/judge-quality/llm-judge-gt`` and
``outputs/judge-quality/lalm-judge-no-gt``, scores each audio judge
against the Qwen3.6-with-gold majority vote, and opens a per-question
viewer (audio, gold, test-taker answer, GT vs LALM chips).

Usage::

    uv run modal run judge-quality/download_judge_quality.py
    uv run python judge-quality/view_judge_quality.py
    uv run python judge-quality/view_judge_quality.py --port 7863
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alt_test import DEFAULT_EPSILON
from view_mmar import (
    CONFIG as MMAR_CONFIG,
    DEFAULT_AUDIO_DIR,
    ensure_mmar_audio,
    resolve_audio,
)

from score_lalm_vs_gt import (
    DEFAULT_LOCAL_DIR,
    GT_PACK_NAME,
    LALM_PACK_NAME,
    print_score_table,
    score_experiment,
    scores_only,
    short_judge_name,
)

CONFIG: dict[str, Any] = {}


def _compact_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    out: dict[str, Any] = {}
    for key in (
        "correct",
        "verdict",
        "output",
        "generation",
        "model_id",
        "prompt",
        "include_gold",
        "n_samples",
    ):
        if key in entry:
            out[key] = entry[key]
    samples = entry.get("samples")
    if isinstance(samples, list) and samples:
        out["samples"] = [
            {
                "correct": item.get("correct") if isinstance(item, dict) else None,
                "verdict": item.get("verdict") if isinstance(item, dict) else None,
                "output": item.get("output") if isinstance(item, dict) else None,
                "generation": item.get("generation") if isinstance(item, dict) else None,
            }
            for item in samples
            if isinstance(item, dict)
        ]
    return out


@lru_cache(maxsize=2)
def load_experiment(gt_dir: str, lalm_dir: str, epsilon: float) -> dict[str, Any]:
    payload = score_experiment(Path(gt_dir), Path(lalm_dir), epsilon=epsilon)
    judges = []
    for key, row in (payload.get("judges") or {}).items():
        judges.append(
            {
                "label": key,
                "short": row.get("short") or short_judge_name(key),
                "overall": row.get("overall") or {},
                "by_model": row.get("by_model") or {},
                "alt_test": row.get("alt_test") or {},
                "n_missing": row.get("n_missing") or 0,
            }
        )
    questions = []
    for row in payload.get("questions") or []:
        questions.append(
            {
                "id": row["id"],
                "question": row.get("question") or "",
                "category": row.get("category"),
                "models": row.get("models") or [],
                "n_models": row.get("n_models") or 0,
                "n_scored": row.get("n_scored") or 0,
                "n_gt_pass": row.get("n_gt_pass") or 0,
                "n_disagree": row.get("n_disagree") or 0,
                "n_compared": row.get("n_compared") or 0,
                "agree_rate": row.get("agree_rate"),
                "disagree_by_judge": row.get("disagree_by_judge") or {},
                "agree_by_judge": row.get("agree_by_judge") or {},
            }
        )
    return {
        "scores": scores_only(payload),
        "gt_judge_key": payload["gt_judge_key"],
        "gt_judge_short": short_judge_name(payload["gt_judge_key"]),
        "lalm_judge_keys": payload["lalm_judge_keys"],
        "judges": judges,
        "models": payload["models"],
        "questions": questions,
        "by_id": payload["by_id"],
        "n_questions": payload["n_questions"],
        "n_pairs": payload["n_pairs"],
        "n_gt_missing": payload["n_gt_missing"],
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LALM vs LLM-GT</title>
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
  button { cursor: pointer; }
  main {
    max-width: 1480px; margin: 0 auto; padding: 1.25rem;
    display: grid; grid-template-columns: 400px 1fr; gap: 1rem;
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
  .acc-wrap { padding: 0.4rem 0.55rem 0.7rem; overflow: auto; max-height: 280px; }
  .acc-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
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
  #qlist {
    list-style: none; margin: 0; padding: 0;
    max-height: calc(100vh - 520px); overflow: auto;
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
  .mode-badge {
    display: inline-flex; align-items: center;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.04em; text-transform: uppercase;
    padding: 0.2rem 0.55rem; border-radius: 999px;
    border: 1px solid #d4b88a;
    color: #5a3a12; background: #f3e6cf;
  }
  .brand-row { display: flex; flex-wrap: wrap; gap: 0.55rem; align-items: center; }
  .model-block {
    border: 1px solid var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; margin: 0.75rem 0; background: #fff;
  }
  .model-block h3 {
    margin: 0 0 0.5rem; font-size: 1rem;
    display: flex; gap: 0.6rem; align-items: baseline; flex-wrap: wrap;
  }
  .judge-row {
    display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center;
    margin: 0.35rem 0;
    font-family: "IBM Plex Mono", monospace; font-size: 0.75rem;
  }
  details.accordion {
    border: 1px solid var(--line); border-radius: 10px;
    background: #fff; margin: 0.5rem 0; overflow: hidden;
  }
  details.accordion > summary {
    cursor: pointer; list-style: none;
    padding: 0.55rem 0.75rem;
    font-weight: 600; font-size: 0.85rem;
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
        <h1>LALM vs LLM-GT</h1>
        <span class="mode-badge">neutral_no_gt · first shot</span>
      </div>
      <p>Audio judges scored against Qwen3.6-35B with-gold majority vote</p>
    </div>
    <div class="controls">
      <label>Test-taker
        <select id="model"><option value="">All</option></select>
      </label>
      <label>LALM judge
        <select id="judge"><option value="">All</option></select>
      </label>
      <label>Sort
        <select id="sort">
          <option value="disagree">Most disagreements</option>
          <option value="agree">Lowest agreement</option>
          <option value="gtpass">Fewest GT-correct</option>
          <option value="id">Question id</option>
        </select>
      </label>
      <label>Search
        <input id="search" type="search" placeholder="id or question text" />
      </label>
    </div>
  </div>
</header>
<main>
  <section class="panel">
    <h2>Judges</h2>
    <div id="stats" class="stats"></div>
    <div class="acc-wrap">
      <table class="acc-table" id="acc">
        <thead>
          <tr>
            <th>Judge</th><th>n</th><th>acc</th><th>κ</th><th>F1</th><th>ρ</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
    <h2>Questions</h2>
    <ul id="qlist"></ul>
  </section>
  <section class="panel">
    <h2>Question</h2>
    <div id="detail"><p class="muted">Select a question.</p></div>
  </section>
</main>
<script>
const state = {
  pack: null,
  selectedId: null,
  selectedJudge: "",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

function fmt(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(3);
}

function chip(correct, missingLabel) {
  if (correct === true) return `<span class="pass">correct</span>`;
  if (correct === false) return `<span class="fail">incorrect</span>`;
  return `<span class="pending">${escapeHtml(missingLabel || "missing")}</span>`;
}

function agreeChip(gt, pred) {
  if (gt === null || gt === undefined || pred === null || pred === undefined) {
    return `<span class="pending">no pair</span>`;
  }
  return gt === pred
    ? `<span class="agree">agree</span>`
    : `<span class="disagree">disagree</span>`;
}

async function api(path) {
  const res = await fetch(path);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function filteredQuestions() {
  const pack = state.pack;
  const model = document.getElementById("model").value;
  const judge = document.getElementById("judge").value;
  const sort = document.getElementById("sort").value;
  const q = document.getElementById("search").value.trim().toLowerCase();
  let rows = pack.questions.slice();
  if (model) {
    rows = rows.filter(row => (row.models || []).includes(model));
  }
  if (q) {
    rows = rows.filter(row =>
      String(row.id).toLowerCase().includes(q) ||
      String(row.question || "").toLowerCase().includes(q)
    );
  }
  const disagreeOf = (row) => {
    if (!judge) return row.n_disagree || 0;
    return (row.disagree_by_judge || {})[judge] || 0;
  };
  const comparedOf = (row) => {
    if (!judge) return row.n_compared || 0;
    const d = (row.disagree_by_judge || {})[judge] || 0;
    const a = (row.agree_by_judge || {})[judge] || 0;
    return d + a;
  };
  const agreeRate = (row) => {
    const n = comparedOf(row);
    if (!n) return 1;
    return 1 - (disagreeOf(row) / n);
  };
  rows.sort((a, b) => {
    if (sort === "id") return String(a.id).localeCompare(String(b.id));
    if (sort === "gtpass") return (a.n_gt_pass - b.n_gt_pass) || String(a.id).localeCompare(String(b.id));
    if (sort === "agree") {
      const da = agreeRate(a) - agreeRate(b);
      return da || (disagreeOf(b) - disagreeOf(a));
    }
    return (disagreeOf(b) - disagreeOf(a)) || String(a.id).localeCompare(String(b.id));
  });
  state.filterModel = model;
  state.selectedJudge = judge;
  return rows;
}

function renderAccuracy() {
  const tbody = document.querySelector("#acc tbody");
  const selected = state.selectedJudge;
  tbody.innerHTML = state.pack.judges.map(row => {
    const o = row.overall || {};
    const alt = row.alt_test || {};
    const cls = row.label === selected ? "selected" : "";
    return `<tr class="${cls}" data-judge="${escapeHtml(row.label)}">
      <td class="mono">${escapeHtml(row.short)}</td>
      <td>${o.n ?? 0}</td>
      <td>${fmt(o.accuracy)}</td>
      <td>${fmt(o.kappa)}</td>
      <td>${fmt(o.f1)}</td>
      <td>${fmt(alt.advantage_prob)}</td>
    </tr>`;
  }).join("");
  tbody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", () => {
      const sel = document.getElementById("judge");
      sel.value = sel.value === tr.dataset.judge ? "" : tr.dataset.judge;
      renderList();
      renderAccuracy();
    });
  });
}

function renderList() {
  const rows = filteredQuestions();
  const list = document.getElementById("qlist");
  document.getElementById("stats").innerHTML = [
    `<span><strong>${state.pack.n_questions}</strong> questions</span>`,
    `<span><strong>${state.pack.n_pairs}</strong> GT-labeled shots</span>`,
    `<span><strong>${state.pack.models.length}</strong> test-takers</span>`,
    `<span>showing <strong>${rows.length}</strong></span>`,
  ].join(" · ");
  list.innerHTML = rows.map(row => {
    const judge = state.selectedJudge;
    const disagree = judge
      ? ((row.disagree_by_judge || {})[judge] || 0)
      : (row.n_disagree || 0);
    const compared = judge
      ? disagree + ((row.agree_by_judge || {})[judge] || 0)
      : (row.n_compared || 0);
    const rate = compared ? 1 - (disagree / compared) : null;
    const active = row.id === state.selectedId ? "active" : "";
    return `<li class="${active}" data-id="${escapeHtml(row.id)}">
      <div class="qid">${escapeHtml(row.id)}</div>
      <div class="rate">${fmt(rate)} agree · ${disagree} disagree · ${row.n_gt_pass}/${row.n_models} GT-correct</div>
      <p class="qtext">${escapeHtml(row.question || "")}</p>
    </li>`;
  }).join("");
  list.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => selectQuestion(li.dataset.id));
  });
}

function renderEntry(title, payload) {
  const entry = payload && payload.entry;
  const correct = payload ? payload.correct : null;
  const samples = (entry && entry.samples) || [];
  const sampleHtml = samples.length
    ? `<div class="judge-row">samples ${samples.map(s => chip(s.correct)).join(" ")}</div>`
    : "";
  const gen = entry && (entry.generation || entry.output);
  const body = gen
    ? `<details class="accordion"><summary>Judge text</summary><div class="accordion-body"><pre>${escapeHtml(gen)}</pre></div></details>`
    : "";
  return `<div>
    <div class="judge-row"><strong>${escapeHtml(title)}</strong> ${chip(correct)}</div>
    ${sampleHtml}
    ${body}
  </div>`;
}

async function selectQuestion(id) {
  state.selectedId = id;
  renderList();
  const detail = document.getElementById("detail");
  detail.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const data = await api("/api/question?id=" + encodeURIComponent(id));
    const q = data.question || {};
    const audio = data.audio_url
      ? `<audio controls src="${escapeHtml(data.audio_url)}"></audio>`
      : `<p class="muted">No local wav for this clip.</p>`;
    const modelFilter = document.getElementById("model").value;
    const labels = (data.model_labels || []).filter(label => !modelFilter || label === modelFilter);
    const judgeFilter = state.selectedJudge;
    const blocks = labels.map(label => {
      const model = (q.models || {})[label] || {};
      const gt = model.gt || {};
      const lalms = model.lalms || {};
      const keys = judgeFilter ? [judgeFilter] : (data.lalm_judge_keys || []);
      const lalmHtml = keys.map(key => {
        const row = lalms[key] || {};
        const short = (data.judge_short || {})[key] || key;
        return `<div class="judge-row">
          ${escapeHtml(short)} ${chip(row.correct)} ${agreeChip(gt.correct, row.correct)}
        </div>
        ${row.entry && (row.entry.generation || row.entry.output)
          ? `<details class="accordion"><summary>${escapeHtml(short)} text</summary><div class="accordion-body"><pre>${escapeHtml(row.entry.generation || row.entry.output || "")}</pre></div></details>`
          : ""}`;
      }).join("");
      const pred = model.answer_prediction || "";
      return `<div class="model-block">
        <h3>${escapeHtml(label)}${keys.length === 1 ? " " + agreeChip(gt.correct, (lalms[keys[0]] || {}).correct) : ""}</h3>
        ${renderEntry("LLM-GT " + (data.gt_judge_short || "gt"), gt)}
        <div class="answer-box"><div class="muted">Test-taker answer</div><pre>${escapeHtml(pred)}</pre></div>
        ${lalmHtml || `<p class="muted">No LALM verdicts for this shot.</p>`}
      </div>`;
    }).join("");
    detail.innerHTML = `
      <div class="qid">${escapeHtml(q.id || id)}</div>
      <p>${escapeHtml(q.question || "")}</p>
      ${q.category ? `<p class="muted">${escapeHtml(q.category)}${q.language ? " · " + escapeHtml(q.language) : ""}</p>` : ""}
      <div class="answer-box"><div class="muted">Gold answer (shown to LLM-GT only)</div><pre>${escapeHtml(q.answer || "")}</pre></div>
      ${audio}
      ${blocks || `<p class="banner">No test-taker answers for the current filter.</p>`}
    `;
  } catch (err) {
    detail.innerHTML = `<p class="muted">Failed to load question: ${escapeHtml(String(err))}</p>`;
  }
}

async function boot() {
  const pack = await api("/api/pack");
  state.pack = pack;
  const modelSel = document.getElementById("model");
  modelSel.innerHTML = `<option value="">All</option>` + pack.models.map(m =>
    `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`
  ).join("");
  const judgeSel = document.getElementById("judge");
  judgeSel.innerHTML = `<option value="">All</option>` + pack.judges.map(j =>
    `<option value="${escapeHtml(j.label)}">${escapeHtml(j.short)}</option>`
  ).join("");
  ["model", "judge", "sort"].forEach(id => {
    document.getElementById(id).addEventListener("change", () => {
      renderAccuracy();
      renderList();
    });
  });
  document.getElementById("search").addEventListener("input", renderList);
  renderAccuracy();
  renderList();
  if (pack.questions && pack.questions.length) {
    const first = filteredQuestions()[0];
    if (first) selectQuestion(first.id);
  }
}

boot().catch(err => {
  document.getElementById("detail").innerHTML =
    `<p class="banner">${escapeHtml(String(err))}. Download packs first:<br>
     <code>uv run modal run judge-quality/download_judge_quality.py</code></p>`;
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[view_judge_quality] {self.address_string()} {fmt % args}")

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
        return load_experiment(
            str(CONFIG["gt_dir"]),
            str(CONFIG["lalm_dir"]),
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
                        "judges": bundle["judges"],
                        "models": bundle["models"],
                        "gt_judge_key": bundle["gt_judge_key"],
                        "gt_judge_short": bundle["gt_judge_short"],
                        "lalm_judge_keys": bundle["lalm_judge_keys"],
                        "n_questions": bundle["n_questions"],
                        "n_pairs": bundle["n_pairs"],
                        "n_gt_missing": bundle["n_gt_missing"],
                        "scores": {
                            key: value
                            for key, value in (bundle.get("scores") or {}).items()
                            if key != "questions"
                        },
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
                models = {}
                for label, model in (row.get("models") or {}).items():
                    gt = model.get("gt") or {}
                    lalms = {}
                    for key, item in (model.get("lalms") or {}).items():
                        lalms[key] = {
                            "correct": item.get("correct"),
                            "entry": _compact_entry(item.get("entry")),
                        }
                    models[label] = {
                        "model_label": label,
                        "answer_prediction": model.get("answer_prediction") or "",
                        "thinking_prediction": model.get("thinking_prediction"),
                        "gt": {
                            "key": gt.get("key"),
                            "correct": gt.get("correct"),
                            "entry": _compact_entry(gt.get("entry")),
                        },
                        "lalms": lalms,
                    }
                audio = resolve_audio(row.get("audio_path"))
                audio_url = f"/audio/{audio.name}" if audio is not None else None
                self._send_json(
                    {
                        "question": {
                            "id": row.get("id") or qid,
                            "question": row.get("question"),
                            "answer": row.get("answer"),
                            "category": row.get("category"),
                            "language": row.get("language"),
                            "audio_path": row.get("audio_path"),
                            "models": models,
                        },
                        "audio_url": audio_url,
                        "model_labels": [
                            label
                            for label in bundle["models"]
                            if label in models
                        ],
                        "lalm_judge_keys": bundle["lalm_judge_keys"],
                        "gt_judge_key": bundle["gt_judge_key"],
                        "gt_judge_short": bundle["gt_judge_short"],
                        "judge_short": {
                            item["label"]: item["short"] for item in bundle["judges"]
                        },
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
        except SystemExit as exc:
            self._send_json({"error": str(exc)}, 500)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR / GT_PACK_NAME,
        help="Downloaded llm-judge-gt pack",
    )
    parser.add_argument(
        "--lalm-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR / LALM_PACK_NAME,
        help="Downloaded lalm-judge-no-gt pack",
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
        help="Do not download MMAR wavs if the local cache is incomplete",
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
        help="Alt-Test ε for winning rate ω",
    )
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    gt_dir = args.gt_dir.expanduser().resolve()
    lalm_dir = args.lalm_dir.expanduser().resolve()
    CONFIG["gt_dir"] = gt_dir
    CONFIG["lalm_dir"] = lalm_dir
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
    load_experiment.cache_clear()

    print(f"GT pack:   {gt_dir}")
    print(f"LALM pack: {lalm_dir}")
    print(f"Audio:     {audio_dir}")
    if not gt_dir.is_dir() or not lalm_dir.is_dir():
        print(
            "Pack directory missing. Run:\n"
            "  uv run modal run judge-quality/download_judge_quality.py"
        )
    else:
        bundle = load_experiment(str(gt_dir), str(lalm_dir), float(args.epsilon))
        print_score_table(bundle["scores"] | {"gt_judge_key": bundle["gt_judge_key"]})
        print(
            f"Loaded {bundle['n_questions']} questions, "
            f"{bundle['n_pairs']} GT-labeled shots, "
            f"{len(bundle['judges'])} LALM judges, "
            f"{len(bundle['models'])} models"
        )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
