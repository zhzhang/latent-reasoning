"""Local viewer for MMAR answer-uniformity (Gemini) results.

Shows the original question, audio, MCQ choices (reference only), every
model's extracted answer strings, and Gemini's uniformity verdict.

Usage::

    uv run python answer-variety/view_answer_variety.py
    uv run python answer-variety/view_answer_variety.py --port 7863
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
PACKAGE_DIR = Path(__file__).resolve().parent
for path in (str(PACKAGE_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from answer_variety import (  # noqa: E402
    DEFAULT_SOURCE_RUN_ID,
    configure_view_difficulty,
    discover_variety_runs,
    load_variety_bundle,
)
from view_difficulty import (  # noqa: E402
    DEFAULT_AUDIO_DIR,
    DEFAULT_RESULTS_DIR,
    ensure_mmar_audio,
    resolve_audio,
)

CONFIG: dict[str, Any] = {}


def _sync_vd_config() -> None:
    configure_view_difficulty(
        Path(CONFIG["results_dir"]),
        Path(CONFIG["audio_dir"]),
    )


@lru_cache(maxsize=8)
def _load_bundle(run_id: str) -> dict[str, Any]:
    return load_variety_bundle(Path(CONFIG["results_dir"]), run_id)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MMAR Answer Variety</title>
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
  select, input[type="search"], button {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.45rem 0.65rem;
  }
  select, input[type="search"] { min-width: 12rem; }
  button { cursor: pointer; }
  button.active { background: #e2eef6; border-color: #8fb3c9; }
  main {
    max-width: 1400px; margin: 0 auto; padding: 1.25rem;
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
  .qtext {
    margin: 0.35rem 0 0; font-size: 0.86rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .chip {
    font-size: 0.7rem; font-family: "IBM Plex Mono", monospace;
    padding: 0.15rem 0.4rem; border-radius: 999px;
    background: #e8eef2; color: var(--muted);
  }
  .chip.yes { color: var(--good); background: var(--soft-good); }
  .chip.no { color: var(--bad); background: var(--soft-bad); }
  .chip.yn { color: #5a3a12; background: #f3e6cf; }
  .pass { color: var(--good); background: var(--soft-good); padding: 0.1rem 0.4rem; border-radius: 999px; }
  .fail { color: var(--bad); background: var(--soft-bad); padding: 0.1rem 0.4rem; border-radius: 999px; }
  .judge-pills { display: flex; flex-wrap: wrap; gap: 0.25rem; }
  .judge-pills .chip, .judge-pills .pass, .judge-pills .fail {
    font-size: 0.68rem; font-family: "IBM Plex Mono", monospace;
  }
  #detail { padding: 1rem 1.15rem; max-height: calc(100vh - 160px); overflow: auto; }
  .muted { color: var(--muted); }
  .box {
    border: 1px solid var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; margin: 0.75rem 0; background: #fff;
  }
  .box h3 { margin: 0 0 0.45rem; font-size: 0.95rem; }
  .choice-box {
    border: 1px dashed var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; margin: 0.75rem 0; background: #faf7f2;
  }
  .choice-box .note {
    font-size: 0.78rem; color: var(--muted); margin: 0.25rem 0 0.45rem;
  }
  .verdict-hero {
    border-radius: 12px; padding: 1rem 1.1rem; margin: 0.4rem 0 0.9rem;
    border: 1px solid var(--line);
  }
  .verdict-hero.yes { background: var(--soft-good); border-color: #b7dcc8; }
  .verdict-hero.no { background: var(--soft-bad); border-color: #e3b6b6; }
  .verdict-hero.unknown { background: #f2f6f9; }
  .verdict-label {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--muted); margin: 0 0 0.2rem;
  }
  .verdict-value {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.6rem; font-weight: 600; letter-spacing: -0.03em;
    margin: 0;
  }
  .verdict-hero.yes .verdict-value { color: var(--good); }
  .verdict-hero.no .verdict-value { color: var(--bad); }
  table.answers {
    width: 100%; border-collapse: collapse; font-size: 0.86rem;
  }
  table.answers th, table.answers td {
    text-align: left; vertical-align: top;
    padding: 0.45rem 0.5rem; border-bottom: 1px solid var(--line);
  }
  table.answers th {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--muted); font-weight: 600;
  }
  table.answers .model {
    font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; white-space: nowrap;
  }
  audio { width: 100%; margin-top: 0.5rem; }
  pre {
    white-space: pre-wrap; word-break: break-word;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem; margin: 0.25rem 0 0;
    background: #f2f6f9; padding: 0.55rem 0.65rem; border-radius: 8px;
  }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <h1>MMAR Answer Variety</h1>
      <p>Gemini uniformity over extracted freeform answers (all models × all shots)</p>
    </div>
    <div class="controls">
      <label>Run
        <select id="run"></select>
      </label>
      <label>Search
        <input id="search" type="search" placeholder="id / question text" />
      </label>
      <label>&nbsp;
        <button id="filter-varied" type="button">Varied only</button>
      </label>
      <label>&nbsp;
        <button id="filter-uniform" type="button">Uniform only</button>
      </label>
      <label>&nbsp;
        <button id="filter-hide-yn" type="button">Hide yes/no</button>
      </label>
      <label>&nbsp;
        <button id="filter-yn" type="button">Yes/no only</button>
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
  questions: [],
  selectedId: null,
  filterVaried: false,
  filterUniform: false,
  filterHideYn: false,
  filterYn: false,
  scores: {},
  judge: "",
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

function verdictChip(verdict) {
  if (verdict === "Yes") return `<span class="chip yes">Uniform</span>`;
  if (verdict === "No") return `<span class="chip no">Varied</span>`;
  return `<span class="chip">Unparsed</span>`;
}

function yesNoChip(item) {
  if (item && item.binary_implied && !item.yes_no_heuristic) {
    return `<span class="chip yn">Binary</span>`;
  }
  if (item && item.yes_no) return `<span class="chip yn">Yes/No</span>`;
  return "";
}

function shortJudge(label) {
  const text = String(label || "");
  if (text.includes("35b")) return "35B";
  if (text.includes("27b")) return "27B";
  if (/3b/.test(text) && !text.includes("35b") && !text.includes("27b")) return "3B";
  if (text.includes("string-match")) return "match";
  return text.replace(/-instruct$/, "").split("-").slice(-2).join("-") || "judge";
}

function judgePills(ans, judgeList, primaryJudge) {
  const judges = ans.judges || {};
  const labels = (judgeList && judgeList.length)
    ? judgeList.map(j => (typeof j === "string" ? j : j.label)).filter(Boolean)
    : Object.keys(judges);
  if (!labels.length) {
    if (ans.correct === true) return `<span class="pass">pass</span>`;
    if (ans.correct === false) return `<span class="fail">fail</span>`;
    return `<span class="chip">—</span>`;
  }
  return `<div class="judge-pills">${labels.map(label => {
    const entry = judges[label] || {};
    const ok = entry.correct;
    const isPrimary = entry.primary || label === primaryJudge || label === ans.primary_judge;
    const tag = shortJudge(label) + (isPrimary ? "*" : "");
    if (ok === true) return `<span class="pass" title="${escapeHtml(label)}">${escapeHtml(tag)} pass</span>`;
    if (ok === false) return `<span class="fail" title="${escapeHtml(label)}">${escapeHtml(tag)} fail</span>`;
    return `<span class="chip" title="${escapeHtml(label)}">${escapeHtml(tag)} —</span>`;
  }).join("")}</div>`;
}

function fmtRate(v) {
  if (v === null || v === undefined) return "—";
  return (100 * Number(v)).toFixed(1) + "%";
}

async function loadRuns() {
  const data = await api("/api/runs");
  state.runs = data.runs || [];
  const sel = document.getElementById("run");
  sel.innerHTML = state.runs.map(r => {
    const rate = fmtRate(r.uniform_rate);
    return `<option value="${escapeHtml(r.id)}">${escapeHtml(r.id)} · ${rate} uniform</option>`;
  }).join("");
  if (!state.runs.length) {
    document.getElementById("stats").textContent = "No answer-variety runs found.";
    return;
  }
  const preferred = new URLSearchParams(location.search).get("run") || state.runs[0].id;
  sel.value = preferred;
  await loadRun(preferred);
}

async function loadRun(runId) {
  state.runId = runId;
  const data = await api(`/api/run?run_id=${encodeURIComponent(runId)}`);
  state.questions = data.questions || [];
  state.scores = data.scores || {};
  state.judge = data.judge_model_id || "";
  renderList();
  if (state.questions.length) {
    selectQuestion(state.questions[0].id);
  } else {
    document.getElementById("detail").innerHTML = `<p class="muted">No scored questions in this run.</p>`;
  }
}

function filteredQuestions() {
  const q = (document.getElementById("search").value || "").trim().toLowerCase();
  return state.questions.filter(item => {
    if (state.filterVaried && item.verdict !== "No") return false;
    if (state.filterUniform && item.verdict !== "Yes") return false;
    if (state.filterHideYn && item.yes_no) return false;
    if (state.filterYn && !item.yes_no) return false;
    if (!q) return true;
    return (item.id || "").toLowerCase().includes(q)
      || (item.question || "").toLowerCase().includes(q);
  });
}

function renderList() {
  const items = filteredQuestions();
  const s = state.scores || {};
  document.getElementById("stats").innerHTML =
    `<span><strong>${items.length}</strong> shown</span>` +
    `<span>${state.questions.length} total</span>` +
    `<span>${s.n_varied ?? "—"} varied</span>` +
    `<span>${s.n_uniform ?? "—"} uniform (${fmtRate(s.uniform_rate)})</span>` +
    `<span>${s.n_yes_no ?? "—"} yes/no (${fmtRate(s.uniform_rate_yes_no)})</span>` +
    `<span>open ${fmtRate(s.uniform_rate_open)}</span>` +
    (state.judge ? `<span>${escapeHtml(state.judge)}</span>` : "");
  const ul = document.getElementById("qlist");
  ul.innerHTML = items.map(item => {
    return `<li data-id="${escapeHtml(item.id)}" class="${item.id === state.selectedId ? "active" : ""}">
      <div class="qid">${escapeHtml(item.id)} ${verdictChip(item.verdict)} ${yesNoChip(item)}</div>
      <div class="qtext">${escapeHtml(item.question)}</div>
    </li>`;
  }).join("");
  ul.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => selectQuestion(li.dataset.id));
  });
}

async function selectQuestion(id) {
  state.selectedId = id;
  renderList();
  const detail = document.getElementById("detail");
  detail.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const data = await api(`/api/question?run_id=${encodeURIComponent(state.runId)}&id=${encodeURIComponent(id)}`);
    const verdict = data.verdict;
    const heroClass = verdict === "Yes" ? "yes" : verdict === "No" ? "no" : "unknown";
    const heroLabel = verdict === "Yes"
      ? "Uniform — same concept"
      : verdict === "No"
        ? "Varied — not the same concept"
        : "Unparsed verdict";
    const choices = (data.choices || []).map((c, i) =>
      `<div>(${String.fromCharCode(65+i)}) ${escapeHtml(c)}</div>`
    ).join("");
    const byModel = {};
    for (const ans of (data.answers || [])) {
      const label = ans.model || "unknown";
      (byModel[label] ||= []).push(ans);
    }
    const modelOrder = data.model_labels && data.model_labels.length
      ? data.model_labels
      : Object.keys(byModel);
    const judgeList = data.source_judges || [];
    const primaryJudge = data.source_primary_judge || "";
    const rows = [];
    let n = 1;
    for (const label of modelOrder) {
      const shots = (byModel[label] || []).slice().sort((a, b) => (a.shot_index ?? 0) - (b.shot_index ?? 0));
      for (const ans of shots) {
        rows.push(`<tr>
          <td class="muted">${n}</td>
          <td class="model">${escapeHtml(label)}</td>
          <td class="muted">s${ans.shot_index ?? "—"}</td>
          <td>${judgePills(ans, judgeList, primaryJudge)}</td>
          <td>${escapeHtml(ans.answer_prediction || "")}</td>
        </tr>`);
        n += 1;
      }
    }
    detail.innerHTML = `
      <div class="qid">${escapeHtml(data.id)} ${yesNoChip(data)}</div>
      <div class="verdict-hero ${heroClass}">
        <div class="verdict-label">Gemini uniformity</div>
        <p class="verdict-value">${escapeHtml(heroLabel)}</p>
        ${data.reason ? `<pre>${escapeHtml(data.reason)}</pre>` : ""}
      </div>
      <h3 style="margin:0.2rem 0 0.35rem;font-family:Space Grotesk,sans-serif">${escapeHtml(data.question || "")}</h3>
      <p class="muted">${escapeHtml(data.modality || "")} · ${escapeHtml(data.category || "")}</p>
      ${data.audio_url ? `<audio controls preload="none" src="${escapeHtml(data.audio_url)}"></audio>` : `<p class="muted">Audio not found locally.</p>`}
      <div class="choice-box">
        <strong>Choices</strong>
        <div class="note">Not shown to the model in freeform mode (gold answer reference only).${
          data.yes_no_heuristic
            ? " Tagged as a yes/no MCQ."
            : data.binary_implied
              ? " Question text implies a binary answer."
              : ""
        }</div>
        ${choices || `<span class="muted">No choices stored.</span>`}
      </div>
      <div class="box">
        <h3>Gold</h3>
        <div>${escapeHtml(data.answer || "")}</div>
      </div>
      <div class="box">
        <h3>Extracted answer strings</h3>
        <p class="muted" style="margin:0 0 0.4rem;font-size:0.82rem">Numbered as sent to Gemini (no model names in the judge prompt). * = primary freeform judge.</p>
        <table class="answers">
          <thead><tr><th>#</th><th>Model</th><th>Shot</th><th>Judge</th><th>Answer</th></tr></thead>
          <tbody>${rows.join("") || `<tr><td colspan="5" class="muted">No extracted answers.</td></tr>`}</tbody>
        </table>
      </div>
      ${data.raw_response ? `<details class="box"><summary class="muted">Raw Gemini response</summary><pre>${escapeHtml(data.raw_response)}</pre></details>` : ""}
    `;
  } catch (err) {
    detail.innerHTML = `<p class="muted">Failed to load question: ${escapeHtml(String(err))}</p>`;
  }
}

function syncFilterButtons() {
  document.getElementById("filter-varied").classList.toggle("active", state.filterVaried);
  document.getElementById("filter-uniform").classList.toggle("active", state.filterUniform);
  document.getElementById("filter-hide-yn").classList.toggle("active", state.filterHideYn);
  document.getElementById("filter-yn").classList.toggle("active", state.filterYn);
}

document.getElementById("run").addEventListener("change", (e) => loadRun(e.target.value));
document.getElementById("search").addEventListener("input", renderList);
document.getElementById("filter-varied").addEventListener("click", () => {
  state.filterVaried = !state.filterVaried;
  if (state.filterVaried) state.filterUniform = false;
  syncFilterButtons();
  renderList();
});
document.getElementById("filter-uniform").addEventListener("click", () => {
  state.filterUniform = !state.filterUniform;
  if (state.filterUniform) state.filterVaried = false;
  syncFilterButtons();
  renderList();
});
document.getElementById("filter-hide-yn").addEventListener("click", () => {
  state.filterHideYn = !state.filterHideYn;
  if (state.filterHideYn) state.filterYn = false;
  syncFilterButtons();
  renderList();
});
document.getElementById("filter-yn").addEventListener("click", () => {
  state.filterYn = !state.filterYn;
  if (state.filterYn) state.filterHideYn = false;
  syncFilterButtons();
  renderList();
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
        print(f"[view_answer_variety] {self.address_string()} - {fmt % args}")

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

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
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
                self._send_json({"runs": discover_variety_runs(Path(CONFIG["results_dir"]))})
                return
            if path == "/api/run":
                run_id = (qs.get("run_id") or [""])[0]
                if not run_id:
                    self._send_json({"error": "run_id required"}, 400)
                    return
                bundle = _load_bundle(run_id)
                self._send_json(
                    {
                        "run_id": bundle["run_id"],
                        "source_run_id": bundle.get("source_run_id"),
                        "judge_model_id": bundle.get("judge_model_id"),
                        "questions": bundle["questions"],
                        "scores": bundle.get("scores") or {},
                        "manifest": bundle.get("manifest") or {},
                        "model_labels": bundle.get("model_labels") or [],
                        "source_judges": bundle.get("source_judges") or [],
                        "source_primary_judge": bundle.get("source_primary_judge"),
                    }
                )
                return
            if path == "/api/question":
                run_id = (qs.get("run_id") or [""])[0]
                qid = (qs.get("id") or [""])[0]
                if not run_id or not qid:
                    self._send_json({"error": "run_id and id required"}, 400)
                    return
                bundle = _load_bundle(run_id)
                item = bundle["by_id"].get(qid)
                if item is None:
                    self._send_json({"error": "question not found"}, 404)
                    return
                audio = resolve_audio(item.get("audio_path"))
                audio_url = f"/audio/{audio.name}" if audio is not None else None
                self._send_json(
                    {
                        **item,
                        "audio_url": audio_url,
                        "model_labels": bundle.get("model_labels") or [],
                        "source_judges": bundle.get("source_judges")
                        or item.get("source_judges")
                        or [],
                        "source_primary_judge": bundle.get("source_primary_judge")
                        or item.get("source_primary_judge"),
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
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--skip-audio-download", action="store_true")
    args = parser.parse_args()

    CONFIG["results_dir"] = args.results_dir.expanduser().resolve()
    audio_dir = args.audio_dir.expanduser().resolve()
    if not args.skip_audio_download:
        try:
            audio_dir = ensure_mmar_audio(audio_dir)
        except SystemExit as exc:
            print(f"Audio setup failed: {exc}", flush=True)
            print("Continuing without local audio; pass --skip-audio-download to silence.")
    CONFIG["audio_dir"] = audio_dir
    _sync_vd_config()
    _load_bundle.cache_clear()

    runs = discover_variety_runs(CONFIG["results_dir"])
    print(f"Results: {CONFIG['results_dir']}")
    print(f"Audio:   {CONFIG['audio_dir']}")
    print(f"Found {len(runs)} answer-variety run(s)")
    for run in runs:
        rate = run.get("uniform_rate")
        rate_s = f"{100 * rate:.1f}%" if rate is not None else "—"
        print(f"  {run['id']}: {run.get('n_questions') or '—'} q, {rate_s} uniform")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    preferred = args.run_id or (runs[0]["id"] if runs else DEFAULT_SOURCE_RUN_ID)
    print(f"Open {url}/?run={preferred}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
