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
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs" / "exp-mmar-question-difficulty"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "mmar"
DEFAULT_AUDIO_DIR = DEFAULT_DATA_DIR / "audio"

MODEL_LABELS = (
    "af-next-think",
    "step-audio-2-mini",
    "mimo-audio-7b",
    "interactive-omni-8b",
)

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
    return {
        "difficulty": difficulty,
        "by_id": by_id,
        "predictions": predictions,
        "scores": load_json(run_dir / "scores.json") or {},
        "manifest": load_json(run_dir / "manifest.json") or {},
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
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <h1>MMAR Question Difficulty</h1>
      <p>Hardest-first by mean shot success rate across models</p>
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
  "step-audio-2-mini",
  "mimo-audio-7b",
  "interactive-omni-8b",
];
const state = { runs: [], runId: "", questions: [], selectedId: null };

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function fmtRate(v) {
  if (v === null || v === undefined) return "—";
  return (100 * Number(v)).toFixed(0) + "%";
}

function renderStats(scores) {
  const by = scores.by_model || {};
  const parts = [
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

async function selectQuestion(id) {
  state.selectedId = id;
  renderList();
  const detail = document.getElementById("detail");
  detail.innerHTML = `<p class="muted">Loading…</p>`;
  const data = await api(`/api/question?run=${encodeURIComponent(state.runId)}&id=${encodeURIComponent(id)}`);
  const row = data.difficulty;
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
      return `<div class="shot">
        <div class="shot-head">
          <span>shot ${shot.shot_index}</span>
          <span class="${ok ? "pass" : "fail"}">${ok ? "pass" : "fail"}</span>
          <span class="muted">parsed: ${escapeHtml(shot.answer_prediction || "")}</span>
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
  detail.innerHTML = `
    <div class="qid">${escapeHtml(row.id)}</div>
    <h3 style="margin:0.4rem 0 0.2rem">${escapeHtml(row.question || "")}</h3>
    <p class="muted">${escapeHtml(row.modality || "")} · ${escapeHtml(row.category || "")} · avg ${fmtRate(row.avg_success_rate)}</p>
    ${audio}
    <div class="choice-box"><strong>Choices</strong>${choices}</div>
    <div class="answer-box"><strong>Gold</strong><div>${escapeHtml(row.answer || "")}</div></div>
    ${modelsHtml}
  `;
}

async function loadRun(runId) {
  state.runId = runId;
  const data = await api(`/api/questions?run=${encodeURIComponent(runId)}`);
  state.questions = data.questions || [];
  renderStats(data.scores || {});
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
  sel.innerHTML = state.runs.map(r =>
    `<option value="${r.id}">${r.id}${r.has_difficulty ? "" : " (no difficulty yet)"}</option>`
  ).join("");
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
            self._send_json(
                {
                    "difficulty": row,
                    "predictions": preds,
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
