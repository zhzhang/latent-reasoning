"""Local viewer for cascade audio→caption generations.

Reads ``outputs/mmar-descriptions`` (or ``--pack-dir``): audio, MMAR
question/gold for context, and every model's caption shots.

Usage::

    uv run python cascade/view_descriptions.py
    uv run python cascade/view_descriptions.py --port 7864
    uv run python cascade/view_descriptions.py \\
      --pack-dir ./outputs/mmar-descriptions
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aggregate import is_dropped_model  # noqa: E402
from mmar_common import build_mmar_description_prompt, load_jsonl  # noqa: E402
from view_mmar import ensure_mmar_audio  # noqa: E402

DEFAULT_PACK_DIR = REPO_ROOT / "outputs" / "mmar-descriptions"
DEFAULT_AUDIO_DIR = REPO_ROOT / "data" / "mmar" / "audio"

CONFIG: dict[str, Any] = {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def discover_model_labels(pack_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Models with predictions on disk, preferring manifest order."""
    models_root = pack_dir / "models"
    on_disk: list[str] = []
    if models_root.is_dir():
        on_disk = [
            child.name
            for child in sorted(models_root.iterdir())
            if child.is_dir() and (child / "predictions.jsonl").is_file()
            and not is_dropped_model(child.name)
        ]
    preferred = [str(x) for x in (manifest.get("models") or [])]
    ordered = [label for label in preferred if label in on_disk]
    for label in on_disk:
        if label not in ordered:
            ordered.append(label)
    return ordered


def resolve_audio(audio_path: str | None) -> Path | None:
    if not audio_path:
        return None
    path = Path(audio_path)
    if path.is_file():
        return path
    name = path.name
    for candidate in (
        Path(CONFIG["audio_dir"]) / name,
        REPO_ROOT / "data" / "mmar" / "audio" / name,
    ):
        if candidate.is_file():
            return candidate
    return None


def _shot_caption(shot: dict[str, Any]) -> str:
    text = shot.get("answer_prediction")
    if text is None or str(text).strip() == "":
        text = shot.get("model_output")
    return str(text or "").strip()


def _compact_shot(shot: dict[str, Any]) -> dict[str, Any]:
    try:
        shot_index = int(shot.get("shot_index", 0))
    except (TypeError, ValueError):
        shot_index = 0
    return {
        "shot_index": shot_index,
        "caption": _shot_caption(shot),
        "thinking_prediction": shot.get("thinking_prediction") or "",
        "model_output": shot.get("model_output") or "",
    }


@lru_cache(maxsize=2)
def load_pack(pack_dir_s: str) -> dict[str, Any]:
    pack_dir = Path(pack_dir_s)
    if not pack_dir.is_dir():
        raise FileNotFoundError(pack_dir)

    manifest = _load_json(pack_dir / "manifest.json")
    ids_payload = _load_json(pack_dir / "question_ids.json")
    model_labels = discover_model_labels(pack_dir, manifest)

    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for label in model_labels:
        rows = load_jsonl(pack_dir / "models" / label / "predictions.jsonl")
        label_map: dict[str, dict[str, Any]] = {}
        for record in rows:
            qid = str(record.get("id") or "")
            if not qid:
                continue
            shots = [_compact_shot(shot) for shot in (record.get("shots") or [])]
            shots.sort(key=lambda s: int(s.get("shot_index") or 0))
            label_map[qid] = {
                "id": qid,
                "n_shots": len(shots),
                "shots": shots,
            }
            if qid not in by_id:
                by_id[qid] = {
                    "id": qid,
                    "question": record.get("question") or "",
                    "answer": record.get("answer") or "",
                    "category": record.get("category") or "",
                    "modality": record.get("modality") or "",
                    "audio_path": record.get("audio_path") or "",
                    "choices": list(record.get("choices") or []),
                }
        predictions[label] = label_map

    ordered_ids = [str(x) for x in (ids_payload.get("ids") or [])]
    if not ordered_ids:
        ordered_ids = list(by_id.keys())
    # Keep pack order, then any extras found only in predictions.
    seen = set(ordered_ids)
    for qid in by_id:
        if qid not in seen:
            ordered_ids.append(qid)
            seen.add(qid)

    questions: list[dict[str, Any]] = []
    modalities: set[str] = set()
    categories: set[str] = set()
    for qid in ordered_ids:
        row = by_id.get(qid)
        if row is None:
            continue
        n_models = sum(1 for label in model_labels if qid in predictions.get(label, {}))
        n_captions = sum(
            len((predictions.get(label) or {}).get(qid, {}).get("shots") or [])
            for label in model_labels
        )
        questions.append(
            {
                "id": qid,
                "question": row.get("question") or "",
                "answer": row.get("answer") or "",
                "category": row.get("category") or "",
                "modality": row.get("modality") or "",
                "n_models": n_models,
                "n_captions": n_captions,
                "complete": n_models == len(model_labels) and len(model_labels) > 0,
            }
        )
        if row.get("modality"):
            modalities.add(str(row["modality"]))
        if row.get("category"):
            categories.add(str(row["category"]))

    coverage = []
    for label in model_labels:
        n_have = len(predictions.get(label) or {})
        n_shots = sum(
            len(rec.get("shots") or [])
            for rec in (predictions.get(label) or {}).values()
        )
        coverage.append(
            {
                "model": label,
                "n_questions": n_have,
                "n_captions": n_shots,
                "complete": n_have >= len(questions) and len(questions) > 0,
            }
        )

    n_shots_manifest = int(manifest.get("n_shots") or 0)
    max_shots = max(
        (len(rec.get("shots") or []) for preds in predictions.values() for rec in preds.values()),
        default=0,
    )

    return {
        "pack_dir": str(pack_dir),
        "manifest": manifest,
        "model_labels": model_labels,
        "predictions": predictions,
        "by_id": by_id,
        "questions": questions,
        "modalities": sorted(modalities),
        "categories": sorted(categories),
        "coverage": coverage,
        "n_questions": len(questions),
        "n_complete": sum(1 for q in questions if q["complete"]),
        "n_shots": n_shots_manifest or max_shots,
        "description_prompt": build_mmar_description_prompt(),
        "enable_thinking": bool(manifest.get("enable_thinking")),
        "experiment": manifest.get("experiment") or "mmar-descriptions",
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Cascade Descriptions</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg: #e8efe8;
    --ink: #1a2420;
    --muted: #5a6b62;
    --line: #b7c9bc;
    --card: #f6faf6;
    --accent: #2a6b4f;
    --accent-soft: #d5ebe0;
    --shadow: 0 1px 0 rgba(26,36,32,0.04), 0 10px 28px rgba(26,36,32,0.07);
    --radius: 12px;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    font-family: "IBM Plex Sans", system-ui, sans-serif;
    color: var(--ink);
    background:
      linear-gradient(155deg, #dce8df 0%, transparent 40%),
      linear-gradient(340deg, #c9d9ce 0%, transparent 35%),
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
    max-width: 1500px; margin: 0 auto;
    display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem;
    align-items: baseline; justify-content: space-between;
  }
  h1 {
    margin: 0; font-family: "Space Grotesk", sans-serif;
    font-size: 1.35rem; font-weight: 600; letter-spacing: -0.02em;
  }
  #run-meta, #stats { color: var(--muted); font-size: 0.88rem; }
  .layout {
    max-width: 1500px; margin: 0 auto; padding: 1rem 1.25rem 2.5rem;
    display: grid; grid-template-columns: minmax(280px, 360px) 1fr; gap: 1rem;
  }
  @media (max-width: 960px) {
    .layout { grid-template-columns: 1fr; }
  }
  .panel {
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--radius); box-shadow: var(--shadow);
  }
  .sidebar { display: flex; flex-direction: column; max-height: calc(100vh - 7rem);
    position: sticky; top: 5.2rem; overflow: hidden; }
  @media (max-width: 960px) {
    .sidebar { position: static; max-height: 50vh; }
  }
  .filters {
    padding: 0.85rem; border-bottom: 1px solid var(--line);
    display: flex; flex-direction: column; gap: 0.45rem;
  }
  .filter-row {
    display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center;
  }
  input[type="search"], select {
    font: inherit; color: var(--ink); background: #fff;
    border: 1px solid var(--line); border-radius: 8px; padding: 0.4rem 0.55rem;
  }
  input[type="search"] { flex: 1; min-width: 0; }
  select { min-width: 0; }
  button.chip {
    font: inherit; font-size: 0.8rem; cursor: pointer;
    border: 1px solid var(--line); background: #fff; color: var(--ink);
    border-radius: 999px; padding: 0.28rem 0.65rem;
  }
  button.chip.active {
    background: var(--accent); color: #fff; border-color: var(--accent);
  }
  #qlist {
    list-style: none; margin: 0; padding: 0; overflow: auto; flex: 1;
  }
  #qlist li {
    padding: 0.7rem 0.85rem; border-bottom: 1px solid var(--line);
    cursor: pointer;
  }
  #qlist li:hover { background: color-mix(in srgb, var(--accent-soft) 55%, transparent); }
  #qlist li.active { background: var(--accent-soft); }
  .qid {
    font-family: "IBM Plex Mono", monospace; font-size: 0.72rem;
    color: var(--muted); word-break: break-all;
  }
  .qtext {
    margin-top: 0.2rem; font-size: 0.9rem; line-height: 1.35;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .badge {
    display: inline-block; font-size: 0.72rem; font-weight: 500;
    padding: 0.1rem 0.4rem; border-radius: 6px; margin-left: 0.25rem;
    background: var(--accent-soft); color: var(--accent);
  }
  .badge.warn { background: #f0e6d4; color: #7a5520; }
  #detail { padding: 1.1rem 1.25rem; min-height: 60vh; }
  #detail h2 {
    margin: 0 0 0.35rem; font-family: "Space Grotesk", sans-serif;
    font-size: 1.2rem; font-weight: 600; letter-spacing: -0.02em;
  }
  .muted { color: var(--muted); font-size: 0.88rem; }
  audio { width: 100%; margin: 0.65rem 0 0.85rem; }
  .box {
    background: #fff; border: 1px solid var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; margin: 0.65rem 0;
  }
  .box h3 {
    margin: 0 0 0.4rem; font-size: 0.82rem; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--muted); font-weight: 600;
  }
  .gold { font-size: 1.02rem; font-weight: 500; }
  .model-block { margin: 1rem 0 0; }
  .model-head {
    display: flex; flex-wrap: wrap; gap: 0.4rem 0.85rem; align-items: baseline;
    margin-bottom: 0.45rem;
  }
  .model-head .name {
    font-family: "Space Grotesk", sans-serif; font-weight: 600; font-size: 1.02rem;
  }
  .shots { display: grid; gap: 0.5rem; }
  .shot {
    background: #fff; border: 1px solid var(--line); border-radius: 10px;
    padding: 0.65rem 0.8rem;
  }
  .shot-label {
    font-family: "IBM Plex Mono", monospace; font-size: 0.75rem;
    color: var(--muted); margin-bottom: 0.3rem;
  }
  .caption {
    white-space: pre-wrap; line-height: 1.45; font-size: 0.92rem;
  }
  .caption.empty { color: var(--muted); font-style: italic; }
  details.raw summary {
    cursor: pointer; color: var(--muted); font-size: 0.8rem; margin-top: 0.35rem;
  }
  details.raw pre {
    white-space: pre-wrap; font-family: "IBM Plex Mono", monospace;
    font-size: 0.75rem; background: #f0f4f1; padding: 0.5rem; border-radius: 6px;
    overflow: auto; max-height: 240px;
  }
  .prompt-box {
    font-size: 0.88rem; line-height: 1.4; color: var(--muted);
    white-space: pre-wrap;
  }
  .coverage {
    display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.35rem;
  }
  .coverage span {
    font-size: 0.75rem; padding: 0.15rem 0.45rem; border-radius: 6px;
    background: #fff; border: 1px solid var(--line);
  }
  .coverage span.ok { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div>
      <h1>Cascade Descriptions</h1>
      <div id="run-meta">Loading…</div>
    </div>
    <div id="stats"></div>
  </div>
</header>
<div class="layout">
  <aside class="panel sidebar">
    <div class="filters">
      <div class="filter-row">
        <input id="search" type="search" placeholder="Search id / question / gold…" />
      </div>
      <div class="filter-row">
        <select id="category"><option value="">All categories</option></select>
        <select id="modality"><option value="">All modalities</option></select>
      </div>
      <div class="filter-row">
        <button type="button" class="chip" id="filter-incomplete">Incomplete</button>
        <select id="model-filter"><option value="">All models</option></select>
      </div>
      <div class="coverage" id="coverage"></div>
    </div>
    <ul id="qlist"></ul>
  </aside>
  <main class="panel" id="detail">
    <p class="muted">Select a question.</p>
  </main>
</div>
<script>
const state = {
  questions: [],
  modelLabels: [],
  modalities: [],
  categories: [],
  coverage: [],
  nShots: 0,
  descriptionPrompt: "",
  enableThinking: false,
  experiment: "",
  nComplete: 0,
  selectedId: null,
  filterIncomplete: false,
  modelFilter: "",
};

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  })[c]);
}

async function api(path) {
  const res = await fetch(path);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function fillSelect(el, values, allLabel) {
  const cur = el.value;
  el.innerHTML = `<option value="">${escapeHtml(allLabel)}</option>` +
    values.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
  if (values.includes(cur)) el.value = cur;
}

function filteredQuestions() {
  const q = (document.getElementById("search").value || "").trim().toLowerCase();
  const cat = document.getElementById("category").value;
  const mod = document.getElementById("modality").value;
  const modelFilter = state.modelFilter;
  return state.questions.filter(row => {
    if (state.filterIncomplete && row.complete) return false;
    if (cat && row.category !== cat) return false;
    if (mod && row.modality !== mod) return false;
    if (modelFilter && (row.n_models || 0) < 1) return false;
    if (!q) return true;
    const hay = `${row.id} ${row.question} ${row.answer} ${row.category} ${row.modality}`.toLowerCase();
    return hay.includes(q);
  });
}

function renderCoverage() {
  const el = document.getElementById("coverage");
  el.innerHTML = (state.coverage || []).map(row => {
    const cls = row.complete ? "ok" : "";
    return `<span class="${cls}">${escapeHtml(row.model)} · ${row.n_questions}q / ${row.n_captions} caps</span>`;
  }).join("");
}

function renderList() {
  const items = filteredQuestions();
  document.getElementById("stats").textContent =
    `${items.length} shown · ${state.questions.length} total · ${state.nComplete || 0} complete`;
  const ul = document.getElementById("qlist");
  ul.innerHTML = items.map(item => {
    const badge = item.complete
      ? `<span class="badge">${item.n_models} models</span>`
      : `<span class="badge warn">${item.n_models}/${state.modelLabels.length}</span>`;
    return `<li data-id="${escapeHtml(item.id)}" class="${item.id === state.selectedId ? "active" : ""}">
      <div class="qid">${escapeHtml(item.id)} ${badge}</div>
      <div class="qtext">${escapeHtml(item.question || "(no question text)")}</div>
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
    const data = await api(`/api/question?id=${encodeURIComponent(id)}`);
    const q = data.question || {};
    const labels = data.model_labels || state.modelLabels;
    const filter = state.modelFilter;
    const modelBlocks = labels
      .filter(label => !filter || label === filter)
      .map(label => {
        const pred = (data.predictions || {})[label];
        const shots = (pred && pred.shots) || [];
        if (!shots.length) {
          return `<div class="model-block">
            <div class="model-head"><span class="name">${escapeHtml(label)}</span>
            <span class="muted">no captions</span></div>
          </div>`;
        }
        const shotHtml = shots.map(shot => {
          const cap = shot.caption || "";
          const empty = !cap ? " empty" : "";
          const body = cap ? escapeHtml(cap) : "(empty caption)";
          const thinking = shot.thinking_prediction
            ? `<details class="raw"><summary>thinking_prediction</summary><pre>${escapeHtml(shot.thinking_prediction)}</pre></details>`
            : "";
          const raw = shot.model_output && shot.model_output !== cap
            ? `<details class="raw"><summary>raw model_output</summary><pre>${escapeHtml(shot.model_output)}</pre></details>`
            : "";
          return `<div class="shot">
            <div class="shot-label">shot ${shot.shot_index}</div>
            <div class="caption${empty}">${body}</div>
            ${thinking}${raw}
          </div>`;
        }).join("");
        return `<div class="model-block">
          <div class="model-head">
            <span class="name">${escapeHtml(label)}</span>
            <span class="muted">${shots.length} caption${shots.length === 1 ? "" : "s"}</span>
          </div>
          <div class="shots">${shotHtml}</div>
        </div>`;
      }).join("");

    detail.innerHTML = `
      <div class="qid">${escapeHtml(q.id || id)}</div>
      <h2>${escapeHtml(q.question || "(no question text)")}</h2>
      <p class="muted">${escapeHtml(q.modality || "—")} · ${escapeHtml(q.category || "—")}</p>
      ${data.audio_url
        ? `<audio controls preload="metadata" src="${escapeHtml(data.audio_url)}"></audio>`
        : `<p class="muted">Audio not found locally.</p>`}
      <div class="box">
        <h3>Gold answer</h3>
        <div class="gold">${escapeHtml(q.answer || "—")}</div>
      </div>
      <div class="box">
        <h3>Caption prompt</h3>
        <div class="prompt-box">${escapeHtml(data.description_prompt || state.descriptionPrompt)}</div>
        <p class="muted" style="margin:0.45rem 0 0">thinking ${
          data.enable_thinking ? "on" : "off"
        } · PREFIX + question + SUFFIX (no choices)</p>
      </div>
      ${modelBlocks || `<p class="muted">No model captions for this question.</p>`}
    `;
  } catch (err) {
    detail.innerHTML = `<p class="muted">Failed to load: ${escapeHtml(String(err))}</p>`;
  }
}

async function init() {
  const data = await api("/api/pack");
  state.questions = data.questions || [];
  state.modelLabels = data.model_labels || [];
  state.modalities = data.modalities || [];
  state.categories = data.categories || [];
  state.coverage = data.coverage || [];
  state.nShots = data.n_shots || 0;
  state.nComplete = data.n_complete || 0;
  state.descriptionPrompt = data.description_prompt || "";
  state.enableThinking = !!data.enable_thinking;
  state.experiment = data.experiment || "";

  const bits = [];
  if (state.experiment) bits.push(state.experiment);
  bits.push(`${state.nShots} shots`);
  bits.push(`${state.modelLabels.length} models`);
  bits.push(`${state.questions.length} questions`);
  bits.push(state.enableThinking ? "thinking on" : "thinking off");
  document.getElementById("run-meta").textContent = bits.join(" · ");

  fillSelect(document.getElementById("category"), state.categories, "All categories");
  fillSelect(document.getElementById("modality"), state.modalities, "All modalities");
  fillSelect(document.getElementById("model-filter"), state.modelLabels, "All models");
  renderCoverage();

  document.getElementById("search").addEventListener("input", renderList);
  document.getElementById("category").addEventListener("change", renderList);
  document.getElementById("modality").addEventListener("change", renderList);
  document.getElementById("model-filter").addEventListener("change", (e) => {
    state.modelFilter = e.target.value || "";
    renderList();
    if (state.selectedId) selectQuestion(state.selectedId);
  });
  document.getElementById("filter-incomplete").addEventListener("click", () => {
    state.filterIncomplete = !state.filterIncomplete;
    document.getElementById("filter-incomplete").classList.toggle("active", state.filterIncomplete);
    renderList();
  });

  if (!state.questions.length) {
    document.getElementById("stats").textContent = "No questions in this pack.";
    document.getElementById("detail").innerHTML =
      `<p class="muted">Pack is empty. Download with:<br>
       <code>uv run modal run download_results.py --volume-name mmar-descriptions</code></p>`;
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
  document.getElementById("run-meta").textContent = String(err);
});
</script>
</body>
</html>
"""


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
                bundle = load_pack(str(CONFIG["pack_dir"]))
                self._send_json(
                    {
                        "questions": bundle["questions"],
                        "manifest": bundle["manifest"],
                        "model_labels": bundle["model_labels"],
                        "modalities": bundle["modalities"],
                        "categories": bundle["categories"],
                        "coverage": bundle["coverage"],
                        "n_shots": bundle["n_shots"],
                        "n_complete": bundle["n_complete"],
                        "n_questions": bundle["n_questions"],
                        "description_prompt": bundle["description_prompt"],
                        "enable_thinking": bundle["enable_thinking"],
                        "experiment": bundle["experiment"],
                    }
                )
                return

            if path == "/api/question":
                qid = (qs.get("id") or [""])[0]
                if not qid:
                    self._send_json({"error": "missing id"}, 400)
                    return
                bundle = load_pack(str(CONFIG["pack_dir"]))
                row = bundle["by_id"].get(qid)
                if row is None:
                    self._send_json({"error": "question not found"}, 404)
                    return
                preds = {
                    label: bundle["predictions"].get(label, {}).get(qid)
                    for label in bundle["model_labels"]
                }
                audio = resolve_audio(row.get("audio_path"))
                audio_url = f"/audio/{audio.name}" if audio is not None else None
                self._send_json(
                    {
                        "question": row,
                        "predictions": preds,
                        "audio_url": audio_url,
                        "model_labels": bundle["model_labels"],
                        "n_shots": bundle["n_shots"],
                        "description_prompt": bundle["description_prompt"],
                        "enable_thinking": bundle["enable_thinking"],
                    }
                )
                return

            if path.startswith("/audio/"):
                name = unquote(path[len("/audio/") :])
                # Prevent path traversal.
                name = Path(name).name
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
    parser.add_argument("--port", type=int, default=7864)
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
    load_pack.cache_clear()

    print(f"Pack:  {pack_dir}")
    print(f"Audio: {audio_dir}")
    if not pack_dir.is_dir():
        print("Pack directory not found. Run:")
        print("  uv run modal run download_results.py --volume-name mmar-descriptions")
    else:
        bundle = load_pack(str(pack_dir))
        print(
            f"Loaded {bundle['n_questions']} questions, "
            f"{len(bundle['model_labels'])} models, "
            f"{bundle['n_shots']} shots "
            f"(thinking={'on' if bundle['enable_thinking'] else 'off'})"
        )
        for row in bundle["coverage"]:
            status = "complete" if row["complete"] else "partial"
            print(
                f"  {row['model']}: {row['n_questions']}q / "
                f"{row['n_captions']} captions ({status})"
            )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
