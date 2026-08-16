"""Local viewer for M_JudgeBench preference examples.

Browse multimodal preference pairs from
``data/m_judgebench/m_judgebench_data.jsonl`` (czythu/M_Judger): question +
image, gold answer, chosen vs rejected responses, and error-type metadata.

Usage:

    uv run python view_m_judgebench.py
    uv run python view_m_judgebench.py --port 7862
    uv run python view_m_judgebench.py --data-dir ./data/m_judgebench
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "m_judgebench"
DEFAULT_JSONL = "m_judgebench_data.jsonl"

CONFIG: dict[str, Any] = {}
CACHE: dict[str, Any] = {}


def load_dataset(data_dir: Path, jsonl_name: str = DEFAULT_JSONL) -> dict[str, Any]:
    path = data_dir / jsonl_name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download from "
            "https://github.com/czythu/M_Judger/blob/main/data/m_judgebench_data.jsonl"
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)

    benchmarks: set[str] = set()
    types: set[str] = set()
    fields: set[str] = set()
    by_idx: dict[int, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for row in rows:
        idx = int(row["idx"])
        by_idx[idx] = row
        benchmarks.add(str(row.get("benchmark_name") or ""))
        types.add(str(row.get("type") or ""))
        fields.add(str(row.get("field") or ""))
        summaries.append(
            {
                "idx": idx,
                "question": row.get("question") or "",
                "answer": row.get("answer") or "",
                "chosen_answer": row.get("chosen_answer") or "",
                "rejected_answer": row.get("rejected_answer") or "",
                "field": row.get("field") or "",
                "source": row.get("source") or "",
                "benchmark_name": row.get("benchmark_name") or "",
                "type": row.get("type") or "",
                "has_image": bool(row.get("image")),
            }
        )

    return {
        "path": str(path),
        "n": len(rows),
        "summaries": summaries,
        "by_idx": by_idx,
        "benchmarks": sorted(b for b in benchmarks if b),
        "types": sorted(t for t in types if t),
        "fields": sorted(f for f in fields if f),
    }


def resolve_image(rel: str | None) -> Path | None:
    if not rel:
        return None
    rel_path = Path(rel)
    data_dir = Path(CONFIG["data_dir"])
    candidates = [
        data_dir / rel_path,
        data_dir / "images" / rel_path.name,
        data_dir / rel_path.name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>M_JudgeBench Viewer</title>
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
    max-width: 1500px; margin: 0 auto;
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
    border-radius: 8px; padding: 0.45rem 0.65rem; min-width: 11rem;
  }
  main {
    max-width: 1500px; margin: 0 auto; padding: 1.25rem;
    display: grid; grid-template-columns: 380px 1fr; gap: 1rem;
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
  .chip {
    font-size: 0.7rem; font-family: "IBM Plex Mono", monospace;
    padding: 0.15rem 0.4rem; border-radius: 999px;
    background: #e8eef2; color: var(--muted);
  }
  .chip.chosen { color: var(--good); background: var(--soft-good); }
  .chip.rejected { color: var(--bad); background: var(--soft-bad); }
  .chip.type { color: #5a3a12; background: #f3e6cf; }
  .mini-chips {
    display: flex; flex-wrap: wrap; gap: 0.35rem;
    margin-top: 0.35rem;
  }
  .qtext {
    margin: 0.35rem 0 0; font-size: 0.86rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  #detail { padding: 1rem 1.15rem; max-height: calc(100vh - 160px); overflow: auto; }
  .muted { color: var(--muted); }
  .meta-row {
    display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center;
    margin: 0.45rem 0 0.75rem;
  }
  .example-image {
    display: block; max-width: 100%; max-height: 420px;
    margin: 0.75rem 0; border: 1px solid var(--line); border-radius: 10px;
    background: #fff; object-fit: contain;
  }
  .answer-box, .pair-col, .explain-box {
    border: 1px solid var(--line); border-radius: 10px;
    padding: 0.75rem 0.9rem; margin: 0.75rem 0; background: #fff;
  }
  .pair-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;
    margin: 0.75rem 0;
  }
  @media (max-width: 900px) {
    .pair-grid { grid-template-columns: 1fr; }
  }
  .pair-col.chosen { border-color: #8fbfa4; background: #f4faf7; }
  .pair-col.rejected { border-color: #d0a0a0; background: #faf5f5; }
  .pair-col h3 {
    margin: 0 0 0.45rem; font-size: 0.95rem;
    display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;
  }
  .short-answer {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.85rem; margin: 0 0 0.55rem;
    padding: 0.4rem 0.55rem; border-radius: 8px;
    background: #f2f6f9;
  }
  pre {
    white-space: pre-wrap; word-break: break-word;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem; margin: 0.25rem 0 0;
    background: #f2f6f9; padding: 0.55rem 0.65rem; border-radius: 8px;
    max-height: 420px; overflow: auto;
  }
  .nav-btns { display: flex; gap: 0.4rem; }
  button {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.45rem 0.7rem; cursor: pointer;
  }
  button:hover { background: #eef5fa; }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <h1>M_JudgeBench</h1>
      <p>Multimodal preference pairs · chosen vs rejected</p>
    </div>
    <div class="controls">
      <label>Benchmark
        <select id="benchmark"><option value="">All</option></select>
      </label>
      <label>Type
        <select id="type"><option value="">All</option></select>
      </label>
      <label>Field
        <select id="field"><option value="">All</option></select>
      </label>
      <label>Search
        <input id="search" type="search" placeholder="question, answer, idx…" />
      </label>
      <div class="nav-btns">
        <button id="prev" type="button" title="Previous">←</button>
        <button id="next" type="button" title="Next">→</button>
      </div>
    </div>
  </div>
</header>
<main>
  <section class="panel">
    <h2>Examples</h2>
    <div class="stats" id="stats">Loading…</div>
    <ul id="qlist"></ul>
  </section>
  <section class="panel">
    <h2>Detail</h2>
    <div id="detail"><p class="muted">Select an example.</p></div>
  </section>
</main>
<script>
const state = {
  summaries: [],
  filtered: [],
  selectedIdx: null,
  meta: {},
};

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function fillSelect(id, values) {
  const el = document.getElementById(id);
  const current = el.value;
  el.innerHTML = `<option value="">All</option>` + values.map(v =>
    `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`
  ).join("");
  if (values.includes(current)) el.value = current;
}

function applyFilters() {
  const q = document.getElementById("search").value.trim().toLowerCase();
  const bench = document.getElementById("benchmark").value;
  const typ = document.getElementById("type").value;
  const field = document.getElementById("field").value;
  state.filtered = state.summaries.filter(row => {
    if (bench && row.benchmark_name !== bench) return false;
    if (typ && row.type !== typ) return false;
    if (field && row.field !== field) return false;
    if (!q) return true;
    const hay = [
      String(row.idx),
      row.question,
      row.answer,
      row.chosen_answer,
      row.rejected_answer,
      row.field,
      row.source,
      row.benchmark_name,
      row.type,
    ].join("\n").toLowerCase();
    return hay.includes(q);
  });
  document.getElementById("stats").innerHTML =
    `<span>Showing <strong>${state.filtered.length}</strong> / ${state.summaries.length}</span>` +
    (bench || typ || field || q ? ` <span class="chip">filtered</span>` : "");
  renderList();
}

function renderList() {
  const ul = document.getElementById("qlist");
  if (!state.filtered.length) {
    ul.innerHTML = `<li class="muted">No matching examples.</li>`;
    return;
  }
  ul.innerHTML = state.filtered.map(row => {
    const active = row.idx === state.selectedIdx ? "active" : "";
    return `<li class="${active}" data-idx="${row.idx}">
      <div class="qid">#${row.idx} · ${escapeHtml(row.benchmark_name)}</div>
      <div class="qtext">${escapeHtml(row.question)}</div>
      <div class="mini-chips">
        <span class="chip type">${escapeHtml(row.type)}</span>
        <span class="chip">${escapeHtml(row.field)}</span>
      </div>
    </li>`;
  }).join("");
  ul.querySelectorAll("li[data-idx]").forEach(li => {
    li.addEventListener("click", () => selectExample(Number(li.dataset.idx)));
  });
}

async function selectExample(idx) {
  state.selectedIdx = idx;
  renderList();
  const detail = document.getElementById("detail");
  detail.innerHTML = `<p class="muted">Loading…</p>`;
  const data = await api(`/api/example?idx=${encodeURIComponent(idx)}`);
  const row = data.example;
  const img = data.image_url
    ? `<img class="example-image" src="${escapeHtml(data.image_url)}" alt="example image" />`
    : `<p class="muted">Image not found locally (${escapeHtml(row.image || "")}).</p>`;
  detail.innerHTML = `
    <div class="qid">idx ${row.idx}</div>
    <h3 style="margin:0.35rem 0 0.15rem">${escapeHtml(row.question || "")}</h3>
    <div class="meta-row">
      <span class="chip type">${escapeHtml(row.type || "")}</span>
      <span class="chip">${escapeHtml(row.benchmark_name || "")}</span>
      <span class="chip">${escapeHtml(row.field || "")}</span>
      ${row.source ? `<span class="chip">${escapeHtml(row.source)}</span>` : ""}
    </div>
    ${img}
    <div class="answer-box">
      <strong>Gold answer</strong>
      <div class="short-answer" style="margin-top:0.45rem">${escapeHtml(row.answer || "")}</div>
    </div>
    <div class="pair-grid">
      <div class="pair-col chosen">
        <h3><span class="chip chosen">chosen</span></h3>
        <div class="short-answer">${escapeHtml(row.chosen_answer || "")}</div>
        <pre>${escapeHtml(row.chosen || "")}</pre>
      </div>
      <div class="pair-col rejected">
        <h3><span class="chip rejected">rejected</span></h3>
        <div class="short-answer">${escapeHtml(row.rejected_answer || "")}</div>
        <pre>${escapeHtml(row.rejected || "")}</pre>
      </div>
    </div>
    ${row.explanation ? `<div class="explain-box"><strong>Explanation</strong><pre>${escapeHtml(row.explanation)}</pre></div>` : ""}
  `;
  const active = document.querySelector(`#qlist li[data-idx="${idx}"]`);
  if (active) active.scrollIntoView({ block: "nearest" });
}

function move(delta) {
  if (!state.filtered.length || state.selectedIdx == null) return;
  const i = state.filtered.findIndex(r => r.idx === state.selectedIdx);
  if (i < 0) return;
  const next = state.filtered[Math.max(0, Math.min(state.filtered.length - 1, i + delta))];
  if (next) selectExample(next.idx);
}

async function init() {
  const data = await api("/api/meta");
  state.meta = data;
  state.summaries = data.summaries || [];
  fillSelect("benchmark", data.benchmarks || []);
  fillSelect("type", data.types || []);
  fillSelect("field", data.fields || []);
  ["benchmark", "type", "field", "search"].forEach(id => {
    document.getElementById(id).addEventListener("input", () => {
      applyFilters();
      if (state.filtered.length && !state.filtered.some(r => r.idx === state.selectedIdx)) {
        selectExample(state.filtered[0].idx);
      } else {
        renderList();
      }
    });
  });
  document.getElementById("prev").addEventListener("click", () => move(-1));
  document.getElementById("next").addEventListener("click", () => move(1));
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === "ArrowLeft") move(-1);
    if (e.key === "ArrowRight") move(1);
  });
  applyFilters();
  if (state.filtered.length) await selectExample(state.filtered[0].idx);
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
        print(f"[m-judgebench] {self.address_string()} {fmt % args}")

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
        ds = CACHE["dataset"]

        if path in ("/", "/index.html"):
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/meta":
            self._send_json(
                {
                    "n": ds["n"],
                    "path": ds["path"],
                    "summaries": ds["summaries"],
                    "benchmarks": ds["benchmarks"],
                    "types": ds["types"],
                    "fields": ds["fields"],
                }
            )
            return

        if path == "/api/example":
            raw = (qs.get("idx") or [""])[0]
            try:
                idx = int(raw)
            except ValueError:
                self._send_json({"error": "invalid idx"}, 400)
                return
            row = ds["by_idx"].get(idx)
            if row is None:
                self._send_json({"error": "example not found"}, 404)
                return
            image = resolve_image(row.get("image"))
            image_url = None
            if image is not None:
                # Serve via relative path under data dir.
                rel = image.relative_to(Path(CONFIG["data_dir"])).as_posix()
                image_url = f"/image/{rel}"
            self._send_json({"example": row, "image_url": image_url})
            return

        if path.startswith("/image/"):
            rel = unquote(path[len("/image/") :])
            # Prevent path escape.
            image = (Path(CONFIG["data_dir"]) / rel).resolve()
            data_root = Path(CONFIG["data_dir"]).resolve()
            if not str(image).startswith(str(data_root)) or not image.is_file():
                self.send_error(404, "image not found")
                return
            data = image.read_bytes()
            ctype = mimetypes.guess_type(str(image))[0] or "application/octet-stream"
            self._send(200, data, ctype)
            return

        self.send_error(404, "not found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing m_judgebench_data.jsonl and images/",
    )
    parser.add_argument(
        "--jsonl",
        default=DEFAULT_JSONL,
        help="JSONL filename inside --data-dir",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    CONFIG["data_dir"] = data_dir
    dataset = load_dataset(data_dir, args.jsonl)
    CACHE["dataset"] = dataset

    print(f"Data:  {dataset['path']}", flush=True)
    print(f"Images:{data_dir / 'images'}", flush=True)
    print(f"Loaded {dataset['n']} examples", flush=True)
    print(f"  benchmarks: {', '.join(dataset['benchmarks'])}", flush=True)
    print(f"  types: {len(dataset['types'])}", flush=True)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
