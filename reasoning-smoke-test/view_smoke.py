"""Viewer for reasoning-smoke-test traces (raw text + token boundaries).

Usage::

    uv run python reasoning-smoke-test/view_smoke.py
    uv run python reasoning-smoke-test/view_smoke.py --run-id <id>
    uv run python reasoning-smoke-test/view_smoke.py --port 7864
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from view_mmar import (  # noqa: E402
    DEFAULT_AUDIO_DIR,
    ensure_mmar_audio,
    resolve_audio,
)
from view_mmar import CONFIG as MMAR_CONFIG  # noqa: E402

DEFAULT_RUNS_DIR = REPO_ROOT / "outputs" / "reasoning-smoke-test"
CONFIG: dict[str, Any] = {}

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Reasoning smoke — token trace</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg: #e7eef3;
    --ink: #14202a;
    --muted: #5a6b78;
    --line: #b7c7d2;
    --card: #f7fafc;
    --accent: #1f5f8b;
    --prompt: #d9e7f2;
    --prompt-ink: #1a3d55;
    --out: #e7f3ea;
    --out-ink: #1d4a32;
    --special: #ece4f6;
    --special-ink: #5a3d86;
    --repeat: #f3efe4;
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
    font-weight: 600; font-size: 1.35rem;
    margin: 0 0 0.15rem; letter-spacing: -0.03em;
  }
  .brand p { margin: 0; color: var(--muted); font-size: 0.9rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: end; }
  label { font-size: 0.75rem; color: var(--muted); display: grid; gap: 0.25rem; }
  select, button {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.45rem 0.65rem;
  }
  button { cursor: pointer; }
  button.active { background: #e2eef6; border-color: #8fb3c9; }
  main { max-width: 1280px; margin: 0 auto; padding: 1.25rem; display: grid; gap: 1rem; }
  .card {
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--radius); box-shadow: var(--shadow);
    padding: 1rem 1.1rem;
  }
  .meta { color: var(--muted); font-size: 0.85rem; display: flex; flex-wrap: wrap; gap: 0.75rem; }
  .question { font-size: 1.05rem; margin: 0.4rem 0 0.75rem; }
  audio { width: 100%; margin-top: 0.4rem; }
  .tabs { display: flex; flex-wrap: wrap; gap: 0.4rem; }
  .tabs button { font-size: 0.85rem; }
  h2 {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.95rem; font-weight: 600;
    margin: 0 0 0.45rem; letter-spacing: -0.02em;
  }
  .row-head {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 0.75rem; margin-bottom: 0.45rem;
  }
  .count { color: var(--muted); font-size: 0.8rem; }
  pre.raw {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.78rem; line-height: 1.45;
    white-space: pre-wrap; word-break: break-word;
    background: #f1f5f8; border: 1px solid var(--line);
    border-radius: 8px; padding: 0.7rem 0.8rem; margin: 0 0 0.75rem;
    max-height: 14rem; overflow: auto;
  }
  .tokens {
    display: flex; flex-wrap: wrap; gap: 2px; align-items: stretch;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.78rem; line-height: 1.3;
  }
  .tok {
    display: inline-flex; align-items: center;
    border: 1px solid color-mix(in srgb, var(--ink) 12%, transparent);
    border-radius: 4px; padding: 0.12rem 0.28rem;
    background: var(--prompt); color: var(--prompt-ink);
    white-space: pre; max-width: 100%;
  }
  .tokens.output .tok { background: var(--out); color: var(--out-ink); }
  .tok.special { background: var(--special); color: var(--special-ink); }
  .tok.repeat { background: var(--repeat); color: var(--muted); font-size: 0.72rem; }
  .tok.empty { opacity: 0.85; font-style: italic; }
  .think { color: var(--muted); font-size: 0.88rem; }
  .think strong { color: var(--ink); font-weight: 600; }
  .empty { color: var(--muted); }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <h1>Reasoning smoke</h1>
      <p>Raw prompt / completion as tokenized, with token boundaries.</p>
    </div>
    <div class="controls">
      <label>Run
        <select id="run"></select>
      </label>
      <label>
        <span>&nbsp;</span>
        <button id="collapse" class="active" type="button">Collapse repeats</button>
      </label>
    </div>
  </div>
</header>
<main>
  <div class="card" id="question-card">
    <p class="empty">Loading…</p>
  </div>
  <div class="card">
    <div class="tabs" id="tabs"></div>
  </div>
  <div id="detail"></div>
</main>
<script>
const state = { runs: [], runId: null, pack: null, label: null, collapse: true };

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

function isSpecial(piece) {
  const blob = `${piece.token || ""}${piece.text || ""}`;
  return /<\|/.test(blob) || /<\/?think/i.test(blob) || /^</.test(piece.token || "");
}

function boundedText(pieces) {
  return (pieces || []).map(piece => {
    const text = (piece.text ?? "").length ? piece.text : (piece.token || `id=${piece.id}`);
    return String(text).replaceAll("\n", "\\n");
  }).join("│");
}

function visibleText(piece) {
  const text = piece.text ?? "";
  if (text.length) return text;
  if (piece.token) return piece.token;
  return `id=${piece.id}`;
}

function collapsedPieces(pieces) {
  if (!state.collapse) return pieces.map(p => ({ ...p, count: 1 }));
  const out = [];
  for (const piece of pieces) {
    const prev = out[out.length - 1];
    if (prev && prev.id === piece.id && prev.token === piece.token && prev.text === piece.text) {
      prev.count += 1;
      continue;
    }
    out.push({ ...piece, count: 1 });
  }
  return out;
}

function renderTokens(pieces, kind) {
  const collapsed = collapsedPieces(pieces || []);
  if (!collapsed.length) return `<p class="empty">No token ids captured.</p>`;
  return `<div class="tokens ${kind}">` + collapsed.map(piece => {
    const special = isSpecial(piece) ? " special" : "";
    const empty = !(piece.text || "").length ? " empty" : "";
    const repeat = piece.count > 1 ? " repeat" : "";
    const label = piece.count > 1
      ? `${visibleText(piece)} × ${piece.count}`
      : visibleText(piece);
    const title = `id=${piece.id}` + (piece.token ? `  token=${piece.token}` : "") +
      (piece.count > 1 ? `  ×${piece.count}` : "");
    return `<span class="tok${special}${empty}${repeat}" title="${escapeHtml(title)}">${escapeHtml(label)}</span>`;
  }).join("") + `</div>`;
}

function renderQuestion() {
  const pack = state.pack;
  if (!pack) {
    document.getElementById("question-card").innerHTML = `<p class="empty">No smoke runs in outputs/reasoning-smoke-test.</p>`;
    return;
  }
  const sample = pack.models[state.label] || pack.models[pack.labels[0]] || {};
  const audio = sample.audio_url
    ? `<audio controls src="${escapeHtml(sample.audio_url)}"></audio>`
    : `<p class="empty">Audio not found locally (${escapeHtml(sample.audio_path || "")}).</p>`;
  document.getElementById("question-card").innerHTML = `
    <div class="meta">
      <span>run <strong>${escapeHtml(pack.run_id)}</strong></span>
      <span>question <strong>${escapeHtml(sample.question_id || pack.question_id || "")}</strong></span>
      <span>${escapeHtml(sample.modality || "")}</span>
    </div>
    <p class="question">${escapeHtml(sample.question || "")}</p>
    ${audio}
  `;
}

function renderTabs() {
  const pack = state.pack;
  const tabs = document.getElementById("tabs");
  if (!pack) { tabs.innerHTML = ""; return; }
  tabs.innerHTML = pack.labels.map(label => {
    const rec = pack.models[label] || {};
    const nOut = (rec.output && rec.output.n_tokens) || 0;
    const cls = label === state.label ? "active" : "";
    return `<button class="${cls}" data-label="${escapeHtml(label)}" type="button">${escapeHtml(label)} · ${nOut} tok</button>`;
  }).join("");
  tabs.querySelectorAll("button").forEach(btn => {
    btn.addEventListener("click", () => {
      state.label = btn.dataset.label;
      render();
    });
  });
}

function renderDetail() {
  const rec = state.pack && state.pack.models[state.label];
  const el = document.getElementById("detail");
  if (!rec) { el.innerHTML = ""; return; }
  const sampling = rec.sampling || {};
  const chatKw = rec.chat_kwargs && rec.chat_kwargs.chat_template_kwargs
    ? JSON.stringify(rec.chat_kwargs.chat_template_kwargs)
    : "";
  el.innerHTML = `
    <div class="card">
      <div class="meta">
        <span>${escapeHtml(rec.model_id || "")}</span>
        <span>backend ${escapeHtml(rec.backend || "")}</span>
        <span>enable_thinking=${escapeHtml(String(rec.enable_thinking))}</span>
        <span>${escapeHtml(chatKw)}</span>
        <span>T=${escapeHtml(String(sampling.temperature ?? ""))} max_tokens=${escapeHtml(String(sampling.max_tokens ?? ""))}</span>
        <span>finish=${escapeHtml(rec.finish_reason || "")}</span>
      </div>
    </div>
    <div class="card">
      <div class="row-head">
        <h2>Prompt — raw text</h2>
        <span class="count">${(rec.prompt && rec.prompt.n_tokens) || 0} tokens</span>
      </div>
      <pre class="raw">${escapeHtml((rec.prompt && rec.prompt.text) || "")}</pre>
      <h2>Prompt — same text with token boundaries (│)</h2>
      <pre class="raw">${escapeHtml(boundedText((rec.prompt && rec.prompt.pieces) || []))}</pre>
      <h2>Prompt — token chips</h2>
      ${renderTokens((rec.prompt && rec.prompt.pieces) || [], "prompt")}
    </div>
    <div class="card">
      <div class="row-head">
        <h2>Output — raw text</h2>
        <span class="count">${(rec.output && rec.output.n_tokens) || 0} tokens</span>
      </div>
      <pre class="raw">${escapeHtml((rec.output && rec.output.text) || "")}</pre>
      <h2>Output — same text with token boundaries (│)</h2>
      <pre class="raw">${escapeHtml(boundedText((rec.output && rec.output.pieces) || []))}</pre>
      <h2>Output — token chips</h2>
      ${renderTokens((rec.output && rec.output.pieces) || [], "output")}
      <p class="think" style="margin:0.85rem 0 0">
        parsed thinking: <strong>${escapeHtml((rec.thinking_prediction || "").slice(0, 400) || "—")}</strong>
        <br/>parsed answer: <strong>${escapeHtml(rec.answer_prediction || "—")}</strong>
      </p>
    </div>
  `;
}

function render() {
  renderQuestion();
  renderTabs();
  renderDetail();
}

async function loadRun(runId) {
  const res = await fetch(`/api/run?id=${encodeURIComponent(runId)}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  state.pack = data;
  state.runId = data.run_id;
  state.label = data.labels.includes(state.label) ? state.label : (data.labels[0] || null);
  render();
}

async function init() {
  const res = await fetch("/api/runs");
  const data = await res.json();
  state.runs = data.runs || [];
  const sel = document.getElementById("run");
  sel.innerHTML = state.runs.map(r =>
    `<option value="${escapeHtml(r.id)}">${escapeHtml(r.id)} · ${(r.models || []).length} models</option>`
  ).join("");
  document.getElementById("collapse").addEventListener("click", () => {
    state.collapse = !state.collapse;
    document.getElementById("collapse").classList.toggle("active", state.collapse);
    renderDetail();
  });
  sel.addEventListener("change", () => loadRun(sel.value));
  const preferred = new URLSearchParams(location.search).get("run") || data.selected;
  if (preferred && state.runs.some(r => r.id === preferred)) {
    sel.value = preferred;
    await loadRun(preferred);
  } else if (state.runs.length) {
    await loadRun(state.runs[0].id);
  } else {
    renderQuestion();
  }
}

init().catch(err => {
  document.getElementById("question-card").innerHTML =
    `<p class="empty">${escapeHtml(String(err))}</p>`;
});
</script>
</body>
</html>
"""


def _runs_dir() -> Path:
    return Path(CONFIG["runs_dir"])


def discover_runs(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(runs_dir.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        models = sorted(p.stem for p in path.glob("*.json") if p.name != "manifest.json")
        if not models:
            continue
        found.append({"id": path.name, "models": models})
    return found


def load_run(runs_dir: Path, run_id: str) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    models: dict[str, Any] = {}
    for path in sorted(run_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        audio_path = payload.get("audio_path")
        audio = resolve_audio(audio_path)
        payload["audio_url"] = f"/audio/{audio.name}" if audio is not None else None
        models[path.stem] = payload
    labels = list(models)
    question_id = next(
        (models[label].get("question_id") for label in labels),
        None,
    )
    return {
        "run_id": run_id,
        "labels": labels,
        "question_id": question_id,
        "models": models,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[view_smoke] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/runs":
                runs = discover_runs(_runs_dir())
                selected = CONFIG.get("run_id")
                if selected and not any(run["id"] == selected for run in runs):
                    selected = runs[0]["id"] if runs else None
                elif not selected and runs:
                    selected = runs[0]["id"]
                self._send_json({"runs": runs, "selected": selected})
                return
            if path == "/api/run":
                run_id = (qs.get("id") or [None])[0]
                if not run_id:
                    self._send_json({"error": "missing id"}, 400)
                    return
                self._send_json(load_run(_runs_dir(), run_id))
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
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Local smoke traces (default: outputs/reasoning-smoke-test)",
    )
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--skip-audio-download", action="store_true")
    args = parser.parse_args()

    runs_dir = args.runs_dir.expanduser().resolve()
    CONFIG["runs_dir"] = runs_dir
    CONFIG["run_id"] = args.run_id
    audio_dir = args.audio_dir.expanduser().resolve()
    if not args.skip_audio_download:
        try:
            audio_dir = ensure_mmar_audio(audio_dir)
        except SystemExit as exc:
            print(f"Audio setup failed: {exc}", flush=True)
    CONFIG["audio_dir"] = audio_dir
    MMAR_CONFIG["audio_dir"] = audio_dir

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Reasoning smoke viewer http://{args.host}:{args.port}")
    print(f"Traces: {runs_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
