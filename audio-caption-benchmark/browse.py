"""Local HTTP browser for caption samples + QA benchmark drafting.

Usage:
    uv run python audio-caption-benchmark/browse.py
    uv run python audio-caption-benchmark/browse.py --smoke
    uv run python audio-caption-benchmark/browse.py --smoke --self-check
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from db import (  # noqa: E402
    clear_benchmark,
    connect,
    export_benchmark,
    get_example,
    get_state,
    list_examples,
    set_state,
    upsert_benchmark,
)

DEFAULT_DATA_DIR = PKG_DIR / "data"
SMOKE_DATA_DIR = PKG_DIR / "data" / "smoke"

CONFIG: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--smoke", action="store_true", help="Use data/smoke/")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Hit list/detail APIs once and exit (no long-running server).",
    )
    return parser.parse_args()


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        return args.data_dir.expanduser().resolve()
    if args.smoke:
        return SMOKE_DATA_DIR.resolve()
    return DEFAULT_DATA_DIR.resolve()


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Audio Caption Benchmark</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg: #e8eef2;
    --ink: #14202a;
    --muted: #5a6b78;
    --line: #b7c7d2;
    --card: #f7fafc;
    --accent: #1f5f8b;
    --good: #1f6b4a;
    --soft: #d9e7f0;
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
    padding: 0.85rem 1.25rem;
    display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem;
    align-items: center; justify-content: space-between;
  }
  header h1 {
    margin: 0;
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.02em;
  }
  .controls { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
  select, button, input[type="number"] {
    font: inherit;
    border: 1px solid var(--line);
    background: var(--card);
    color: var(--ink);
    border-radius: 8px;
    padding: 0.35rem 0.65rem;
  }
  button {
    cursor: pointer;
    background: var(--accent);
    color: #fff;
    border-color: transparent;
    font-weight: 500;
  }
  button.secondary { background: var(--card); color: var(--ink); border-color: var(--line); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  main {
    max-width: 920px;
    margin: 0 auto;
    padding: 1.25rem;
    display: grid;
    gap: 1rem;
  }
  .panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 1rem 1.15rem;
  }
  .meta-line {
    color: var(--muted);
    font-size: 0.85rem;
    font-family: "IBM Plex Mono", monospace;
    margin-bottom: 0.75rem;
  }
  audio { width: 100%; margin: 0.5rem 0 1rem; }
  .caption {
    font-size: 1.2rem;
    line-height: 1.45;
    font-weight: 500;
  }
  .caption-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-bottom: 0.35rem;
  }
  details.aux {
    margin-top: 0.25rem;
  }
  details.aux summary {
    cursor: pointer;
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
  }
  .kv {
    display: grid;
    grid-template-columns: minmax(7rem, 30%) 1fr;
    gap: 0.35rem 0.75rem;
    font-size: 0.82rem;
  }
  .kv dt { color: var(--muted); font-family: "IBM Plex Mono", monospace; }
  .kv dd { margin: 0; word-break: break-word; }
  label.field {
    display: grid;
    gap: 0.35rem;
    margin-bottom: 0.85rem;
    font-size: 0.9rem;
    font-weight: 500;
  }
  textarea {
    font: inherit;
    min-height: 4.5rem;
    resize: vertical;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.55rem 0.7rem;
    background: #fff;
  }
  .row-actions { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
  .status { color: var(--muted); font-size: 0.85rem; min-height: 1.2em; }
  .status.ok { color: var(--good); }
  .empty {
    padding: 2rem;
    text-align: center;
    color: var(--muted);
  }
  a.export { color: var(--accent); font-size: 0.9rem; }
</style>
</head>
<body>
<header>
  <h1>Audio Caption Benchmark</h1>
  <div class="controls">
    <label>Source
      <select id="sourceFilter">
        <option value="all">all</option>
        <option value="wavcaps">wavcaps</option>
        <option value="audiocaps">audiocaps</option>
        <option value="clotho">clotho</option>
      </select>
    </label>
    <label>QA
      <select id="annotatedFilter">
        <option value="all">all</option>
        <option value="no">unannotated</option>
        <option value="yes">annotated</option>
      </select>
    </label>
    <button type="button" class="secondary" id="prevBtn">Prev</button>
    <span id="position">0 / 0</span>
    <button type="button" class="secondary" id="nextBtn">Next</button>
    <label># <input type="number" id="jumpInput" min="1" style="width:4.5rem" /></label>
    <button type="button" class="secondary" id="jumpBtn">Go</button>
    <a class="export" href="/api/export" download="benchmark.jsonl">Export JSONL</a>
  </div>
</header>
<main id="main">
  <div class="empty" id="empty">Loading…</div>
  <section class="panel" id="detail" hidden>
    <div class="meta-line" id="metaLine"></div>
    <audio id="player" controls preload="metadata"></audio>
    <div class="caption-label">Caption</div>
    <div class="caption" id="caption"></div>
  </section>
  <section class="panel" id="auxPanel" hidden>
    <details class="aux" open>
      <summary>Metadata</summary>
      <dl class="kv" id="metaKv"></dl>
    </details>
  </section>
  <section class="panel" id="qaPanel" hidden>
    <label class="field">Question
      <textarea id="question" placeholder="Write a question about this audio…"></textarea>
    </label>
    <label class="field">Answer
      <textarea id="answer" placeholder="Corresponding answer…"></textarea>
    </label>
    <div class="row-actions">
      <button type="button" id="saveBtn">Save</button>
      <button type="button" class="secondary" id="clearBtn">Clear</button>
      <span class="status" id="saveStatus"></span>
    </div>
  </section>
</main>
<script>
const state = {
  ids: [],
  index: 0,
  current: null,
};

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res.text();
}

function renderMeta(meta) {
  const dl = document.getElementById("metaKv");
  dl.innerHTML = "";
  const entries = Object.entries(meta || {});
  if (!entries.length) {
    dl.innerHTML = "<dt>—</dt><dd>No extra metadata</dd>";
    return;
  }
  for (const [k, v] of entries) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = typeof v === "string" ? v : JSON.stringify(v);
    dl.appendChild(dt);
    dl.appendChild(dd);
  }
}

function updateNav() {
  const n = state.ids.length;
  const pos = n ? state.index + 1 : 0;
  document.getElementById("position").textContent = `${pos} / ${n}`;
  document.getElementById("prevBtn").disabled = state.index <= 0;
  document.getElementById("nextBtn").disabled = state.index >= n - 1 || n === 0;
  document.getElementById("jumpInput").value = pos || "";
}

async function loadList() {
  const source = document.getElementById("sourceFilter").value;
  const annotated = document.getElementById("annotatedFilter").value;
  const qs = new URLSearchParams({ source, annotated });
  const data = await api("/api/examples?" + qs.toString());
  state.ids = data.ids || [];
  const preferred = data.current_id;
  let idx = 0;
  if (preferred != null) {
    const found = state.ids.indexOf(preferred);
    if (found >= 0) idx = found;
  }
  state.index = idx;
  updateNav();
  if (!state.ids.length) {
    document.getElementById("empty").hidden = false;
    document.getElementById("empty").textContent =
      "No examples yet. Run download_samples.py first.";
    document.getElementById("detail").hidden = true;
    document.getElementById("auxPanel").hidden = true;
    document.getElementById("qaPanel").hidden = true;
    return;
  }
  document.getElementById("empty").hidden = true;
  await loadCurrent();
}

async function loadCurrent() {
  if (!state.ids.length) return;
  const id = state.ids[state.index];
  const item = await api("/api/example/" + id);
  state.current = item;
  document.getElementById("detail").hidden = false;
  document.getElementById("auxPanel").hidden = false;
  document.getElementById("qaPanel").hidden = false;
  document.getElementById("metaLine").textContent =
    `#${item.id} · ${item.source} · ${item.source_id}`;
  document.getElementById("caption").textContent = item.caption || "";
  const player = document.getElementById("player");
  player.src = "/audio/" + encodeURIComponent(item.audio_path);
  player.load();
  renderMeta(item.metadata || {});
  document.getElementById("question").value = item.question || "";
  document.getElementById("answer").value = item.answer || "";
  document.getElementById("saveStatus").textContent = "";
  updateNav();
}

async function saveQa() {
  if (!state.current) return;
  const body = {
    example_id: state.current.id,
    question: document.getElementById("question").value,
    answer: document.getElementById("answer").value,
  };
  const res = await api("/api/benchmark", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const status = document.getElementById("saveStatus");
  status.className = "status ok";
  status.textContent = res.cleared ? "Cleared" : "Saved";
}

async function clearQa() {
  if (!state.current) return;
  await api("/api/benchmark/" + state.current.id, { method: "DELETE" });
  document.getElementById("question").value = "";
  document.getElementById("answer").value = "";
  const status = document.getElementById("saveStatus");
  status.className = "status ok";
  status.textContent = "Cleared";
}

document.getElementById("prevBtn").onclick = async () => {
  if (state.index > 0) { state.index -= 1; await loadCurrent(); }
};
document.getElementById("nextBtn").onclick = async () => {
  if (state.index < state.ids.length - 1) { state.index += 1; await loadCurrent(); }
};
document.getElementById("jumpBtn").onclick = async () => {
  const n = parseInt(document.getElementById("jumpInput").value, 10);
  if (!n || n < 1 || n > state.ids.length) return;
  state.index = n - 1;
  await loadCurrent();
};
document.getElementById("sourceFilter").onchange = () => loadList();
document.getElementById("annotatedFilter").onchange = () => loadList();
document.getElementById("saveBtn").onclick = () => saveQa().catch(err => {
  document.getElementById("saveStatus").textContent = String(err);
});
document.getElementById("clearBtn").onclick = () => clearQa().catch(err => {
  document.getElementById("saveStatus").textContent = String(err);
});

loadList().catch(err => {
  document.getElementById("empty").textContent = String(err);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, data, "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        conn = CONFIG["conn"]
        data_dir: Path = CONFIG["data_dir"]

        if path in {"/", "/index.html"}:
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/examples":
            source = (qs.get("source") or ["all"])[0]
            annotated = (qs.get("annotated") or ["all"])[0]
            rows = list_examples(conn, source=source, annotated=annotated)
            ids = [int(r["id"]) for r in rows]
            current_raw = get_state(conn, "current_example_id")
            current_id = int(current_raw) if current_raw and current_raw.isdigit() else None
            if current_id not in ids and ids:
                current_id = ids[0]
            self._json(
                200,
                {
                    "ids": ids,
                    "count": len(ids),
                    "current_id": current_id,
                    "summaries": [
                        {
                            "id": r["id"],
                            "source": r["source"],
                            "source_id": r["source_id"],
                            "has_qa": bool((r.get("question") or "").strip() and (r.get("answer") or "").strip()),
                        }
                        for r in rows
                    ],
                },
            )
            return

        if path.startswith("/api/example/"):
            try:
                example_id = int(path.rsplit("/", 1)[-1])
            except ValueError:
                self._json(400, {"error": "bad id"})
                return
            item = get_example(conn, example_id)
            if item is None:
                self._json(404, {"error": "not found"})
                return
            set_state(conn, "current_example_id", str(example_id))
            self._json(200, item)
            return

        if path == "/api/export":
            rows = export_benchmark(conn)
            lines = [json.dumps(r, ensure_ascii=False) for r in rows]
            body = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="benchmark.jsonl"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path.startswith("/audio/"):
            rel = unquote(path[len("/audio/") :])
            # Prevent path traversal
            audio_path = (data_dir / rel).resolve()
            if not str(audio_path).startswith(str(data_dir.resolve())):
                self._json(400, {"error": "bad path"})
                return
            if not audio_path.is_file():
                self._json(404, {"error": "audio missing", "path": rel})
                return
            mime, _ = mimetypes.guess_type(str(audio_path))
            data = audio_path.read_bytes()
            self._send(200, data, mime or "application/octet-stream")
            return

        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/benchmark":
            self._json(404, {"error": "not found"})
            return
        try:
            payload = self._read_json()
            example_id = int(payload["example_id"])
            result = upsert_benchmark(
                CONFIG["conn"],
                example_id=example_id,
                question=str(payload.get("question") or ""),
                answer=str(payload.get("answer") or ""),
            )
            self._json(200, result)
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": str(exc)})

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/benchmark/"):
            self._json(404, {"error": "not found"})
            return
        try:
            example_id = int(parsed.path.rsplit("/", 1)[-1])
            clear_benchmark(CONFIG["conn"], example_id)
            self._json(200, {"ok": True, "example_id": example_id})
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": str(exc)})


def self_check(host: str, port: int) -> None:
    import urllib.request

    base = f"http://{host}:{port}"
    with urllib.request.urlopen(base + "/api/examples", timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    ids = payload.get("ids") or []
    print(f"self-check: {len(ids)} examples", flush=True)
    if not ids:
        raise SystemExit("self-check failed: no examples in DB (run download_samples.py --smoke)")
    eid = ids[0]
    with urllib.request.urlopen(base + f"/api/example/{eid}", timeout=10) as resp:
        item = json.loads(resp.read().decode("utf-8"))
    assert item.get("caption"), "missing caption"
    assert item.get("audio_path"), "missing audio_path"
    audio_rel = item["audio_path"]
    with urllib.request.urlopen(base + "/audio/" + audio_rel, timeout=30) as resp:
        audio = resp.read()
    if len(audio) < 100:
        raise SystemExit("self-check failed: audio too small")
    print(
        f"self-check ok: id={eid} source={item.get('source')} "
        f"audio_bytes={len(audio)} caption={item['caption'][:60]!r}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args)
    db_path = data_dir / "browser.db"
    if not db_path.is_file():
        print(
            f"No DB at {db_path}. Run:\n"
            f"  uv run python audio-caption-benchmark/download_samples.py"
            f"{' --smoke' if args.smoke else ''}",
            flush=True,
        )
        if args.self_check:
            raise SystemExit(1)

    conn = connect(db_path)
    CONFIG["conn"] = conn
    CONFIG["data_dir"] = data_dir

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving {data_dir} at {url}", flush=True)

    if args.self_check:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self_check(args.host, args.port)
        finally:
            server.shutdown()
        return

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    finally:
        server.server_close()
        conn.close()


if __name__ == "__main__":
    main()
