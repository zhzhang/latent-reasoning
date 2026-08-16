"""Take the MMAR benchmark as a human.

Flow per question:
  1. Play audio + read the question (no choices) → write a freeform answer
     and mark whether the question is reasonably answerable without choices
  2. Reveal multiple-choice options → pick one
  3. Reveal the gold MC answer → continue

Browse with Prev/Next, revisit answered questions, and update answers
(including the free-response checkbox). Saves upsert into a local JSONL file.

Usage:

    uv run python take_mmar.py
    uv run python take_mmar.py --participant jordan --port 7862
    uv run python take_mmar.py --shuffle --seed 0 --limit 50
    uv run python take_mmar.py --answers-path outputs/human-mmar/me.jsonl
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import random
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from mmar_common import load_jsonl
from view_difficulty import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_DATA_DIR,
    ensure_mmar_audio,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_META = DEFAULT_DATA_DIR / "MMAR-meta.jsonl"
DEFAULT_ANSWERS_DIR = REPO_ROOT / "outputs" / "human-mmar"

CONFIG: dict[str, Any] = {}
STATE_LOCK = threading.Lock()

# In-progress freeform drafts keyed by question id (server-side gate before MC).
PENDING_FREEFORM: dict[str, dict[str, Any]] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def coerce_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def load_answers_by_id(answers_path: Path) -> dict[str, dict]:
    """Latest answer record per question id (later lines win)."""
    if not answers_path.is_file():
        return {}
    by_id: dict[str, dict] = {}
    for row in load_jsonl(answers_path):
        qid = row.get("id")
        if qid is None:
            continue
        by_id[str(qid)] = row
    return by_id


def upsert_answer(record: dict) -> None:
    """Replace the saved answer for ``record['id']`` and rewrite the JSONL."""
    answers_path: Path = CONFIG["answers_path"]
    answers_by_id: dict[str, dict] = CONFIG["answers_by_id"]
    qid = str(record["id"])
    answers_by_id[qid] = record
    CONFIG["answered_ids"].add(qid)

    # Keep file order aligned with the active question set, then extras.
    ordered_ids = [item["id"] for item in CONFIG["items"] if item["id"] in answers_by_id]
    extras = [qid_ for qid_ in answers_by_id if qid_ not in set(ordered_ids)]
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(answers_path, "w", encoding="utf-8") as handle:
        for qid_ in ordered_ids + extras:
            handle.write(json.dumps(answers_by_id[qid_], ensure_ascii=False) + "\n")


def load_items(
    meta_path: Path,
    *,
    shuffle: bool,
    seed: int,
    limit: int | None,
) -> list[dict]:
    rows = load_jsonl(meta_path)
    items: list[dict] = []
    for row in rows:
        qid = row.get("id")
        if qid is None:
            continue
        choices = list(row.get("choices") or [])
        answer = str(row.get("answer") or "").strip()
        if not choices or not answer:
            continue
        items.append(
            {
                "id": str(qid),
                "question": str(row.get("question") or "").strip(),
                "choices": choices,
                "answer": answer,
                "audio_path": row.get("audio_path"),
                "modality": row.get("modality"),
                "category": row.get("category"),
                "sub_category": row.get("sub-category") or row.get("sub_category"),
                "language": row.get("language"),
            }
        )
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    if limit is not None and limit > 0:
        items = items[:limit]
    return items


def audio_url_for(item: dict) -> str | None:
    audio_path = item.get("audio_path")
    if not audio_path:
        return None
    name = Path(str(audio_path)).name
    audio_dir = CONFIG.get("audio_dir")
    if not audio_dir:
        return f"/audio/{name}"
    candidate = Path(audio_dir) / name
    if candidate.is_file():
        return f"/audio/{name}"
    return None


def public_item(item: dict, *, index: int) -> dict:
    return {
        "id": item["id"],
        "question": item["question"],
        "modality": item.get("modality"),
        "category": item.get("category"),
        "sub_category": item.get("sub_category"),
        "language": item.get("language"),
        "audio_url": audio_url_for(item),
        "index": index,
        "total": len(CONFIG["items"]),
        "n_choices": len(item["choices"]),
    }


def first_unanswered_index() -> int | None:
    answered: set[str] = CONFIG["answered_ids"]
    for index, item in enumerate(CONFIG["items"]):
        if item["id"] not in answered:
            return index
    return None


def session_snapshot() -> dict:
    items: list[dict] = CONFIG["items"]
    answered: set[str] = CONFIG["answered_ids"]
    remaining = sum(1 for item in items if item["id"] not in answered)
    return {
        "participant": CONFIG["participant"],
        "answers_path": str(CONFIG["answers_path"]),
        "total": len(items),
        "answered": len(answered),
        "remaining": remaining,
        "first_unanswered_index": first_unanswered_index(),
        "start_index": CONFIG.get("start_index", first_unanswered_index() or 0),
        "shuffle": CONFIG["shuffle"],
        "seed": CONFIG["seed"],
    }


def get_item(qid: str) -> dict | None:
    return CONFIG["by_id"].get(qid)


def saved_public(record: dict, item: dict) -> dict:
    return {
        "freeform_answer": str(record.get("freeform_answer") or ""),
        "freeform_reasonable": coerce_bool(
            record.get("freeform_reasonable"), default=True
        ),
        "mc_choice": str(record.get("mc_choice") or ""),
        "mc_correct": bool(record.get("mc_correct")),
        "gold_answer": item["answer"],
        "choices": list(item["choices"]),
        "timestamp": record.get("timestamp"),
    }


def question_payload(index: int) -> dict:
    items: list[dict] = CONFIG["items"]
    if index < 0 or index >= len(items):
        raise IndexError(index)
    item = items[index]
    qid = item["id"]
    is_answered = qid in CONFIG["answered_ids"]
    payload: dict[str, Any] = {
        **session_snapshot(),
        "item": public_item(item, index=index),
        "is_answered": is_answered,
        "has_prev": index > 0,
        "has_next": index + 1 < len(items),
    }
    if is_answered:
        payload["saved"] = saved_public(CONFIG["answers_by_id"][qid], item)
        payload["stage_hint"] = "reveal"
        return payload

    pending = PENDING_FREEFORM.get(qid)
    if pending:
        payload["draft"] = {
            "freeform_answer": str(pending.get("freeform_answer") or ""),
            "freeform_reasonable": coerce_bool(
                pending.get("freeform_reasonable"), default=True
            ),
        }
        payload["choices"] = list(item["choices"])
        payload["stage_hint"] = "mc"
    else:
        payload["stage_hint"] = "freeform"
    return payload


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Take MMAR</title>
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
  .meta {
    color: var(--muted);
    font-size: 0.85rem;
    font-family: "IBM Plex Mono", monospace;
  }
  .nav {
    display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center;
  }
  main {
    max-width: 720px;
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
    padding: 1.1rem 1.2rem;
  }
  .steps {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.85rem;
  }
  .step {
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    background: #fff;
  }
  .step.active {
    color: #fff;
    background: var(--accent);
    border-color: transparent;
  }
  .step.done {
    color: var(--good);
    background: var(--soft-good);
    border-color: transparent;
  }
  .question {
    font-size: 1.25rem;
    line-height: 1.4;
    font-weight: 500;
    margin: 0.35rem 0 0.85rem;
  }
  .tags {
    display: flex; flex-wrap: wrap; gap: 0.4rem;
    margin-bottom: 0.75rem;
  }
  .tag {
    font-size: 0.75rem;
    font-family: "IBM Plex Mono", monospace;
    color: var(--muted);
    background: var(--soft);
    border-radius: 6px;
    padding: 0.15rem 0.45rem;
  }
  audio { width: 100%; margin: 0.25rem 0 0.85rem; }
  label.field {
    display: grid;
    gap: 0.35rem;
    font-size: 0.9rem;
    font-weight: 500;
  }
  textarea {
    font: inherit;
    min-height: 6.5rem;
    resize: vertical;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.65rem 0.75rem;
    background: #fff;
  }
  .choices { display: grid; gap: 0.5rem; margin: 0.5rem 0 0.85rem; }
  .choice {
    display: flex;
    gap: 0.65rem;
    align-items: flex-start;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.7rem 0.8rem;
    background: #fff;
    cursor: pointer;
  }
  .choice:hover { border-color: var(--accent); }
  .choice.selected {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--soft) 70%, #fff);
  }
  .choice input { margin-top: 0.2rem; }
  .choice .label {
    font-family: "IBM Plex Mono", monospace;
    font-weight: 500;
    color: var(--accent);
    min-width: 1.5rem;
  }
  .choice .text { flex: 1; line-height: 1.35; }
  .row-actions {
    display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
    margin-top: 0.75rem;
  }
  button, select, input[type="text"], input[type="number"] {
    font: inherit;
    border: 1px solid var(--line);
    background: var(--card);
    color: var(--ink);
    border-radius: 8px;
    padding: 0.4rem 0.75rem;
  }
  button {
    cursor: pointer;
    background: var(--accent);
    color: #fff;
    border-color: transparent;
    font-weight: 500;
  }
  button.secondary {
    background: var(--card);
    color: var(--ink);
    border-color: var(--line);
  }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .status { color: var(--muted); font-size: 0.85rem; min-height: 1.2em; }
  .status.err { color: var(--bad); }
  .status.ok { color: var(--good); }
  .reveal-box {
    border-radius: 10px;
    padding: 0.85rem 0.95rem;
    margin-top: 0.5rem;
  }
  .reveal-box.correct { background: var(--soft-good); }
  .reveal-box.wrong { background: var(--soft-bad); }
  .reveal-box h3 {
    margin: 0 0 0.35rem;
    font-size: 0.95rem;
  }
  .reveal-box p { margin: 0.25rem 0; line-height: 1.4; }
  .empty {
    padding: 2rem 1rem;
    text-align: center;
    color: var(--muted);
  }
  .freeform-echo {
    margin-top: 0.75rem;
    padding: 0.7rem 0.8rem;
    background: #fff;
    border: 1px dashed var(--line);
    border-radius: 8px;
    font-size: 0.92rem;
    line-height: 1.4;
  }
  .freeform-echo .caption {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin-bottom: 0.25rem;
  }
  label.check {
    display: flex;
    gap: 0.55rem;
    align-items: flex-start;
    margin-top: 0.75rem;
    font-size: 0.9rem;
    font-weight: 400;
    line-height: 1.35;
    cursor: pointer;
  }
  label.check input { margin-top: 0.2rem; }
  .banner {
    font-size: 0.85rem;
    color: var(--muted);
    background: var(--soft);
    border-radius: 8px;
    padding: 0.45rem 0.7rem;
    margin-bottom: 0.75rem;
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>Take MMAR</h1>
    <div class="meta" id="sessionMeta">Loading…</div>
  </div>
  <div class="nav">
    <button type="button" class="secondary" id="prevBtn" disabled>Prev</button>
    <span class="meta" id="positionMeta">—</span>
    <button type="button" class="secondary" id="nextBtn" disabled>Next</button>
    <label class="meta"># <input type="number" id="jumpInput" min="1" style="width:4.2rem" /></label>
    <button type="button" class="secondary" id="jumpBtn">Go</button>
    <button type="button" class="secondary" id="unansweredBtn">Next unanswered</button>
  </div>
</header>
<main>
  <section class="panel" id="quiz" hidden>
    <div class="banner" id="banner" hidden></div>
    <div class="steps">
      <span class="step" id="step1">1 Freeform</span>
      <span class="step" id="step2">2 Multiple choice</span>
      <span class="step" id="step3">3 Reveal</span>
    </div>
    <div class="tags" id="tags"></div>
    <div class="question" id="question"></div>
    <audio id="player" controls preload="metadata"></audio>

    <label class="field">Your answer
      <textarea id="freeform" placeholder="Listen, then write your best answer…"></textarea>
    </label>
    <label class="check">
      <input type="checkbox" id="freeformReasonable" checked />
      <span>This question can reasonably be answered in free-response form (without seeing the choices).</span>
    </label>

    <div id="mcSection" hidden>
      <div class="choices" id="choices" style="margin-top:0.85rem"></div>
    </div>

    <div id="revealSection" hidden>
      <div class="reveal-box" id="revealBox">
        <h3 id="revealTitle"></h3>
        <p><strong>Correct answer:</strong> <span id="goldAnswer"></span></p>
      </div>
    </div>

    <div class="row-actions">
      <button type="button" id="primaryBtn">Submit freeform</button>
      <button type="button" class="secondary" id="continueBtn" hidden>Next unanswered</button>
      <span class="status" id="status"></span>
    </div>
  </section>

  <section class="panel empty" id="empty">Loading…</section>
</main>
<script>
const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const state = {
  index: 0,
  total: 0,
  item: null,
  answered: false,
  unlockedChoices: false,
  freeform: "",
  freeformReasonable: true,
  choices: [],
  mcChoice: null,
  gold: null,
  correct: null,
  firstUnanswered: null,
  saveTimer: null,
  suppressAutosave: false,
};

async function api(path, opts) {
  const res = await fetch(path, opts);
  const ct = res.headers.get("content-type") || "";
  const payload = ct.includes("application/json")
    ? await res.json()
    : await res.text();
  if (!res.ok) {
    const msg = (payload && payload.error) || payload || res.statusText;
    throw new Error(msg);
  }
  return payload;
}

function setSteps() {
  let cur = 1;
  if (state.unlockedChoices) cur = 2;
  if (state.answered) cur = 3;
  for (let i = 1; i <= 3; i++) {
    const el = document.getElementById("step" + i);
    el.classList.remove("active", "done");
    if (i < cur) el.classList.add("done");
    if (i === cur) el.classList.add("active");
  }
}

function setStatus(text, kind) {
  const el = document.getElementById("status");
  el.textContent = text || "";
  el.className = "status" + (kind ? " " + kind : "");
}

function renderTags(item) {
  const tags = document.getElementById("tags");
  tags.innerHTML = "";
  for (const [label, value] of [
    ["modality", item.modality],
    ["category", item.category],
    ["sub", item.sub_category],
    ["lang", item.language],
  ]) {
    if (!value) continue;
    const span = document.createElement("span");
    span.className = "tag";
    span.textContent = `${label}: ${value}`;
    tags.appendChild(span);
  }
}

function renderChoices(choices, selected) {
  const root = document.getElementById("choices");
  root.innerHTML = "";
  state.mcChoice = selected || null;
  choices.forEach((text, i) => {
    const label = document.createElement("label");
    label.className = "choice";
    if (selected === text) label.classList.add("selected");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "mc";
    input.value = text;
    input.checked = selected === text;
    input.addEventListener("change", () => {
      state.mcChoice = text;
      document.querySelectorAll(".choice").forEach((el) => el.classList.remove("selected"));
      label.classList.add("selected");
      syncPrimaryButton();
      if (state.answered) queueAutosave();
    });
    const lab = document.createElement("span");
    lab.className = "label";
    lab.textContent = `(${LABELS[i] || i})`;
    const body = document.createElement("span");
    body.className = "text";
    body.textContent = text;
    label.appendChild(input);
    label.appendChild(lab);
    label.appendChild(body);
    root.appendChild(label);
  });
}

function syncPrimaryButton() {
  const btn = document.getElementById("primaryBtn");
  if (state.answered) {
    btn.hidden = true;
    return;
  }
  btn.hidden = false;
  if (!state.unlockedChoices) {
    btn.textContent = "Submit freeform";
    btn.disabled = false;
  } else {
    btn.textContent = "Submit choice";
    btn.disabled = !state.mcChoice;
  }
}

function showReveal() {
  const box = document.getElementById("revealBox");
  box.className = "reveal-box " + (state.correct ? "correct" : "wrong");
  document.getElementById("revealTitle").textContent = state.correct
    ? "Correct"
    : "Incorrect";
  document.getElementById("goldAnswer").textContent = state.gold;
  document.getElementById("revealSection").hidden = false;
}

function updateFormVisibility() {
  document.getElementById("mcSection").hidden = !state.unlockedChoices;
  document.getElementById("revealSection").hidden = !state.answered;
  document.getElementById("continueBtn").hidden = state.firstUnanswered == null;
  syncPrimaryButton();
  setSteps();
}

function updateNav(data) {
  state.total = data.total;
  state.firstUnanswered = data.first_unanswered_index;
  document.getElementById("sessionMeta").textContent =
    `participant=${data.participant || "anon"} · ${data.answered}/${data.total} answered · answers=${data.answers_path}`;
  document.getElementById("positionMeta").textContent =
    `Q ${state.index + 1} / ${state.total}`;
  document.getElementById("jumpInput").value = String(state.index + 1);
  document.getElementById("prevBtn").disabled = !data.has_prev;
  document.getElementById("nextBtn").disabled = !data.has_next;
  document.getElementById("unansweredBtn").disabled = data.first_unanswered_index == null;
  document.getElementById("continueBtn").hidden = data.first_unanswered_index == null;
  const banner = document.getElementById("banner");
  if (data.remaining === 0) {
    banner.hidden = false;
    banner.textContent = "All questions answered. Edits save automatically.";
  } else {
    banner.hidden = true;
  }
}

function readForm() {
  return {
    freeform: document.getElementById("freeform").value.trim(),
    freeformReasonable: document.getElementById("freeformReasonable").checked,
    mcChoice: state.mcChoice,
  };
}

function fillForm() {
  state.suppressAutosave = true;
  document.getElementById("freeform").value = state.freeform || "";
  document.getElementById("freeformReasonable").checked = !!state.freeformReasonable;
  state.suppressAutosave = false;
}

async function loadIndex(index) {
  if (state.saveTimer) {
    clearTimeout(state.saveTimer);
    state.saveTimer = null;
  }
  setStatus("");
  const data = await api("/api/question?index=" + encodeURIComponent(index));
  state.index = data.item.index;
  state.item = data.item;
  state.answered = !!data.is_answered;
  state.unlockedChoices = false;
  state.choices = [];
  state.mcChoice = null;
  state.gold = null;
  state.correct = null;

  document.getElementById("quiz").hidden = false;
  document.getElementById("empty").hidden = true;
  document.getElementById("question").textContent = data.item.question;
  renderTags(data.item);
  updateNav(data);

  const player = document.getElementById("player");
  if (data.item.audio_url) {
    player.src = data.item.audio_url;
    player.load();
  } else {
    player.removeAttribute("src");
    player.load();
  }

  if (data.is_answered && data.saved) {
    state.freeform = data.saved.freeform_answer || "";
    state.freeformReasonable = !!data.saved.freeform_reasonable;
    state.choices = data.saved.choices || [];
    state.mcChoice = data.saved.mc_choice || null;
    state.gold = data.saved.gold_answer;
    state.correct = !!data.saved.mc_correct;
    state.unlockedChoices = true;
    fillForm();
    renderChoices(state.choices, state.mcChoice);
    showReveal();
    updateFormVisibility();
    return;
  }

  if (data.draft) {
    state.freeform = data.draft.freeform_answer || "";
    state.freeformReasonable = data.draft.freeform_reasonable !== false;
    state.choices = data.choices || [];
    state.unlockedChoices = true;
    fillForm();
    renderChoices(state.choices, null);
    updateFormVisibility();
    return;
  }

  state.freeform = "";
  state.freeformReasonable = true;
  fillForm();
  updateFormVisibility();
}

async function boot() {
  const session = await api("/api/session");
  if (!session.total) {
    document.getElementById("empty").textContent = "No questions in this set.";
    return;
  }
  const start =
    session.start_index != null
      ? session.start_index
      : session.first_unanswered_index != null
        ? session.first_unanswered_index
        : Math.max(0, session.total - 1);
  await loadIndex(start);
}

async function submitFreeform() {
  if (!state.item) return;
  const form = readForm();
  if (!form.freeform) {
    setStatus("Write something first.", "err");
    return;
  }
  setStatus("Saving…");
  try {
    const data = await api("/api/freeform", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: state.item.id,
        freeform_answer: form.freeform,
        freeform_reasonable: form.freeformReasonable,
      }),
    });
    state.freeform = form.freeform;
    state.freeformReasonable = form.freeformReasonable;
    state.choices = data.choices || [];
    state.unlockedChoices = true;
    renderChoices(state.choices, null);
    updateFormVisibility();
    setStatus("");
  } catch (err) {
    setStatus(String(err.message || err), "err");
  }
}

async function saveAnswer({ quiet } = {}) {
  if (!state.item) return;
  const form = readForm();
  if (!form.freeform || !form.mcChoice) {
    if (!quiet) setStatus("Freeform and a choice are required.", "err");
    return;
  }
  if (!quiet) setStatus("Saving…");
  try {
    // Unlock gate for first-time save path.
    if (!state.answered) {
      await api("/api/freeform", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: state.item.id,
          freeform_answer: form.freeform,
          freeform_reasonable: form.freeformReasonable,
        }),
      });
    }
    const data = await api("/api/mc", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: state.item.id,
        freeform_answer: form.freeform,
        freeform_reasonable: form.freeformReasonable,
        mc_choice: form.mcChoice,
      }),
    });
    state.freeform = form.freeform;
    state.freeformReasonable = form.freeformReasonable;
    state.mcChoice = form.mcChoice;
    state.gold = data.gold_answer;
    state.correct = !!data.mc_correct;
    state.answered = true;
    state.unlockedChoices = true;
    showReveal();
    updateFormVisibility();
    setStatus(data.updated ? "Updated" : "Saved", "ok");
    const session = await api("/api/session");
    updateNav({
      ...session,
      has_prev: state.index > 0,
      has_next: state.index + 1 < state.total,
      is_answered: true,
    });
  } catch (err) {
    setStatus(String(err.message || err), "err");
  }
}

function queueAutosave() {
  if (state.suppressAutosave || !state.answered) return;
  if (state.saveTimer) clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => {
    state.saveTimer = null;
    saveAnswer({ quiet: true });
  }, 350);
}

async function onPrimary() {
  if (state.answered) return;
  if (!state.unlockedChoices) {
    await submitFreeform();
    return;
  }
  await saveAnswer();
}

document.getElementById("primaryBtn").onclick = () => onPrimary();
document.getElementById("freeform").addEventListener("input", () => {
  if (state.answered) queueAutosave();
});
document.getElementById("freeformReasonable").addEventListener("change", () => {
  if (state.answered) queueAutosave();
});
function goPrev() {
  if (state.index > 0) loadIndex(state.index - 1).catch(showBootError);
}
function goNext() {
  if (state.index + 1 < state.total) loadIndex(state.index + 1).catch(showBootError);
}
document.getElementById("prevBtn").onclick = () => goPrev();
document.getElementById("nextBtn").onclick = () => goNext();
document.addEventListener("keydown", (event) => {
  if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
    return;
  }
  const tag = (event.target && event.target.tagName) || "";
  if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT" || (event.target && event.target.isContentEditable)) {
    return;
  }
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    goPrev();
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    goNext();
  }
});
document.getElementById("jumpBtn").onclick = () => {
  const n = parseInt(document.getElementById("jumpInput").value, 10);
  if (!n || n < 1 || n > state.total) return;
  loadIndex(n - 1).catch(showBootError);
};
document.getElementById("unansweredBtn").onclick = () => {
  if (state.firstUnanswered == null) return;
  loadIndex(state.firstUnanswered).catch(showBootError);
};
document.getElementById("continueBtn").onclick = () => {
  if (state.firstUnanswered == null) return;
  loadIndex(state.firstUnanswered).catch(showBootError);
};

function showBootError(err) {
  document.getElementById("empty").hidden = false;
  document.getElementById("quiz").hidden = true;
  document.getElementById("empty").textContent = String(err.message || err);
}

boot().catch(showBootError);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        print(f"[take-mmar] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/session":
            with STATE_LOCK:
                self._send_json(session_snapshot())
            return

        if path == "/api/question":
            raw = (qs.get("index") or [""])[0]
            try:
                index = int(raw)
            except ValueError:
                self._send_json({"error": "index must be an integer"}, 400)
                return
            with STATE_LOCK:
                try:
                    self._send_json(question_payload(index))
                except IndexError:
                    self._send_json({"error": "index out of range"}, 404)
            return

        # Back-compat alias: first unanswered (or last if complete).
        if path == "/api/next":
            with STATE_LOCK:
                index = first_unanswered_index()
                if index is None:
                    index = max(0, len(CONFIG["items"]) - 1)
                    if not CONFIG["items"]:
                        self._send_json({"done": True, **session_snapshot()})
                        return
                self._send_json({"done": False, **question_payload(index)})
            return

        if path.startswith("/audio/"):
            name = unquote(path[len("/audio/") :])
            if Path(name).name != name:
                self.send_error(400, "bad path")
                return
            audio = Path(CONFIG["audio_dir"]) / name
            if not audio.is_file():
                self.send_error(404, "audio not found")
                return
            data = audio.read_bytes()
            ctype = mimetypes.guess_type(str(audio))[0] or "audio/wav"
            self._send(200, data, ctype)
            return

        self.send_error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            payload = self._read_json()
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"invalid json: {exc}"}, 400)
            return

        if path == "/api/freeform":
            qid = str(payload.get("id") or "").strip()
            freeform = str(payload.get("freeform_answer") or "").strip()
            freeform_reasonable = coerce_bool(
                payload.get("freeform_reasonable"), default=True
            )
            if not qid or not freeform:
                self._send_json({"error": "id and freeform_answer required"}, 400)
                return
            with STATE_LOCK:
                item = get_item(qid)
                if item is None:
                    self._send_json({"error": "unknown id"}, 404)
                    return
                PENDING_FREEFORM[qid] = {
                    "freeform_answer": freeform,
                    "freeform_reasonable": freeform_reasonable,
                }
                self._send_json(
                    {
                        "id": qid,
                        "choices": list(item["choices"]),
                        "freeform_reasonable": freeform_reasonable,
                        "updating": qid in CONFIG["answered_ids"],
                    }
                )
            return

        if path == "/api/mc":
            qid = str(payload.get("id") or "").strip()
            freeform = str(payload.get("freeform_answer") or "").strip()
            mc_choice = str(payload.get("mc_choice") or "").strip()
            freeform_reasonable = coerce_bool(
                payload.get("freeform_reasonable"), default=True
            )
            if not qid or not freeform or not mc_choice:
                self._send_json(
                    {"error": "id, freeform_answer, and mc_choice required"},
                    400,
                )
                return
            with STATE_LOCK:
                item = get_item(qid)
                if item is None:
                    self._send_json({"error": "unknown id"}, 404)
                    return
                pending = PENDING_FREEFORM.get(qid) or {}
                pending_text = str(pending.get("freeform_answer") or "").strip()
                # Allow direct update of an already-saved answer without a
                # fresh freeform unlock (e.g. race / refresh), but prefer pending.
                if pending_text:
                    freeform = pending_text
                    if "freeform_reasonable" in pending:
                        freeform_reasonable = coerce_bool(
                            pending.get("freeform_reasonable"), default=True
                        )
                elif qid not in CONFIG["answered_ids"]:
                    self._send_json(
                        {"error": "submit freeform answer before multiple choice"},
                        400,
                    )
                    return
                if mc_choice not in item["choices"]:
                    self._send_json({"error": "mc_choice not in choices"}, 400)
                    return
                gold = item["answer"]
                correct = mc_choice == gold
                updated = qid in CONFIG["answered_ids"]
                record = {
                    "id": qid,
                    "participant": CONFIG["participant"],
                    "question": item["question"],
                    "freeform_answer": freeform,
                    "freeform_reasonable": freeform_reasonable,
                    "mc_choice": mc_choice,
                    "mc_correct": correct,
                    "gold_answer": gold,
                    "choices": list(item["choices"]),
                    "modality": item.get("modality"),
                    "category": item.get("category"),
                    "sub_category": item.get("sub_category"),
                    "language": item.get("language"),
                    "timestamp": utc_now(),
                }
                upsert_answer(record)
                PENDING_FREEFORM.pop(qid, None)
                self._send_json(
                    {
                        "id": qid,
                        "gold_answer": gold,
                        "mc_correct": correct,
                        "freeform_reasonable": freeform_reasonable,
                        "saved": True,
                        "updated": updated,
                        "answers_path": str(CONFIG["answers_path"]),
                    }
                )
            return

        self._send_json({"error": "not found"}, 404)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_META,
        help="Path to MMAR-meta.jsonl",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Local MMAR wav directory",
    )
    parser.add_argument(
        "--answers-path",
        type=Path,
        default=None,
        help="JSONL file to upsert answers into "
        "(default: outputs/human-mmar/answers.jsonl or "
        "outputs/human-mmar/<participant>.jsonl)",
    )
    parser.add_argument(
        "--participant",
        default="",
        help="Optional participant label stored with each answer",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle question order (deterministic with --seed)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only present the first N questions after shuffle/order",
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
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Ignore already-saved answers when choosing the starting question "
        "(answers remain on disk and can still be reviewed/updated)",
    )
    return parser.parse_args()


def resolve_answers_path(args: argparse.Namespace) -> Path:
    if args.answers_path is not None:
        return args.answers_path.expanduser().resolve()
    DEFAULT_ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    participant = (args.participant or "").strip()
    if participant:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in participant)
        return (DEFAULT_ANSWERS_DIR / f"{safe}.jsonl").resolve()
    return (DEFAULT_ANSWERS_DIR / "answers.jsonl").resolve()


def main() -> None:
    args = parse_args()
    meta_path = args.meta.expanduser().resolve()
    if not meta_path.is_file():
        raise SystemExit(f"MMAR meta not found: {meta_path}")

    audio_dir = args.audio_dir.expanduser().resolve()
    if not args.skip_audio_download:
        try:
            audio_dir = ensure_mmar_audio(
                audio_dir, force=args.force_audio_download
            )
        except SystemExit as exc:
            print(f"Audio setup failed: {exc}", flush=True)
            print("Continuing without ensuring audio; pass --skip-audio-download to silence.")

    items = load_items(
        meta_path,
        shuffle=args.shuffle,
        seed=args.seed,
        limit=args.limit,
    )
    if not items:
        raise SystemExit(f"No usable MMAR items in {meta_path}")

    answers_path = resolve_answers_path(args)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    answers_by_id = load_answers_by_id(answers_path)
    active_ids = {item["id"] for item in items}
    # Track answered status only for the active set; keep all file rows for upsert.
    answered_ids = {qid for qid in answers_by_id if qid in active_ids}

    CONFIG["items"] = items
    CONFIG["by_id"] = {item["id"]: item for item in items}
    CONFIG["answers_by_id"] = answers_by_id
    CONFIG["answered_ids"] = answered_ids
    CONFIG["answers_path"] = answers_path
    CONFIG["audio_dir"] = audio_dir
    CONFIG["participant"] = (args.participant or "").strip()
    CONFIG["shuffle"] = bool(args.shuffle)
    CONFIG["seed"] = args.seed
    # Prefer question 0 when --redo; otherwise first unanswered (or last if done).
    CONFIG["start_index"] = (
        0
        if args.redo
        else (
            first_unanswered_index()
            if first_unanswered_index() is not None
            else max(0, len(items) - 1)
        )
    )

    remaining = sum(1 for item in items if item["id"] not in answered_ids)
    print(f"Meta:     {meta_path}")
    print(f"Audio:    {audio_dir}")
    print(f"Answers:  {answers_path}")
    print(
        f"Set:      {len(items)} questions "
        f"({len(answered_ids)} already answered, {remaining} remaining)"
    )
    if CONFIG["participant"]:
        print(f"Participant: {CONFIG['participant']}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
