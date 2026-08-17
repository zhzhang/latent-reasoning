"""Label whether MMAR open-ended questions have one correct answer, and which
model generations a judge should accept.

All raters use the same frozen 100-question sample. Audio and question text
only — no gold answers, no MCQ choices, no model names.

Flow per question:
  1. Play audio + read the question → unique vs multiple possible answers
  2. If multiple: check which unique generation strings a judge should mark
     correct (zero selections allowed)

Usage::

    uv run python label_answers.py --participant jordan
    uv run python label_answers.py --participant jordan --port 7864
    uv run python label_answers.py --build-sample
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import random
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from collate_mmar_freeform import DEFAULT_OUT_DIR
from mmar_common import load_question_ids_csv, write_json
from view_mmar import (
    DEFAULT_AUDIO_DIR,
    LABEL_ORDER,
    ensure_mmar_audio,
    load_pack,
)

REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = REPO_ROOT / "answer-variety"
OPEN_ENDED_IDS_PATH = PACKAGE_DIR / "open_ended_question_ids.csv"
SAMPLE_IDS_PATH = PACKAGE_DIR / "label_sample_question_ids.csv"
SAMPLE_JSON_PATH = PACKAGE_DIR / "label_sample.json"
DEFAULT_LABELS_DIR = REPO_ROOT / "outputs" / "answer-labels"

SAMPLE_N = 100
SAMPLE_SEED = 42

CONFIG: dict[str, Any] = {}
STATE_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_id_csv(path: Path, ids: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id"])
        for qid in ids:
            writer.writerow([qid])
    return path


def unique_answer_key(text: str) -> str:
    return " ".join(str(text or "").split())


def ordered_model_labels(labels: list[str]) -> list[str]:
    found = list(dict.fromkeys(str(x) for x in labels if x))
    known = [label for label in LABEL_ORDER if label in found]
    rest = sorted(label for label in found if label not in set(LABEL_ORDER))
    return known + rest


def _shot_index(shot: dict[str, Any]) -> int:
    raw = shot.get("shot_index")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def unique_answers_for(qid: str, pack: dict[str, Any]) -> list[str]:
    """First-seen unique ``answer_prediction`` strings for ``qid``."""
    predictions = pack.get("predictions") or {}
    labels = ordered_model_labels(list(pack.get("model_labels") or []))
    seen: set[str] = set()
    answers: list[str] = []
    for label in labels:
        record = (predictions.get(label) or {}).get(qid)
        if not record:
            continue
        shots = list(record.get("shots") or [])
        shots.sort(key=_shot_index)
        for shot in shots:
            original = str(shot.get("answer_prediction") or "")
            key = unique_answer_key(original)
            if not key or key in seen:
                continue
            seen.add(key)
            answers.append(original)
    return answers


def question_fields_for(qid: str, pack: dict[str, Any]) -> tuple[str, str]:
    row = (pack.get("by_id") or {}).get(qid) or {}
    question = str(row.get("question") or "").strip()
    audio_path = Path(str(row.get("audio_path") or "")).name
    if question and audio_path:
        return question, audio_path
    predictions = pack.get("predictions") or {}
    for label in ordered_model_labels(list(pack.get("model_labels") or [])):
        record = (predictions.get(label) or {}).get(qid) or {}
        if not question:
            question = str(record.get("question") or "").strip()
        if not audio_path:
            audio_path = Path(str(record.get("audio_path") or "")).name
        if question and audio_path:
            break
    return question, audio_path


def build_sample(
    *,
    source_ids_csv: Path,
    pack_dir: Path,
    ids_out: Path,
    json_out: Path,
    n: int = SAMPLE_N,
    seed: int = SAMPLE_SEED,
) -> dict[str, Any]:
    source_ids = load_question_ids_csv(source_ids_csv)
    if len(source_ids) < n:
        raise SystemExit(
            f"Need at least {n} open-ended ids in {source_ids_csv}, "
            f"found {len(source_ids)}"
        )
    sampled = list(source_ids)
    random.Random(seed).shuffle(sampled)
    sampled = sampled[:n]

    pack = load_pack(str(pack_dir.expanduser().resolve()))
    items: list[dict[str, Any]] = []
    for qid in sampled:
        question, audio_path = question_fields_for(qid, pack)
        items.append(
            {
                "id": qid,
                "question": question,
                "audio_path": audio_path,
                "answers": unique_answers_for(qid, pack),
            }
        )

    payload = {
        "seed": seed,
        "n": len(items),
        "source_ids_csv": _relpath(source_ids_csv),
        "pack_dir": _relpath(pack_dir),
        "unique_key": "whitespace-collapsed answer_prediction (case-sensitive)",
        "order": (
            "first appearance: LABEL_ORDER models, then remaining alpha, "
            "shots by shot_index"
        ),
        "items": items,
    }
    write_id_csv(ids_out, sampled)
    write_json(json_out, payload)
    return payload


def load_sample(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(
            f"Frozen sample not found: {path}\n"
            "Generate it with: uv run python label_answers.py --build-sample"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise SystemExit(f"No items in {path}")
    out: list[dict[str, Any]] = []
    for raw in items:
        qid = str((raw or {}).get("id") or "").strip()
        if not qid:
            continue
        answers = [str(text) for text in (raw.get("answers") or [])]
        out.append(
            {
                "id": qid,
                "question": str(raw.get("question") or "").strip(),
                "audio_path": Path(str(raw.get("audio_path") or "")).name,
                "answers": answers,
            }
        )
    if not out:
        raise SystemExit(f"No usable items in {path}")
    return out


def load_labels_by_id(labels_path: Path) -> dict[str, dict]:
    if not labels_path.is_file():
        return {}
    by_id: dict[str, dict] = {}
    with open(labels_path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            qid = row.get("id")
            if qid is None:
                continue
            by_id[str(qid)] = row
    return by_id


def upsert_label(record: dict) -> None:
    labels_path: Path = CONFIG["labels_path"]
    labels_by_id: dict[str, dict] = CONFIG["labels_by_id"]
    qid = str(record["id"])
    labels_by_id[qid] = record
    CONFIG["labeled_ids"].add(qid)

    ordered_ids = [item["id"] for item in CONFIG["items"] if item["id"] in labels_by_id]
    extras = [qid_ for qid_ in labels_by_id if qid_ not in set(ordered_ids)]
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as handle:
        for qid_ in ordered_ids + extras:
            handle.write(json.dumps(labels_by_id[qid_], ensure_ascii=False) + "\n")


def audio_url_for(item: dict) -> str | None:
    name = str(item.get("audio_path") or "").strip()
    if not name or Path(name).name != name:
        return None
    audio_dir = CONFIG.get("audio_dir")
    if not audio_dir:
        return f"/audio/{name}"
    candidate = Path(audio_dir) / name
    if candidate.is_file():
        return f"/audio/{name}"
    return None


def public_item(item: dict, *, index: int, include_answers: bool) -> dict:
    payload = {
        "id": item["id"],
        "question": item["question"],
        "audio_url": audio_url_for(item),
        "index": index,
        "total": len(CONFIG["items"]),
        "n_answers": len(item["answers"]),
    }
    if include_answers:
        payload["answers"] = list(item["answers"])
    return payload


def first_unlabeled_index() -> int | None:
    labeled: set[str] = CONFIG["labeled_ids"]
    for index, item in enumerate(CONFIG["items"]):
        if item["id"] not in labeled:
            return index
    return None


def session_snapshot() -> dict:
    items: list[dict] = CONFIG["items"]
    labeled: set[str] = CONFIG["labeled_ids"]
    remaining = sum(1 for item in items if item["id"] not in labeled)
    return {
        "participant": CONFIG["participant"],
        "labels_path": str(CONFIG["labels_path"]),
        "total": len(items),
        "labeled": len(labeled),
        "remaining": remaining,
        "first_unlabeled_index": first_unlabeled_index(),
        "start_index": CONFIG.get("start_index", first_unlabeled_index() or 0),
    }


def saved_public(record: dict) -> dict:
    cardinality = str(record.get("answer_cardinality") or "").strip()
    if cardinality not in {"unique", "multiple"}:
        cardinality = ""
    indices = [
        int(i)
        for i in (record.get("accepted_answer_indices") or [])
        if isinstance(i, int) or str(i).isdigit()
    ]
    answers = [str(text) for text in (record.get("accepted_answers") or [])]
    return {
        "answer_cardinality": cardinality,
        "accepted_answer_indices": indices,
        "accepted_answers": answers,
        "timestamp": record.get("timestamp"),
    }


def question_payload(index: int) -> dict:
    items: list[dict] = CONFIG["items"]
    if index < 0 or index >= len(items):
        raise IndexError(index)
    item = items[index]
    qid = item["id"]
    is_labeled = qid in CONFIG["labeled_ids"]
    saved = None
    if is_labeled:
        saved = saved_public(CONFIG["labels_by_id"][qid])
    include_answers = bool(
        saved and saved.get("answer_cardinality") == "multiple"
    )
    payload: dict[str, Any] = {
        **session_snapshot(),
        "item": public_item(item, index=index, include_answers=include_answers),
        "is_labeled": is_labeled,
        "has_prev": index > 0,
        "has_next": index + 1 < len(items),
    }
    if saved is not None:
        payload["saved"] = saved
    return payload


def parse_cardinality(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"unique", "single", "one"}:
        return "unique"
    if text in {"multiple", "multi", "varied"}:
        return "multiple"
    return None


def normalize_accepted(
    item: dict,
    *,
    cardinality: str,
    raw_indices: Any,
) -> tuple[list[int], list[str]]:
    if cardinality == "unique":
        return [], []
    answers = list(item["answers"])
    indices: list[int] = []
    seen: set[int] = set()
    for raw in raw_indices or []:
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(answers) or index in seen:
            continue
        seen.add(index)
        indices.append(index)
    indices.sort()
    return indices, [answers[i] for i in indices]


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Label MMAR answers</title>
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
    padding: 0.2rem 0.55rem;
  }
  .step.active { color: var(--accent); border-color: #8fb3c9; background: #e2eef6; }
  .step.done { color: var(--good); border-color: #b7dcc8; background: var(--soft-good); }
  .question {
    font-size: 1.15rem;
    line-height: 1.45;
    margin: 0.35rem 0 0.75rem;
  }
  audio { width: 100%; margin: 0.35rem 0 0.85rem; }
  .field-label {
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 0.45rem;
  }
  .choices {
    display: grid;
    gap: 0.45rem;
  }
  label.choice, label.check {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.7rem 0.8rem;
    background: #fff;
    cursor: pointer;
    line-height: 1.4;
  }
  label.choice:hover, label.check:hover { background: #eef5fa; }
  label.choice.selected {
    border-color: #8fb3c9;
    background: #e2eef6;
  }
  label.choice input, label.check input { margin-top: 0.2rem; }
  .choice .title { font-weight: 500; }
  .choice .hint {
    display: block;
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 0.15rem;
  }
  .answer-text {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.9rem;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .row-actions {
    display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
    margin-top: 0.85rem;
  }
  button, select, input[type="number"] {
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
  .banner {
    font-size: 0.85rem;
    color: var(--muted);
    background: var(--soft);
    border-radius: 8px;
    padding: 0.45rem 0.7rem;
    margin-bottom: 0.75rem;
  }
  .empty {
    padding: 2rem 1rem;
    text-align: center;
    color: var(--muted);
  }
  .muted { color: var(--muted); font-size: 0.9rem; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Label MMAR answers</h1>
    <div class="meta" id="sessionMeta">Loading…</div>
  </div>
  <div class="nav">
    <button type="button" class="secondary" id="prevBtn" disabled>Prev</button>
    <span class="meta" id="positionMeta">—</span>
    <button type="button" class="secondary" id="nextBtn" disabled>Next</button>
    <label class="meta"># <input type="number" id="jumpInput" min="1" style="width:4.2rem" /></label>
    <button type="button" class="secondary" id="jumpBtn">Go</button>
    <button type="button" class="secondary" id="unansweredBtn">Next unlabeled</button>
  </div>
</header>
<main>
  <section class="panel" id="quiz" hidden>
    <div class="banner" id="banner" hidden></div>
    <div class="steps">
      <span class="step" id="step1">1 Unique or multiple</span>
      <span class="step" id="step2">2 Acceptable answers</span>
    </div>
    <div class="question" id="question"></div>
    <audio id="player" controls preload="metadata"></audio>

    <p class="field-label">This question</p>
    <div class="choices" id="cardinality">
      <label class="choice" id="choiceUnique">
        <input type="radio" name="cardinality" value="unique" />
        <span>
          <span class="title">Only one possible correct answer</span>
          <span class="hint">A judge should accept one specific answer (wording can vary).</span>
        </span>
      </label>
      <label class="choice" id="choiceMultiple">
        <input type="radio" name="cardinality" value="multiple" />
        <span>
          <span class="title">Different answers might be possible</span>
          <span class="hint">More than one distinct answer could reasonably be correct.</span>
        </span>
      </label>
    </div>

    <div id="answersSection" hidden>
      <p class="field-label" style="margin-top:1rem">Which generations should a judge mark correct?</p>
      <p class="muted" id="answersHint">Select every string that is actually correct given the audio. None is allowed.</p>
      <div class="choices" id="answers"></div>
    </div>

    <div class="row-actions">
      <button type="button" id="primaryBtn">Save</button>
      <button type="button" class="secondary" id="continueBtn" hidden>Next unlabeled</button>
      <span class="status" id="status"></span>
    </div>
  </section>
  <section class="panel empty" id="empty">Loading…</section>
</main>
<script>
const state = {
  index: 0,
  total: 0,
  item: null,
  labeled: false,
  cardinality: null,
  accepted: [],
  answers: [],
  firstUnlabeled: null,
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
  const el1 = document.getElementById("step1");
  const el2 = document.getElementById("step2");
  el1.classList.remove("active", "done");
  el2.classList.remove("active", "done");
  if (!state.cardinality) {
    el1.classList.add("active");
    return;
  }
  el1.classList.add("done");
  if (state.cardinality === "unique") {
    el2.classList.add("done");
    return;
  }
  el2.classList.add(state.labeled ? "done" : "active");
}

function setStatus(text, kind) {
  const el = document.getElementById("status");
  el.textContent = text || "";
  el.className = "status" + (kind ? " " + kind : "");
}

function updateNav(data) {
  state.total = data.total;
  state.firstUnlabeled = data.first_unlabeled_index;
  document.getElementById("sessionMeta").textContent =
    `participant=${data.participant || "anon"} · ${data.labeled}/${data.total} labeled · labels=${data.labels_path}`;
  document.getElementById("positionMeta").textContent =
    `Q ${state.index + 1} / ${state.total}`;
  document.getElementById("jumpInput").value = String(state.index + 1);
  document.getElementById("prevBtn").disabled = !data.has_prev;
  document.getElementById("nextBtn").disabled = !data.has_next;
  document.getElementById("unansweredBtn").disabled = data.first_unlabeled_index == null;
  document.getElementById("continueBtn").hidden = data.first_unlabeled_index == null;
  const banner = document.getElementById("banner");
  if (data.remaining === 0) {
    banner.hidden = false;
    banner.textContent = "All questions labeled. Edits save automatically.";
  } else {
    banner.hidden = true;
  }
}

function syncCardinalityUi() {
  document.getElementById("choiceUnique").classList.toggle("selected", state.cardinality === "unique");
  document.getElementById("choiceMultiple").classList.toggle("selected", state.cardinality === "multiple");
  document.querySelectorAll('input[name="cardinality"]').forEach((el) => {
    el.checked = el.value === state.cardinality;
  });
  const showAnswers = state.cardinality === "multiple";
  document.getElementById("answersSection").hidden = !showAnswers;
  document.getElementById("primaryBtn").disabled = !state.cardinality;
  setSteps();
}

function renderAnswers(answers, selected) {
  const root = document.getElementById("answers");
  root.innerHTML = "";
  const selectedSet = new Set(selected || []);
  state.answers = answers || [];
  if (!state.answers.length) {
    root.innerHTML = `<p class="muted">No model generations in the frozen sample for this question.</p>`;
    return;
  }
  state.answers.forEach((text, i) => {
    const label = document.createElement("label");
    label.className = "check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = String(i);
    input.checked = selectedSet.has(i);
    input.addEventListener("change", () => {
      if (input.checked) {
        if (!state.accepted.includes(i)) state.accepted.push(i);
      } else {
        state.accepted = state.accepted.filter((x) => x !== i);
      }
      state.accepted.sort((a, b) => a - b);
      if (state.labeled) queueAutosave();
    });
    const body = document.createElement("span");
    body.className = "answer-text";
    body.textContent = text;
    label.appendChild(input);
    label.appendChild(body);
    root.appendChild(label);
  });
}

async function ensureAnswersLoaded() {
  if (state.answers && state.answers.length) return state.answers;
  if (!state.item) return [];
  const data = await api("/api/answers?id=" + encodeURIComponent(state.item.id));
  state.answers = data.answers || [];
  return state.answers;
}

async function onCardinality(value) {
  const previous = state.cardinality;
  state.cardinality = value;
  if (value === "unique") {
    state.accepted = [];
  }
  syncCardinalityUi();
  if (value === "multiple" && previous !== "multiple") {
    const answers = await ensureAnswersLoaded();
    renderAnswers(answers, state.accepted);
    syncCardinalityUi();
  }
  if (state.labeled) queueAutosave();
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
  state.labeled = !!data.is_labeled;
  state.cardinality = (data.saved && data.saved.answer_cardinality) || null;
  state.accepted = (data.saved && data.saved.accepted_answer_indices) || [];
  state.answers = data.item.answers || [];

  document.getElementById("quiz").hidden = false;
  document.getElementById("empty").hidden = true;
  document.getElementById("question").textContent = data.item.question || "";
  updateNav(data);

  const player = document.getElementById("player");
  if (data.item.audio_url) {
    player.src = data.item.audio_url;
    player.load();
  } else {
    player.removeAttribute("src");
    player.load();
  }

  state.suppressAutosave = true;
  if (state.cardinality === "multiple") {
    const answers = await ensureAnswersLoaded();
    renderAnswers(answers, state.accepted);
  } else {
    document.getElementById("answers").innerHTML = "";
    state.answers = data.item.answers || [];
  }
  syncCardinalityUi();
  state.suppressAutosave = false;
}

async function saveLabel({ quiet, advance } = {}) {
  if (!state.item || !state.cardinality) {
    if (!quiet) setStatus("Pick unique or multiple first.", "err");
    return;
  }
  if (!quiet) setStatus("Saving…");
  try {
    const data = await api("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: state.item.id,
        answer_cardinality: state.cardinality,
        accepted_answer_indices: state.cardinality === "multiple" ? state.accepted : [],
      }),
    });
    state.labeled = true;
    setStatus(data.updated ? "Updated" : "Saved", "ok");
    const session = await api("/api/session");
    updateNav({
      ...session,
      has_prev: state.index > 0,
      has_next: state.index + 1 < state.total,
    });
    syncCardinalityUi();
    if (advance && session.first_unlabeled_index != null) {
      await loadIndex(session.first_unlabeled_index);
    }
  } catch (err) {
    setStatus(String(err.message || err), "err");
  }
}

function queueAutosave() {
  if (state.suppressAutosave || !state.labeled) return;
  if (state.saveTimer) clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => {
    state.saveTimer = null;
    saveLabel({ quiet: true });
  }, 350);
}

document.querySelectorAll('input[name="cardinality"]').forEach((el) => {
  el.addEventListener("change", () => onCardinality(el.value));
});
document.getElementById("primaryBtn").onclick = () => {
  const advance = !state.labeled;
  saveLabel({ advance });
};
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
  if (state.firstUnlabeled == null) return;
  loadIndex(state.firstUnlabeled).catch(showBootError);
};
document.getElementById("continueBtn").onclick = () => {
  if (state.firstUnlabeled == null) return;
  loadIndex(state.firstUnlabeled).catch(showBootError);
};

function showBootError(err) {
  document.getElementById("empty").hidden = false;
  document.getElementById("quiz").hidden = true;
  document.getElementById("empty").textContent = String(err.message || err);
}

async function boot() {
  const session = await api("/api/session");
  const start = session.start_index != null ? session.start_index : 0;
  await loadIndex(start);
}
boot().catch(showBootError);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        print(f"[label-answers] {self.address_string()} {fmt % args}", flush=True)

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

        if path == "/api/answers":
            qid = str((qs.get("id") or [""])[0]).strip()
            with STATE_LOCK:
                item = CONFIG["by_id"].get(qid)
                if item is None:
                    self._send_json({"error": "unknown id"}, 404)
                    return
                self._send_json({"id": qid, "answers": list(item["answers"])})
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

        if path == "/api/save":
            qid = str(payload.get("id") or "").strip()
            cardinality = parse_cardinality(payload.get("answer_cardinality"))
            if not qid or cardinality is None:
                self._send_json(
                    {"error": "id and answer_cardinality (unique|multiple) required"},
                    400,
                )
                return
            with STATE_LOCK:
                item = CONFIG["by_id"].get(qid)
                if item is None:
                    self._send_json({"error": "unknown id"}, 404)
                    return
                indices, answers = normalize_accepted(
                    item,
                    cardinality=cardinality,
                    raw_indices=payload.get("accepted_answer_indices"),
                )
                updated = qid in CONFIG["labeled_ids"]
                record = {
                    "id": qid,
                    "participant": CONFIG["participant"],
                    "answer_cardinality": cardinality,
                    "accepted_answer_indices": indices,
                    "accepted_answers": answers,
                    "timestamp": utc_now(),
                }
                upsert_label(record)
                self._send_json(
                    {
                        "id": qid,
                        "saved": True,
                        "updated": updated,
                        "answer_cardinality": cardinality,
                        "accepted_answer_indices": indices,
                        "accepted_answers": answers,
                        "labels_path": str(CONFIG["labels_path"]),
                    }
                )
            return

        self._send_json({"error": "not found"}, 404)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument(
        "--participant",
        default="",
        help="Participant label stored with each record "
        "(also selects outputs/answer-labels/<participant>.jsonl)",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help="JSONL file to upsert labels into",
    )
    parser.add_argument(
        "--sample-json",
        type=Path,
        default=SAMPLE_JSON_PATH,
        help="Frozen sample JSON (default: answer-variety/label_sample.json)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Local MMAR wav directory",
    )
    parser.add_argument(
        "--force-audio-download",
        action="store_true",
        help="Re-download the MMAR wav archive even if wavs are present",
    )
    parser.add_argument(
        "--build-sample",
        action="store_true",
        help="Regenerate the frozen 100-id sample from the freeform pack and exit",
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Collated freeform pack used by --build-sample",
    )
    parser.add_argument(
        "--source-ids-csv",
        type=Path,
        default=OPEN_ENDED_IDS_PATH,
        help="Open-ended ID list to sample from",
    )
    parser.add_argument("--sample-n", type=int, default=SAMPLE_N)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    return parser.parse_args()


def resolve_labels_path(args: argparse.Namespace) -> Path:
    if args.labels_path is not None:
        return args.labels_path.expanduser().resolve()
    DEFAULT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    participant = (args.participant or "").strip()
    if participant:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in participant)
        return (DEFAULT_LABELS_DIR / f"{safe}.jsonl").resolve()
    return (DEFAULT_LABELS_DIR / "labels.jsonl").resolve()


def main() -> None:
    args = parse_args()
    if args.build_sample:
        payload = build_sample(
            source_ids_csv=args.source_ids_csv.expanduser().resolve(),
            pack_dir=args.pack_dir.expanduser().resolve(),
            ids_out=SAMPLE_IDS_PATH,
            json_out=SAMPLE_JSON_PATH,
            n=args.sample_n,
            seed=args.sample_seed,
        )
        n_answers = sum(len(item["answers"]) for item in payload["items"])
        print(f"Wrote {SAMPLE_IDS_PATH}")
        print(f"Wrote {SAMPLE_JSON_PATH}")
        print(
            f"Sampled {payload['n']} questions, "
            f"{n_answers} unique generation strings"
        )
        return

    audio_dir = ensure_mmar_audio(
        args.audio_dir.expanduser().resolve(),
        force=args.force_audio_download,
    )

    items = load_sample(args.sample_json.expanduser().resolve())
    labels_path = resolve_labels_path(args)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_by_id = load_labels_by_id(labels_path)
    active_ids = {item["id"] for item in items}
    labeled_ids = {qid for qid in labels_by_id if qid in active_ids}

    CONFIG["items"] = items
    CONFIG["by_id"] = {item["id"]: item for item in items}
    CONFIG["labels_by_id"] = labels_by_id
    CONFIG["labeled_ids"] = labeled_ids
    CONFIG["labels_path"] = labels_path
    CONFIG["audio_dir"] = audio_dir
    CONFIG["participant"] = (args.participant or "").strip()
    CONFIG["start_index"] = (
        first_unlabeled_index()
        if first_unlabeled_index() is not None
        else max(0, len(items) - 1)
    )

    remaining = sum(1 for item in items if item["id"] not in labeled_ids)
    print(f"Sample:   {args.sample_json}")
    print(f"Audio:    {audio_dir}")
    print(f"Labels:   {labels_path}")
    print(
        f"Set:      {len(items)} questions "
        f"({len(labeled_ids)} already labeled, {remaining} remaining)"
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
