"""Local viewer for MMAR question-difficulty experiment results.

Browse questions sorted hardest-first by mean shot success rate across
models, and inspect each model's sampled responses.

Reads from ``outputs/`` (default), discovering run folders under
``exp-mmar-question-difficulty/<run_id>/``:

    difficulty.jsonl
    scores.json
    manifest.json
    models/<label>/predictions.jsonl

Missing ``difficulty.jsonl`` / ``scores.json`` are built on demand via
``aggregate.aggregate_difficulty``.

Usage:

    uv run python view_difficulty.py
    uv run python view_difficulty.py --port 7860
    uv run python view_difficulty.py --results-dir ./outputs
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import tarfile
import tempfile
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from aggregate import aggregate_difficulty, discover_model_labels
from mmar_common import (
    AF_NEXT_THINK_SUFFIX,
    build_mmar_freeform_prompt,
    build_mmar_prompt,
)
from mmar_models import AF_NEXT_SYSTEM

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs"
EXPERIMENT_SUBDIR = "exp-mmar-question-difficulty"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "mmar"
DEFAULT_AUDIO_DIR = DEFAULT_DATA_DIR / "audio"
MMAR_REPO = "BoJack/MMAR"
MMAR_AUDIO_ARCHIVE = "mmar-audio.tar.gz"
MIN_MMAR_WAVS = 1000

CONFIG: dict[str, Any] = {}


def infer_run_mode(
    manifest: dict | None = None,
    scores: dict | None = None,
    *,
    predictions: dict[str, dict[str, dict]] | None = None,
) -> str:
    """Return ``freeform`` or ``mc`` from manifest / scores / prediction stamps."""
    manifest = manifest or {}
    scores = scores or {}
    saw_freeform_judge = False
    saw_pending_grade = False
    if predictions:
        for per_model in predictions.values():
            for record in per_model.values():
                scoring = str(record.get("scoring") or "").lower()
                if "freeform" in scoring or "qwen_freeform" in scoring:
                    saw_freeform_judge = True
                if record.get("pending_grade"):
                    saw_pending_grade = True
                for shot in record.get("shots") or []:
                    judges = shot.get("judges") or {}
                    if any(label != "string-match" for label in judges):
                        saw_freeform_judge = True
                    # Legacy flat grader fields imply a freeform LLM judge,
                    # but ignore synthetic string-match stamps.
                    elif shot.get("grader") or shot.get("grader_output"):
                        saw_freeform_judge = True
                    if shot.get("pending_grade"):
                        saw_pending_grade = True
    if saw_freeform_judge:
        return "freeform"
    mode = str(manifest.get("mode") or scores.get("mode") or "").strip().lower()
    if mode in {"freeform", "free_form", "free-form", "open"}:
        return "freeform"
    if mode in {"mc", "multiple_choice", "multiple-choice", "mcq", "choice"}:
        return "mc"
    scoring = str(
        manifest.get("scoring") or scores.get("scoring") or ""
    ).lower()
    if "freeform" in scoring or "qwen_freeform" in scoring:
        return "freeform"
    if saw_pending_grade:
        return "freeform"
    return "mc"


def collect_judges(
    manifest: dict | None = None,
    scores: dict | None = None,
    *,
    predictions: dict[str, dict[str, dict]] | None = None,
) -> list[dict]:
    """Ordered judge entries ``[{label, model_id, primary}, ...]`` for a run."""
    manifest = manifest or {}
    scores = scores or {}
    entries: list[dict] = []
    seen: set[str] = set()
    primary = scores.get("primary_judge") or manifest.get("primary_judge")

    def _add(label: str | None, model_id: Any = None, is_primary: bool = False) -> None:
        if not label or label in seen:
            return
        seen.add(str(label))
        entries.append(
            {
                "label": str(label),
                "model_id": model_id,
                "primary": bool(is_primary) or str(label) == primary,
            }
        )

    for raw in scores.get("judges") or manifest.get("judges") or []:
        if isinstance(raw, dict):
            _add(raw.get("label"), raw.get("model_id"), raw.get("primary"))
        else:
            _add(raw)

    if predictions:
        for per_model in predictions.values():
            for record in per_model.values():
                for label in record.get("judges") or []:
                    _add(label)
                for shot in record.get("shots") or []:
                    for label, entry in (shot.get("judges") or {}).items():
                        _add(
                            label,
                            (entry or {}).get("model_id") if isinstance(entry, dict) else None,
                        )

    if not primary and entries:
        primary = entries[0]["label"]
    for entry in entries:
        entry["primary"] = entry["label"] == primary
    if primary:
        entries.sort(key=lambda e: (0 if e["label"] == primary else 1, e["label"]))
    return entries


def mode_label(mode: str) -> str:
    return "Freeform" if mode == "freeform" else "MCQ"


def build_base_prompt(item: dict, mode: str, *, think_suffix: str | None = None) -> str:
    if mode == "freeform":
        return build_mmar_freeform_prompt(item, think_suffix=think_suffix)
    return build_mmar_prompt(item, think_suffix=think_suffix)


def build_model_prompts(item: dict, mode: str) -> dict[str, str]:
    """Reconstruct the text prompts sent to each model for this question."""
    base = build_base_prompt(item, mode)
    af_base = build_base_prompt(item, mode, think_suffix=AF_NEXT_THINK_SUFFIX)
    if mode == "freeform":
        step_system = (
            "You are an expert in audio analysis. "
            "Listen carefully and answer the question accurately.\n"
            f"{base}"
        )
    else:
        step_system = (
            "You are an expert in audio analysis. "
            "Listen carefully and answer the multiple-choice question accurately.\n"
            f"{base}"
        )
    return {
        "shared": base,
        "af-next-think": (
            f"<|im_start|>system\n{AF_NEXT_SYSTEM}<|im_end|>\n"
            "<|im_start|>user\n"
            f"<sound>{af_base}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "mimo-audio-7b": (
            "<|im_start|>user\n"
            f"<|sosp|><|empty|><|eosp|>{base}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n"
        ),
        "interactive-omni-8b": (
            "[audio attached]\n"
            f"{base}"
        ),
        "qwen3-omni": (
            "<|im_start|>user\n"
            f"<|audio_start|><|audio_pad|><|audio_end|>{base}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "voxtral-small-24b": (
            "[audio attached]\n"
            f"{base}"
        ),
        "step-audio-2-mini": (
            f"<|im_start|>system\n{step_system}<|im_end|>\n"
            "<|im_start|>user\n<audio_patch><|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    }


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


def count_wavs(audio_dir: Path) -> int:
    if not audio_dir.is_dir():
        return 0
    return sum(
        1
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def ensure_mmar_audio(audio_dir: Path, *, force: bool = False) -> Path:
    """Download and extract the MMAR wav archive into ``audio_dir`` if needed."""
    audio_dir = audio_dir.expanduser().resolve()
    wav_count = count_wavs(audio_dir)
    if wav_count >= MIN_MMAR_WAVS and not force:
        print(f"MMAR audio ready: {wav_count} wav files in {audio_dir}", flush=True)
        return audio_dir

    from huggingface_hub import hf_hub_download

    cache_root = audio_dir.parent
    archive_cache = cache_root / MMAR_AUDIO_ARCHIVE
    cache_root.mkdir(parents=True, exist_ok=True)

    print(
        f"MMAR audio missing or incomplete ({wav_count} wavs); "
        f"downloading {MMAR_AUDIO_ARCHIVE} from {MMAR_REPO} ...",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="mmar-audio-") as tmp:
        tmp_root = Path(tmp)
        archive_tmp = Path(
            hf_hub_download(
                repo_id=MMAR_REPO,
                filename=MMAR_AUDIO_ARCHIVE,
                repo_type="dataset",
                local_dir=str(tmp_root / "download"),
            )
        )
        print(f"Extracting {archive_tmp.name} ...", flush=True)
        extract_dir = tmp_root / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_tmp, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        candidate_audio = extract_dir / "audio"
        if not candidate_audio.is_dir():
            matches = [path for path in extract_dir.rglob("audio") if path.is_dir()]
            if not matches:
                raise SystemExit(
                    f"No audio/ directory found after extracting {archive_tmp}"
                )
            candidate_audio = matches[0]

        if audio_dir.exists():
            shutil.rmtree(audio_dir)
        shutil.copytree(candidate_audio, audio_dir)
        shutil.copy2(archive_tmp, archive_cache)

    wav_count = count_wavs(audio_dir)
    if wav_count < MIN_MMAR_WAVS:
        raise SystemExit(
            f"Expected at least {MIN_MMAR_WAVS} wav files in {audio_dir}, "
            f"found {wav_count}."
        )
    print(f"MMAR audio ready: {wav_count} wav files in {audio_dir}", flush=True)
    return audio_dir


def is_experiment_run_dir(path: Path) -> bool:
    """True for a run_experiment.py output folder."""
    if not path.is_dir():
        return False
    if (path / "difficulty.jsonl").is_file() or (path / "manifest.json").is_file():
        return True
    models = path / "models"
    return models.is_dir() and any(
        (child / "predictions.jsonl").is_file()
        for child in models.iterdir()
        if child.is_dir()
    )


def resolve_run_roots(results_dir: Path) -> list[Path]:
    """Directories that directly contain experiment run folders."""
    results_dir = results_dir.expanduser().resolve()
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        if not path.is_dir():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(resolved)

    exp = results_dir / EXPERIMENT_SUBDIR
    if exp.is_dir():
        _add(exp)
    if results_dir.name == EXPERIMENT_SUBDIR:
        _add(results_dir)
    elif any(
        is_experiment_run_dir(child)
        for child in results_dir.iterdir()
        if child.is_dir()
    ):
        _add(results_dir)
    if not roots and results_dir.is_dir():
        # Default layout even before the first download.
        _add(exp if exp.parent == results_dir else results_dir)
    return roots


def needs_aggregation(run_dir: Path) -> bool:
    return not (run_dir / "difficulty.jsonl").is_file() or not (
        run_dir / "scores.json"
    ).is_file()


def ensure_aggregated(run_dir: Path, model_labels: list[str] | None = None) -> bool:
    """Build difficulty.jsonl / scores.json when missing. Returns True if written."""
    if not needs_aggregation(run_dir):
        return False
    labels = model_labels or discover_model_labels(run_dir)
    if not labels:
        return False
    has_preds = any(
        (run_dir / "models" / label / "predictions.jsonl").is_file()
        for label in labels
    )
    if not has_preds:
        return False
    print(f"[viewer] aggregating scores for {run_dir.name} ...", flush=True)
    aggregate_difficulty(run_dir, model_labels=labels)
    load_run_bundle.cache_clear()
    return True


def discover_runs(results_dir: Path) -> list[dict]:
    runs: list[dict] = []
    seen_ids: set[str] = set()
    for root in resolve_run_roots(results_dir):
        for path in sorted(root.iterdir(), reverse=True):
            if not is_experiment_run_dir(path) or path.name in seen_ids:
                continue
            seen_ids.add(path.name)
            manifest = load_json(path / "manifest.json") or {}
            models = discover_model_labels(path, manifest=manifest)
            ensure_aggregated(path, models)
            scores = load_json(path / "scores.json") or {}
            difficulty = path / "difficulty.jsonl"
            # Lightweight mode hint from one prediction record when available.
            sample_preds: dict[str, dict[str, dict]] = {}
            for label in models[:1]:
                rows = load_jsonl(path / "models" / label / "predictions.jsonl")
                if rows:
                    sample_preds[label] = {str(rows[0].get("id") or "0"): rows[0]}
            mode = infer_run_mode(manifest, scores, predictions=sample_preds or None)
            runs.append(
                {
                    "id": path.name,
                    "path": str(path),
                    "has_difficulty": difficulty.exists(),
                    "n_questions": scores.get("n_questions")
                    or manifest.get("n_questions")
                    or manifest.get("num_samples"),
                    "n_questions_scored": scores.get("n_questions_scored"),
                    "n_questions_pending": scores.get("n_questions_pending"),
                    "avg_success_rate": scores.get("avg_success_rate"),
                    "models": models,
                    "seed": manifest.get("seed"),
                    "n_shots": manifest.get("n_shots"),
                    "temperature": manifest.get("temperature"),
                    "mode": mode,
                    "mode_label": mode_label(mode),
                    "scoring": manifest.get("scoring") or scores.get("scoring"),
                    "grader_model_id": manifest.get("grader_model_id")
                    or scores.get("grader_model_id"),
                    "judges": collect_judges(manifest, scores, predictions=sample_preds or None),
                    "primary_judge": scores.get("primary_judge")
                    or manifest.get("primary_judge"),
                    "source_run_id": manifest.get("source_run_id")
                    or scores.get("source_run_id"),
                }
            )
    runs.sort(key=lambda row: row["id"], reverse=True)
    return runs


def run_dir_for(run_id: str) -> Path:
    results_dir = Path(CONFIG["results_dir"])
    for root in resolve_run_roots(results_dir):
        candidate = root / run_id
        if candidate.is_dir():
            return candidate
    return results_dir / EXPERIMENT_SUBDIR / run_id


@lru_cache(maxsize=8)
def load_run_bundle(run_id: str) -> dict[str, Any]:
    run_dir = run_dir_for(run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    manifest = load_json(run_dir / "manifest.json") or {}
    model_labels = discover_model_labels(run_dir, manifest=manifest)
    ensure_aggregated(run_dir, model_labels)
    difficulty = load_jsonl(run_dir / "difficulty.jsonl")
    # Already hardest-first from aggregate; keep order.
    by_id = {str(row["id"]): row for row in difficulty}
    scores = load_json(run_dir / "scores.json") or {}
    predictions: dict[str, dict[str, dict]] = {}
    for label in model_labels:
        preds = load_jsonl(run_dir / "models" / label / "predictions.jsonl")
        predictions[label] = {str(p["id"]): p for p in preds if p.get("id")}
    mode = infer_run_mode(manifest, scores, predictions=predictions)
    judges = collect_judges(manifest, scores, predictions=predictions)
    primary_judge = (
        scores.get("primary_judge")
        or manifest.get("primary_judge")
        or (judges[0]["label"] if judges else None)
    )
    return {
        "difficulty": difficulty,
        "by_id": by_id,
        "predictions": predictions,
        "scores": scores,
        "manifest": manifest,
        "mode": mode,
        "mode_label": mode_label(mode),
        "model_labels": model_labels,
        "judges": judges,
        "primary_judge": primary_judge,
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
  .mode-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.04em; text-transform: uppercase;
    padding: 0.2rem 0.55rem; border-radius: 999px;
    border: 1px solid var(--line);
    vertical-align: middle;
  }
  .mode-badge.mc {
    color: #1a4a6e; background: #d9e8f3; border-color: #a9c4d8;
  }
  .mode-badge.freeform {
    color: #5a3a12; background: #f3e6cf; border-color: #d4b88a;
  }
  .brand-row {
    display: flex; flex-wrap: wrap; gap: 0.55rem; align-items: center;
  }
  .run-meta {
    margin-top: 0.35rem; font-size: 0.82rem; color: var(--muted);
    display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center;
  }
  .stats .mode-badge { margin-right: 0.15rem; }
  details.accordion {
    border: 1px solid var(--line); border-radius: 10px;
    background: #fff; margin: 0.75rem 0; overflow: hidden;
  }
  details.accordion > summary {
    cursor: pointer; list-style: none;
    padding: 0.75rem 0.9rem;
    font-weight: 600; font-size: 0.92rem;
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
  details.prompt-model {
    border: 1px solid var(--line); border-radius: 8px;
    margin: 0.5rem 0; background: #fbfcfd;
  }
  details.prompt-model > summary {
    cursor: pointer; padding: 0.55rem 0.7rem;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem; font-weight: 500;
    list-style: none; display: flex; justify-content: space-between;
  }
  details.prompt-model > summary::-webkit-details-marker { display: none; }
  details.prompt-model > summary::after {
    content: "▸"; color: var(--muted);
  }
  details.prompt-model[open] > summary::after { content: "▾"; }
  details.prompt-model pre {
    margin: 0; border-radius: 0 0 8px 8px;
    border-top: 1px solid var(--line);
  }
  .choice-box.hidden-from-model {
    border-style: dashed; background: #faf7f2;
  }
  .choice-box .note {
    font-size: 0.78rem; color: var(--muted); margin: 0.25rem 0 0.45rem;
  }
  .grader-note {
    font-size: 0.78rem; color: var(--muted);
    font-family: "IBM Plex Mono", monospace;
  }
  .verdict-grid-wrap {
    border: 1px solid var(--line); border-radius: 10px;
    background: #fff; margin: 0.75rem 0; overflow: auto;
  }
  .verdict-grid-wrap > .vg-title {
    padding: 0.55rem 0.8rem; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--line); background: #f2f6f9;
  }
  .verdict-grid {
    display: grid;
    gap: 3px;
    padding: 0.65rem 0.8rem 0.8rem;
    align-items: center;
    min-width: max-content;
  }
  .vg-corner { font-size: 0.72rem; color: var(--muted); }
  .vg-judge {
    text-align: center;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; font-weight: 500;
    padding: 0.15rem 0.25rem;
    border-bottom: 1px solid var(--line);
  }
  .vg-judge.primary { color: var(--accent); }
  .vg-shot {
    text-align: center;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.68rem; color: var(--muted);
  }
  .vg-model {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.75rem; font-weight: 500;
    padding-right: 0.5rem; white-space: nowrap;
  }
  .vg-cell {
    width: 1.05rem; height: 1.05rem; border-radius: 3px;
    justify-self: center;
    border: 1px solid transparent;
  }
  .vg-cell.pass { background: var(--good); }
  .vg-cell.fail { background: var(--bad); }
  .vg-cell.pending { background: #d5dde3; border-color: var(--line); }
  .vg-cell.missing { background: transparent; border: 1px dashed var(--line); }
  .judge-pills { display: flex; flex-wrap: wrap; gap: 0.3rem; align-items: center; }
  .judge-pill {
    font-size: 0.68rem; font-family: "IBM Plex Mono", monospace;
    padding: 0.08rem 0.35rem; border-radius: 999px;
  }
  .judge-pill.pass { color: var(--good); background: var(--soft-good); }
  .judge-pill.fail { color: var(--bad); background: var(--soft-bad); }
  .judge-pill.pending { color: var(--muted); background: #e8eef2; }
  .judge-gens { margin-top: 0.45rem; display: flex; flex-direction: column; gap: 0.35rem; }
  .judge-gen details.accordion { margin: 0; }
  .judge-gen details.accordion > summary {
    font-size: 0.78rem; padding: 0.4rem 0.65rem;
  }
  .judge-gen .accordion-body { padding: 0.55rem 0.65rem 0.7rem; }
  .judge-gen pre { margin-top: 0; }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <div class="brand-row">
        <h1>MMAR Question Difficulty</h1>
        <span id="mode-badge" class="mode-badge mc">MCQ</span>
      </div>
      <p id="brand-sub">Hardest-first by mean shot success rate across models</p>
      <div class="run-meta" id="run-meta"></div>
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
const state = {
  runs: [],
  runId: "",
  mode: "mc",
  modeLabel: "MCQ",
  modelLabels: [],
  judges: [],
  primaryJudge: null,
  manifest: {},
  scores: {},
  questions: [],
  selectedId: null,
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

function modeBadgeHtml(mode, label) {
  const cls = mode === "freeform" ? "freeform" : "mc";
  return `<span class="mode-badge ${cls}">${escapeHtml(label || (mode === "freeform" ? "Freeform" : "MCQ"))}</span>`;
}

function setHeaderMode(mode, modeLabel, manifest, scores) {
  const badge = document.getElementById("mode-badge");
  badge.className = `mode-badge ${mode === "freeform" ? "freeform" : "mc"}`;
  badge.textContent = modeLabel || (mode === "freeform" ? "Freeform" : "MCQ");
  const sub = document.getElementById("brand-sub");
  const judges = state.judges || [];
  if (mode === "freeform") {
    const names = judges.map(j => j.label).filter(Boolean);
    sub.textContent = names.length
      ? `Freeform answers · judged by ${names.join(", ")} · hardest-first`
      : "Freeform answers · graded by judge · hardest-first";
  } else {
    sub.textContent = "Multiple-choice · string-match scoring · hardest-first";
  }
  const metaBits = [];
  const scoring = manifest.scoring || scores.scoring;
  if (scoring) metaBits.push(`scoring: ${scoring}`);
  if (judges.length) {
    const judgeBits = judges.map(j => {
      const star = j.primary || j.label === state.primaryJudge ? "*" : "";
      return `${j.label}${star}`;
    });
    metaBits.push(`judges: ${judgeBits.join(", ")}`);
  } else if (manifest.grader_model_id || scores.grader_model_id) {
    metaBits.push(`grader: ${manifest.grader_model_id || scores.grader_model_id}`);
  }
  if (manifest.source_run_id || scores.source_run_id) {
    metaBits.push(`source: ${manifest.source_run_id || scores.source_run_id}`);
  }
  if (manifest.n_shots || scores.n_shots) {
    metaBits.push(`${manifest.n_shots || scores.n_shots || "—"} shots`);
  }
  if (scores.n_questions_pending) {
    metaBits.push(`${scores.n_questions_pending} pending grade`);
  }
  document.getElementById("run-meta").textContent = metaBits.join(" · ");
}

function renderStats(scores, mode, modeLabel) {
  const by = scores.by_model || {};
  const labels = state.modelLabels.length
    ? state.modelLabels
    : Object.keys(by);
  const parts = [
    modeBadgeHtml(mode, modeLabel),
    `<span><strong>${scores.n_questions ?? state.questions.length ?? "—"}</strong> questions</span>`,
    `<span>avg <strong>${fmtRate(scores.avg_success_rate)}</strong></span>`,
  ];
  if (scores.n_questions_pending) {
    parts.push(`<span><strong>${scores.n_questions_pending}</strong> pending</span>`);
  }
  for (const label of labels) {
    const m = by[label] || {};
    const pending = m.n_pending ? ` (${m.n_pending} pend)` : "";
    parts.push(`<span>${label}: <strong>${fmtRate(m.avg_shot_success_rate)}</strong>${pending}</span>`);
  }
  document.getElementById("stats").innerHTML = parts.join(" · ");
}

function renderList() {
  const q = document.getElementById("search").value.trim().toLowerCase();
  const list = document.getElementById("qlist");
  const labels = state.modelLabels;
  const items = state.questions.filter(row => {
    if (!q) return true;
    return String(row.id).toLowerCase().includes(q)
      || String(row.question || "").toLowerCase().includes(q);
  });
  list.innerHTML = items.map(row => {
    const chips = labels.map(label => {
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

function renderPromptAccordion(prompts, mode) {
  if (!prompts) return "";
  const shared = prompts.shared || "";
  const modelOrder = [
    ...state.modelLabels,
    "af-next-think",
    "mimo-audio-7b",
    "interactive-omni-8b",
    "qwen3-omni",
    "voxtral-small-24b",
    "step-audio-2-mini",
  ];
  const seen = new Set();
  const modelBlocks = modelOrder
    .filter(label => {
      if (!prompts[label] || seen.has(label)) return false;
      seen.add(label);
      return true;
    })
    .map(label => `<details class="prompt-model">
      <summary>${escapeHtml(label)}</summary>
      <pre>${escapeHtml(prompts[label])}</pre>
    </details>`)
    .join("");
  const modeNote = mode === "freeform"
    ? "Question only — multiple-choice options were not shown to the model."
    : "Multiple-choice prompt including the four options.";
  return `<details class="accordion" open>
    <summary><span>Full prompt</span></summary>
    <div class="accordion-body">
      <p class="muted" style="margin:0 0 0.55rem">${escapeHtml(modeNote)}</p>
      <strong style="font-size:0.82rem;color:var(--muted)">Shared question text</strong>
      <pre>${escapeHtml(shared)}</pre>
      <strong style="display:block;margin-top:0.75rem;font-size:0.82rem;color:var(--muted)">Per-model chat wrappers</strong>
      ${modelBlocks}
    </div>
  </details>`;
}

function shotJudgeEntry(shot, judgeLabel) {
  const judges = shot.judges || {};
  if (judges[judgeLabel]) return judges[judgeLabel];
  // Legacy flat fields when this is the only / primary judge.
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
  const pills = judges.map(j => {
    const entry = shotJudgeEntry(shot, j.label);
    if (!entry || entry.correct === null || entry.correct === undefined) {
      return `<span class="judge-pill pending" title="${escapeHtml(j.label)}: pending">${escapeHtml(j.label)}?</span>`;
    }
    const ok = !!entry.correct;
    const tip = `${j.label}: ${judgeVerdictLabel(entry)}`;
    return `<span class="judge-pill ${ok ? "pass" : "fail"}" title="${escapeHtml(tip)}">${escapeHtml(j.label)} ${ok ? "✓" : "✗"}</span>`;
  }).join("");
  return `<span class="judge-pills">${pills}</span>`;
}

function renderJudgeGenerations(shot, judges) {
  const list = (judges && judges.length) ? judges : [];
  const blocks = [];
  const seen = new Set();
  for (const j of list) {
    const entry = shotJudgeEntry(shot, j.label);
    const gen = judgeGenerationText(entry);
    if (!gen) continue;
    seen.add(j.label);
    const verdict = judgeVerdictLabel(entry);
    const open = list.length === 1 ? " open" : "";
    blocks.push(`<div class="judge-gen">
      <details class="accordion"${open}>
        <summary><span>Judge ${escapeHtml(j.label)} · ${escapeHtml(verdict)}</span></summary>
        <div class="accordion-body"><pre>${escapeHtml(gen)}</pre></div>
      </details>
    </div>`);
  }
  // Also show any shot-level judges not listed in metadata.
  for (const [label, entry] of Object.entries(shot.judges || {})) {
    if (seen.has(label)) continue;
    const gen = judgeGenerationText(entry);
    if (!gen) continue;
    const verdict = judgeVerdictLabel(entry);
    blocks.push(`<div class="judge-gen">
      <details class="accordion">
        <summary><span>Judge ${escapeHtml(label)} · ${escapeHtml(verdict)}</span></summary>
        <div class="accordion-body"><pre>${escapeHtml(gen)}</pre></div>
      </details>
    </div>`);
  }
  if (!blocks.length) return "";
  return `<div class="judge-gens">${blocks.join("")}</div>`;
}

function renderVerdictGrid(modelLabels, predictions, judges) {
  const labels = modelLabels || [];
  let judgeList = (judges && judges.length) ? judges : [];
  // Infer judges / n_shots from predictions when metadata is empty.
  let nShots = 0;
  for (const label of labels) {
    const shots = ((predictions || {})[label] || {}).shots || [];
    nShots = Math.max(nShots, shots.length);
    if (!judgeList.length) {
      for (const shot of shots) {
        for (const j of Object.keys(shot.judges || {})) {
          if (!judgeList.find(x => x.label === j)) judgeList.push({ label: j });
        }
      }
    }
  }
  if (!labels.length || !nShots) return "";
  if (!judgeList.length) {
    judgeList = [{ label: state.mode === "freeform" ? "judge" : "string-match", primary: true }];
  }
  const cols = 1 + judgeList.length * nShots;
  const judgeHeaders = judgeList.map(j => {
    const primary = j.primary || j.label === state.primaryJudge;
    return `<div class="vg-judge${primary ? " primary" : ""}" style="grid-column: span ${nShots}">${escapeHtml(j.label)}${primary ? " *" : ""}</div>`;
  }).join("");
  const shotHeaders = judgeList.map(() =>
    Array.from({ length: nShots }, (_, i) => `<div class="vg-shot">s${i}</div>`).join("")
  ).join("");
  const rows = labels.map(label => {
    const pred = (predictions || {})[label] || {};
    const shots = pred.shots || [];
    const cells = judgeList.map(j => {
      return Array.from({ length: nShots }, (_, i) => {
        const shot = shots.find(s => Number(s.shot_index) === i) || shots[i];
        if (!shot) return `<div class="vg-cell missing" title="${escapeHtml(label)} / ${escapeHtml(j.label)} s${i}: missing"></div>`;
        const entry = shotJudgeEntry(shot, j.label);
        if (!entry || entry.correct === null || entry.correct === undefined) {
          const pending = shot.pending_grade || shot.correct === null || shot.correct === undefined;
          const cls = pending ? "pending" : "missing";
          return `<div class="vg-cell ${cls}" title="${escapeHtml(label)} / ${escapeHtml(j.label)} s${i}: ${cls}"></div>`;
        }
        const ok = !!entry.correct;
        const tip = `${label} / ${j.label} s${i}: ${judgeVerdictLabel(entry)}`;
        return `<div class="vg-cell ${ok ? "pass" : "fail"}" title="${escapeHtml(tip)}"></div>`;
      }).join("");
    }).join("");
    return `<div class="vg-model">${escapeHtml(label)}</div>${cells}`;
  }).join("");
  return `<div class="verdict-grid-wrap">
    <div class="vg-title">Verdict grid · model × judge × shot</div>
    <div class="verdict-grid" style="grid-template-columns: max-content repeat(${cols - 1}, 1.15rem)">
      <div class="vg-corner"></div>
      ${judgeHeaders}
      <div class="vg-corner"></div>
      ${shotHeaders}
      ${rows}
    </div>
  </div>`;
}

async function selectQuestion(id) {
  state.selectedId = id;
  renderList();
  const detail = document.getElementById("detail");
  detail.innerHTML = `<p class="muted">Loading…</p>`;
  const data = await api(`/api/question?run=${encodeURIComponent(state.runId)}&id=${encodeURIComponent(id)}`);
  const row = data.difficulty;
  const mode = data.mode || state.mode;
  const modeLabel = data.mode_label || state.modeLabel;
  const judges = (data.judges && data.judges.length) ? data.judges : state.judges;
  if (data.primary_judge) state.primaryJudge = data.primary_judge;
  const choices = (row.choices || []).map((c, i) =>
    `<div>(${String.fromCharCode(65+i)}) ${escapeHtml(c)}</div>`
  ).join("");
  let modelsHtml = "";
  const labels = (data.model_labels && data.model_labels.length)
    ? data.model_labels
    : state.modelLabels;
  const verdictGrid = renderVerdictGrid(labels, data.predictions || {}, judges);
  for (const label of labels) {
    const pred = (data.predictions || {})[label];
    const pm = (row.per_model || {})[label] || {};
    if (!pred) {
      modelsHtml += `<div class="model-block"><h3>${label} <span class="muted">missing</span></h3></div>`;
      continue;
    }
    const shots = pred.shots || [];
    const shotsHtml = shots.map(shot => {
      const verdict = renderJudgePills(shot, judges);
      const judgeGens = renderJudgeGenerations(shot, judges);
      return `<div class="shot">
        <div class="shot-head">
          <span>shot ${shot.shot_index}</span>
          ${verdict}
          <span class="muted">parsed: ${escapeHtml(shot.answer_prediction || "")}</span>
        </div>
        <pre>${escapeHtml(shot.model_output || "")}</pre>
        ${judgeGens}
      </div>`;
    }).join("");
    const rateChip = pm.pending_grade
      ? `<span class="chip">pending grade</span>`
      : `<span class="chip">${fmtRate(pm.shot_success_rate)} (${pm.n_shot_correct ?? "—"}/${pm.n_shots ?? "—"})</span>`;
    modelsHtml += `<div class="model-block">
      <h3>${label}
        ${rateChip}
      </h3>
      ${shotsHtml || "<p class='muted'>No shots stored.</p>"}
    </div>`;
  }
  const audio = data.audio_url
    ? `<audio controls preload="none" src="${data.audio_url}"></audio>`
    : `<p class="muted">Audio not found locally.</p>`;
  const choicesClass = mode === "freeform" ? "choice-box hidden-from-model" : "choice-box";
  const choicesTitle = mode === "freeform"
    ? `<strong>Choices</strong><div class="note">Not shown to the model in freeform mode (gold answer reference only).</div>`
    : `<strong>Choices</strong>`;
  detail.innerHTML = `
    <div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap">
      <div class="qid">${escapeHtml(row.id)}</div>
      ${modeBadgeHtml(mode, modeLabel)}
    </div>
    <h3 style="margin:0.4rem 0 0.2rem">${escapeHtml(row.question || "")}</h3>
    <p class="muted">${escapeHtml(row.modality || "")} · ${escapeHtml(row.category || "")} · avg ${fmtRate(row.avg_success_rate)}</p>
    ${verdictGrid}
    ${audio}
    ${renderPromptAccordion(data.prompts, mode)}
    <div class="${choicesClass}">${choicesTitle}${choices}</div>
    <div class="answer-box"><strong>Gold</strong><div>${escapeHtml(row.answer || "")}</div></div>
    ${modelsHtml}
  `;
}

async function loadRun(runId) {
  state.runId = runId;
  const data = await api(`/api/questions?run=${encodeURIComponent(runId)}`);
  state.questions = data.questions || [];
  state.mode = data.mode || "mc";
  state.modeLabel = data.mode_label || (state.mode === "freeform" ? "Freeform" : "MCQ");
  state.modelLabels = data.model_labels || data.manifest?.models || [];
  state.judges = data.judges || [];
  state.primaryJudge = data.primary_judge || (state.judges[0] && state.judges[0].label) || null;
  state.manifest = data.manifest || {};
  state.scores = data.scores || {};
  setHeaderMode(state.mode, state.modeLabel, state.manifest, state.scores);
  renderStats(state.scores, state.mode, state.modeLabel);
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
  sel.innerHTML = state.runs.map(r => {
    const tag = r.mode_label || (r.mode === "freeform" ? "Freeform" : "MCQ");
    const pending = r.n_questions_pending ? ` · ${r.n_questions_pending} pending` : "";
    const avail = r.has_difficulty ? "" : " (no scores yet)";
    return `<option value="${r.id}">[${tag}] ${r.id}${pending}${avail}</option>`;
  }).join("");
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
                    "mode": bundle["mode"],
                    "mode_label": bundle["mode_label"],
                    "model_labels": bundle["model_labels"],
                    "judges": bundle["judges"],
                    "primary_judge": bundle["primary_judge"],
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
            model_labels = bundle["model_labels"]
            preds = {
                label: bundle["predictions"].get(label, {}).get(qid)
                for label in model_labels
            }
            audio = resolve_audio(row.get("audio_path"))
            audio_url = None
            if audio is not None:
                audio_url = f"/audio/{audio.name}"
            # Prefer a prediction record (has full item fields) when building prompts.
            sample = next(
                (preds[label] for label in model_labels if preds.get(label)),
                row,
            )
            prompts = build_model_prompts(sample, bundle["mode"])
            self._send_json(
                {
                    "difficulty": row,
                    "predictions": preds,
                    "audio_url": audio_url,
                    "mode": bundle["mode"],
                    "mode_label": bundle["mode_label"],
                    "model_labels": model_labels,
                    "judges": bundle["judges"],
                    "primary_judge": bundle["primary_judge"],
                    "prompts": prompts,
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
    parser.add_argument("--port", type=int, default=7860)
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

    CONFIG["results_dir"] = args.results_dir.expanduser().resolve()
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
    load_run_bundle.cache_clear()

    print(f"Results: {CONFIG['results_dir']}")
    for root in resolve_run_roots(CONFIG["results_dir"]):
        print(f"  run root: {root}")
    print(f"Audio:   {CONFIG['audio_dir']}")
    runs = discover_runs(CONFIG["results_dir"])
    print(f"Found {len(runs)} run(s)")
    for run in runs:
        rate = run.get("avg_success_rate")
        rate_s = f"{100 * rate:.1f}%" if rate is not None else "—"
        pending = run.get("n_questions_pending") or 0
        pending_s = f", {pending} pending" if pending else ""
        print(
            f"  [{run['mode_label']}] {run['id']}: "
            f"{len(run['models'])} models, avg {rate_s}{pending_s}"
        )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
