"""Local web UI for MMAR-Rubrics examples + AF3 generations / grades.

On startup, ensures MMAR-meta.jsonl and the MMAR wav archive are persisted
under ``data/mmar/`` so examples can be played in-browser.

Reads:
  - MMAR-meta.jsonl (questions, GT reasoning, instance rubrics)
  - Local wavs under ``data/mmar/audio/``
  - Run folders under ``outputs/`` (direct or nested ``mmar/af3/<run>/``):
      predictions.jsonl, predictions.evaluated.jsonl, scores.json, manifest.json
  - Optional ``attentions/<id>.npz`` artifacts captured on demand

Usage:

    uv run python view_mmar_results.py
    uv run python view_mmar_results.py --port 7860
    uv run python view_mmar_results.py --results-dir ./outputs
    uv run python view_mmar_results.py --meta ./data/mmar/MMAR-meta.jsonl

Capture attentions for the selected example via the detail-panel button
(runs ``capture_mmar_attention.py`` on Modal, then syncs artifacts locally).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from audio_flamingo_runtime import (
    attention_artifact_id,
    audio_token_indices,
    load_attention_audio_gen_layer_matrix,
    load_attention_meta,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "outputs"
DEFAULT_DATA_DIR = ROOT / "data" / "mmar"
DEFAULT_META = DEFAULT_DATA_DIR / "MMAR-meta.jsonl"
DEFAULT_AUDIO_DIR = DEFAULT_DATA_DIR / "audio"
RESULTS_VOLUME_NAME = "latent-reasoning-results"
MMAR_META_URL = (
    "https://raw.githubusercontent.com/ddlBoJack/MMAR/main/MMAR-meta.jsonl"
)
MMAR_REPO = "BoJack/MMAR"
MMAR_AUDIO_ARCHIVE = "mmar-audio.tar.gz"
MIN_MMAR_WAVS = 1000

# Populated in main() before the server starts.
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


def count_wavs(audio_dir: Path) -> int:
    if not audio_dir.is_dir():
        return 0
    return sum(
        1
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def ensure_meta(meta_path: Path) -> Path:
    """Download MMAR-meta.jsonl (with rubrics) if missing."""
    if meta_path.exists() and meta_path.stat().st_size > 0:
        return meta_path
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MMAR-meta.jsonl -> {meta_path}")
    try:
        urllib.request.urlretrieve(MMAR_META_URL, meta_path)
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Failed to download MMAR meta from {MMAR_META_URL}: {exc}\n"
            f"Pass --meta PATH to an existing MMAR-meta.jsonl."
        ) from exc
    return meta_path


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


@lru_cache(maxsize=1)
def meta_by_id() -> dict[str, dict]:
    path = Path(CONFIG["meta"])
    return {item["id"]: item for item in load_jsonl(path) if "id" in item}


def is_run_dir(path: Path) -> bool:
    """True if ``path`` looks like an AF3 MMAR run folder."""
    return path.is_dir() and (
        (path / "manifest.json").is_file() or (path / "predictions.jsonl").is_file()
    )


def iter_run_dirs(results_dir: Path) -> list[Path]:
    """Find run folders under ``results_dir``.

    Supports both layouts produced by ``download_results.py``:
      - outputs/<run_id>/                  (download of mmar/af3)
      - outputs/mmar/af3/<run_id>/         (full volume sync)
    Also works when ``--results-dir`` points directly at ``mmar/af3``.
    """
    if not results_dir.is_dir():
        return []

    found: dict[str, Path] = {}

    def _add_from(parent: Path) -> None:
        if not parent.is_dir():
            return
        for child in parent.iterdir():
            if is_run_dir(child):
                found[child.name] = child

    _add_from(results_dir)
    _add_from(results_dir / "mmar" / "af3")
    _add_from(results_dir / "af3")
    _add_from(results_dir / "mmar" / "af-next-think")
    _add_from(results_dir / "af-next-think")

    return sorted(found.values(), key=lambda p: p.name, reverse=True)


def list_runs(results_dir: Path) -> list[dict]:
    runs = []
    for child in iter_run_dirs(results_dir):
        manifest = load_json(child / "manifest.json") or {}
        scores = load_json(child / "scores.json")
        preds = child / "predictions.jsonl"
        evaluated = child / "predictions.evaluated.jsonl"
        n_preds = sum(1 for _ in open(preds, encoding="utf-8")) if preds.exists() else 0
        n_eval = (
            sum(1 for _ in open(evaluated, encoding="utf-8")) if evaluated.exists() else 0
        )
        runs.append(
            {
                "id": child.name,
                "path": str(child),
                "manifest": manifest,
                "scores": scores,
                "n_predictions": n_preds,
                "n_evaluated": n_eval,
                "has_predictions": n_preds > 0,
                "has_evaluated": n_eval > 0,
            }
        )
    return runs


def resolve_run_dir(results_dir: Path, run_id: str | None) -> Path | None:
    run_dirs = iter_run_dirs(results_dir)
    if run_id:
        for path in run_dirs:
            if path.name == run_id:
                return path
        # Back-compat: allow an explicit relative path under results_dir.
        path = results_dir / run_id
        return path if is_run_dir(path) else None
    return run_dirs[0] if run_dirs else None


def index_by_id(path: Path) -> dict[str, dict]:
    return {item["id"]: item for item in load_jsonl(path) if "id" in item}


def merge_example(
    meta: dict,
    prediction: dict | None,
    evaluated: dict | None,
) -> dict:
    """Prefer evaluated > prediction > meta for overlapping fields."""
    out = dict(meta)
    if prediction:
        out.update(prediction)
    if evaluated:
        out.update(evaluated)
    out["has_generation"] = bool(
        (prediction or evaluated)
        and (
            (prediction or evaluated or {}).get("raw_tokens")
            or (prediction or evaluated or {}).get("model_output")
            or (prediction or evaluated or {}).get("answer_prediction")
            or (prediction or evaluated or {}).get("thinking_prediction")
        )
    )
    out["has_grade"] = evaluated is not None and "score" in (evaluated or {})
    return out


def build_examples(run_dir: Path | None, *, only_generated: bool = False) -> list[dict]:
    meta = meta_by_id()
    predictions: dict[str, dict] = {}
    evaluated: dict[str, dict] = {}
    if run_dir is not None:
        predictions = index_by_id(run_dir / "predictions.jsonl")
        evaluated = index_by_id(run_dir / "predictions.evaluated.jsonl")

    ids: list[str] = []
    seen: set[str] = set()
    # Prefer run order (evaluated, then predictions), then remaining meta.
    for source in (evaluated, predictions):
        for item_id in source:
            if item_id not in seen:
                ids.append(item_id)
                seen.add(item_id)
    if not only_generated:
        for item_id in meta:
            if item_id not in seen:
                ids.append(item_id)
                seen.add(item_id)

    examples = []
    for item_id in ids:
        base = meta.get(item_id) or predictions.get(item_id) or evaluated.get(item_id) or {}
        if "id" not in base:
            continue
        example = merge_example(
            base if item_id in meta else {**base, **{k: v for k, v in (meta.get(item_id) or {}).items()}},
            predictions.get(item_id),
            evaluated.get(item_id),
        )
        # Ensure rubric / thinking from meta win when prediction overwrote with missing keys.
        if item_id in meta:
            for key in ("rubric", "thinking", "cue", "question", "choices", "answer"):
                if key in meta[item_id] and (
                    key not in example or example.get(key) in (None, "", [])
                ):
                    example[key] = meta[item_id][key]
                elif key in meta[item_id] and key in ("rubric", "thinking", "cue"):
                    # Always prefer full meta rubrics / GT CoT when present.
                    example[key] = meta[item_id][key]
        if only_generated and not example.get("has_generation"):
            continue
        examples.append(summarize_example(example))
    return examples


def summarize_example(item: dict) -> dict:
    """Strip bulky fields for the list API; keep enough for the card."""
    n_shots = item.get("n_shots")
    n_shot_correct = item.get("n_shot_correct")
    shot_success_rate = item.get("shot_success_rate")
    shots = item.get("shots") or []
    if n_shots is None and shots:
        n_shots = len(shots)
    if n_shot_correct is None and shots:
        n_shot_correct = sum(1 for shot in shots if shot.get("correct"))
    if shot_success_rate is None and n_shots:
        shot_success_rate = (n_shot_correct or 0) / n_shots
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "choices": item.get("choices") or [],
        "modality": item.get("modality"),
        "category": item.get("category"),
        "sub_category": item.get("sub-category") or item.get("sub_category"),
        "language": item.get("language"),
        "url": item.get("url"),
        "audio_path": item.get("audio_path"),
        "has_generation": bool(item.get("has_generation")),
        "has_grade": bool(item.get("has_grade")),
        "correct": item.get("correct"),
        "score": item.get("score"),
        "answer_prediction": item.get("answer_prediction"),
        "n_shots": n_shots,
        "n_shot_correct": n_shot_correct,
        "shot_success_rate": shot_success_rate,
        "any_shot_success": bool(item.get("correct"))
        if item.get("correct") is not None
        else None,
        "n_rubric": len(item.get("rubric") or []),
        "n_cues": len(item.get("cue") or []),
    }


def full_example(item_id: str, run_dir: Path | None) -> dict | None:
    meta = meta_by_id()
    prediction = None
    evaluated = None
    if run_dir is not None:
        prediction = index_by_id(run_dir / "predictions.jsonl").get(item_id)
        evaluated = index_by_id(run_dir / "predictions.evaluated.jsonl").get(item_id)
    base = meta.get(item_id) or prediction or evaluated
    if not base:
        return None
    example = merge_example(dict(base), prediction, evaluated)
    if item_id in meta:
        for key in ("rubric", "thinking", "cue", "question", "choices", "answer", "url", "audio_path"):
            if key in meta[item_id]:
                example[key] = meta[item_id][key]
    example["local_audio"] = resolve_local_audio(example.get("audio_path"))
    attn_meta = load_attention_meta(run_dir, item_id) if run_dir is not None else None
    example["has_attention"] = attn_meta is not None
    example["attention_meta"] = attn_meta
    return example


def infer_results_subdir(run_dir: Path, manifest: dict | None = None) -> str:
    """Infer Modal volume subdir (mmar/af3 or mmar/af-next-think) for a local run."""
    manifest = manifest or load_json(run_dir / "manifest.json") or {}
    parts = list(run_dir.resolve().parts)
    if "mmar" in parts:
        idx = parts.index("mmar")
        if idx + 1 < len(parts) and parts[idx + 1] in {"af3", "af-next-think"}:
            return f"mmar/{parts[idx + 1]}"
    label = str(manifest.get("model_label") or "").lower()
    model_id = str(manifest.get("model_id") or "").lower()
    if "next" in label or "next" in model_id:
        return "mmar/af-next-think"
    return "mmar/af3"


def attention_paths(run_dir: Path, sample_id: str) -> tuple[Path, Path, Path, str]:
    artifact_id = attention_artifact_id(sample_id)
    attn_dir = run_dir / "attentions"
    return (
        attn_dir,
        attn_dir / f"{artifact_id}.npz",
        attn_dir / f"{artifact_id}.json",
        artifact_id,
    )


def remote_attention_prefix(
    run_dir: Path,
    sample_id: str,
    manifest: dict | None = None,
) -> str:
    """Volume-relative path prefix for an attention artifact (no extension)."""
    results_subdir = infer_results_subdir(run_dir, manifest)
    artifact_id = attention_artifact_id(sample_id)
    return f"{results_subdir}/{run_dir.name}/attentions/{artifact_id}"


def sync_attention_from_volume(
    run_dir: Path,
    sample_id: str,
    *,
    remote_prefix: str | None = None,
    manifest: dict | None = None,
) -> dict:
    """Download attention artifacts for ``sample_id`` into ``run_dir/attentions``."""
    manifest = manifest if manifest is not None else (load_json(run_dir / "manifest.json") or {})
    prefix = remote_prefix or remote_attention_prefix(run_dir, sample_id, manifest)
    artifact_id = attention_artifact_id(sample_id)
    saved = download_attention_artifacts(prefix, run_dir, artifact_id)
    meta = load_attention_meta(run_dir, sample_id) or {}
    return {
        "status": "ok",
        "downloaded": True,
        "attentions_remote": prefix,
        "files": saved,
        "attention_meta": meta,
    }


def download_attention_artifacts(
    remote_prefix: str,
    run_dir: Path,
    artifact_id: str,
    *,
    max_attempts: int = 10,
    initial_delay_s: float = 1.0,
) -> dict[str, str]:
    """Fetch ``attentions/<id>.{npz,json}`` from the results Volume into ``run_dir``.

    Large npz writes can be briefly listed on the Volume before their blocks are
    readable (``404 block not found``). Retry with backoff, and prefer the
    Python Volume API over ``modal volume get``.
    """
    import time

    import modal

    dest = run_dir / "attentions"
    dest.mkdir(parents=True, exist_ok=True)
    vol = modal.Volume.from_name(RESULTS_VOLUME_NAME)
    saved: dict[str, str] = {}

    # JSON first (small); then the large npz.
    for ext in (".json", ".npz"):
        remote = f"{remote_prefix}{ext}"
        local = dest / f"{artifact_id}{ext}"
        tmp = local.with_suffix(local.suffix + ".partial")
        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                print(
                    f"[viewer] downloading {remote} -> {local} "
                    f"(attempt {attempt}/{max_attempts})",
                    flush=True,
                )
                with open(tmp, "wb") as handle:
                    nbytes = vol.read_file_into_fileobj(remote, handle)
                tmp.replace(local)
                print(f"[viewer] wrote {nbytes} bytes to {local}", flush=True)
                saved[ext.lstrip(".")] = str(local)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 — retry volume races
                last_err = exc
                msg = str(exc).lower()
                retryable = (
                    "404" in msg
                    or "not found" in msg
                    or "block not found" in msg
                    or "no such file" in msg
                )
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                if not retryable or attempt >= max_attempts:
                    break
                delay = min(initial_delay_s * (2 ** (attempt - 1)), 30.0)
                print(
                    f"[viewer] download not ready ({exc}); retrying in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        if last_err is not None:
            raise RuntimeError(
                f"Failed to download {remote} after {max_attempts} attempts: {last_err}"
            ) from last_err
    return saved


def run_capture_attention_job(
    run_id: str,
    sample_id: str,
    results_subdir: str,
) -> dict:
    """Spawn Modal capture via ``modal run`` and parse the CAPTURE_RESULT line."""
    cmd = [
        "uv",
        "run",
        "modal",
        "run",
        str(ROOT / "capture_mmar_attention.py"),
        "--run-id",
        run_id,
        "--sample-id",
        sample_id,
        "--results-subdir",
        results_subdir,
    ]
    print(f"[viewer] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if stdout:
        print(stdout, flush=True)
    if stderr:
        print(stderr, flush=True)
    result = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("CAPTURE_RESULT:"):
            result = json.loads(line[len("CAPTURE_RESULT:") :])
            break
    if proc.returncode != 0:
        raise RuntimeError(
            f"modal capture failed (exit {proc.returncode}). "
            f"{(stderr or stdout)[-2000:]}"
        )
    if not isinstance(result, dict):
        raise RuntimeError(
            "modal capture finished without CAPTURE_RESULT JSON in stdout"
        )
    return result


def resolve_local_audio(audio_path: str | None) -> str | None:
    if not audio_path:
        return None
    audio_dir = Path(CONFIG["audio_dir"])
    candidates = [
        Path(audio_path),
        audio_dir / Path(audio_path).name,
        ROOT / audio_path.lstrip("./"),
        audio_dir.parent / audio_path.lstrip("./"),
    ]
    for path in candidates:
        if path.is_file():
            return f"/audio/{path.name}"
    return None


def find_audio_file(name: str) -> Path | None:
    name = Path(name).name
    audio_dir = Path(CONFIG["audio_dir"])
    candidates = [
        audio_dir / name,
        DEFAULT_AUDIO_DIR / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MMAR Rubrics Viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg: #e8eef2;
    --bg-2: #d9e3ea;
    --ink: #14202a;
    --muted: #5a6b78;
    --line: #b7c7d2;
    --card: #f7fafc;
    --accent: #1f5f8b;
    --accent-soft: #d5e8f4;
    --good: #1f6b4a;
    --bad: #9b3a3a;
    --warn: #7a5b16;
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
    font-weight: 600; font-size: 1.45rem;
    margin: 0 0 0.15rem; letter-spacing: -0.03em;
  }
  .brand p { margin: 0; color: var(--muted); font-size: 0.92rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; }
  label { font-size: 0.75rem; color: var(--muted); display: grid; gap: 0.25rem; }
  select, input[type="search"] {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.45rem 0.65rem; min-width: 10rem;
  }
  input[type="search"] { min-width: 14rem; }
  main {
    max-width: 1280px; margin: 0 auto; padding: 1.25rem;
    display: flex; flex-direction: column; gap: 1rem;
  }
  .content {
    display: grid; grid-template-columns: 340px 1fr; gap: 1rem;
    min-width: 0;
  }
  @media (max-width: 960px) {
    .content { grid-template-columns: 1fr; }
  }
  .panel {
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--radius); box-shadow: var(--shadow);
  }
  .stats {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem;
  }
  @media (max-width: 720px) { .stats { grid-template-columns: repeat(2, 1fr); } }
  .stat {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 10px; padding: 0.85rem 1rem;
  }
  .stat .k { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .stat .v { font-family: "Space Grotesk", sans-serif; font-size: 1.4rem; margin-top: 0.15rem; }
  .list-panel { overflow: hidden; display: flex; flex-direction: column; max-height: calc(100vh - 12rem); }
  .list-head {
    padding: 0.85rem 1rem; border-bottom: 1px solid var(--line);
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 0.85rem; color: var(--muted);
  }
  .list { overflow: auto; padding: 0.4rem; }
  .item {
    width: 100%; text-align: left; border: 0; background: transparent;
    border-radius: 8px; padding: 0.75rem 0.8rem; cursor: pointer;
    font: inherit; color: inherit; display: grid; gap: 0.35rem;
  }
  .item:hover { background: var(--bg-2); }
  .item.active { background: var(--accent-soft); }
  .item .qid {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.72rem; color: var(--muted);
  }
  .item .q { font-size: 0.92rem; line-height: 1.35; }
  .tags { display: flex; flex-wrap: wrap; gap: 0.3rem; }
  .tag {
    font-size: 0.68rem; padding: 0.15rem 0.45rem; border-radius: 4px;
    background: var(--bg-2); color: var(--muted); border: 1px solid var(--line);
  }
  .tag.good { background: #dff0e8; color: var(--good); border-color: #a9d4c0; }
  .tag.bad { background: #f3e0e0; color: var(--bad); border-color: #dfb4b4; }
  .tag.neutral { background: #ebe4cf; color: var(--warn); border-color: #d4c69a; }
  .detail { padding: 1.15rem 1.25rem 1.5rem; min-height: 60vh; }
  .detail.empty {
    display: grid; place-items: center; color: var(--muted); min-height: 40vh;
  }
  .detail h2 {
    font-family: "Space Grotesk", sans-serif; font-weight: 600;
    font-size: 1.3rem; margin: 0 0 0.75rem; line-height: 1.25; letter-spacing: -0.02em;
  }
  .meta-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1rem; }
  .section { margin-top: 1.25rem; }
  .section h3 {
    margin: 0 0 0.55rem; font-size: 0.78rem; letter-spacing: 0.05em;
    text-transform: uppercase; color: var(--muted); font-weight: 600;
  }
  .section-head {
    display: flex; flex-wrap: wrap; align-items: center; gap: 0.55rem;
    margin-bottom: 0.55rem;
  }
  .section-head h3 { margin: 0; }
  .section-head select {
    font: inherit; color: var(--ink);
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 0.25rem 0.5rem; min-width: 8rem;
    text-transform: none; letter-spacing: normal; font-size: 0.82rem;
  }
  .choices { display: grid; gap: 0.4rem; }
  .choice {
    border: 1px solid var(--line); border-radius: 8px; padding: 0.55rem 0.75rem;
    background: var(--bg);
  }
  .choice.correct { border-color: #8fcbb4; background: #eef8f3; }
  .choice.pred { outline: 2px solid color-mix(in srgb, var(--accent) 45%, transparent); }
  .choice .label {
    font-family: "IBM Plex Mono", monospace; font-size: 0.75rem; color: var(--muted);
    margin-right: 0.4rem;
  }
  .box {
    border: 1px solid var(--line); border-radius: 10px; padding: 0.85rem 1rem;
    background: var(--bg); white-space: pre-wrap; line-height: 1.5; font-size: 0.95rem;
  }
  .box.mono {
    font-family: "IBM Plex Mono", monospace; font-size: 0.85rem;
    overflow-x: auto; max-height: 28rem; overflow-y: auto;
  }
  .box.muted { color: var(--muted); }
  .section-hint {
    font-size: 0.75rem; color: var(--muted); font-weight: 400;
    margin-left: 0.35rem;
  }
  .token-legend {
    display: flex; flex-wrap: wrap; gap: 0.55rem; align-items: center;
    margin-bottom: 0.55rem; font-size: 0.75rem; color: var(--muted);
  }
  .token-legend .swatch {
    display: inline-flex; align-items: center; gap: 0.3rem;
  }
  .token-legend .tok {
    margin: 0; padding: 0.05rem 0.35rem; font-size: 0.72rem;
  }
  .token-viewer {
    border: 1px solid var(--line); border-radius: 10px; padding: 0.75rem 0.85rem;
    background: var(--bg); max-height: 28rem; overflow: auto;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.8rem; line-height: 1.85; word-break: break-word;
  }
  .tok {
    display: inline; border-radius: 3px; padding: 0.08em 0.18em;
    margin: 0 1px; border: 1px solid transparent; white-space: pre-wrap;
  }
  .tok.input {
    background: #e4edf4; border-color: #c2d3e0; color: #2a4050;
  }
  .tok.generated {
    background: #e7f3ea; border-color: #b5d6be; color: #1f4d32;
  }
  .tok.special {
    background: #f3ebe0; border-color: #d8c4a4; color: #6a4a1a;
    font-weight: 500;
  }
  .tok.special.input { background: #efe6d8; }
  .tok.special.generated { background: #e9efd9; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem; }
  @media (max-width: 720px) { .grid-2 { grid-template-columns: 1fr; } }
  .rubric { display: grid; gap: 0.55rem; }
  .rubric-item {
    border: 1px solid var(--line); border-radius: 10px; padding: 0.75rem 0.9rem;
    background: var(--card);
  }
  .rubric-item.pass { border-color: #8fcbb4; background: #f3faf7; }
  .rubric-item.fail { border-color: #dfb4b4; background: #faf3f3; }
  .rubric-item .name { font-weight: 600; margin-bottom: 0.25rem; }
  .rubric-item .point { color: var(--muted); font-size: 0.9rem; }
  .score-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-family: "IBM Plex Mono", monospace; font-size: 0.8rem;
  }
  audio {
    width: 100%;
    margin-top: 0.35rem;
    height: 2.5rem;
  }
  .audio-player {
    border: 1px solid var(--line); border-radius: 10px; padding: 0.75rem 1rem;
    background: var(--bg);
  }
  .audio-player .missing { color: var(--muted); font-size: 0.9rem; }
  a { color: var(--accent); }
  .btn {
    font: inherit; cursor: pointer; color: #fff;
    background: var(--accent); border: 1px solid var(--accent);
    border-radius: 8px; padding: 0.5rem 0.85rem;
  }
  .btn:hover { filter: brightness(1.05); }
  .btn:disabled { opacity: 0.55; cursor: wait; }
  .btn.secondary {
    background: var(--card); color: var(--ink); border-color: var(--line);
  }
  .attn-controls {
    display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: end;
    margin-bottom: 0.75rem;
  }
  .attn-controls label { min-width: 5rem; }
  .attn-controls select { min-width: 5.5rem; }
  .attn-status { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.65rem; }
  .attn-heatmap-wrap {
    border: 1px solid var(--line); border-radius: 10px; padding: 0.75rem;
    background: var(--bg); overflow: auto; max-height: min(70vh, 48rem);
  }
  .attn-axis-label {
    font-size: 0.72rem; color: var(--muted); margin: 0 0 0.35rem;
  }
  .attn-legend {
    display: flex; align-items: flex-start; gap: 0.55rem;
    margin: 0 0 0.55rem; font-family: "IBM Plex Mono", monospace;
    font-size: 0.72rem; color: var(--muted);
  }
  .attn-legend-scale {
    display: flex; flex-direction: column; gap: 0.2rem;
    width: 10rem; flex: 0 0 10rem;
  }
  .attn-legend-bar {
    width: 100%; height: 0.7rem; border-radius: 3px;
    border: 1px solid var(--line);
    background: linear-gradient(
      90deg,
      rgb(70, 60, 140) 0%,
      rgb(50, 120, 115) 50%,
      rgb(30, 180, 40) 100%
    );
  }
  .attn-legend-labels {
    display: flex; justify-content: space-between; width: 100%;
  }
  .attn-grid {
    display: grid;
    grid-template-columns: minmax(7rem, max-content) 1fr;
    gap: 0.35rem 0.55rem;
    align-items: start;
  }
  .attn-layer-labels {
    grid-column: 2;
    display: flex; font-family: "IBM Plex Mono", monospace;
    font-size: 0.62rem; color: var(--muted); line-height: 1;
  }
  .attn-layer-labels span {
    flex: 1 1 0; text-align: center; min-width: 0;
  }
  .attn-token-col {
    display: flex; flex-direction: column;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.72rem; line-height: 1;
  }
  .attn-token-col .tok {
    display: block; margin: 0; padding: 0 0.35rem;
    height: 16px; line-height: 16px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    max-width: 14rem; border-radius: 2px;
  }
  .attn-canvas {
    display: block;
    image-rendering: pixelated;
    cursor: crosshair;
  }
  .attn-tooltip {
    position: fixed; z-index: 40; pointer-events: none;
    display: none; padding: 0.35rem 0.5rem; border-radius: 6px;
    background: rgba(18, 18, 22, 0.92); color: #f2f2f4;
    border: 1px solid rgba(255, 255, 255, 0.12);
    font-family: "IBM Plex Mono", monospace; font-size: 0.72rem;
    line-height: 1.35; white-space: nowrap; box-shadow: 0 6px 18px rgba(0,0,0,0.25);
  }
  .attn-tooltip.visible { display: block; }
</style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div class="brand">
        <h1>MMAR Rubrics Viewer</h1>
        <p>Examples, AF3 generations, and OpenAI rubric grades</p>
      </div>
      <div class="controls">
        <label>Run
          <select id="run"></select>
        </label>
        <label>Filter
          <select id="filter">
            <option value="all">All examples</option>
            <option value="generated">With generations</option>
            <option value="graded">Graded</option>
            <option value="correct">Correct</option>
            <option value="incorrect">Incorrect</option>
            <option value="ungenerated">No generation yet</option>
          </select>
        </label>
        <label>Modality
          <select id="modality"><option value="">All</option></select>
        </label>
        <label>Search
          <input id="search" type="search" placeholder="id, question, answer…" />
        </label>
      </div>
    </div>
  </header>
  <main>
    <div class="stats" id="stats"></div>
    <div class="content">
      <div class="panel list-panel">
        <div class="list-head">
          <span id="list-count">0 examples</span>
          <span id="run-label"></span>
        </div>
        <div class="list" id="list"></div>
      </div>
      <div class="panel detail empty" id="detail">Select an example</div>
    </div>
  </main>
<script>
const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
let state = { runs: [], examples: [], selectedId: null, runId: null, selectedShotIndex: 0, currentExample: null };

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function fmtScore(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(3);
}

function tag(text, cls="") {
  return `<span class="tag ${cls}">${escapeHtml(text)}</span>`;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  }[c]));
}

function renderStats(run) {
  const scores = run?.scores || {};
  const m = run?.manifest || {};
  const cards = [
    ["Examples", state.examples.length],
    ["Generated", state.examples.filter(e => e.has_generation).length],
    ["Accuracy", scores.accuracy != null ? (100*scores.accuracy).toFixed(1)+"%" : "—"],
    [
      scores.avg_shot_success_rate != null ? "Avg shot rate" : "Avg rubric",
      scores.avg_shot_success_rate != null
        ? (100*scores.avg_shot_success_rate).toFixed(1)+"%"
        : (scores.avg_score != null ? fmtScore(scores.avg_score) : "—"),
    ],
  ];
  document.getElementById("stats").innerHTML = cards.map(([k,v]) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`
  ).join("");
  const nShots = scores.n_shots ?? m.n_shots;
  const model = m.model_id ? m.model_id.split("/").pop() : (run?.id || "");
  document.getElementById("run-label").textContent =
    nShots && nShots > 1 ? `${model} · ${nShots}-shot` : model;
}

function filteredExamples() {
  const filter = document.getElementById("filter").value;
  const modality = document.getElementById("modality").value;
  const q = document.getElementById("search").value.trim().toLowerCase();
  return state.examples.filter(e => {
    if (modality && e.modality !== modality) return false;
    if (filter === "generated" && !e.has_generation) return false;
    if (filter === "graded" && !e.has_grade) return false;
    if (filter === "correct" && e.correct !== true) return false;
    if (filter === "incorrect" && e.correct !== false) return false;
    if (filter === "ungenerated" && e.has_generation) return false;
    if (q) {
      const hay = [e.id, e.question, e.answer, e.answer_prediction, e.category, e.modality]
        .join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function renderList() {
  const items = filteredExamples();
  document.getElementById("list-count").textContent = `${items.length} examples`;
  const list = document.getElementById("list");
  list.innerHTML = items.map(e => {
    const tags = [];
    if (e.modality) tags.push(tag(e.modality));
    if (e.category) tags.push(tag(e.category));
    if (e.correct === true || e.correct === false) {
      const label = e.n_shots && e.n_shots > 1
        ? (e.correct ? "any success" : "no success")
        : (e.correct ? "correct" : "incorrect");
      tags.push(tag(label, e.correct ? "good" : "bad"));
    }
    if (e.n_shots && e.n_shots > 1 && e.n_shot_correct != null) {
      tags.push(tag(`${e.n_shot_correct}/${e.n_shots} shots`, "neutral"));
    } else if (e.has_grade) {
      tags.push(tag(`score ${fmtScore(e.score)}`, "neutral"));
    } else if (e.has_generation) {
      tags.push(tag("generated", "neutral"));
    }
    return `<button class="item ${e.id===state.selectedId?"active":""}" data-id="${escapeHtml(e.id)}">
      <div class="qid">${escapeHtml(e.id)}</div>
      <div class="q">${escapeHtml(e.question || "")}</div>
      <div class="tags">${tags.join("")}</div>
    </button>`;
  }).join("") || `<div style="padding:1rem;color:var(--muted)">No examples match.</div>`;
  list.querySelectorAll(".item").forEach(btn => {
    btn.addEventListener("click", () => selectExample(btn.dataset.id));
  });
}

function choiceClass(choice, answer, pred) {
  const classes = ["choice"];
  if (choice === answer) classes.push("correct");
  if (pred && choice === pred) classes.push("pred");
  return classes.join(" ");
}

function exampleShots(example) {
  return Array.isArray(example?.shots) ? example.shots : [];
}

function defaultShotIndex(example) {
  const shots = exampleShots(example);
  if (!shots.length) return 0;
  const firstSuccess = shots.findIndex(s => s.correct);
  return firstSuccess >= 0 ? firstSuccess : 0;
}

function viewForShot(example, shotIndex) {
  const shots = exampleShots(example);
  if (!shots.length) return example;
  const idx = Math.max(0, Math.min(shotIndex, shots.length - 1));
  const shot = shots[idx] || {};
  return {
    ...example,
    selected_shot_index: idx,
    answer_prediction: shot.answer_prediction ?? example.answer_prediction,
    thinking_prediction: shot.thinking_prediction ?? example.thinking_prediction,
    model_output: shot.model_output ?? example.model_output,
    raw_tokens: shot.raw_tokens ?? example.raw_tokens,
    shot_correct: shot.correct,
  };
}

function renderRawText(example) {
  const tokens = example.raw_tokens;
  if (Array.isArray(tokens) && tokens.length) {
    const nIn = tokens.filter(t => t.role === "input").length;
    const nGen = tokens.filter(t => t.role === "generated").length;
    const nSp = tokens.filter(t => t.special).length;
    const chips = tokens.map(t => {
      const role = t.role === "generated" ? "generated" : "input";
      const cls = ["tok", role].concat(t.special ? ["special"] : []).join(" ");
      const title = `id=${t.id} · ${role}${t.special ? " · special" : ""}`;
      return `<span class="${cls}" title="${escapeHtml(title)}">${escapeHtml(String(t.token ?? ""))}</span>`;
    }).join("");
    return `
      <div class="token-legend">
        <span class="swatch"><span class="tok input">in</span> input (${nIn})</span>
        <span class="swatch"><span class="tok generated">gen</span> generated (${nGen})</span>
        <span class="swatch"><span class="tok special">sp</span> special (${nSp})</span>
        <span>${tokens.length} tokens</span>
      </div>
      <div class="token-viewer">${chips}</div>`;
  }
  if (example.model_output) {
    return `<div class="box mono">${escapeHtml(example.model_output)}</div>
      <div class="section-hint" style="margin:0.4rem 0 0;display:block">
        Continuous decode only (no per-token list on this record).
      </div>`;
  }
  if (example.has_generation) {
    return `<div class="box mono muted">No raw_tokens / model_output on this record (older run may have omitted them).</div>`;
  }
  return `<div class="box mono muted">No generation in this run.</div>`;
}

function renderRubric(example) {
  const rubric = example.rubric || [];
  const results = example.rubric_results || [];
  const byName = Object.fromEntries((results || []).map(r => [r.name, r]));
  if (!rubric.length) return `<div class="box">No rubric on this example.</div>`;
  return `<div class="rubric">${rubric.map(r => {
    const res = byName[r.name];
    let cls = "rubric-item";
    let badge = "";
    const passed = (typeof res?.pass === "boolean") ? res.pass
      : (typeof res?.passed === "boolean") ? res.passed : null;
    if (passed === true || passed === false) {
      cls += passed ? " pass" : " fail";
      badge = `<span class="score-pill">${passed ? "pass" : "fail"}</span>`;
    } else if (res && res.score != null) {
      badge = `<span class="score-pill">score ${escapeHtml(res.score)}</span>`;
    }
    return `<div class="${cls}">
      <div class="name">${escapeHtml(r.name)} ${badge}</div>
      <div class="point">${escapeHtml(r.scoring_point || "")}</div>
      ${r.note ? `<div class="point" style="margin-top:0.35rem">${escapeHtml(r.note)}</div>` : ""}
      ${res?.justification ? `<div class="point" style="margin-top:0.45rem"><em>${escapeHtml(res.justification)}</em></div>` : ""}
    </div>`;
  }).join("")}</div>`;
}

function renderShotSelectOptions(example, selectedIndex) {
  const shots = exampleShots(example);
  if (shots.length <= 1) return "";
  return shots.map((shot, i) => {
    const idx = shot.shot_index ?? i;
    const ok = !!shot.correct;
    const label = `Shot ${idx}${ok ? " · success" : " · fail"}`;
    return `<option value="${i}" ${i === selectedIndex ? "selected" : ""}>${escapeHtml(label)}</option>`;
  }).join("");
}

function renderAttentionSection(example) {
  const meta = example.attention_meta;
  const matchBadge = meta
    ? (meta.token_match
        ? tag("token match", "good")
        : tag("token mismatch", "bad"))
    : "";
  const status = meta
    ? `${meta.num_steps || "?"} gen steps · ${meta.num_layers || "?"} layers · ${meta.num_heads || "?"} heads · seed ${meta.seed ?? "?"} ${matchBadge}`
    : "No attention artifact yet. Capture re-runs this example on Modal with eager attention.";
  const layers = meta?.num_layers || 28;
  const layerAxis = Array.from({length: layers}, (_, i) =>
    `<span title="layer ${i}">${i}</span>`).join("");
  return `
    <div class="section" id="attention-section">
      <h3>Attention<span class="section-hint">generated tokens × layers · audio attention</span></h3>
      <div class="attn-status" id="attn-status">${status}</div>
      <div class="attn-controls">
        <button type="button" class="btn" id="capture-attn">
          ${meta ? "Re-capture attention" : "Capture attention"}
        </button>
        <button type="button" class="btn secondary" id="sync-attn">
          Pull from volume
        </button>
        ${meta ? `
        <label>Reduce
          <select id="attn-reduce">
            <option value="avg" selected>avg</option>
            <option value="sum">sum</option>
            <option value="max">max</option>
          </select>
        </label>
        ` : ""}
      </div>
      ${meta ? `
        <div class="attn-heatmap-wrap">
          <div class="attn-axis-label">Horizontal: layers 0…${layers - 1} · Vertical: generated tokens · cell = avg/sum/max over audio tokens and heads (sum ÷ heads)</div>
          <div class="attn-legend" id="attn-legend">
            <div class="attn-legend-scale">
              <div class="attn-legend-bar" aria-hidden="true"></div>
              <div class="attn-legend-labels">
                <span id="attn-legend-min">0</span>
                <span id="attn-legend-mid">0.5</span>
                <span id="attn-legend-max">1</span>
              </div>
            </div>
          </div>
          <div class="attn-grid">
            <div></div>
            <div class="attn-layer-labels" id="attn-layer-labels">${layerAxis}</div>
            <div class="attn-token-col" id="attn-tokens"></div>
            <canvas class="attn-canvas" id="attn-canvas"></canvas>
          </div>
        </div>
        <div class="attn-tooltip" id="attn-tooltip" role="tooltip"></div>
      ` : ""}
    </div>`;
}

function generatedTokensForAttention(example) {
  const tokens = example.raw_tokens || [];
  const generated = tokens.filter(t => t.role === "generated");
  if (generated.length) return generated;
  const ids = example.attention_meta?.generated_ids || [];
  return ids.map((id, i) => ({ id, token: `[${i}]`, role: "generated", special: false }));
}

function attnColor(t) {
  const u = Math.max(0, Math.min(1, t));
  const r = Math.round(30 + 40 * (1 - u));
  const g = Math.round(60 + 120 * u);
  const b = Math.round(90 + 50 * (1 - u));
  return `rgb(${r},${g},${b})`;
}

function paintAttentionLegend() {
  const minEl = document.getElementById("attn-legend-min");
  const midEl = document.getElementById("attn-legend-mid");
  const maxEl = document.getElementById("attn-legend-max");
  const bar = document.querySelector(".attn-legend-bar");
  if (!minEl || !midEl || !maxEl) return;
  minEl.textContent = "0";
  midEl.textContent = "0.5";
  maxEl.textContent = "1";
  if (bar) {
    bar.style.background = `linear-gradient(90deg, ${attnColor(0)}, ${attnColor(0.5)}, ${attnColor(1)})`;
  }
}

function formatAttnValue(v) {
  if (!Number.isFinite(v)) return "—";
  return Number(v).toFixed(4);
}

function hideAttnTooltip() {
  const tip = document.getElementById("attn-tooltip");
  if (tip) tip.classList.remove("visible");
}

function bindAttentionHover(canvas, hit) {
  canvas._attnHit = hit;
  if (canvas._attnHoverBound) return;
  canvas._attnHoverBound = true;

  canvas.addEventListener("mousemove", (ev) => {
    const h = canvas._attnHit;
    const tip = document.getElementById("attn-tooltip");
    if (!h || !tip) return;
    const rect = canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const li = Math.floor(x / h.colW);
    const ti = Math.floor(y / h.rowH);
    if (li < 0 || ti < 0 || li >= h.numLayers || ti >= h.numSteps) {
      tip.classList.remove("visible");
      return;
    }
    const value = h.matrix[ti]?.[li];
    const tok = h.tokens[ti];
    const label = tok?.token != null ? String(tok.token) : `[${ti}]`;
    tip.innerHTML =
      `<strong>${formatAttnValue(value)}</strong>` +
      `<br>layer ${li} · gen ${ti} · ${escapeHtml(label)}`;
    const pad = 12;
    let left = ev.clientX + pad;
    let top = ev.clientY + pad;
    tip.classList.add("visible");
    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    if (left + tw > window.innerWidth - 8) left = ev.clientX - tw - pad;
    if (top + th > window.innerHeight - 8) top = ev.clientY - th - pad;
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  });

  canvas.addEventListener("mouseleave", hideAttnTooltip);
}

function paintAttentionHeatmap(matrix, tokens) {
  const canvas = document.getElementById("attn-canvas");
  const tokenCol = document.getElementById("attn-tokens");
  if (!canvas || !matrix?.length) return;
  const numSteps = matrix.length;
  const numLayers = matrix[0]?.length || 0;
  if (!numLayers) return;

  // Measure from the scroll wrap (stable), not the grid — sizing the canvas
  // from the grid's clientWidth feeds back and grows on every repaint.
  canvas.style.width = "0px";
  canvas.style.height = "0px";
  const wrap = canvas.closest(".attn-heatmap-wrap");
  const tokenW = tokenCol?.offsetWidth || 112;
  const availW = Math.max(
    numLayers * 10,
    (wrap?.clientWidth || 320) - tokenW - 36
  );
  const rowH = 16;
  const colW = Math.max(10, Math.floor(availW / numLayers));
  canvas.width = numLayers * colW;
  canvas.height = numSteps * rowH;
  canvas.style.width = `${canvas.width}px`;
  canvas.style.height = `${canvas.height}px`;

  // Fixed 0–1 scale: avg/max are means/maxima over audio×heads; sum is
  // total audio mass averaged over heads (also in [0,1]).
  paintAttentionLegend();

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let ti = 0; ti < numSteps; ti++) {
    const row = matrix[ti];
    for (let li = 0; li < numLayers; li++) {
      ctx.fillStyle = attnColor(row[li]);
      ctx.fillRect(li * colW, ti * rowH, colW, rowH);
    }
  }

  const slice = tokens.slice(0, numSteps);
  while (slice.length < numSteps) {
    slice.push({ token: `[${slice.length}]`, role: "generated", special: false });
  }
  if (tokenCol) {
    tokenCol.innerHTML = slice.map((t, i) => {
      const cls = ["tok", "generated"].concat(t.special ? ["special"] : []).join(" ");
      const label = String(t.token ?? "");
      return `<span class="${cls}" title="${escapeHtml(label)} · gen ${i}">${escapeHtml(label)}</span>`;
    }).join("");
  }

  bindAttentionHover(canvas, {
    matrix, tokens: slice, colW, rowH, numLayers, numSteps,
  });
  hideAttnTooltip();
}

async function loadAttentionHeatmap(example) {
  const reduce = document.getElementById("attn-reduce")?.value || "avg";
  const run = encodeURIComponent(state.runId || "");
  const status = document.getElementById("attn-status");
  try {
    const data = await api(
      `/api/attention-data?run=${run}&id=${encodeURIComponent(example.id)}&reduce=${encodeURIComponent(reduce)}`
    );
    paintAttentionHeatmap(data.matrix || [], generatedTokensForAttention(example));
    if (status && example.attention_meta) {
      const meta = example.attention_meta;
      const matchBadge = meta.token_match
        ? tag("token match", "good")
        : tag("token mismatch", "bad");
      status.innerHTML =
        `${reduce} · ${data.num_steps} gen tokens × ${data.num_layers} layers · ` +
        `${data.num_audio_keys} audio keys · ${data.num_heads} heads · seed ${meta.seed ?? "?"} ${matchBadge}`;
    }
  } catch (err) {
    if (status) status.textContent = String(err);
  }
}

async function syncAttention(example) {
  const btn = document.getElementById("sync-attn");
  const status = document.getElementById("attn-status");
  if (btn) { btn.disabled = true; btn.textContent = "Pulling…"; }
  if (status) status.textContent = "Downloading attentions from Modal volume…";
  try {
    const res = await fetch("/api/sync-attention", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ run: state.runId, id: example.id }),
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || res.statusText);
    if (status) status.textContent = "Downloaded attention artifact from volume.";
    await selectExample(example.id);
  } catch (err) {
    if (status) status.textContent = String(err);
    if (btn) { btn.disabled = false; btn.textContent = "Pull from volume"; }
  }
}

async function captureAttention(example) {
  const btn = document.getElementById("capture-attn");
  const status = document.getElementById("attn-status");
  if (btn) { btn.disabled = true; btn.textContent = "Capturing on Modal…"; }
  if (status) status.textContent = "Running capture on Modal (model load + generate). This can take several minutes…";
  try {
    const res = await fetch("/api/capture-attention", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ run: state.runId, id: example.id }),
    });
    const payload = await res.json();
    if (!res.ok && res.status !== 202) throw new Error(payload.error || res.statusText);
    if (payload.downloaded === false) {
      if (status) {
        status.textContent =
          `Capture succeeded on Modal, but local download failed: ${payload.download_error || "unknown"}. Use Pull from volume.`;
      }
      const syncBtn = document.getElementById("sync-attn");
      if (syncBtn) syncBtn.disabled = false;
      if (btn) { btn.disabled = false; btn.textContent = "Re-capture attention"; }
      return;
    }
    if (status) {
      status.textContent = `Captured · token_match=${payload.token_match} · ${payload.num_steps} steps · ${(payload.estimated_bytes/1e6).toFixed(1)} MB`;
    }
    await selectExample(example.id);
  } catch (err) {
    if (status) status.textContent = String(err);
    if (btn) { btn.disabled = false; btn.textContent = "Capture attention"; }
  }
}

async function selectExample(id, preferredShotIndex = null) {
  state.selectedId = id;
  renderList();
  const detail = document.getElementById("detail");
  detail.classList.remove("empty");
  detail.innerHTML = `<div style="color:var(--muted)">Loading…</div>`;
  const run = encodeURIComponent(state.runId || "");
  const example = await api(`/api/example?id=${encodeURIComponent(id)}&run=${run}`);
  state.currentExample = example;
  const shots = exampleShots(example);
  if (preferredShotIndex != null && preferredShotIndex >= 0 && preferredShotIndex < shots.length) {
    state.selectedShotIndex = preferredShotIndex;
  } else {
    state.selectedShotIndex = defaultShotIndex(example);
  }
  renderExampleDetail(example, state.selectedShotIndex);
}

function renderExampleDetail(example, shotIndex) {
  const detail = document.getElementById("detail");
  detail.classList.remove("empty");
  const view = viewForShot(example, shotIndex);
  const choices = example.choices || [];
  const cues = example.cue || [];
  const shots = exampleShots(example);
  const shotOptions = renderShotSelectOptions(example, view.selected_shot_index);
  detail.innerHTML = `
    <h2>${escapeHtml(example.question || "")}</h2>
    <div class="meta-row">
      ${example.modality ? tag(example.modality) : ""}
      ${example.category ? tag(example.category) : ""}
      ${example["sub-category"] ? tag(example["sub-category"]) : ""}
      ${example.correct === true || example.correct === false
        ? tag(
            (example.n_shots && example.n_shots > 1)
              ? (example.correct ? "any success" : "no success")
              : (example.correct ? "correct" : "incorrect"),
            example.correct ? "good" : "bad"
          )
        : ""}
      ${example.n_shots && example.n_shots > 1 && example.n_shot_correct != null
        ? tag(`${example.n_shot_correct}/${example.n_shots} shots`, "neutral")
        : ""}
      ${example.has_grade ? tag("score " + fmtScore(example.score), "neutral") : ""}
      ${example.has_attention ? tag("attention", "good") : ""}
      ${tag(example.id, "")}
    </div>

    <div class="section">
      <h3>Audio</h3>
      <div class="audio-player">
        ${
          example.local_audio
            ? `<audio controls preload="metadata" src="${escapeHtml(example.local_audio)}"></audio>`
            : `<div class="missing">Local wav not found for this example. Restart the viewer to download MMAR audio, or pass --audio-dir.</div>`
        }
      </div>
    </div>

    <div class="section">
      <h3>Choices</h3>
      <div class="choices">
        ${choices.map((c,i) => `
          <div class="${choiceClass(c, example.answer, view.answer_prediction)}">
            <span class="label">(${LABELS[i] || i})</span>${escapeHtml(c)}
            ${c === example.answer ? " · GT" : ""}
            ${view.answer_prediction && c === view.answer_prediction ? " · pred" : ""}
          </div>`).join("")}
      </div>
    </div>

    <div class="section grid-2">
      <div>
        <h3>Ground-truth reasoning</h3>
        <div class="box">${escapeHtml(example.thinking || "—")}</div>
      </div>
      <div>
        <div class="section-head">
          <h3>Model reasoning</h3>
          ${shotOptions ? `<select id="shot-select" aria-label="Select shot">${shotOptions}</select>` : ""}
        </div>
        <div class="box">${escapeHtml(view.thinking_prediction || (example.has_generation ? "—" : "No generation in this run"))}</div>
      </div>
    </div>

    <div class="section grid-2">
      <div>
        <h3>Ground-truth answer</h3>
        <div class="box">${escapeHtml(example.answer || "—")}</div>
      </div>
      <div>
        <h3>Model answer</h3>
        <div class="box">${escapeHtml(view.answer_prediction || "—")}</div>
      </div>
    </div>

    ${cues.length ? `<div class="section"><h3>Reasoning cues</h3><div class="tags">${cues.map(c => tag(c)).join("")}</div></div>` : ""}

    <div class="section">
      <h3>Raw text<span class="section-hint">${shots.length > 1 ? `shot ${view.selected_shot_index} · ` : ""}all tokens · input + specials + model-generated</span></h3>
      ${renderRawText(view)}
    </div>

    ${example.has_generation ? renderAttentionSection(example) : ""}

    <div class="section">
      <h3>Instance rubric${example.has_grade ? " + grades" : ""}</h3>
      ${renderRubric(example)}
    </div>
  `;
  const audio = detail.querySelector("audio");
  if (audio) audio.volume = 0.5;

  const shotSelect = document.getElementById("shot-select");
  if (shotSelect) {
    shotSelect.addEventListener("change", () => {
      state.selectedShotIndex = Number(shotSelect.value);
      renderExampleDetail(example, state.selectedShotIndex);
    });
  }

  const captureBtn = document.getElementById("capture-attn");
  if (captureBtn) {
    captureBtn.addEventListener("click", () => captureAttention(example));
  }
  const syncBtn = document.getElementById("sync-attn");
  if (syncBtn) {
    syncBtn.addEventListener("click", () => syncAttention(example));
  }
  const reduceSelect = document.getElementById("attn-reduce");
  if (reduceSelect) {
    reduceSelect.addEventListener("change", () => loadAttentionHeatmap(view));
    loadAttentionHeatmap(view);
  }
}

function fillModalities() {
  const mods = [...new Set(state.examples.map(e => e.modality).filter(Boolean))].sort();
  const sel = document.getElementById("modality");
  const current = sel.value;
  sel.innerHTML = `<option value="">All</option>` + mods.map(m =>
    `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");
  sel.value = mods.includes(current) ? current : "";
}

async function loadRun(runId) {
  state.runId = runId;
  const q = runId ? `?run=${encodeURIComponent(runId)}` : "";
  const data = await api(`/api/examples${q}`);
  state.examples = data.examples;
  const run = state.runs.find(r => r.id === runId) || data.run || null;
  renderStats(run);
  fillModalities();
  renderList();
  if (state.examples.length) {
    const first = filteredExamples()[0] || state.examples[0];
    await selectExample(first.id);
  } else {
    document.getElementById("detail").classList.add("empty");
    document.getElementById("detail").textContent = "No examples found.";
  }
}

async function init() {
  const data = await api("/api/runs");
  state.runs = data.runs;
  const sel = document.getElementById("run");
  if (!state.runs.length) {
    sel.innerHTML = `<option value="">(no runs — showing meta only)</option>`;
  } else {
    sel.innerHTML = state.runs.map(r => {
      const label = `${r.id} · ${r.n_predictions} preds · ${r.n_evaluated} graded`;
      return `<option value="${escapeHtml(r.id)}">${escapeHtml(label)}</option>`;
    }).join("");
  }
  const preferred = data.default_run || state.runs[0]?.id || "";
  if (preferred) sel.value = preferred;
  sel.addEventListener("change", () => loadRun(sel.value || null));
  ["filter","modality","search"].forEach(id => {
    document.getElementById(id).addEventListener("input", renderList);
    document.getElementById(id).addEventListener("change", renderList);
  });
  await loadRun(sel.value || null);
}

init().catch(err => {
  document.getElementById("detail").textContent = String(err);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "MMARViewer/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[viewer] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        results_dir = Path(CONFIG["results_dir"])

        if path in ("/", "/index.html"):
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/runs":
            runs = list_runs(results_dir)
            default = runs[0]["id"] if runs else None
            self._json(
                {
                    "runs": runs,
                    "default_run": default,
                    "meta": str(CONFIG["meta"]),
                    "n_meta": len(meta_by_id()),
                }
            )
            return

        if path == "/api/examples":
            run_id = (qs.get("run") or [None])[0] or None
            only = (qs.get("only_generated") or ["0"])[0] in ("1", "true", "yes")
            run_dir = resolve_run_dir(results_dir, run_id)
            examples = build_examples(run_dir, only_generated=only)
            run_info = None
            if run_dir is not None:
                run_info = next(
                    (r for r in list_runs(results_dir) if r["id"] == run_dir.name),
                    {
                        "id": run_dir.name,
                        "manifest": load_json(run_dir / "manifest.json"),
                        "scores": load_json(run_dir / "scores.json"),
                    },
                )
            self._json({"run": run_info, "examples": examples, "n": len(examples)})
            return

        if path == "/api/example":
            item_id = (qs.get("id") or [None])[0]
            if not item_id:
                self._json({"error": "missing id"}, 400)
                return
            run_id = (qs.get("run") or [None])[0] or None
            run_dir = resolve_run_dir(results_dir, run_id)
            example = full_example(item_id, run_dir)
            if example is None:
                self._json({"error": "not found"}, 404)
                return
            self._json(example)
            return

        if path == "/api/attention":
            item_id = (qs.get("id") or [None])[0]
            if not item_id:
                self._json({"error": "missing id"}, 400)
                return
            run_id = (qs.get("run") or [None])[0] or None
            run_dir = resolve_run_dir(results_dir, run_id)
            if run_dir is None:
                self._json({"error": "run not found"}, 404)
                return
            meta = load_attention_meta(run_dir, item_id)
            if meta is None:
                self._json({"error": "attention artifact not found", "has_attention": False}, 404)
                return
            self._json({"has_attention": True, **meta})
            return

        if path == "/api/attention-data":
            item_id = (qs.get("id") or [None])[0]
            if not item_id:
                self._json({"error": "missing id"}, 400)
                return
            run_id = (qs.get("run") or [None])[0] or None
            run_dir = resolve_run_dir(results_dir, run_id)
            if run_dir is None:
                self._json({"error": "run not found"}, 404)
                return
            reduce = (qs.get("reduce") or ["avg"])[0] or "avg"
            example = full_example(item_id, run_dir)
            if example is None:
                self._json({"error": "example not found"}, 404)
                return
            indices = audio_token_indices(example.get("raw_tokens"))
            if not indices:
                self._json({"error": "no <sound> audio tokens in raw_tokens"}, 400)
                return
            try:
                payload = load_attention_audio_gen_layer_matrix(
                    run_dir,
                    item_id,
                    audio_indices=indices,
                    reduce=reduce,
                )
            except FileNotFoundError as exc:
                self._json({"error": str(exc)}, 404)
                return
            except (KeyError, IndexError, ValueError) as exc:
                self._json({"error": str(exc)}, 400)
                return
            self._json(
                {
                    "id": item_id,
                    "run": run_dir.name,
                    **payload,
                }
            )
            return

        if path.startswith("/audio/"):
            name = unquote(path[len("/audio/") :])
            audio = find_audio_file(name)
            if audio is None:
                self._json({"error": "audio not found"}, 404)
                return
            data = audio.read_bytes()
            mime = mimetypes.guess_type(str(audio))[0] or "audio/wav"
            self._send(200, data, mime)
            return

        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path not in ("/api/capture-attention", "/api/sync-attention"):
            self._json({"error": "not found"}, 404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid JSON body"}, 400)
            return

        item_id = payload.get("id")
        run_id = payload.get("run")
        if not item_id or not run_id:
            self._json({"error": "missing id or run"}, 400)
            return

        results_dir = Path(CONFIG["results_dir"])
        run_dir = resolve_run_dir(results_dir, run_id)
        if run_dir is None:
            self._json({"error": f"run not found: {run_id}"}, 404)
            return

        example = full_example(item_id, run_dir)
        if example is None or not example.get("has_generation"):
            self._json(
                {"error": "example not found or has no generation in this run"},
                404,
            )
            return

        manifest = load_json(run_dir / "manifest.json") or {}
        results_subdir = infer_results_subdir(run_dir, manifest)

        if path == "/api/sync-attention":
            try:
                remote = payload.get("attentions_remote") or remote_attention_prefix(
                    run_dir, item_id, manifest
                )
                result = sync_attention_from_volume(
                    run_dir,
                    item_id,
                    remote_prefix=remote,
                    manifest=manifest,
                )
                self._json({**result, "local_run": str(run_dir)})
            except Exception as exc:  # noqa: BLE001 — surface to UI
                self._json({"error": str(exc)}, 500)
            return

        try:
            result = run_capture_attention_job(
                run_id=run_dir.name,
                sample_id=item_id,
                results_subdir=results_subdir,
            )
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self._json({"error": str(exc)}, 500)
            return

        remote = result.get("attentions_remote")
        artifact_id = attention_artifact_id(item_id)
        downloaded = False
        download_error = None
        if remote:
            try:
                download_attention_artifacts(remote, run_dir, artifact_id)
                downloaded = True
            except Exception as exc:  # noqa: BLE001 — capture still succeeded
                download_error = str(exc)
                print(f"[viewer] post-capture download failed: {exc}", flush=True)

        meta = load_attention_meta(run_dir, item_id) or {}
        self._json(
            {
                **result,
                "downloaded": downloaded,
                "download_error": download_error,
                "attention_meta": meta,
                "local_run": str(run_dir),
            },
            code=200 if downloaded else 202,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing AF3 MMAR run folders (default: outputs)",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_META,
        help="Path to MMAR-meta.jsonl (with rubrics)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Local wav directory for in-browser playback (default: data/mmar/audio)",
    )
    parser.add_argument(
        "--skip-meta-download",
        action="store_true",
        help="Do not auto-download meta if missing",
    )
    parser.add_argument(
        "--skip-audio-download",
        action="store_true",
        help="Do not auto-download MMAR wavs if missing/incomplete",
    )
    parser.add_argument(
        "--force-audio-download",
        action="store_true",
        help="Re-download and replace local MMAR audio even if present",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = args.meta.expanduser().resolve()
    audio_dir = args.audio_dir.expanduser().resolve()

    if not args.skip_meta_download:
        ensure_meta(meta_path)
    elif not meta_path.exists():
        raise SystemExit(f"Meta not found: {meta_path}")

    if not args.skip_audio_download:
        ensure_mmar_audio(audio_dir, force=args.force_audio_download)
    elif count_wavs(audio_dir) == 0:
        print(
            f"Warning: no wav files in {audio_dir}; "
            "audio playback will be unavailable.",
            flush=True,
        )

    CONFIG["results_dir"] = str(args.results_dir.expanduser().resolve())
    CONFIG["meta"] = str(meta_path)
    CONFIG["audio_dir"] = str(audio_dir)

    # Warm caches / validate.
    n_meta = len(meta_by_id())
    runs = list_runs(Path(CONFIG["results_dir"]))
    print(f"Loaded {n_meta} MMAR examples from {meta_path}", flush=True)
    print(f"Audio dir: {audio_dir} ({count_wavs(audio_dir)} wavs)", flush=True)
    print(f"Found {len(runs)} run(s) under {CONFIG['results_dir']}", flush=True)
    for run in runs[:5]:
        print(
            f"  - {run['id']}: {run['n_predictions']} predictions, "
            f"{run['n_evaluated']} evaluated",
            flush=True,
        )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Open {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
