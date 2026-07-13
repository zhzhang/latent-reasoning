"""Shared MMAR data helpers, rubric scoring, and Modal eval pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from modal_cache import RESULTS_MOUNT, RESULTS_VOLUME_NAME, results_volume, volume

# HF BoJack/MMAR MMAR-meta.json omits these; GitHub jsonl includes them.
MMAR_META_URL = "https://raw.githubusercontent.com/ddlBoJack/MMAR/main/MMAR-meta.jsonl"
MMAR_RUBRIC_KEYS = ("thinking", "rubric", "cue")

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
AF3_THINK_SUFFIX = "Please think and reason about the input audio before you respond."
AF_NEXT_THINK_SUFFIX = (
    "Reason step by step with timestamps, then give the final answer."
)
ANSWER_MARKERS = (
    r"therefore[, ]+the answer is[:\s]*",
    r"the answer is[:\s]*",
    r"final answer[:\s]*",
)
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path, records, mode="a"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def resolve_path(data_root, value):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(data_root) / path).resolve())


def count_wavs(audio_dir: Path) -> int:
    if not audio_dir.is_dir():
        return 0
    return sum(
        1
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )


# ---------------------------------------------------------------------------
# Prompts / answer parsing
# ---------------------------------------------------------------------------


def format_choices(choices):
    lines = []
    for index, choice in enumerate(choices):
        label = CHOICE_LABELS[index] if index < len(CHOICE_LABELS) else str(index)
        lines.append(f"({label}) {choice}")
    return "\n".join(lines)


def build_mmar_prompt(item, think_suffix: str | None = None):
    question = str(item["question"]).strip()
    choices_block = format_choices(item["choices"])
    prompt = (
        f"{question}\n"
        "Choose the correct option from the following options:\n"
        f"{choices_block}"
    )
    if think_suffix:
        prompt += f"\n{think_suffix}"
    return prompt


def _match_choice_in_text(text, choices):
    matched = []
    for index, choice in enumerate(choices):
        label = CHOICE_LABELS[index] if index < len(CHOICE_LABELS) else str(index)
        patterns = (
            rf"\({label}\)\s*{re.escape(choice)}",
            rf"\b{re.escape(choice)}\b",
            rf"\({label}\)",
        )
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                matched.append((index, choice))
                break

    if not matched:
        return None

    # Prefer the last-mentioned choice in the output tail.
    return matched[-1][1]


def parse_choice_output(raw_text, choices):
    """Split free-form model text into (thinking, answer) for MMAR choices."""
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    search_regions = []
    if lines:
        search_regions.append(lines[-1])
        search_regions.append("\n".join(lines[-3:]))
    search_regions.append(text)

    answer_prediction = None
    for region in search_regions:
        answer_prediction = _match_choice_in_text(region, choices)
        if answer_prediction:
            break

    if not answer_prediction:
        for marker in ANSWER_MARKERS:
            match = re.search(marker + r"(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            candidate = match.group(1).strip().splitlines()[0].strip()
            answer_prediction = _match_choice_in_text(candidate, choices) or candidate
            break

    if not answer_prediction and lines:
        answer_prediction = _match_choice_in_text(lines[-1], choices) or lines[-1]

    if not answer_prediction:
        answer_prediction = text
        return "", answer_prediction

    answer_index = text.lower().rfind(answer_prediction.lower())
    if answer_index > 0:
        thinking_prediction = text[:answer_index].strip()
    elif len(lines) > 1:
        thinking_prediction = "\n".join(lines[:-1]).strip()
    else:
        thinking_prediction = ""

    return thinking_prediction, answer_prediction


def parse_think_tagged_output(raw_text, choices):
    """Parse outputs that may wrap CoT in ``<think>...</think>`` tags."""
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    match = THINK_BLOCK_RE.search(text)
    if match:
        thinking_prediction = match.group(1).strip()
        remainder = (text[: match.start()] + text[match.end() :]).strip()
        if remainder:
            _, answer_prediction = parse_choice_output(remainder, choices)
            return thinking_prediction, answer_prediction
        # Think block present but no trailing answer text; fall back to choice match.
        answer_prediction = _match_choice_in_text(thinking_prediction, choices)
        if answer_prediction:
            return thinking_prediction, answer_prediction
        return thinking_prediction, remainder or text

    return parse_choice_output(text, choices)


# ---------------------------------------------------------------------------
# Rubric metadata / scoring
# ---------------------------------------------------------------------------


def load_completed_ids(predictions_path):
    if not predictions_path.exists():
        return set()
    completed = set()
    for item in load_jsonl(predictions_path):
        record_id = item.get("id")
        if record_id:
            completed.add(record_id)
    return completed


def meta_has_rubrics(meta_path: Path) -> bool:
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        return False
    for item in load_jsonl(meta_path):
        return all(key in item for key in MMAR_RUBRIC_KEYS)
    return False


def ensure_rubric_meta(meta_path: Path, dest: Path | None = None) -> Path:
    """Return a meta path that includes thinking/rubric/cue for MMAR-Rubrics.

    Downloads the GitHub MMAR-meta.jsonl when the volume copy is the HF
    variant that omits those fields.
    """
    if meta_has_rubrics(meta_path):
        return meta_path

    import urllib.request

    dest = dest or meta_path
    print(
        f"MMAR meta at {meta_path} is missing {MMAR_RUBRIC_KEYS}; "
        f"downloading rubric-complete meta from {MMAR_META_URL} -> {dest}"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(MMAR_META_URL, tmp)
    if not meta_has_rubrics(tmp):
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"Downloaded MMAR meta still missing {MMAR_RUBRIC_KEYS}. "
            "Cannot run rubric scoring."
        )
    tmp.replace(dest)
    return dest


def unscored_prediction_ids(predictions_path: Path, evaluated_path: Path) -> set[str]:
    """Return prediction ids that do not yet appear in the evaluated file."""
    if not predictions_path.exists() or predictions_path.stat().st_size == 0:
        return set()
    pred_ids = {item["id"] for item in load_jsonl(predictions_path) if item.get("id")}
    eval_ids: set[str] = set()
    if evaluated_path.exists() and evaluated_path.stat().st_size > 0:
        eval_ids = {
            item["id"]
            for item in load_jsonl(evaluated_path)
            if item.get("id") and "score" in item
        }
    return pred_ids - eval_ids


def run_rubrics_scoring(
    predictions_path, meta_path, evaluated_path, *, required: bool = True
):
    """Run OpenAI rubric grading; write ``predictions.evaluated.jsonl``."""
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key.strip():
        message = (
            "OPENAI_API_KEY is not set; cannot run MMAR-Rubrics grading.\n"
            "Ensure Modal Secret ``openai-secret`` exists with key OPENAI_API_KEY:\n"
            "  uv run modal secret create openai-secret OPENAI_API_KEY=...\n"
            "Or pass --no-score to skip grading."
        )
        if required:
            raise SystemExit(message)
        print(message)
        return False

    if not predictions_path.exists() or predictions_path.stat().st_size == 0:
        message = f"No predictions to score at {predictions_path}"
        if required:
            raise SystemExit(message)
        print(message)
        return False

    remaining = unscored_prediction_ids(Path(predictions_path), Path(evaluated_path))
    if not remaining:
        print(
            f"All predictions already graded at {evaluated_path}; "
            "skipping rubric subprocess."
        )
        return True

    meta_path = ensure_rubric_meta(
        Path(meta_path),
        dest=Path(predictions_path).parent / "MMAR-meta.rubrics.jsonl",
    )

    rubrics_script = Path("/root/evaluation_rubrics.py")
    if not rubrics_script.exists():
        # add_local_python_source places modules on sys.path; resolve via import.
        import evaluation_rubrics as rubrics_mod

        rubrics_script = Path(rubrics_mod.__file__)

    cmd = [
        sys.executable,
        str(rubrics_script),
        "--input",
        str(predictions_path),
        "--meta",
        str(meta_path),
        "--output",
        str(evaluated_path),
    ]
    print(
        f"Running MMAR-Rubrics OpenAI evaluation "
        f"({len(remaining)} remaining of "
        f"{len(load_jsonl(predictions_path))} predictions) ..."
    )
    subprocess.run(cmd, check=True)
    return True


class StreamingRubricGrader:
    """Grade predictions on a background asyncio loop while generation continues.

    Spreads OpenAI calls across the generation window (fewer burst rate-limit
    hits) and overlaps GPU work with API latency.
    """

    def __init__(
        self,
        evaluated_path: Path,
        *,
        qps: float | None = None,
        max_workers: int | None = None,
    ):
        # Import only when scoring: evaluation_rubrics requires OPENAI_API_KEY.
        import evaluation_rubrics as rubrics

        self._rubrics = rubrics
        self.evaluated_path = Path(evaluated_path)
        self.evaluated_path.parent.mkdir(parents=True, exist_ok=True)

        self._completed_ids: set[str] = set()
        if self.evaluated_path.exists() and self.evaluated_path.stat().st_size > 0:
            for item in load_jsonl(self.evaluated_path):
                if item.get("id") and "score" in item:
                    self._completed_ids.add(item["id"])

        self._inflight: set[str] = set()
        self._futures: list[Future] = []
        self._submit_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._errors: list[dict] = []
        self._graded = 0

        self._scorer = rubrics.Scorer(
            api_max_retries=rubrics.DEFAULT_API_MAX_RETRIES,
            api_retry_interval=rubrics.DEFAULT_API_RETRY_INTERVAL,
            max_workers=max_workers
            if max_workers is not None
            else rubrics.DEFAULT_MAX_WORKERS,
            qps=qps if qps is not None else rubrics.DEFAULT_QPS,
            timeout=rubrics.DEFAULT_TIMEOUT,
        )
        self._semaphore: asyncio.Semaphore | None = None
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="streaming-rubric-grader",
            daemon=True,
        )

    def _thread_main(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._semaphore = asyncio.Semaphore(self._scorer.max_workers)
        self._ready.set()
        self._loop.run_forever()

    def start(self) -> None:
        print(
            f"Starting pipelined MMAR-Rubrics grader "
            f"(qps={self._scorer.qps}, max_workers={self._scorer.max_workers}) ..."
        )
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise RuntimeError("Streaming rubric grader failed to start")

    def submit(self, records: list[dict]) -> int:
        """Enqueue records for grading. Returns how many were newly submitted."""
        required = self._rubrics.InputItem.__required_keys__
        submitted = 0
        for record in records:
            record_id = record.get("id")
            if not record_id:
                continue
            with self._submit_lock:
                if record_id in self._completed_ids or record_id in self._inflight:
                    continue
                if not required <= set(record.keys()):
                    missing = sorted(required - set(record.keys()))
                    print(
                        f"Skipping grade for {record_id}: "
                        f"missing required keys {missing}"
                    )
                    continue
                self._inflight.add(record_id)

            future = asyncio.run_coroutine_threadsafe(
                self._evaluate_and_write(record),
                self._loop,
            )
            with self._submit_lock:
                self._futures.append(future)
            submitted += 1
        return submitted

    async def _evaluate_and_write(self, record: dict) -> None:
        record_id = record["id"]
        assert self._semaphore is not None
        try:
            result = await self._rubrics.evaluate_one_record(
                self._scorer,
                self._semaphore,
                record_id,
                record["question"],
                record["answer"],
                record["thinking"],
                record["cue"],
                record["choices"],
                record["rubric"],
                record["thinking_prediction"],
                record["answer_prediction"],
            )
            if result.exception is not None:
                with self._submit_lock:
                    self._inflight.discard(record_id)
                    self._errors.append({"id": record_id, "error": result.exception})
                print(f"Rubric grade failed for {record_id}: {result.exception}")
                return

            evaluated = {
                **record,
                "new": True,
                "score": result.score,
                "correct": result.correct,
                "raw_responses": result.raw_responses,
                "rubric_results": result.rubric_results,
            }
            with self._write_lock:
                write_jsonl(self.evaluated_path, [evaluated], mode="a")
            with self._submit_lock:
                self._inflight.discard(record_id)
                self._completed_ids.add(record_id)
                self._graded += 1
        except Exception as exc:
            with self._submit_lock:
                self._inflight.discard(record_id)
                self._errors.append({"id": record_id, "error": str(exc)})
            print(f"Rubric grade failed for {record_id}: {exc}")

    def drain(self) -> dict:
        """Wait for all submitted grades to finish. Returns summary stats."""
        with self._submit_lock:
            futures = list(self._futures)
            self._futures.clear()

        for future in futures:
            try:
                future.result()
            except Exception as exc:
                print(f"Rubric grade future error: {exc}")

        with self._submit_lock:
            return {
                "graded": self._graded,
                "errors": len(self._errors),
                "completed_ids": len(self._completed_ids),
            }

    def close(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=60)


def summarize_evaluated(evaluated_path: Path) -> dict:
    """Aggregate accuracy / rubric score from an evaluated predictions file."""
    if not evaluated_path.exists():
        return {}

    records = load_jsonl(evaluated_path)
    if not records:
        return {"n": 0}

    n = len(records)
    n_correct = sum(1 for item in records if item.get("correct"))
    score_sum = sum(float(item.get("score") or 0.0) for item in records)
    by_modality: dict[str, dict] = {}
    by_category: dict[str, dict] = {}
    for item in records:
        for key, bucket in (
            (item.get("modality") or "unknown", by_modality),
            (item.get("category") or "unknown", by_category),
        ):
            stats = bucket.setdefault(key, {"n": 0, "correct": 0, "score_sum": 0.0})
            stats["n"] += 1
            stats["correct"] += int(bool(item.get("correct")))
            stats["score_sum"] += float(item.get("score") or 0.0)

    def finalize(bucket: dict) -> dict:
        out = {}
        for key, stats in bucket.items():
            total = stats["n"] or 1
            out[key] = {
                "n": stats["n"],
                "accuracy": stats["correct"] / total,
                "avg_score": stats["score_sum"] / total,
            }
        return out

    return {
        "n": n,
        "accuracy": n_correct / n,
        "avg_score": score_sum / n,
        "by_modality": finalize(by_modality),
        "by_category": finalize(by_category),
    }


# ---------------------------------------------------------------------------
# Run artifacts
# ---------------------------------------------------------------------------


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_output_dir(output_dir: str | Path, run_id: str | None = None) -> Path:
    """Place this run under ``<dataset>/<model>/<run_id>/`` on the results Volume."""
    base = Path(output_dir).expanduser()
    rid = run_id or make_run_id()
    return (base / rid).resolve()


def find_newest_run_dir(output_dir: str | Path) -> Path | None:
    """Return the newest timestamped run folder under ``output_dir``, if any."""
    base = Path(output_dir).expanduser().resolve()
    if not base.is_dir():
        return None
    runs = [p for p in base.iterdir() if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.name, reverse=True)[0]


def write_manifest(run_dir: Path, payload: dict) -> Path:
    path = run_dir / "manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def volume_relative_path(path: Path, mount: Path) -> str:
    """Return a Volume-relative path even when Modal resolves mounts under /__modal."""
    path_r = path.resolve()
    mount_r = mount.resolve()
    try:
        return str(path_r.relative_to(mount_r))
    except ValueError:
        parts = path_r.parts
        for marker in ("mmar", "aha", "mcr"):
            if marker in parts:
                idx = parts.index(marker)
                return str(Path(*parts[idx:]))
        return str(path_r)


def publish_run_artifacts(run_dir: Path, manifest: dict | None = None) -> dict:
    """Commit the results Volume and summarize artifacts present in ``run_dir``."""
    if manifest is not None:
        write_manifest(run_dir, manifest)

    artifacts = {}
    for name in (
        "predictions.jsonl",
        "predictions.evaluated.jsonl",
        "scores.json",
        "manifest.json",
    ):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            artifacts[name] = str(path)

    results_volume.commit()

    volume_rel = volume_relative_path(run_dir, RESULTS_MOUNT)
    print(f"Published results to volume:{RESULTS_VOLUME_NAME}/{volume_rel}")
    return {
        "run_dir": str(run_dir),
        "volume_path": volume_rel,
        "artifacts": artifacts,
    }


def finalize_scored_run(
    run_dir: Path,
    predictions_path: Path,
    evaluated_path: Path,
    meta_path: Path,
    base_manifest: dict,
    *,
    score: bool,
) -> dict:
    """Ensure rubric grading + summary scores are on disk, then publish."""
    scored = False
    if score:
        scored = run_rubrics_scoring(
            predictions_path,
            meta_path,
            evaluated_path,
            required=True,
        )
        summary = summarize_evaluated(evaluated_path)
        if summary:
            write_json(run_dir / "scores.json", summary)
            print(
                f"Rubric summary: n={summary.get('n')} "
                f"acc={summary.get('accuracy', 0):.3f} "
                f"avg_score={summary.get('avg_score', 0):.3f}"
            )

    manifest = {
        **base_manifest,
        "scored": scored,
        "predictions": str(predictions_path),
        "evaluated": str(evaluated_path) if evaluated_path.exists() else None,
        "scores": str(run_dir / "scores.json")
        if (run_dir / "scores.json").exists()
        else None,
        "fields": {
            "model_output": "full prompt + generation decode (special tokens retained)",
            "raw_tokens": "per-token list covering input + specials + generated tokens",
            "thinking_prediction": "parsed CoT / reasoning trace",
            "answer_prediction": "parsed final answer",
            "score": "MMAR-Rubrics aggregate score (evaluated file)",
            "correct": "answer correctness from rubric grader",
            "raw_responses": "OpenAI grader raw responses",
            "rubric_results": "per-criterion rubric judgments",
        },
    }
    return publish_run_artifacts(run_dir, manifest)


# ---------------------------------------------------------------------------
# Shared MMAR eval pipeline
# ---------------------------------------------------------------------------


def run_mmar_evaluation(
    *,
    args: SimpleNamespace,
    load_model: Callable,
    generate_batch_fn: Callable,
    model_label: str,
    manifest_extra: dict | None = None,
) -> dict:
    """Shared MMAR inference + pipelined MMAR-Rubrics grading loop."""
    volume.reload()
    results_volume.reload()

    run_id = args.run_id or make_run_id()
    args.run_id = run_id
    random.seed(args.seed)

    data_root_path = Path(args.data_root).expanduser().resolve()
    meta_path = Path(args.meta).expanduser().resolve()
    run_dir = resolve_run_output_dir(args.output_dir, run_id=run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    evaluated_path = run_dir / "predictions.evaluated.jsonl"

    base_manifest = {
        "run_id": run_id,
        "benchmark": "mmar",
        "model_id": args.model_id,
        "model_label": model_label,
        "num_samples": args.num_samples,
        "start": args.start,
        "seed": args.seed,
        "torch_dtype": args.torch_dtype,
        "max_new_tokens": args.max_new_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **(manifest_extra or {}),
    }

    if args.score_only:
        if not predictions_path.exists():
            newest = find_newest_run_dir(args.output_dir)
            newest_preds = newest / "predictions.jsonl" if newest else None
            if newest_preds is not None and newest_preds.exists():
                run_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(newest_preds, predictions_path)
            else:
                raise SystemExit(
                    f"No predictions found at {predictions_path}"
                    + (
                        f" (and no run under {args.output_dir})"
                        if newest is None
                        else ""
                    )
                )
        published = finalize_scored_run(
            run_dir,
            predictions_path,
            evaluated_path,
            meta_path,
            {**base_manifest, "status": "scored", "score_only": True},
            score=True,
        )
        return {"status": "scored", **published}

    if not meta_path.exists():
        raise SystemExit(
            f"MMAR metadata not found: {meta_path}\n"
            "Seed first: uv run modal run seed_volume.py --datasets mmar --models none"
        )

    refreshed_meta = not meta_has_rubrics(meta_path)
    meta_path = ensure_rubric_meta(meta_path)
    if refreshed_meta:
        volume.commit()

    meta_items = load_jsonl(meta_path)
    expected_wavs = len(meta_items)
    resolved_audio_dir = (
        Path(args.audio_dir) if args.audio_dir else data_root_path / "audio"
    )
    wav_count = count_wavs(resolved_audio_dir)
    if wav_count < expected_wavs:
        raise SystemExit(
            f"MMAR audio missing in {resolved_audio_dir} "
            f"({wav_count}/{expected_wavs} wav files).\n"
            "Seed first: uv run modal run seed_volume.py --datasets mmar --models none"
        )

    pool = meta_items[args.start :]
    if args.num_samples < 0 or args.num_samples >= len(pool):
        selected_items = pool
    else:
        # Index-sample a subset so we avoid contiguous prefix bias.
        selected_indices = random.sample(range(len(pool)), args.num_samples)
        selected_items = [pool[i] for i in selected_indices]
    completed_ids = load_completed_ids(predictions_path)
    pending_items = []
    for item in selected_items:
        audio_path = resolve_path(data_root_path, item["audio_path"])
        if not os.path.exists(audio_path):
            print(f"Skipping {item['id']}: missing audio at {audio_path}")
            continue
        if item["id"] in completed_ids:
            continue
        pending_items.append({**item, "audio_path": audio_path})

    if not pending_items:
        print("No pending MMAR items to evaluate.")
        if not predictions_path.exists() or predictions_path.stat().st_size == 0:
            raise SystemExit(
                f"No predictions available to score at {predictions_path}. "
                "Re-run inference first."
            )
        published = finalize_scored_run(
            run_dir,
            predictions_path,
            evaluated_path,
            meta_path,
            {**base_manifest, "status": "noop", "pending": 0},
            score=args.score,
        )
        return {"status": "noop", "pending": 0, **published}

    print(f"Evaluating {len(pending_items)} MMAR items with {args.model_id} ...")
    print(f"Writing run artifacts to {run_dir}")
    model, processor = load_model(args)

    grader: StreamingRubricGrader | None = None
    grade_submitted = 0
    if args.score:
        api_key = os.environ.get("OPENAI_API_KEY") or ""
        if not api_key.strip():
            raise SystemExit(
                "OPENAI_API_KEY is not set; cannot run MMAR-Rubrics grading.\n"
                "Ensure Modal Secret ``openai-secret`` exists with key OPENAI_API_KEY:\n"
                "  uv run modal secret create openai-secret OPENAI_API_KEY=...\n"
                "Or pass --no-score to skip grading."
            )
        grader = StreamingRubricGrader(evaluated_path)
        grader.start()
        if predictions_path.exists() and predictions_path.stat().st_size > 0:
            n = grader.submit(load_jsonl(predictions_path))
            grade_submitted += n
            if n:
                print(f"Queued {n} existing predictions for pipelined grading")

    start_time = time.time()
    completed = 0
    failures = []
    try:
        for batch_start in range(0, len(pending_items), args.batch_size):
            batch = pending_items[batch_start : batch_start + args.batch_size]
            try:
                outputs = generate_batch_fn(model, processor, batch, args)
            except (OSError, ValueError, RuntimeError) as exc:
                ids = ", ".join(item["id"] for item in batch)
                print(f"Skipping batch ({ids}): {exc}")
                failures.append(
                    {"ids": [item["id"] for item in batch], "error": str(exc)}
                )
                continue

            records = []
            for item, output in zip(batch, outputs):
                record = {**item, **output}
                records.append(record)
                completed += 1
                if args.print_every > 0 and completed % args.print_every == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[{completed}/{len(pending_items)}] "
                        f"id={item['id']} answer={output['answer_prediction']!r} "
                        f"({elapsed:.1f}s elapsed)"
                    )

            write_jsonl(predictions_path, records, mode="a")
            if grader is not None:
                grade_submitted += grader.submit(records)
            results_volume.commit()

        print(f"Wrote predictions to {predictions_path}")
        if completed == 0:
            raise SystemExit(
                f"All {len(pending_items)} generations failed; not publishing empty results.\n"
                f"First errors: {failures[:3]}"
            )
    finally:
        if grader is not None:
            print(
                f"Waiting for pipelined rubric grades "
                f"({grade_submitted} submitted this run) ..."
            )
            try:
                grade_stats = grader.drain()
                print(
                    f"Pipelined grading done: graded={grade_stats['graded']} "
                    f"errors={grade_stats['errors']} "
                    f"on_disk={grade_stats['completed_ids']}"
                )
                results_volume.commit()
            finally:
                grader.close()

    elapsed_s = round(time.time() - start_time, 1)
    published = finalize_scored_run(
        run_dir,
        predictions_path,
        evaluated_path,
        meta_path,
        {
            **base_manifest,
            "status": "ok",
            "completed": completed,
            "failed_batches": len(failures),
            "elapsed_s": elapsed_s,
            "pipelined_grading": bool(args.score),
        },
        score=args.score,
    )
    return {
        "status": "ok",
        "completed": completed,
        "failed_batches": len(failures),
        "elapsed_s": elapsed_s,
        **published,
    }
