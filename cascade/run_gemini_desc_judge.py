"""Search for an audio description that lets a text-only judge match humans.

For every MMAR question that has a triple-rated generation in
``exports/labels.csv``, Gemini 3.7 Flash (1) captions the clip with
``build_mmar_description_prompt`` and (2) grades every triple-rated
generation with the ``judge_no_gt`` template, substituting that caption
for the audio. Up to ``--max-attempts`` captions are tried per question;
a question stops early at 100% agreement with the human majority vote.
The caption with the highest accuracy is kept.

Identical generation texts on a question are graded once per caption.

Usage::

    export GEMINI_API_KEY=...

    uv run python cascade/run_gemini_desc_judge.py --limit 5
    uv run python cascade/run_gemini_desc_judge.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aggregate import is_dropped_model  # noqa: E402
from alt_test import MIN_HUMANS_PER_INSTANCE, majority_label  # noqa: E402
from grader import JUDGE_FORMATS, parse_grade_verdict  # noqa: E402
from mmar_api import gemini_response_text  # noqa: E402
from mmar_common import (  # noqa: E402
    build_mmar_description_prompt,
    load_jsonl,
    resolve_path,
    write_json,
)

DEFAULT_MODEL = "gemini-3.7-flash"
DEFAULT_LABELS = REPO_ROOT / "exports" / "labels.csv"
DEFAULT_GENERATIONS = REPO_ROOT / "exports" / "generations.csv"
DEFAULT_META = REPO_ROOT / "data" / "mmar" / "MMAR-meta.jsonl"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "mmar"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "gemini-desc-judge"
JUDGE_PROMPT_NAME = "judge_no_gt"
MAX_ATTEMPTS = 5
DESCRIPTIONS_NAME = "descriptions.jsonl"
GRADES_NAME = "grades.jsonl"
QUESTIONS_NAME = "questions.jsonl"
MANIFEST_NAME = "manifest.json"
SUMMARY_NAME = "summary.json"

logger = logging.getLogger(__name__)


def _parse_ratings_cell(raw: object) -> list[bool]:
    if isinstance(raw, list):
        values = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            values = json.loads(text)
        except json.JSONDecodeError:
            return []
    if not isinstance(values, list) or not values:
        return []
    out: list[bool] = []
    for item in values:
        if isinstance(item, bool):
            out.append(item)
        else:
            return []
    return out


def normalize_prediction(text: str) -> str:
    """Identity used to reuse a grade across identical generation strings."""
    return str(text or "").strip().lower()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        shutil.move(str(tmp_path), path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        shutil.move(str(tmp_path), path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def load_jsonl_dicts(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt line in %s", path)
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def build_description_grade_prompt(
    *,
    question: str,
    prediction: str,
    description: str,
) -> str:
    """``judge_no_gt`` text with the caption standing in for the clip."""
    fmt = JUDGE_FORMATS[JUDGE_PROMPT_NAME]
    parts = [
        fmt.prompt,
        f'Audio description: "{description}"',
        fmt.fields(question=question, answer="", prediction=prediction),
    ]
    if fmt.closer:
        parts.append(fmt.closer)
    return "\n\n".join(parts)


def load_mmar_meta(meta_path: Path, data_root: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for row in load_jsonl(meta_path):
        if not isinstance(row, dict):
            continue
        qid = str(row.get("id") or "").strip()
        if not qid:
            continue
        record = dict(row)
        record["id"] = qid
        record["question"] = str(row.get("question") or "").strip()
        record["answer"] = str(row.get("answer") or "").strip()
        record["audio_path"] = resolve_path(data_root, row.get("audio_path") or "")
        by_id[qid] = record
    if not by_id:
        raise SystemExit(f"No MMAR-meta rows in {meta_path}")
    return by_id


def load_generation_index(path: Path) -> tuple[dict[str, dict], dict[tuple[str, str, int], dict]]:
    by_id: dict[str, dict] = {}
    by_key: dict[tuple[str, str, int], dict] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            qid = str(raw.get("question_id") or "").strip()
            model = str(raw.get("model_label") or "").strip()
            if not qid or not model or is_dropped_model(model):
                continue
            try:
                shot_index = int(raw.get("shot_index", 0))
            except (TypeError, ValueError):
                continue
            prediction = raw.get("answer_prediction")
            row = {
                "question_id": qid,
                "generation_id": str(raw.get("generation_id") or "").strip(),
                "model_label": model,
                "shot_index": shot_index,
                "answer_prediction": "" if prediction is None else str(prediction),
            }
            by_key[(qid, model, shot_index)] = row
            if row["generation_id"]:
                by_id[row["generation_id"]] = row
    return by_id, by_key


def load_work(
    labels_path: Path,
    generations_path: Path,
    meta_path: Path,
    data_root: Path,
    *,
    min_raters: int = MIN_HUMANS_PER_INSTANCE,
) -> list[dict[str, Any]]:
    """Questions with triple-rated generations, joined to answers and audio."""
    by_id, by_key = load_generation_index(generations_path)
    meta = load_mmar_meta(meta_path, data_root)

    gens_by_qid: dict[str, list[dict]] = defaultdict(list)
    seen_q: list[str] = []
    seen_set: set[str] = set()
    n_missing_gen = 0
    n_no_majority = 0
    with labels_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            qid = str(raw.get("question_id") or "").strip()
            model = str(raw.get("model_label") or "").strip()
            if is_dropped_model(model):
                continue
            ratings = _parse_ratings_cell(raw.get("ratings"))
            if not qid or not model or len(ratings) < min_raters:
                continue
            majority = majority_label(ratings)
            if majority is None:
                n_no_majority += 1
                continue
            try:
                shot_index = int(raw.get("shot_index", 0))
            except (TypeError, ValueError):
                continue
            gid = str(raw.get("generation_id") or "").strip()
            gen = by_key.get((qid, model, shot_index))
            if gen is None and gid:
                gen = by_id.get(gid)
            if gen is None:
                n_missing_gen += 1
                continue
            if qid not in seen_set:
                seen_set.add(qid)
                seen_q.append(qid)
            gens_by_qid[qid].append(
                {
                    "generation_id": gid or gen.get("generation_id") or "",
                    "model_label": model,
                    "shot_index": shot_index,
                    "answer_prediction": str(gen.get("answer_prediction") or ""),
                    "ratings": ratings,
                    "human_majority": bool(majority),
                }
            )

    if n_missing_gen:
        logger.warning("Dropped %s labeled rows with no generation text", n_missing_gen)
    if n_no_majority:
        logger.warning("Dropped %s labeled rows with no majority vote", n_no_majority)

    questions: list[dict[str, Any]] = []
    for qid in seen_q:
        item = meta.get(qid)
        if not item or not item.get("question"):
            logger.warning("Skipping %s: missing MMAR-meta / question", qid)
            continue
        audio_path = str(item.get("audio_path") or "")
        if not audio_path or not Path(audio_path).is_file():
            logger.warning("Skipping %s: missing audio %s", qid, audio_path)
            continue
        generations = gens_by_qid[qid]
        unique_keys: list[str] = []
        unique_seen: set[str] = set()
        for gen in generations:
            key = normalize_prediction(gen["answer_prediction"])
            if key not in unique_seen:
                unique_seen.add(key)
                unique_keys.append(key)
        questions.append(
            {
                "question_id": qid,
                "question": item["question"],
                "answer": item.get("answer") or "",
                "category": item.get("category"),
                "modality": item.get("modality"),
                "audio_path": audio_path,
                "generations": generations,
                "unique_keys": unique_keys,
            }
        )
    if not questions:
        raise SystemExit(
            f"No triple-labeled questions with audio under {data_root} "
            f"(labels={labels_path})"
        )
    return questions


class GeminiClient:
    """Throttled Gemini client for one audio caption or one text-only grade."""

    def __init__(
        self,
        *,
        model_id: str,
        qps: float,
        timeout: float,
        retries: int,
        retry_interval: float,
        max_output_tokens: int,
        describe_temperature: float,
        judge_temperature: float,
        describe_thinking_level: str | None,
        judge_thinking_level: str | None,
    ):
        from google import genai
        from google.genai import types

        api_key = (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        ).strip()
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY to call Gemini.")
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.timeout = timeout
        self.retries = retries
        self.retry_interval = retry_interval
        self.max_output_tokens = max_output_tokens
        self.describe_temperature = describe_temperature
        self.judge_temperature = judge_temperature
        self.describe_thinking_level = describe_thinking_level
        self.judge_thinking_level = judge_thinking_level
        self._interval = 1.0 / max(float(qps), 0.1)
        self._next_time = 0.0
        self._lock = asyncio.Lock()

    def _thinking(self, level: str | None):
        if not level:
            return None
        return self.types.ThinkingConfig(thinking_level=level)

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
                now = time.monotonic()
            self._next_time = now + self._interval

    def _is_rate_limit(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (429, 418):
            return True
        text = str(exc).lower()
        return "rate" in text or "resource exhausted" in text or "429" in text

    async def _complete(
        self,
        contents: list[Any],
        *,
        temperature: float,
        thinking_level: str | None,
    ) -> str:
        config = self.types.GenerateContentConfig(
            temperature=float(temperature),
            max_output_tokens=int(self.max_output_tokens),
            thinking_config=self._thinking(thinking_level),
        )
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                await self._throttle()
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=self.model_id,
                        contents=contents,
                        config=config,
                    ),
                    timeout=self.timeout,
                )
            except TimeoutError as exc:
                last_exc = exc
                logger.warning("gemini attempt %s timeout: %s", attempt, exc)
                await asyncio.sleep(self.retry_interval)
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = self.retry_interval * attempt if self._is_rate_limit(exc) else self.retry_interval
                logger.warning("gemini attempt %s failed: %s", attempt, exc)
                await asyncio.sleep(wait)
                continue
            text = gemini_response_text(response)
            if text:
                return text
            logger.warning("gemini attempt %s empty response", attempt)
            await asyncio.sleep(self.retry_interval)
        raise RuntimeError(f"Gemini retries exhausted: {last_exc}")

    async def describe(self, audio_path: str, prompt: str) -> str:
        contents = [
            self.types.Part.from_bytes(
                data=Path(audio_path).read_bytes(),
                mime_type="audio/wav",
            ),
            prompt,
        ]
        return await self._complete(
            contents,
            temperature=self.describe_temperature,
            thinking_level=self.describe_thinking_level,
        )

    async def judge(self, prompt: str) -> str:
        return await self._complete(
            [prompt],
            temperature=self.judge_temperature,
            thinking_level=self.judge_thinking_level,
        )


def score_attempt(
    generations: list[dict],
    verdict_by_key: dict[str, bool | None],
) -> dict[str, Any]:
    n = len(generations)
    n_parsed = 0
    n_correct = 0
    for gen in generations:
        key = normalize_prediction(gen["answer_prediction"])
        verdict = verdict_by_key.get(key)
        if verdict is None:
            continue
        n_parsed += 1
        if verdict is bool(gen["human_majority"]):
            n_correct += 1
    accuracy = (n_correct / n) if n else 0.0
    parse_rate = (n_parsed / n) if n else 0.0
    return {
        "n_generations": n,
        "n_parsed": n_parsed,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "parse_rate": parse_rate,
        "perfect": n > 0 and n_correct == n,
    }


def question_record(
    item: dict,
    *,
    attempts: list[dict],
    max_attempts: int,
) -> dict[str, Any]:
    best = max(attempts, key=lambda row: (row["accuracy"], row["parse_rate"]))
    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "category": item.get("category"),
        "modality": item.get("modality"),
        "n_generations": len(item["generations"]),
        "n_unique": len(item["unique_keys"]),
        "n_attempts": len(attempts),
        "stopped_early": bool(best.get("perfect")),
        "complete": bool(best.get("perfect")) or len(attempts) >= max_attempts,
        "best_attempt": best["attempt"],
        "best_accuracy": best["accuracy"],
        "best_parse_rate": best["parse_rate"],
        "n_correct": best["n_correct"],
        "n_parsed": best["n_parsed"],
        "description": best["description"],
    }


def load_resume_state(out_dir: Path) -> tuple[dict[str, dict[int, str]], dict[tuple[str, int, str], dict]]:
    descriptions: dict[str, dict[int, str]] = defaultdict(dict)
    for row in load_jsonl_dicts(out_dir / DESCRIPTIONS_NAME):
        qid = str(row.get("question_id") or "")
        try:
            attempt = int(row.get("attempt", 0))
        except (TypeError, ValueError):
            continue
        text = str(row.get("description") or "").strip()
        if qid and attempt >= 1 and text:
            descriptions[qid][attempt] = text

    grades: dict[tuple[str, int, str], dict] = {}
    for row in load_jsonl_dicts(out_dir / GRADES_NAME):
        qid = str(row.get("question_id") or "")
        key = normalize_prediction(row.get("prediction_key") or row.get("prediction") or "")
        try:
            attempt = int(row.get("attempt", 0))
        except (TypeError, ValueError):
            continue
        if not qid or attempt < 1 or not key:
            continue
        if row.get("judge_correct") is None and row.get("raw") is None:
            continue
        grades[(qid, attempt, key)] = row
    return descriptions, grades


def summarize(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    n_complete = sum(1 for row in records if row.get("complete"))
    n_perfect = sum(1 for row in records if row.get("stopped_early"))
    accs = [float(row["best_accuracy"]) for row in records]
    parses = [float(row["best_parse_rate"]) for row in records]
    attempts = [int(row["n_attempts"]) for row in records]
    return {
        "n_questions": n,
        "n_complete": n_complete,
        "n_perfect": n_perfect,
        "mean_best_accuracy": (sum(accs) / n) if n else 0.0,
        "mean_best_parse_rate": (sum(parses) / n) if n else 0.0,
        "mean_attempts": (sum(attempts) / n) if n else 0.0,
        "updated_at": datetime.now(UTC).isoformat(),
    }


async def run_question(
    item: dict,
    client: GeminiClient,
    *,
    out_dir: Path,
    max_attempts: int,
    max_workers: int,
    write_lock: asyncio.Lock,
    descriptions: dict[str, dict[int, str]],
    grades: dict[tuple[str, int, str], dict],
    question_rows: dict[str, dict],
) -> dict[str, Any]:
    qid = item["question_id"]
    prior_attempts = _attempts_from_disk(qid, item, descriptions, grades)
    if prior_attempts and (
        any(row.get("perfect") for row in prior_attempts)
        or len(prior_attempts) >= max_attempts
    ):
        record = question_record(item, attempts=prior_attempts, max_attempts=max_attempts)
        question_rows[qid] = record
        return record

    semaphore = asyncio.Semaphore(max(int(max_workers), 1))
    attempts = list(prior_attempts)
    start = len(attempts) + 1
    desc_prompt = build_mmar_description_prompt()

    for attempt in range(start, max_attempts + 1):
        description = descriptions.get(qid, {}).get(attempt, "")
        if not description:
            logger.info("[%s] attempt %s: describing audio", qid, attempt)
            description = (await client.describe(item["audio_path"], desc_prompt)).strip()
            if not description:
                raise RuntimeError(f"{qid} attempt {attempt}: empty description")
            descriptions.setdefault(qid, {})[attempt] = description
            async with write_lock:
                append_jsonl(
                    out_dir / DESCRIPTIONS_NAME,
                    {
                        "question_id": qid,
                        "attempt": attempt,
                        "description": description,
                    },
                )

        missing = [
            key
            for key in item["unique_keys"]
            if (qid, attempt, key) not in grades
        ]
        logger.info(
            "[%s] attempt %s: grading %s unique / %s labeled (cached %s)",
            qid,
            attempt,
            len(missing),
            len(item["generations"]),
            len(item["unique_keys"]) - len(missing),
        )
        example_by_key = {
            normalize_prediction(gen["answer_prediction"]): gen
            for gen in item["generations"]
        }

        async def _grade_one(key: str, *, _attempt: int = attempt, _desc: str = description) -> None:
            gen = example_by_key[key]
            prompt = build_description_grade_prompt(
                question=item["question"],
                prediction=gen["answer_prediction"],
                description=_desc,
            )
            async with semaphore:
                raw = await client.judge(prompt)
            verdict = parse_grade_verdict(raw)
            row = {
                "question_id": qid,
                "attempt": _attempt,
                "prediction_key": key,
                "prediction": gen["answer_prediction"],
                "judge_correct": verdict,
                "verdict": (
                    "pass" if verdict is True else "fail" if verdict is False else None
                ),
                "raw": raw,
            }
            grades[(qid, _attempt, key)] = row
            async with write_lock:
                append_jsonl(out_dir / GRADES_NAME, row)

        if missing:
            await asyncio.gather(*[_grade_one(key) for key in missing])

        verdict_by_key = {
            key: grades[(qid, attempt, key)].get("judge_correct")
            for key in item["unique_keys"]
            if (qid, attempt, key) in grades
        }
        stats = score_attempt(item["generations"], verdict_by_key)
        attempt_row = {
            "attempt": attempt,
            "description": description,
            **stats,
        }
        attempts.append(attempt_row)
        record = question_record(item, attempts=attempts, max_attempts=max_attempts)
        question_rows[qid] = record
        async with write_lock:
            atomic_write_jsonl(out_dir / QUESTIONS_NAME, list(question_rows.values()))
            atomic_write_json(out_dir / SUMMARY_NAME, summarize(list(question_rows.values())))
        logger.info(
            "[%s] attempt %s: acc=%.3f parse=%.3f (%s/%s)%s",
            qid,
            attempt,
            stats["accuracy"],
            stats["parse_rate"],
            stats["n_correct"],
            stats["n_generations"],
            " PERFECT — stopping" if stats["perfect"] else "",
        )
        if stats["perfect"]:
            break

    return question_rows[qid]


def _attempts_from_disk(
    qid: str,
    item: dict,
    descriptions: dict[str, dict[int, str]],
    grades: dict[tuple[str, int, str], dict],
) -> list[dict]:
    """Rebuild finished attempts (description + every unique text graded)."""
    needed = set(item["unique_keys"])
    attempts: list[dict] = []
    for attempt, description in sorted(descriptions.get(qid, {}).items()):
        have = {
            key
            for key in needed
            if (qid, attempt, key) in grades
        }
        if have != needed:
            continue
        verdict_by_key = {
            key: grades[(qid, attempt, key)].get("judge_correct") for key in needed
        }
        attempts.append(
            {
                "attempt": attempt,
                "description": description,
                **score_attempt(item["generations"], verdict_by_key),
            }
        )
    return attempts


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_work(
        Path(args.labels),
        Path(args.generations),
        Path(args.meta),
        Path(args.data_root),
    )
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
        print(f"[desc-judge] --limit {args.limit}: {len(questions)} questions")
    else:
        print(f"[desc-judge] {len(questions)} triple-labeled questions")

    n_gens = sum(len(item["generations"]) for item in questions)
    n_unique = sum(len(item["unique_keys"]) for item in questions)
    print(
        f"[desc-judge] {n_gens} labeled generations, {n_unique} unique texts "
        f"(dedup saves {n_gens - n_unique} grade calls per attempt)"
    )

    desc_prompt = build_mmar_description_prompt()
    manifest_path = out_dir / MANIFEST_NAME
    prior_prompt = None
    if manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}
        if isinstance(prior, dict):
            prior_prompt = prior.get("description_prompt")
    captions_path = out_dir / DESCRIPTIONS_NAME
    has_captions = captions_path.is_file() and captions_path.stat().st_size > 0
    if has_captions and prior_prompt != desc_prompt:
        raise SystemExit(
            "description prompt changed; resume would mix captions. "
            f"Move {out_dir} aside and re-run."
        )

    write_json(
        out_dir / MANIFEST_NAME,
        {
            "model_id": args.gemini_model,
            "judge_prompt": JUDGE_PROMPT_NAME,
            "description_prompt": desc_prompt,
            "max_attempts": args.max_attempts,
            "n_questions": len(questions),
            "n_generations": n_gens,
            "n_unique": n_unique,
            "labels": str(Path(args.labels).resolve()),
            "generations": str(Path(args.generations).resolve()),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )

    descriptions, grades = load_resume_state(out_dir)
    question_rows = {
        str(row["question_id"]): row
        for row in load_jsonl_dicts(out_dir / QUESTIONS_NAME)
        if row.get("question_id")
    }
    for item in questions:
        qid = item["question_id"]
        rebuilt = _attempts_from_disk(qid, item, descriptions, grades)
        if rebuilt:
            question_rows[qid] = question_record(
                item, attempts=rebuilt, max_attempts=args.max_attempts
            )

    pending = [
        item
        for item in questions
        if not (question_rows.get(item["question_id"]) or {}).get("complete")
    ]
    print(
        f"[desc-judge] resume: {len(questions) - len(pending)} complete, "
        f"{len(pending)} pending -> {out_dir}"
    )
    if not pending:
        summary = summarize([question_rows[item["question_id"]] for item in questions])
        atomic_write_json(out_dir / SUMMARY_NAME, summary)
        print("[desc-judge] already done:", summary)
        return summary

    client = GeminiClient(
        model_id=args.gemini_model,
        qps=args.qps,
        timeout=args.timeout,
        retries=args.retries,
        retry_interval=args.retry_interval,
        max_output_tokens=args.max_tokens,
        describe_temperature=args.describe_temperature,
        judge_temperature=args.judge_temperature,
        describe_thinking_level=args.describe_thinking_level or None,
        judge_thinking_level=args.judge_thinking_level or None,
    )
    write_lock = asyncio.Lock()
    question_sema = asyncio.Semaphore(max(int(args.question_workers), 1))

    async def _one(item: dict) -> None:
        async with question_sema:
            try:
                record = await run_question(
                    item,
                    client,
                    out_dir=out_dir,
                    max_attempts=args.max_attempts,
                    max_workers=args.max_workers,
                    write_lock=write_lock,
                    descriptions=descriptions,
                    grades=grades,
                    question_rows=question_rows,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("question %s failed: %s", item["question_id"], exc)
                return
            print(
                f"[desc-judge] {record['question_id']} "
                f"acc={record['best_accuracy']:.3f} "
                f"parse={record['best_parse_rate']:.3f} "
                f"attempts={record['n_attempts']}"
                f"{' (100%)' if record['stopped_early'] else ''}"
            )

    await asyncio.gather(*[_one(item) for item in pending])
    ordered = [question_rows[item["question_id"]] for item in questions if item["question_id"] in question_rows]
    atomic_write_jsonl(out_dir / QUESTIONS_NAME, ordered)
    summary = summarize(ordered)
    atomic_write_json(out_dir / SUMMARY_NAME, summary)
    print("[desc-judge] done:", summary)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--generations", default=str(DEFAULT_GENERATIONS))
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max questions (0 = all). Smoke-test with --limit 5.",
    )
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--gemini-model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-interval", type=float, default=1.0)
    parser.add_argument("--qps", type=float, default=4.0)
    parser.add_argument(
        "--max-workers",
        "-j",
        type=int,
        default=8,
        help="Concurrent grade calls within a question.",
    )
    parser.add_argument(
        "--question-workers",
        type=int,
        default=2,
        help="Questions in flight at once (each holds the clip in memory).",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--describe-temperature",
        type=float,
        default=1.0,
        help="Sampling for captions so retries can differ.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Sampling for judge_no_gt grades.",
    )
    parser.add_argument(
        "--describe-thinking-level",
        default="",
        help="Gemini thinking_level for captions (empty = off).",
    )
    parser.add_argument(
        "--judge-thinking-level",
        default="medium",
        help="Gemini thinking_level for grades: low, medium, or high.",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
