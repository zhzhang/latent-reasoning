"""Shared OpenAI / Gemini audio clients for MMAR test-taking and judging."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmar_common import (
    ensure_judge_schema,
    recompute_multi_judge_scores,
    write_json,
    write_jsonl,
)

logger = logging.getLogger(__name__)

API_SPECS: dict[str, dict[str, Any]] = {
    "gemini-3.7-flash": {
        "model_id": "gemini-3.7-flash",
        "backend": "gemini",
        "native_thinking": True,
        "sampling": {"temperature": 1.0, "max_tokens": 1024},
    },
    "gpt-5.6-luna": {
        "model_id": "gpt-5.6-luna",
        "backend": "openai_batch",
        "native_thinking": True,
        "batch": True,
        "endpoint": "/v1/responses",
        "sampling": {"max_output_tokens": 8192, "reasoning_effort": "medium"},
    },
    "claude-sonnet-5": {
        "model_id": "claude-sonnet-5",
        "backend": "anthropic_batch",
        "native_thinking": True,
        "batch": True,
        "sampling": {"max_tokens": 8192, "effort": "medium"},
    },
}

BATCH_API_GRADE_PROMPT = "with_gt"
OPENAI_BATCH_DIR_NAME = "_openai_batch"
OPENAI_BATCH_ENDPOINT = "/v1/responses"
OPENAI_BATCH_COMPLETION_WINDOW = "24h"
OPENAI_BATCH_TERMINAL = frozenset(
    {"completed", "failed", "expired", "cancelled"}
)
OPENAI_BATCH_ACTIVE = frozenset(
    {"validating", "in_progress", "finalizing", "cancelling"}
)
ANTHROPIC_BATCH_DIR_NAME = "_anthropic_batch"
ANTHROPIC_BATCH_TERMINAL = frozenset({"ended"})
ANTHROPIC_BATCH_ACTIVE = frozenset({"in_progress", "canceling", "cancelling"})
BATCH_DIR_BY_BACKEND = {
    "openai_batch": OPENAI_BATCH_DIR_NAME,
    "anthropic_batch": ANTHROPIC_BATCH_DIR_NAME,
}


BATCH_BACKENDS = frozenset({"openai_batch", "anthropic_batch"})
# Documented caps: OpenAI 50k requests; Anthropic 100k requests or 256 MB.
BATCH_MAX_REQUESTS: dict[str, int] = {
    "openai_batch": 50_000,
    "anthropic_batch": 80_000,
}


def _is_batch_spec(spec: dict[str, Any] | None) -> bool:
    if not spec:
        return False
    return bool(spec.get("batch")) or spec.get("backend") in BATCH_BACKENDS


ALL_API_LABELS = tuple(
    key for key, spec in API_SPECS.items() if not _is_batch_spec(spec)
)
ALL_BATCH_API_LABELS = tuple(
    key for key, spec in API_SPECS.items() if _is_batch_spec(spec)
)
API_JUDGE_ALIASES: dict[str, str] = {
    "gemini-3.7-flash": "gemini-3.7-flash",
    "gemini-3.7-mini": "gemini-3.7-flash",
    "gemini": "gemini-3.7-flash",
    "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt5.6-luna": "gpt-5.6-luna",
    "luna": "gpt-5.6-luna",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-sonnet": "claude-sonnet-5",
    "sonnet-5": "claude-sonnet-5",
    "sonnet": "claude-sonnet-5",
    "claude": "claude-sonnet-5",
}
API_JUDGE_GROUP_ALIASES = frozenset({"api", "all-api"})
BATCH_API_JUDGE_GROUP_ALIASES = frozenset({"batch", "all-batch"})
OPENAI_BATCH_GROUP_ALIASES = frozenset({"openai-batch"})
ANTHROPIC_BATCH_GROUP_ALIASES = frozenset({"anthropic-batch", "claude-batch"})
GEMINI_CACHE_TTL = "900s"
GEMINI_CACHE_MAX_BYTES = 10 * 1024 * 1024
JUDGE_SAMPLING: dict[str, Any] = {"temperature": 0.0, "max_tokens": 2048}
JUDGE_CACHE_KEY_PREFIX = "mmar-judge"


@dataclass(frozen=True)
class CompletionResult:
    text: str
    cached_tokens: int | None = None
    prompt_tokens: int | None = None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _openai_usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    prompt_tokens = _int_or_none(getattr(usage, "prompt_tokens", None))
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = None
    if details is not None:
        cached_tokens = _int_or_none(getattr(details, "cached_tokens", None))
    return cached_tokens, prompt_tokens


def _gemini_usage(response: Any) -> tuple[int | None, int | None]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None, None
    cached_tokens = _int_or_none(getattr(meta, "cached_content_token_count", None))
    if cached_tokens is None:
        cached_tokens = _int_or_none(getattr(meta, "cached_tokens", None))
    prompt_tokens = _int_or_none(getattr(meta, "prompt_token_count", None))
    return cached_tokens, prompt_tokens


def gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            piece = getattr(part, "text", None)
            if piece:
                parts.append(str(piece))
    return "\n".join(parts).strip()


def parse_api_model_list(value: str) -> list[str]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw or any(item.lower() == "all" for item in raw):
        return list(ALL_API_LABELS)
    unknown = [item for item in raw if item not in API_SPECS]
    if unknown:
        raise SystemExit(
            f"Unknown API model label(s): {unknown}. "
            f"Choose from {list(ALL_API_LABELS)} or 'all'."
        )
    return raw


def resolve_api_judge_label(raw: str | None) -> str | None:
    """Map a CLI token to an API judge label, or None if it is not an API judge."""
    key = str(raw or "").strip().lower()
    if not key or key in API_JUDGE_GROUP_ALIASES:
        return None
    if key in API_SPECS:
        return key
    return API_JUDGE_ALIASES.get(key)


def is_api_judge(model_id: str | None) -> bool:
    return resolve_api_judge_label(model_id) is not None


def is_batch_api_judge(model_id: str | None) -> bool:
    label = resolve_api_judge_label(model_id)
    if not label:
        return False
    return _is_batch_spec(API_SPECS.get(label))


def split_api_judges(labels: list[str]) -> tuple[list[str], list[str]]:
    """Split API labels into live (Gemini / sync) vs Batch API judges."""
    live: list[str] = []
    batch: list[str] = []
    for label in labels:
        if is_batch_api_judge(label):
            batch.append(label)
        else:
            live.append(label)
    return live, batch


def expand_api_judge_token(raw: str) -> list[str] | None:
    """Return API labels for ``api`` / aliases, or None when the token is not API."""
    key = str(raw or "").strip().lower()
    if key in API_JUDGE_GROUP_ALIASES:
        return list(ALL_API_LABELS)
    if key in BATCH_API_JUDGE_GROUP_ALIASES:
        return list(ALL_BATCH_API_LABELS)
    if key in OPENAI_BATCH_GROUP_ALIASES:
        return [
            name
            for name, spec in API_SPECS.items()
            if spec.get("backend") == "openai_batch"
        ]
    if key in ANTHROPIC_BATCH_GROUP_ALIASES:
        return [
            name
            for name, spec in API_SPECS.items()
            if spec.get("backend") == "anthropic_batch"
        ]
    label = resolve_api_judge_label(key)
    return [label] if label else None


def make_api_taker(
    label: str,
    *,
    temperature: float,
    max_tokens: int,
    qps: float = 4.0,
    timeout: float = 180.0,
    retries: int = 20,
    retry_interval: float = 1.0,
) -> OpenAIAudioTaker | GeminiAudioTaker:
    spec = API_SPECS.get(label) or API_SPECS.get(resolve_api_judge_label(label) or "")
    if spec is None:
        raise SystemExit(f"Unknown API model label: {label!r}")
    kwargs = dict(
        model_id=spec["model_id"],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        qps=qps,
        timeout=timeout,
        retries=retries,
        retry_interval=retry_interval,
    )
    if spec["backend"] == "openai":
        return OpenAIAudioTaker(**kwargs)
    if spec["backend"] == "gemini":
        return GeminiAudioTaker(**kwargs)
    if spec["backend"] in BATCH_BACKENDS:
        raise SystemExit(
            f"{label} uses the Batch API; run "
            f"uv run run_judges.py --judge-model-id {label}"
        )
    raise SystemExit(f"Unknown backend for {label}")


class OpenAIAudioTaker:
    def __init__(
        self,
        *,
        model_id: str,
        temperature: float,
        max_tokens: int,
        qps: float,
        timeout: float,
        retries: int,
        retry_interval: float,
    ):
        from openai import AsyncOpenAI

        if not (os.environ.get("OPENAI_API_KEY") or "").strip():
            raise SystemExit("Set OPENAI_API_KEY to call OpenAI audio models.")
        self.client = AsyncOpenAI()
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.retry_interval = retry_interval
        self._interval = 1.0 / max(float(qps), 0.1)
        self._next_time = 0.0
        self._lock = asyncio.Lock()

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
                now = time.monotonic()
            self._next_time = now + self._interval

    async def begin_prefix(self, audio_path: str | None, prompt: str) -> str | None:
        del audio_path, prompt
        return None

    async def end_prefix(self, cache_name: str | None) -> None:
        del cache_name

    async def complete(
        self,
        prompt: str,
        audio_path: str | None,
        seed: int,
        *,
        question_id: str,
        cached_content: str | None = None,
        prompt_cache_key: str | None = None,
    ) -> CompletionResult:
        del cached_content
        content: list[dict[str, Any]] = []
        if audio_path:
            audio_b64 = base64.b64encode(Path(audio_path).read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                }
            )
        content.append({"type": "text", "text": prompt})
        cache_key = prompt_cache_key or f"mmar:{question_id}"
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                await self._throttle()
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model_id,
                        modalities=["text"],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        seed=int(seed),
                        prompt_cache_key=cache_key,
                        messages=[{"role": "user", "content": content}],
                    ),
                    timeout=self.timeout,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("openai attempt %s failed: %s", attempt, exc)
                await asyncio.sleep(self.retry_interval * attempt)
                continue
            choice = (response.choices or [None])[0]
            text = ""
            if choice is not None and choice.message is not None:
                text = str(choice.message.content or "").strip()
            if text:
                cached_tokens, prompt_tokens = _openai_usage(response)
                return CompletionResult(
                    text=text,
                    cached_tokens=cached_tokens,
                    prompt_tokens=prompt_tokens,
                )
            logger.warning("openai attempt %s empty response", attempt)
            await asyncio.sleep(self.retry_interval)
        raise RuntimeError(f"OpenAI retries exhausted: {last_exc}")


class GeminiAudioTaker:
    def __init__(
        self,
        *,
        model_id: str,
        temperature: float,
        max_tokens: int,
        qps: float,
        timeout: float,
        retries: int,
        retry_interval: float,
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
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.retry_interval = retry_interval
        self._interval = 1.0 / max(float(qps), 0.1)
        self._next_time = 0.0
        self._lock = asyncio.Lock()
        self._logged_cache_fallback = False

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
                now = time.monotonic()
            self._next_time = now + self._interval

    def _log_cache_fallback(self, reason: str) -> None:
        if self._logged_cache_fallback:
            logger.debug("gemini explicit cache fallback: %s", reason)
            return
        self._logged_cache_fallback = True
        logger.warning(
            "gemini explicit cache unavailable (%s); falling back without explicit cache",
            reason,
        )

    async def begin_prefix(self, audio_path: str | None, prompt: str) -> str | None:
        contents: list[Any]
        if audio_path:
            path = Path(audio_path)
            size = path.stat().st_size
            if size > GEMINI_CACHE_MAX_BYTES:
                self._log_cache_fallback(
                    f"audio {size} bytes exceeds {GEMINI_CACHE_MAX_BYTES}"
                )
                return None
            contents = [
                self.types.Part.from_bytes(
                    data=path.read_bytes(), mime_type="audio/wav"
                )
            ]
        else:
            text = str(prompt or "").strip()
            if not text:
                return None
            contents = [text]
        try:
            await self._throttle()
            cache = await asyncio.wait_for(
                self.client.aio.caches.create(
                    model=self.model_id,
                    config=self.types.CreateCachedContentConfig(
                        contents=contents,
                        ttl=GEMINI_CACHE_TTL,
                    ),
                ),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            self._log_cache_fallback(str(exc))
            return None
        name = getattr(cache, "name", None)
        if not name:
            self._log_cache_fallback("caches.create returned no name")
            return None
        return str(name)

    async def end_prefix(self, cache_name: str | None) -> None:
        if not cache_name:
            return
        try:
            await self._throttle()
            await asyncio.wait_for(
                self.client.aio.caches.delete(name=cache_name),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("gemini cache delete %s failed: %s", cache_name, exc)

    async def complete(
        self,
        prompt: str,
        audio_path: str | None,
        seed: int,
        *,
        question_id: str,
        cached_content: str | None = None,
        prompt_cache_key: str | None = None,
    ) -> CompletionResult:
        del seed, question_id, prompt_cache_key
        if cached_content:
            contents: list[Any] = [prompt]
            config = self.types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=int(self.max_tokens),
                cached_content=cached_content,
            )
        elif audio_path:
            contents = [
                self.types.Part.from_bytes(
                    data=Path(audio_path).read_bytes(), mime_type="audio/wav"
                ),
                prompt,
            ]
            config = self.types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=int(self.max_tokens),
            )
        else:
            contents = [prompt]
            config = self.types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=int(self.max_tokens),
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
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("gemini attempt %s failed: %s", attempt, exc)
                await asyncio.sleep(self.retry_interval * attempt)
                continue
            text = gemini_response_text(response)
            if text:
                cached_tokens, prompt_tokens = _gemini_usage(response)
                return CompletionResult(
                    text=text,
                    cached_tokens=cached_tokens,
                    prompt_tokens=prompt_tokens,
                )
            logger.warning("gemini attempt %s empty response", attempt)
            await asyncio.sleep(self.retry_interval)
        raise RuntimeError(f"Gemini retries exhausted: {last_exc}")


@dataclass
class _PendingJob:
    qid: str
    question: str
    answer: str
    prediction: str
    audio_path: str | None
    reuse_key: tuple[str, str, str]
    owners: list[tuple[str, int, int]] = field(default_factory=list)


def _gold_prefix(question: str, answer: str, prompt_name: str | None = None) -> str:
    from grader import build_grade_gold_prefix

    return build_grade_gold_prefix(
        question=question, answer=answer, prompt=prompt_name
    )


def _verdict_entry(
    result: CompletionResult,
    *,
    model_id: str,
    prompt_name: str,
    include_gold: bool,
) -> dict[str, Any]:
    from grader import format_grade_output, parse_grade_verdict

    verdict = parse_grade_verdict(result.text)
    entry: dict[str, Any] = {
        "correct": bool(verdict) if verdict is not None else False,
        "verdict": (
            "pass" if verdict is True else "fail" if verdict is False else None
        ),
        "output": format_grade_output(verdict),
        "generation": result.text,
        "model_id": model_id,
        "prompt": prompt_name,
        "include_gold": include_gold,
    }
    if result.cached_tokens is not None:
        entry["cached_tokens"] = result.cached_tokens
    if result.prompt_tokens is not None:
        entry["prompt_tokens"] = result.prompt_tokens
    return entry


def _majority_verdict_entry(
    texts: list[str],
    *,
    model_id: str,
    prompt_name: str,
    include_gold: bool,
) -> dict[str, Any]:
    """Match ``grade_predictions_file`` majority-vote shot entries."""
    from grader import _verdict_fields, format_grade_output, majority_grade_verdict

    sample_fields = [_verdict_fields(text) for text in texts]
    raw = [item["grader_verdict_raw"] for item in sample_fields]
    majority = majority_grade_verdict(raw)
    generation = ""
    for item, verdict in zip(sample_fields, raw):
        if majority is not None and verdict is majority:
            generation = item["generation"]
            break
    if not generation and sample_fields:
        generation = sample_fields[0]["generation"]
    return {
        "correct": bool(majority) if majority is not None else False,
        "verdict": (
            "pass" if majority is True else "fail" if majority is False else None
        ),
        "output": format_grade_output(majority),
        "generation": generation,
        "model_id": model_id,
        "prompt": prompt_name,
        "include_gold": include_gold,
        "samples": [
            {
                "correct": item["correct"],
                "verdict": item["verdict"],
                "generation": item["generation"],
                "output": item["grader_output"],
            }
            for item in sample_fields
        ],
        "n_samples": len(sample_fields),
    }


def _load_prediction_records(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


async def grade_pack_with_api_judge(
    pack_dir: Path,
    *,
    label: str,
    model_labels: list[str],
    prompt: str | None = None,
    include_gold: bool = True,
    force: bool = False,
    n_questions: int | None = None,
    question_ids: list[str] | None = None,
    make_primary: bool = False,
    primary_judge: str | None = None,
    qps: float = 4.0,
    max_workers: int = 8,
    timeout: float = 180.0,
    retries: int = 20,
    retry_interval: float = 1.0,
    print_every: int = 5,
) -> dict[str, Any]:
    """Grade pack predictions with one API judge; writes ``predictions.jsonl``."""
    from grader import (
        compose_judge_key,
        get_judge_format,
        normalize_grade_prompt,
        require_audio_nongold_judge,
        resolve_grade_allowed_ids,
        resolve_grade_audio_path,
        _grade_reuse_key,
        _record_needs_grade,
        _shot_needs_grade,
        _shot_prediction_text,
        _shot_judge_entry,
    )

    resolved = resolve_api_judge_label(label) or label
    spec = API_SPECS[resolved]
    model_id = spec["model_id"]
    fmt = get_judge_format(prompt, include_gold=include_gold)
    include_gold = fmt.include_gold
    prompt_name = normalize_grade_prompt(prompt, include_gold=include_gold)
    require_audio_nongold_judge(
        model_id=resolved,
        include_gold=include_gold,
        audio_required=fmt.audio_included,
    )
    use_gold_prefix = include_gold and not fmt.audio_included
    key = compose_judge_key(resolved, prompt=prompt_name, include_gold=include_gold)
    gradees = [item for item in model_labels if item != resolved]
    files: dict[str, list[dict]] = {}
    all_ids: list[str] = []
    for gradee in gradees:
        records = _load_prediction_records(
            pack_dir / "models" / gradee / "predictions.jsonl"
        )
        files[gradee] = records
        for record in records:
            ensure_judge_schema(
                record, fallback_label=key, fallback_model_id=model_id
            )
            qid = str(record.get("id") or "")
            if qid:
                all_ids.append(qid)

    allowed_ids = resolve_grade_allowed_ids(
        all_ids, question_ids=question_ids, n_questions=n_questions
    )

    def _in_sample(record: dict) -> bool:
        if allowed_ids is None:
            return True
        return str(record.get("id") or "").strip() in allowed_ids

    reuse_cache: dict[tuple[str, str, str], dict] = {}
    if not force:
        for gradee, records in files.items():
            for record in records:
                if not _in_sample(record):
                    continue
                question = str(record.get("question") or "")
                answer = str(record.get("answer") or "")
                for shot in record.get("shots") or []:
                    entry = _shot_judge_entry(shot, key)
                    if entry is None:
                        continue
                    reuse_cache.setdefault(
                        _grade_reuse_key(
                            question,
                            answer,
                            _shot_prediction_text(shot, model_label=gradee),
                            include_gold=include_gold,
                        ),
                        dict(entry),
                    )

    pending_by_key: dict[tuple[str, str, str], _PendingJob] = {}
    reuse_owners: list[tuple[str, int, int, dict]] = []
    for gradee, records in files.items():
        pred_path = pack_dir / "models" / gradee / "predictions.jsonl"
        if not pred_path.is_file():
            continue
        for record_index, record in enumerate(records):
            if not _in_sample(record):
                continue
            if not force and not _record_needs_grade(record, key):
                continue
            question = str(record.get("question") or "")
            answer = str(record.get("answer") or "")
            audio_raw = str(record.get("audio_path") or "") or None
            for shot in record.get("shots") or []:
                shot_index = int(shot.get("shot_index", 0))
                if not force and not _shot_needs_grade(shot, key):
                    continue
                prediction = _shot_prediction_text(shot, model_label=gradee)
                cache_key = _grade_reuse_key(
                    question, answer, prediction, include_gold=include_gold
                )
                cached = reuse_cache.get(cache_key)
                if cached is not None:
                    reuse_owners.append((gradee, record_index, shot_index, cached))
                    continue
                job = pending_by_key.get(cache_key)
                if job is None:
                    job = _PendingJob(
                        qid=str(record.get("id") or ""),
                        question=question,
                        answer=answer,
                        prediction=prediction,
                        audio_path=audio_raw,
                        reuse_key=cache_key,
                    )
                    pending_by_key[cache_key] = job
                job.owners.append((gradee, record_index, shot_index))

    graded = 0
    reused = 0

    def _apply(gradee: str, record_index: int, shot_index: int, entry: dict) -> None:
        nonlocal graded
        record = files[gradee][record_index]
        for shot in record.get("shots") or []:
            if int(shot.get("shot_index", -1)) != shot_index:
                continue
            shot.setdefault("judges", {})[key] = dict(entry)
            graded += 1
            break

    for gradee, record_index, shot_index, entry in reuse_owners:
        _apply(gradee, record_index, shot_index, entry)
        reused += 1

    by_qid: dict[str, list[_PendingJob]] = {}
    for job in pending_by_key.values():
        by_qid.setdefault(job.qid, []).append(job)

    taker = None
    total_cached = 0
    total_prompt = 0
    n_api = 0
    if by_qid:
        taker = make_api_taker(
            resolved,
            temperature=float(JUDGE_SAMPLING["temperature"]),
            max_tokens=int(JUDGE_SAMPLING["max_tokens"]),
            qps=qps,
            timeout=timeout,
            retries=retries,
            retry_interval=retry_interval,
        )
        assert taker is not None
        shot_sem = asyncio.Semaphore(max_workers)
        multi = any(len(jobs) > 1 for jobs in by_qid.values())
        question_sem = asyncio.Semaphore(2 if multi else max_workers)
        qids = list(by_qid)
        done_questions = 0

        async def _complete_job(
            job: _PendingJob,
            *,
            cached_content: str | None,
            prefix: str | None,
            audio: str | None,
        ) -> tuple[_PendingJob, CompletionResult]:
            from grader import build_grade_prompt as _build

            full = _build(
                question=job.question,
                answer=job.answer,
                prediction=job.prediction,
                prompt=prompt_name,
                include_gold=include_gold,
            )
            send = full
            if cached_content and prefix and use_gold_prefix and full.startswith(prefix):
                send = full[len(prefix) :]
            async with shot_sem:
                result = await taker.complete(
                    send,
                    audio if fmt.audio_included else None,
                    0,
                    question_id=job.qid,
                    cached_content=cached_content,
                    prompt_cache_key=f"{JUDGE_CACHE_KEY_PREFIX}:{job.qid}",
                )
            return job, result

        async def _one_question(qid: str) -> None:
            nonlocal graded, reused, total_cached, total_prompt, n_api, done_questions
            jobs = by_qid[qid]
            audio: str | None = None
            if fmt.audio_included:
                raw = next((job.audio_path for job in jobs if job.audio_path), None)
                resolved_audio = resolve_grade_audio_path(raw)
                if resolved_audio is None:
                    raise SystemExit(
                        f"Audio API judge {resolved!r} missing audio for id={qid!r} "
                        f"path={raw!r}"
                    )
                audio = str(resolved_audio)
            prefix = None
            if use_gold_prefix:
                sample = jobs[0]
                prefix = _gold_prefix(sample.question, sample.answer, prompt_name)
            cache_name = None
            async with question_sem:
                try:
                    if len(jobs) > 1:
                        cache_name = await taker.begin_prefix(
                            audio if fmt.audio_included else None,
                            prefix or "",
                        )
                    pairs = list(
                        await asyncio.gather(
                            *(
                                _complete_job(
                                    job,
                                    cached_content=cache_name,
                                    prefix=prefix,
                                    audio=audio,
                                )
                                for job in jobs
                            )
                        )
                    )
                finally:
                    await taker.end_prefix(cache_name)
            for job, result in pairs:
                n_api += 1
                total_cached += int(result.cached_tokens or 0)
                total_prompt += int(result.prompt_tokens or 0)
                entry = _verdict_entry(
                    result,
                    model_id=model_id,
                    prompt_name=prompt_name,
                    include_gold=include_gold,
                )
                reuse_cache[job.reuse_key] = entry
                for extra_index, (gradee, record_index, shot_index) in enumerate(
                    job.owners
                ):
                    _apply(gradee, record_index, shot_index, entry)
                    if extra_index:
                        reused += 1
            done_questions += 1
            if done_questions % print_every == 0 or done_questions == len(qids):
                hit = (total_cached / total_prompt) if total_prompt else 0.0
                print(
                    f"[run-judges-api {resolved}] {done_questions}/{len(qids)} "
                    f"qid={qid} api={n_api} cache={total_cached}/{total_prompt} "
                    f"({hit:.0%})"
                )

        await asyncio.gather(*(_one_question(qid) for qid in qids))

    by_model: dict[str, dict] = {}
    for gradee, records in files.items():
        pred_path = pack_dir / "models" / gradee / "predictions.jsonl"
        if not pred_path.is_file():
            by_model[gradee] = {"status": "missing", "n_records": 0}
            continue
        for record in records:
            if allowed_ids is not None and str(record.get("id") or "").strip() not in allowed_ids:
                continue
            existing_primary = record.get("primary_judge")
            use_primary = key if make_primary else (existing_primary or primary_judge)
            existing = [str(x) for x in (record.get("judges") or []) if x]
            ordered: list[str] = []
            if use_primary:
                ordered.append(str(use_primary))
            for item in existing:
                if item not in ordered:
                    ordered.append(item)
            if key not in ordered:
                ordered.append(key)
            record["judges"] = ordered
            record["scoring"] = "qwen_freeform_judge"
            recompute_multi_judge_scores(record, use_primary)
        write_jsonl(pred_path, records, mode="w")
        by_model[gradee] = {
            "status": "ok",
            "n_records": len(records),
            "predictions_path": str(pred_path),
        }

    hit = (total_cached / total_prompt) if total_prompt else 0.0
    print(
        f"[run-judges-api {resolved}] done key={key} graded={graded} "
        f"reused={reused} api={n_api} cache={total_cached}/{total_prompt} "
        f"({hit:.1%})"
    )
    return {
        "status": "ok",
        "judge_label": resolved,
        "judge_key": key,
        "model_id": model_id,
        "backend": spec["backend"],
        "gradees": gradees,
        "by_model": by_model,
        "n_shots_graded": graded,
        "n_shots_reused": reused,
        "n_api_calls": n_api,
        "cached_tokens": total_cached,
        "prompt_tokens": total_prompt,
        "cache_hit_fraction": hit,
        "prompt": prompt_name,
        "include_gold": include_gold,
        "n_questions": n_questions,
        "n_labeled": len(question_ids) if question_ids is not None else None,
    }


async def grade_pack_with_api_judges(
    pack_dir: Path,
    *,
    labels: list[str],
    model_labels: list[str],
    prompts: list[str],
    include_gold: bool,
    force: bool,
    n_questions: int | None,
    question_ids: list[str] | None = None,
    make_primary: bool,
    primary_judge: str | None,
    qps: float = 4.0,
    max_workers: int = 8,
    timeout: float = 180.0,
    retries: int = 20,
    retry_interval: float = 1.0,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    first = True
    for label in labels:
        if is_batch_api_judge(label):
            continue
        for prompt_name in prompts:
            this_primary = bool(make_primary) and first
            first = False
            result = await grade_pack_with_api_judge(
                pack_dir,
                label=label,
                model_labels=model_labels,
                prompt=prompt_name,
                include_gold=include_gold,
                force=force,
                n_questions=n_questions,
                question_ids=question_ids,
                make_primary=this_primary,
                primary_judge=primary_judge,
                qps=qps,
                max_workers=max_workers,
                timeout=timeout,
                retries=retries,
                retry_interval=retry_interval,
            )
            results.append(result)
            print("API graded:", result)
    return results


def _openai_sync_client():
    from openai import OpenAI

    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        raise SystemExit("Set OPENAI_API_KEY to call OpenAI batch models.")
    return OpenAI()


def _anthropic_sync_client():
    from anthropic import Anthropic

    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        raise SystemExit("Set ANTHROPIC_API_KEY to call Anthropic batch models.")
    return Anthropic()


def _make_batch_client(backend: str):
    if backend == "openai_batch":
        return _openai_sync_client()
    if backend == "anthropic_batch":
        return _anthropic_sync_client()
    raise SystemExit(f"Unknown batch backend {backend!r}")


def _batch_id_for_backend(backend: str, batch_id: str | None) -> str | None:
    raw = str(batch_id or "").strip()
    if not raw:
        return None
    if raw.startswith("msgbatch_"):
        return raw if backend == "anthropic_batch" else None
    if raw.startswith("batch_"):
        return raw if backend == "openai_batch" else None
    return raw


def _batch_work_dir(pack_dir: Path, judge_key: str, backend: str = "openai_batch") -> Path:
    folder = BATCH_DIR_BY_BACKEND.get(backend, "_batch")
    dest = pack_dir / folder / judge_key
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(obj, "to_dict"):
        dumped = obj.to_dict()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _batch_request_body(
    spec: dict[str, Any],
    text: str,
    *,
    temperature: float | None = None,
) -> dict[str, Any]:
    backend = str(spec.get("backend") or "")
    sampling = dict(spec.get("sampling") or {})
    model_id = str(spec.get("model_id") or "")
    if backend == "openai_batch":
        body: dict[str, Any] = {
            "model": model_id,
            "input": text,
            "max_output_tokens": int(sampling.get("max_output_tokens") or 8192),
            "reasoning": {
                "effort": str(sampling.get("reasoning_effort") or "medium")
            },
        }
        if temperature is not None:
            body["temperature"] = float(temperature)
        return body
    if backend == "anthropic_batch":
        body = {
            "model": model_id,
            "max_tokens": int(sampling.get("max_tokens") or 8192),
            "messages": [{"role": "user", "content": text}],
        }
        effort = sampling.get("effort") or sampling.get("reasoning_effort")
        if effort:
            body["output_config"] = {"effort": str(effort)}
        if temperature is not None:
            body["temperature"] = float(temperature)
        return body
    raise SystemExit(f"Unknown batch backend {backend!r}")


def _max_jobs_per_batch(backend: str, n_samples: int) -> int:
    cap = int(BATCH_MAX_REQUESTS.get(backend) or 50_000)
    return max(1, cap // max(1, int(n_samples)))


def _batch_requests_for_jobs(
    jobs: list[dict[str, Any]],
    *,
    backend: str,
    endpoint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        ids = job.get("sample_custom_ids") or [job["custom_id"]]
        body = job["body"]
        for cid in ids:
            if backend == "anthropic_batch":
                rows.append({"custom_id": str(cid), "params": body})
            else:
                rows.append(
                    {
                        "custom_id": str(cid),
                        "method": "POST",
                        "url": endpoint,
                        "body": body,
                    }
                )
    return rows


def _responses_output_text(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    text = body.get("output_text")
    if text:
        return str(text).strip()
    parts: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "")
        if kind == "message":
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"}:
                    piece = content.get("text")
                    if piece:
                        parts.append(str(piece))
        elif kind in {"output_text", "text"}:
            piece = item.get("text")
            if piece:
                parts.append(str(piece))
    return "\n".join(parts).strip()


def _batch_response_text(row: dict[str, Any]) -> str:
    error = row.get("error")
    if error:
        return ""
    response = row.get("response") or {}
    if not isinstance(response, dict):
        return ""
    status = response.get("status_code")
    if status not in (None, 200):
        return ""
    body = response.get("body")
    text = _responses_output_text(body)
    if text:
        return text
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            if isinstance(message, dict):
                return str(message.get("content") or "").strip()
    return ""


def _iter_jsonl_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return _iter_jsonl_text(path.read_text(encoding="utf-8"))


def _write_jsonl_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _batch_status(batch: Any) -> str:
    return str(getattr(batch, "status", "") or "")


def _batch_counts(batch: Any) -> tuple[int, int, int]:
    counts = getattr(batch, "request_counts", None)
    total = int(getattr(counts, "total", 0) or 0)
    completed = int(getattr(counts, "completed", 0) or 0)
    failed = int(getattr(counts, "failed", 0) or 0)
    return total, completed, failed


def _download_batch_file(client: Any, file_id: str | None, dest: Path) -> list[dict[str, Any]]:
    if not file_id:
        return []
    payload = client.files.content(file_id).text
    dest.write_text(payload, encoding="utf-8")
    return _iter_jsonl_text(payload)


def _anthropic_message_text(message: Any) -> str:
    payload = _as_dict(message) if not isinstance(message, dict) else message
    parts: list[str] = []
    for block in payload.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            piece = block.get("text")
            if piece:
                parts.append(str(piece))
    return "\n".join(parts).strip()


def _anthropic_batch_response_text(row: dict[str, Any]) -> str:
    result = row.get("result")
    if not isinstance(result, dict):
        result = _as_dict(result)
    if str(result.get("type") or "") != "succeeded":
        return ""
    return _anthropic_message_text(result.get("message") or {})


def _provider_response_text(backend: str, row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    if backend == "anthropic_batch":
        return _anthropic_batch_response_text(row)
    return _batch_response_text(row)


def _provider_batch_status(backend: str, batch: Any) -> str:
    if backend == "anthropic_batch":
        return str(getattr(batch, "processing_status", "") or "")
    return _batch_status(batch)


def _provider_batch_active(backend: str) -> frozenset[str]:
    if backend == "anthropic_batch":
        return ANTHROPIC_BATCH_ACTIVE
    return OPENAI_BATCH_ACTIVE


def _provider_batch_terminal(backend: str) -> frozenset[str]:
    if backend == "anthropic_batch":
        return ANTHROPIC_BATCH_TERMINAL
    return OPENAI_BATCH_TERMINAL


def _provider_batch_succeeded(backend: str, status: str) -> bool:
    if backend == "anthropic_batch":
        return status == "ended"
    return status == "completed"


def _retrieve_provider_batch(backend: str, client: Any, batch_id: str) -> Any:
    if backend == "anthropic_batch":
        return client.messages.batches.retrieve(batch_id)
    return client.batches.retrieve(batch_id)


def _cancel_provider_batch(backend: str, client: Any, batch_id: str) -> None:
    if backend == "anthropic_batch":
        client.messages.batches.cancel(batch_id)
        return
    client.batches.cancel(batch_id)


def _print_batch_progress(backend: str, batch_id: str, batch: Any) -> str:
    status = _provider_batch_status(backend, batch)
    counts = getattr(batch, "request_counts", None)
    if backend == "anthropic_batch":
        processing = int(getattr(counts, "processing", 0) or 0)
        succeeded = int(getattr(counts, "succeeded", 0) or 0)
        errored = int(getattr(counts, "errored", 0) or 0)
        canceled = int(getattr(counts, "canceled", 0) or 0)
        expired = int(getattr(counts, "expired", 0) or 0)
        print(
            f"[run-judges-batch] {batch_id} status={status} "
            f"succeeded={succeeded} processing={processing} "
            f"errored={errored} canceled={canceled} expired={expired}"
        )
    else:
        total, completed, failed = _batch_counts(batch)
        print(
            f"[run-judges-batch] {batch_id} status={status} "
            f"completed={completed}/{total} failed={failed}"
        )
    return status


def _poll_provider_batch(
    backend: str, client: Any, batch_id: str, *, poll_interval: float
) -> Any:
    interval = max(float(poll_interval), 1.0)
    terminal = _provider_batch_terminal(backend)
    while True:
        batch = _retrieve_provider_batch(backend, client, batch_id)
        status = _print_batch_progress(backend, batch_id, batch)
        if status in terminal:
            return batch
        time.sleep(interval)


def _download_anthropic_results(
    client: Any, batch_id: str, dest: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in client.messages.batches.results(batch_id):
        row = _as_dict(item)
        if row:
            rows.append(row)
    _write_jsonl_dicts(dest, rows)
    return rows


def _download_provider_results(
    backend: str,
    client: Any,
    batch: Any,
    *,
    output_path: Path,
    error_path: Path,
) -> list[dict[str, Any]]:
    if backend == "anthropic_batch":
        batch_id = str(getattr(batch, "id", "") or "")
        return _download_anthropic_results(client, batch_id, output_path)
    rows = _download_batch_file(
        client, getattr(batch, "output_file_id", None), output_path
    )
    _download_batch_file(client, getattr(batch, "error_file_id", None), error_path)
    return rows


def _labeled_shot_keys(labeled_rows: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for row in labeled_rows:
        model = str(row.get("model_label") or "").strip()
        qid = str(row.get("question_id") or "").strip()
        if not model or not qid:
            continue
        try:
            shot_index = int(row.get("shot_index", 0))
        except (TypeError, ValueError):
            continue
        keys.add((model, qid, shot_index))
    return keys


def _pack_shot_keys(
    files: dict[str, list[dict]],
    *,
    question_ids: list[str] | None = None,
    shot_indices: tuple[int, ...] | list[int] | None = None,
) -> set[tuple[str, str, int]]:
    allowed_ids = (
        {str(qid) for qid in question_ids} if question_ids is not None else None
    )
    allowed_shots = (
        {int(i) for i in shot_indices} if shot_indices is not None else None
    )
    keys: set[tuple[str, str, int]] = set()
    for gradee, records in files.items():
        for record in records:
            qid = str(record.get("id") or "").strip()
            if not qid:
                continue
            if allowed_ids is not None and qid not in allowed_ids:
                continue
            for shot in record.get("shots") or []:
                try:
                    shot_index = int(shot.get("shot_index", 0))
                except (TypeError, ValueError):
                    continue
                if allowed_shots is not None and shot_index not in allowed_shots:
                    continue
                keys.add((gradee, qid, shot_index))
    return keys


def grade_pack_with_batch_api(
    pack_dir: Path,
    *,
    label: str,
    model_labels: list[str],
    labeled_rows: list[dict[str, Any]] | None = None,
    prompt: str = BATCH_API_GRADE_PROMPT,
    force: bool = False,
    make_primary: bool = False,
    primary_judge: str | None = None,
    poll_interval: float = 30.0,
    batch_id: str | None = None,
    n_samples: int = 1,
    temperature: float | None = None,
    shot_indices: tuple[int, ...] | list[int] | None = None,
    question_ids: list[str] | None = None,
    on_checkpoint: Any | None = None,
) -> dict[str, Any]:
    """Grade shots via OpenAI or Anthropic Batch APIs.

    ``labeled_rows`` limits work to triple-labeled shots (``run_judges``).
    Omit it to grade every matching shot in the pack (``question_ids`` /
    ``shot_indices``). ``n_samples > 1`` submits that many independent
    requests per unique prompt and majority-votes the parsed verdicts.
    """
    from grader import (
        compose_judge_key,
        get_judge_format,
        normalize_grade_prompt,
        _grade_reuse_key,
        _shot_needs_grade,
        _shot_prediction_text,
        _shot_judge_entry,
        build_grade_prompt,
    )

    resolved = resolve_api_judge_label(label) or label
    spec = API_SPECS.get(resolved)
    if spec is None or not _is_batch_spec(spec):
        raise SystemExit(f"{label!r} is not a Batch API judge")
    backend = str(spec.get("backend") or "")
    model_id = str(spec["model_id"])
    n_samples = max(1, int(n_samples))
    fmt = get_judge_format(prompt, include_gold=True)
    if fmt.audio_included:
        raise SystemExit(
            f"Batch judge {resolved!r} only supports text formats "
            f"(got {prompt!r})"
        )
    include_gold = fmt.include_gold
    prompt_name = normalize_grade_prompt(prompt, include_gold=include_gold)
    key = compose_judge_key(resolved, prompt=prompt_name, include_gold=include_gold)

    gradees = [item for item in model_labels if item != resolved]
    files: dict[str, list[dict]] = {}
    for gradee in gradees:
        records = _load_prediction_records(
            pack_dir / "models" / gradee / "predictions.jsonl"
        )
        files[gradee] = records
        for record in records:
            ensure_judge_schema(
                record, fallback_label=key, fallback_model_id=model_id
            )

    if labeled_rows is not None:
        wanted = _labeled_shot_keys(labeled_rows)
        if not wanted:
            raise SystemExit("No labeled shots with 3 reviewer ratings to batch-grade")
    else:
        wanted = _pack_shot_keys(
            files, question_ids=question_ids, shot_indices=shot_indices
        )
        if not wanted:
            raise SystemExit("No shots to batch-grade")

    reuse_cache: dict[tuple[str, str, str], dict] = {}
    if not force:
        for gradee, records in files.items():
            for record in records:
                qid = str(record.get("id") or "")
                question = str(record.get("question") or "")
                answer = str(record.get("answer") or "")
                for shot in record.get("shots") or []:
                    shot_index = int(shot.get("shot_index", 0))
                    if (gradee, qid, shot_index) not in wanted:
                        continue
                    entry = _shot_judge_entry(shot, key)
                    if entry is None:
                        continue
                    reuse_cache.setdefault(
                        _grade_reuse_key(
                            question,
                            answer,
                            _shot_prediction_text(shot, model_label=gradee),
                            include_gold=include_gold,
                        ),
                        dict(entry),
                    )

    pending_by_key: dict[tuple[str, str, str], _PendingJob] = {}
    reuse_owners: list[tuple[str, str, int, dict]] = []
    n_labeled = 0
    for gradee, records in files.items():
        for record in records:
            qid = str(record.get("id") or "")
            question = str(record.get("question") or "")
            answer = str(record.get("answer") or "")
            for shot in record.get("shots") or []:
                shot_index = int(shot.get("shot_index", 0))
                if (gradee, qid, shot_index) not in wanted:
                    continue
                n_labeled += 1
                if not force and not _shot_needs_grade(shot, key):
                    continue
                prediction = _shot_prediction_text(shot, model_label=gradee)
                cache_key = _grade_reuse_key(
                    question, answer, prediction, include_gold=include_gold
                )
                cached = reuse_cache.get(cache_key)
                if cached is not None and not force:
                    reuse_owners.append((gradee, qid, shot_index, cached))
                    continue
                job = pending_by_key.get(cache_key)
                if job is None:
                    job = _PendingJob(
                        qid=qid,
                        question=question,
                        answer=answer,
                        prediction=prediction,
                        audio_path=None,
                        reuse_key=cache_key,
                    )
                    pending_by_key[cache_key] = job
                job.owners.append((gradee, qid, shot_index))

    graded = 0
    reused = 0

    def _find_shot(gradee: str, qid: str, shot_index: int) -> dict | None:
        for record in files.get(gradee) or []:
            if str(record.get("id") or "") != qid:
                continue
            for shot in record.get("shots") or []:
                if int(shot.get("shot_index", -1)) == shot_index:
                    return shot
        return None

    def _apply(gradee: str, qid: str, shot_index: int, entry: dict) -> None:
        nonlocal graded
        shot = _find_shot(gradee, qid, shot_index)
        if shot is None:
            return
        shot.setdefault("judges", {})[key] = dict(entry)
        graded += 1

    def _checkpoint() -> None:
        if on_checkpoint is not None:
            on_checkpoint()

    def _persist() -> None:
        for gradee, records in files.items():
            pred_path = pack_dir / "models" / gradee / "predictions.jsonl"
            if not pred_path.is_file():
                continue
            for record in records:
                qid = str(record.get("id") or "")
                if not any(
                    (gradee, qid, int(shot.get("shot_index", 0))) in wanted
                    for shot in record.get("shots") or []
                ):
                    continue
                existing_primary = record.get("primary_judge")
                use_primary = key if make_primary else (existing_primary or primary_judge)
                existing = [str(x) for x in (record.get("judges") or []) if x]
                ordered: list[str] = []
                if use_primary:
                    ordered.append(str(use_primary))
                for item in existing:
                    if item not in ordered:
                        ordered.append(item)
                if key not in ordered:
                    ordered.append(key)
                record["judges"] = ordered
                record["scoring"] = "qwen_freeform_judge"
                recompute_multi_judge_scores(record, use_primary)
            write_jsonl(pred_path, records, mode="w")
        _checkpoint()

    for gradee, qid, shot_index, entry in reuse_owners:
        _apply(gradee, qid, shot_index, entry)
        reused += 1

    work_dir = _batch_work_dir(pack_dir, key, backend)
    state_path = work_dir / "state.json"
    jobs_path = work_dir / "jobs.jsonl"
    input_path = work_dir / "input.jsonl"
    output_path = work_dir / "output.jsonl"
    error_path = work_dir / "errors.jsonl"
    endpoint = str(spec.get("endpoint") or OPENAI_BATCH_ENDPOINT)
    max_jobs = _max_jobs_per_batch(backend, n_samples)

    def _jobs_from_pending() -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for index, job in enumerate(pending_by_key.values(), start=1):
            custom_id = f"req-{index:06d}"
            text = build_grade_prompt(
                question=job.question,
                answer=job.answer,
                prediction=job.prediction,
                prompt=prompt_name,
                include_gold=include_gold,
            )
            sample_ids = (
                [custom_id]
                if n_samples <= 1
                else [f"{custom_id}-s{i:02d}" for i in range(n_samples)]
            )
            jobs.append(
                {
                    "custom_id": custom_id,
                    "sample_custom_ids": sample_ids,
                    "owners": [
                        {"model": owner[0], "qid": owner[1], "shot_index": int(owner[2])}
                        for owner in job.owners
                    ],
                    "qid": job.qid,
                    "body": _batch_request_body(
                        spec, text, temperature=temperature
                    ),
                }
            )
        return jobs

    n_api = 0
    n_failed = 0
    explicit_batch_id = _batch_id_for_backend(backend, batch_id)
    state: dict[str, Any] = {}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    state_batch_id = str(state.get("batch_id") or "").strip() or None
    used_batch_id = explicit_batch_id or (
        state_batch_id if pending_by_key and not force else None
    )

    client = None
    if pending_by_key or used_batch_id:
        client = _make_batch_client(backend)

    def _write_state(batch: Any, *, n_requests: int, extra: dict[str, Any] | None = None) -> None:
        payload = {
            **state,
            "batch_id": getattr(batch, "id", None) or extra and extra.get("batch_id"),
            "backend": backend,
            "judge_key": key,
            "model_id": model_id,
            "n_requests": n_requests,
            "n_samples": n_samples,
            "status": _provider_batch_status(backend, batch) if batch is not None else "",
            "output_file_id": getattr(batch, "output_file_id", None) if batch is not None else None,
            "error_file_id": getattr(batch, "error_file_id", None) if batch is not None else None,
            "results_url": getattr(batch, "results_url", None) if batch is not None else None,
        }
        if extra:
            payload.update(extra)
        write_json(state_path, payload)
        _checkpoint()

    def _submit(jobs: list[dict[str, Any]]) -> Any:
        assert client is not None
        _write_jsonl_dicts(jobs_path, jobs)
        requests = _batch_requests_for_jobs(
            jobs, backend=backend, endpoint=endpoint
        )
        _write_jsonl_dicts(input_path, requests)
        if backend == "anthropic_batch":
            batch = client.messages.batches.create(requests=requests)
            _write_state(batch, n_requests=len(requests))
            print(
                f"[run-judges-batch {resolved}] submitted {batch.id} "
                f"jobs={len(jobs)} requests={len(requests)} n_samples={n_samples}"
            )
            return batch
        with input_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint=endpoint,
            completion_window=OPENAI_BATCH_COMPLETION_WINDOW,
            metadata={
                "pack": pack_dir.name,
                "judge_key": key,
                "description": f"MMAR {key}",
            },
        )
        _write_state(
            batch,
            n_requests=len(requests),
            extra={"input_file_id": uploaded.id},
        )
        print(
            f"[run-judges-batch {resolved}] submitted {batch.id} "
            f"jobs={len(jobs)} requests={len(requests)} n_samples={n_samples} "
            f"file={uploaded.id}"
        )
        return batch

    def _apply_output(jobs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
        nonlocal n_api, n_failed, reused
        by_id = {
            str(row.get("custom_id") or ""): row
            for row in rows
            if row.get("custom_id")
        }
        for job in jobs:
            sample_ids = [
                str(cid)
                for cid in (job.get("sample_custom_ids") or [job.get("custom_id")])
                if cid
            ]
            if not sample_ids:
                n_failed += 1
                continue
            texts = [
                _provider_response_text(backend, by_id.get(cid))
                for cid in sample_ids
            ]
            if n_samples <= 1:
                text = texts[0] if texts else ""
                if not text:
                    n_failed += 1
                    continue
                n_api += 1
                entry = _verdict_entry(
                    CompletionResult(text=text),
                    model_id=model_id,
                    prompt_name=prompt_name,
                    include_gold=include_gold,
                )
            else:
                if not any(texts):
                    n_failed += 1
                    continue
                n_api += sum(1 for text in texts if text)
                n_failed += sum(1 for text in texts if not text)
                entry = _majority_verdict_entry(
                    texts,
                    model_id=model_id,
                    prompt_name=prompt_name,
                    include_gold=include_gold,
                )
            for extra_index, owner in enumerate(job.get("owners") or []):
                if isinstance(owner, dict):
                    gradee = str(owner.get("model") or "")
                    qid = str(owner.get("qid") or "")
                    shot_index = int(owner.get("shot_index", 0))
                else:
                    gradee, qid, shot_index = owner
                    qid = str(qid)
                    shot_index = int(shot_index)
                if not force:
                    shot = _find_shot(str(gradee), qid, shot_index)
                    if shot is not None and not _shot_needs_grade(shot, key):
                        continue
                _apply(str(gradee), qid, shot_index, entry)
                if extra_index:
                    reused += 1

    def _rebuild_pending() -> None:
        nonlocal pending_by_key
        still: dict[tuple[str, str, str], _PendingJob] = {}
        for cache_key, job in pending_by_key.items():
            remaining = []
            for gradee, qid, shot_index in job.owners:
                shot = _find_shot(str(gradee), str(qid), int(shot_index))
                if shot is None or _shot_needs_grade(shot, key):
                    remaining.append((gradee, qid, shot_index))
            if remaining:
                job.owners = remaining
                still[cache_key] = job
        pending_by_key = still

    def _run_batch(jobs: list[dict[str, Any]], *, existing_id: str | None = None) -> None:
        assert client is not None
        if existing_id:
            batch = _poll_provider_batch(
                backend, client, existing_id, poll_interval=poll_interval
            )
            batch_id_used = existing_id
        else:
            batch = _submit(jobs)
            batch_id_used = str(batch.id)
            batch = _poll_provider_batch(
                backend, client, batch.id, poll_interval=poll_interval
            )
        status = _provider_batch_status(backend, batch)
        n_requests = sum(
            len(job.get("sample_custom_ids") or [job.get("custom_id")])
            for job in jobs
        )
        _write_state(
            batch,
            n_requests=n_requests,
            extra={"batch_id": batch_id_used},
        )
        if not _provider_batch_succeeded(backend, status):
            raise SystemExit(f"Batch {batch_id_used} ended with status={status}")
        rows = _download_provider_results(
            backend, client, batch, output_path=output_path, error_path=error_path
        )
        _apply_output(jobs, rows)
        _persist()

    jobs = _load_jsonl_dicts(jobs_path) if jobs_path.is_file() else []
    if force and state_batch_id and client is not None and not explicit_batch_id:
        try:
            existing = _retrieve_provider_batch(backend, client, state_batch_id)
            if _provider_batch_status(backend, existing) in _provider_batch_active(
                backend
            ):
                _cancel_provider_batch(backend, client, state_batch_id)
                print(f"[run-judges-batch {resolved}] cancelled {state_batch_id}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("cancel %s failed: %s", state_batch_id, exc)
        used_batch_id = None
        jobs = []

    if used_batch_id and client is not None:
        if not jobs:
            raise SystemExit(
                f"Batch {used_batch_id} completed but {jobs_path} is missing"
            )
        _run_batch(jobs, existing_id=used_batch_id)
        _rebuild_pending()
        used_batch_id = None

    while pending_by_key:
        pending_items = list(pending_by_key.items())
        chunk_items = pending_items[:max_jobs]
        rest_items = pending_items[max_jobs:]
        pending_by_key = dict(chunk_items)
        jobs = _jobs_from_pending()
        if client is None:
            client = _make_batch_client(backend)
        _run_batch(jobs)
        pending_by_key = dict(rest_items)

    _persist()

    print(
        f"[run-judges-batch {resolved}] done key={key} labeled={n_labeled} "
        f"graded={graded} reused={reused} api={n_api} failed={n_failed} "
        f"n_samples={n_samples}"
    )
    return {
        "status": "ok",
        "judge_label": resolved,
        "judge_key": key,
        "model_id": model_id,
        "backend": spec["backend"],
        "gradees": gradees,
        "n_labeled_shots": n_labeled,
        "n_shots_graded": graded,
        "n_shots_reused": reused,
        "n_api_calls": n_api,
        "n_failed": n_failed,
        "n_samples": n_samples,
        "prompt": prompt_name,
        "include_gold": include_gold,
        "n_labeled": len(wanted),
    }


grade_pack_with_openai_batch = grade_pack_with_batch_api
