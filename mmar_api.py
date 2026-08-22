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
    select_grade_question_ids,
    write_jsonl,
)

logger = logging.getLogger(__name__)

API_SPECS: dict[str, dict[str, Any]] = {
    "gpt-audio-mini": {
        "model_id": "gpt-audio-mini",
        "backend": "openai",
        "native_thinking": False,
        "sampling": {"temperature": 1.0, "max_tokens": 1024},
    },
    "gemini-3.7-flash": {
        "model_id": "gemini-3.7-flash",
        "backend": "gemini",
        "native_thinking": True,
        "sampling": {"temperature": 1.0, "max_tokens": 1024},
    },
}

ALL_API_LABELS = tuple(API_SPECS.keys())
API_JUDGE_ALIASES: dict[str, str] = {
    "gpt-audio-mini": "gpt-audio-mini",
    "gemini-3.7-flash": "gemini-3.7-flash",
    "gemini-3.7-mini": "gemini-3.7-flash",
    "gemini": "gemini-3.7-flash",
}
API_JUDGE_GROUP_ALIASES = frozenset({"api", "all-api"})
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


def expand_api_judge_token(raw: str) -> list[str] | None:
    """Return API labels for ``api`` / aliases, or None when the token is not API."""
    key = str(raw or "").strip().lower()
    if key in API_JUDGE_GROUP_ALIASES:
        return list(ALL_API_LABELS)
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
            raise SystemExit("Set OPENAI_API_KEY to call gpt-audio-mini.")
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

    del prompt_name
    return build_grade_gold_prefix(question=question, answer=answer)


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
        grade_prompt_name,
        require_audio_nongold_judge,
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
    include_gold = bool(include_gold)
    prompt_name = grade_prompt_name(include_gold)
    del prompt
    require_audio_nongold_judge(model_id=resolved, include_gold=include_gold)
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

    selected = select_grade_question_ids(all_ids, n_questions)
    allowed_ids = set(selected) if selected is not None else None

    def _in_sample(record: dict) -> bool:
        if allowed_ids is None:
            return True
        return str(record.get("id") or "").strip() in allowed_ids

    reuse_cache: dict[tuple[str, str, str], dict] = {}
    if not force:
        for records in files.values():
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
                            _shot_prediction_text(shot),
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
                prediction = _shot_prediction_text(shot)
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
            if cached_content and prefix and include_gold and full.startswith(prefix):
                send = full[len(prefix) :]
            async with shot_sem:
                result = await taker.complete(
                    send,
                    None if include_gold else audio,
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
            if not include_gold:
                raw = next((job.audio_path for job in jobs if job.audio_path), None)
                resolved_audio = resolve_grade_audio_path(raw)
                if resolved_audio is None:
                    raise SystemExit(
                        f"NO_GOLD API judge {resolved!r} missing audio for id={qid!r} "
                        f"path={raw!r}"
                    )
                audio = str(resolved_audio)
            prefix = None
            if include_gold:
                sample = jobs[0]
                prefix = _gold_prefix(sample.question, sample.answer, prompt_name)
            cache_name = None
            async with question_sem:
                try:
                    if len(jobs) > 1:
                        cache_name = await taker.begin_prefix(
                            None if include_gold else audio,
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
