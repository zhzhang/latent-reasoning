"""Shared helpers for audio-groundedness eval of LALM thinking traces.

Judges hear the clip and read the student's output only — no question stem
and no answer options. Verdict is Yes (every factual claim is supported by
the audio) or No (at least one claim is not).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmar_common import (
    THINK_BLOCK_RE,
    load_jsonl,
    summarize_evaluated,
    write_json,
)
from mmar_rubrics import (
    DEFAULT_LIMIT,
    SOURCE_EXPERIMENT,
    append_evaluated,
    first_shot_fields,
    is_fully_graded,
    judge_model_dir,
    load_completed_ids,
    load_predictions_by_id,
    load_question_ids,
    prune_incomplete_evaluations,
)

GROUNDEDNESS_EXPERIMENT = "exp-mmar-groundedness"
GROUNDEDNESS_API_EXPERIMENT = "exp-mmar-groundedness-api"
DEFAULT_MODEL_LABEL = "af-next-think"
DEFAULT_JUDGE_LABEL = "qwen3-omni"
QWEN3_OMNI_JUDGE_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
QWEN3_OMNI_SAMPLE_RATE = 16000
GEMINI_JUDGE_ID = "gemini-3.1-pro-preview"
GEMINI_JUDGE_ALIASES = frozenset(
    {
        "gemini",
        "gemini-3.1",
        "gemini-3.1-pro",
        "gemini-3.1-pro-preview",
    }
)
GEMINI_AUDIO_MIME = "audio/wav"
# Gemini inline generateContent payload cap is 20 MB; leave headroom for text.
GEMINI_INLINE_AUDIO_MAX_BYTES = 18 * 1024 * 1024

GROUNDEDNESS_JUDGE_PROMPT = (
    "You are an expert annotator checking for factual groundedness in student "
    "reasoning over a set of audio reasoning tasks. You will be given the audio "
    "student and the student's output. Determine whether all factual claims in "
    "the output are directly supported by the real contents of the audio. In "
    "doing so you should list each factual claim the student makes and assess "
    "whether the claim is supported by the audio, before giving a final verdict "
    '"Yes" meaning all factual claims made by the students are supported by the '
    'audio, and "No" meaning that there is at least one factual claim made by '
    "the student that is not supported by the audio."
)

_VERDICT_PHRASE_RE = re.compile(
    r"(?:final\s+)?verdict\s*[:\-]?\s*[\"']?(yes|no)[\"']?\b",
    re.IGNORECASE,
)
_BARE_VERDICT_RE = re.compile(r"^[\"']?(yes|no)[\"']?[.\s]*$", re.IGNORECASE)


def _norm_verdict(token: str) -> str:
    return "Yes" if token.lower() == "yes" else "No"


def visible_judge_text(text: str) -> str:
    """Drop ``<think>`` blocks so the verdict is read from the final answer."""
    raw = str(text or "")
    match = THINK_BLOCK_RE.search(raw)
    if not match:
        return raw.strip()
    remainder = (raw[: match.start()] + raw[match.end() :]).strip()
    return remainder or raw.strip()


def parse_groundedness_verdict(text: str) -> str | None:
    """Return ``Yes`` / ``No`` from a judge reply, or None if unparseable."""
    region = visible_judge_text(text)
    if not region:
        return None
    matches = list(_VERDICT_PHRASE_RE.finditer(region))
    if matches:
        return _norm_verdict(matches[-1].group(1))
    lines = [line.strip() for line in region.splitlines() if line.strip()]
    for line in reversed(lines[-12:]):
        bare = _BARE_VERDICT_RE.match(line)
        if bare:
            return _norm_verdict(bare.group(1))
        phrase = _VERDICT_PHRASE_RE.search(line)
        if phrase:
            return _norm_verdict(phrase.group(1))
    return None


def create_groundedness_user_prompt(
    student_output: str, *, include_instructions: bool = True
) -> str:
    """User-turn text: student's output, optionally preceded by the judge prompt.

    Qwen3-Omni has no system turn, so Modal passes ``include_instructions=True``.
    Gemini gets the judge prompt as ``system_instruction``.
    """
    output = str(student_output or "").strip() or "(empty)"
    body = f"Student's output:\n{output}"
    if include_instructions:
        return f"{GROUNDEDNESS_JUDGE_PROMPT}\n\n{body}"
    return body


def format_qwen3_omni_audio_prompt(user_text: str) -> str:
    """Qwen3-Omni Thinking chat turn with an audio placeholder, no system turn."""
    return (
        "<|im_start|>user\n"
        f"<|audio_start|><|audio_pad|><|audio_end|>{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def student_output_from_item(item: dict) -> str:
    thinking = str(item.get("thinking_prediction") or "").strip()
    if thinking:
        return thinking
    raw = str(item.get("model_output") or "").strip()
    if raw:
        return raw
    return str(item.get("answer_prediction") or "").strip()


def resolve_eval_audio_path(
    audio_path: str | None,
    *,
    data_root: Path,
    audio_dir: Path | None = None,
) -> Path | None:
    """Resolve a prediction audio path against the MMAR data root / audio dir."""
    audio_dir = audio_dir or (Path(data_root) / "audio")
    candidates: list[Path] = []
    if audio_path:
        path = Path(audio_path)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(Path(data_root) / path)
        candidates.append(audio_dir / path.name)
        candidates.append(Path(data_root) / "audio" / path.name)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def build_groundedness_input_items(
    source_dir: Path,
    *,
    model_label: str = DEFAULT_MODEL_LABEL,
    data_root: Path,
    limit: int = DEFAULT_LIMIT,
    question_ids: list[str] | None = None,
    audio_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build records for the first ``limit`` question ids of one test-taker."""
    all_ids = question_ids if question_ids is not None else load_question_ids(source_dir)
    selected_ids = list(all_ids[: max(0, int(limit))])
    predictions_path = source_dir / "models" / model_label / "predictions.jsonl"
    preds = load_predictions_by_id(predictions_path)

    items: list[dict[str, Any]] = []
    missing: list[str] = []
    for record_id in selected_ids:
        pred = preds.get(record_id)
        if pred is None:
            missing.append(record_id)
            continue
        thinking_prediction, answer_prediction, shot_index = first_shot_fields(pred)
        audio = resolve_eval_audio_path(
            pred.get("audio_path"),
            data_root=data_root,
            audio_dir=audio_dir,
        )
        if audio is None:
            missing.append(record_id)
            continue
        items.append(
            {
                "id": pred["id"],
                "question": pred.get("question") or "",
                "answer": pred.get("answer") or "",
                "choices": pred.get("choices") or [],
                "thinking_prediction": thinking_prediction,
                "answer_prediction": answer_prediction,
                "shot_index": shot_index,
                "model_output": pred.get("model_output") or "",
                "modality": pred.get("modality") or "unknown",
                "category": pred.get("category") or "unknown",
                "sub-category": pred.get("sub-category"),
                "audio_path": str(audio),
                "source_model": model_label,
                "language": pred.get("language"),
                "source": pred.get("source"),
            }
        )

    if missing and len(items) == 0:
        raise SystemExit(
            f"No usable predictions for model={model_label} under {predictions_path}. "
            f"Missing/invalid ids sample: {missing[:5]}"
        )
    if missing:
        print(
            f"[mmar_groundedness] skipping {len(missing)} ids without "
            f"prediction/audio (sample={missing[:3]})"
        )
    return items, selected_ids


def evaluated_record_from_verdict(
    item: dict,
    *,
    raw_response: str,
    verdict: str | None,
) -> dict[str, Any]:
    yes = verdict == "Yes"
    return {
        **item,
        "new": True,
        "verdict": verdict,
        "score": 1.0 if yes else 0.0,
        "correct": yes,
        "raw_responses": [raw_response],
        "scoring": "audio_groundedness",
    }


def write_groundedness_manifest(
    run_dir: Path,
    *,
    source_run_id: str,
    model_label: str,
    question_ids: list[str],
    judge_label: str,
    judge_model_id: str,
    limit: int,
    backend: str,
    experiment: str | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    judges = {
        str(entry.get("label")): dict(entry)
        for entry in (existing.get("judges") or [])
        if isinstance(entry, dict) and entry.get("label")
    }
    judges[judge_label] = {
        "label": judge_label,
        "model_id": judge_model_id,
        "backend": backend,
        "num_raters": 1,
    }

    models: dict[str, dict[str, Any]] = {}
    for entry in existing.get("models") or []:
        if isinstance(entry, str) and entry.strip():
            models[entry.strip()] = {"label": entry.strip()}
        elif isinstance(entry, dict) and entry.get("label"):
            models[str(entry["label"])] = dict(entry)
    legacy_label = existing.get("model_label")
    if legacy_label and str(legacy_label) not in models:
        models[str(legacy_label)] = {"label": str(legacy_label)}
    models[model_label] = {
        **(models.get(model_label) or {}),
        "label": model_label,
        "shot": "first",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    inferred = experiment or existing.get("experiment") or GROUNDEDNESS_EXPERIMENT
    payload = {
        **existing,
        "experiment": inferred,
        "eval_kind": "groundedness",
        "source_run_id": source_run_id,
        "source_experiment": SOURCE_EXPERIMENT,
        "model_label": model_label,
        "models": list(models.values()),
        "shot": "first",
        "limit": limit,
        "n_questions": len(question_ids),
        "num_raters": 1,
        "scoring": "audio_groundedness",
        "judges": list(judges.values()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if "created_at" not in payload:
        payload["created_at"] = payload["updated_at"]
    write_json(path, payload)
    write_json(
        run_dir / "question_ids.json",
        {
            "source_run_id": source_run_id,
            "limit": limit,
            "n": len(question_ids),
            "ids": question_ids,
        },
    )
    return path


def write_groundedness_scores(
    model_dir: Path,
    evaluated_path: Path,
    *,
    judge_label: str,
    judge_model_id: str,
) -> dict:
    summary = summarize_evaluated(evaluated_path)
    records = load_jsonl(evaluated_path) if evaluated_path.is_file() else []
    n_yes = sum(1 for item in records if item.get("verdict") == "Yes")
    n_no = sum(1 for item in records if item.get("verdict") == "No")
    n_unparsed = sum(
        1
        for item in records
        if item.get("verdict") not in {"Yes", "No"}
    )
    n = len(records)
    summary.update(
        {
            "judge_label": judge_label,
            "judge_model_id": judge_model_id,
            "scoring": "audio_groundedness",
            "n_yes": n_yes,
            "n_no": n_no,
            "n_unparsed": n_unparsed,
            "grounded_rate": (n_yes / n) if n else None,
            "evaluated_path": str(evaluated_path),
        }
    )
    write_json(model_dir / "scores.json", summary)
    return summary


logger = logging.getLogger(__name__)


def is_gemini_judge(model_id: str | None) -> bool:
    text = str(model_id or "").strip().lower()
    if text in GEMINI_JUDGE_ALIASES:
        return True
    return "gemini-3.1-pro" in text


def resolve_gemini_model_id(model_id: str | None = None) -> str:
    text = str(model_id or "").strip()
    if not text or text.lower() in GEMINI_JUDGE_ALIASES:
        return GEMINI_JUDGE_ID
    return text


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


def _gemini_client():
    from google import genai

    api_key = (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    ).strip()
    if not api_key:
        raise ValueError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY to call Gemini 3.1 Pro."
        )
    return genai.Client(api_key=api_key)


def _gemini_audio_part(types: Any, audio_bytes: bytes):
    return types.Part.from_bytes(data=audio_bytes, mime_type=GEMINI_AUDIO_MIME)


class GeminiAudioScorer:
    """Gemini 3.1 Pro judge: wav clip + student output, no question/options."""

    def __init__(
        self,
        *,
        model_name: str = GEMINI_JUDGE_ID,
        api_max_retries: int = 20,
        api_retry_interval: float = 1.0,
        qps: float = 4.0,
        timeout: float = 180.0,
        max_output_tokens: int = 4096,
        thinking_level: str = "medium",
    ):
        from google.genai import types

        self.types = types
        self.client = _gemini_client()
        self.model_name = resolve_gemini_model_id(model_name)
        self.api_max_retries = api_max_retries
        self.api_retry_interval = api_retry_interval
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level
        self._next_time = 0.0
        self._interval = 1.0 / max(float(qps), 0.1)
        self._lock = asyncio.Lock()

    def _is_rate_limit(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (429, 418):
            return True
        text = str(exc).lower()
        return "rate" in text or "resource exhausted" in text or "429" in text

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
                now = time.monotonic()
            self._next_time = now + self._interval

    async def _complete(self, user_prompt: str, audio_bytes: bytes) -> str:
        thinking = None
        if self.thinking_level:
            thinking = self.types.ThinkingConfig(thinking_level=self.thinking_level)
        config = self.types.GenerateContentConfig(
            system_instruction=GROUNDEDNESS_JUDGE_PROMPT,
            max_output_tokens=int(self.max_output_tokens),
            thinking_config=thinking,
        )
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=[
                _gemini_audio_part(self.types, audio_bytes),
                user_prompt,
            ],
            config=config,
        )
        return gemini_response_text(response)

    async def call(self, log_id: str, user_prompt: str, audio_bytes: bytes) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self.api_max_retries + 1):
            try:
                await self._throttle()
                text = await asyncio.wait_for(
                    self._complete(user_prompt, audio_bytes),
                    timeout=self.timeout,
                )
            except TimeoutError as exc:
                last_exc = exc
                logger.warning("%s (api_attempt=%s) timeout: %s", log_id, attempt, exc)
                await asyncio.sleep(self.api_retry_interval)
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if self._is_rate_limit(exc):
                    logger.warning(
                        "%s (api_attempt=%s) rate limit: %s", log_id, attempt, exc
                    )
                    await asyncio.sleep(self.api_retry_interval * attempt)
                    continue
                raise
            if text:
                return text
            logger.warning("%s (api_attempt=%s) empty Gemini response", log_id, attempt)
            await asyncio.sleep(self.api_retry_interval)
        raise RuntimeError(
            f"{log_id} Gemini retries exhausted ({self.api_max_retries}): {last_exc}"
        )


async def grade_items_with_gemini(
    items: list[dict[str, Any]],
    evaluated_path: Path,
    *,
    model_id: str = GEMINI_JUDGE_ID,
    max_workers: int = 8,
    qps: float = 4.0,
    timeout: float = 180.0,
    retries: int = 20,
    retry_interval: float = 1.0,
    max_output_tokens: int = 4096,
    thinking_level: str = "medium",
    poll_interval_s: float = 30.0,
) -> dict[str, Any]:
    """Score pending groundedness items with Gemini Batch API (wav + trace)."""
    del max_workers, qps, timeout, retries, retry_interval  # Batch replaces interactive.
    from api_batch import (
        gemini_generate_request,
        gemini_inline_audio_part,
        gemini_text_part,
        gemini_user_contents,
        run_gemini_generate_batch,
    )

    model_id = resolve_gemini_model_id(model_id)
    evaluated_path.parent.mkdir(parents=True, exist_ok=True)
    removed = prune_incomplete_evaluations(evaluated_path)
    if removed:
        print(f"[gemini] pruned {removed} incomplete rows")
    completed = load_completed_ids(evaluated_path)
    pending = [item for item in items if item["id"] not in completed]
    print(
        f"[gemini-batch] judge={model_id} pending={len(pending)} "
        f"completed={len(completed)} -> {evaluated_path}"
    )
    if not pending:
        return {"status": "already_done", "n_ok": 0, "n_fail": 0, "n_pending": 0}

    length_limit = 10000
    requests: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    n_fail = 0
    for item in pending:
        output = student_output_from_item(item)
        if len(output) >= length_limit:
            n_fail += 1
            print(f"[gemini] skipping overlong trace {item['id']}")
            continue
        audio_path = Path(item["audio_path"])
        if not audio_path.is_file():
            n_fail += 1
            print(f"[gemini] missing audio {item['id']}: {audio_path}")
            continue
        audio_bytes = audio_path.read_bytes()
        if len(audio_bytes) > GEMINI_INLINE_AUDIO_MAX_BYTES:
            n_fail += 1
            print(
                f"[gemini] audio too large for batch request {item['id']}: "
                f"{len(audio_bytes)} bytes"
            )
            continue
        user_prompt = create_groundedness_user_prompt(
            output, include_instructions=False
        )
        requests.append(
            gemini_generate_request(
                key=str(item["id"]),
                contents=gemini_user_contents(
                    gemini_inline_audio_part(audio_bytes, mime_type=GEMINI_AUDIO_MIME),
                    gemini_text_part(user_prompt),
                ),
                system_instruction=GROUNDEDNESS_JUDGE_PROMPT,
                max_output_tokens=max_output_tokens,
                thinking_level=thinking_level,
            )
        )
        eligible.append(item)

    texts = run_gemini_generate_batch(
        requests,
        model=model_id,
        work_dir=evaluated_path.parent / "gemini-batch",
        display_name="groundedness",
        poll_interval_s=poll_interval_s,
    )

    n_ok = 0
    n_unparsed = 0
    for item in eligible:
        raw = texts.get(str(item["id"]))
        if not raw:
            n_fail += 1
            print(f"[gemini] failed {item['id']}: missing batch result")
            continue
        verdict = parse_groundedness_verdict(raw)
        if verdict is None:
            n_unparsed += 1
            print(f"[gemini] unparsed verdict for {item['id']}: {raw[:180]!r}")
        append_evaluated(
            evaluated_path,
            [evaluated_record_from_verdict(item, raw_response=raw, verdict=verdict)],
        )
        n_ok += 1
        if n_ok % 10 == 0 or n_ok == len(eligible):
            print(f"[gemini] scored {n_ok}/{len(eligible)}")

    return {
        "status": "ok",
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_unparsed": n_unparsed,
        "n_pending": len(pending),
    }
