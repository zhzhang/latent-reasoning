"""Helpers for MMAR freeform answer-uniformity eval (Gemini).

Per question, collect every model's extracted ``answer_prediction`` strings
(all shots) and ask whether they name the same specific concept.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmar_common import (  # noqa: E402
    THINK_BLOCK_RE,
    load_jsonl,
    write_json,
    write_jsonl,
)
from mmar_rubrics import SOURCE_EXPERIMENT  # noqa: E402
import view_difficulty as vd  # noqa: E402

ANSWER_VARIETY_EXPERIMENT = "exp-mmar-answer-variety"
DEFAULT_SOURCE_RUN_ID = "20260807T145000Z"
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
DEFAULT_LIMIT = 0  # 0 = all questions
PACKAGE_DIR = Path(__file__).resolve().parent
GEMINI_BINARY_IDS_PATH = PACKAGE_DIR / "gemini_binary_question_ids.csv"
OPEN_ENDED_IDS_PATH = PACKAGE_DIR / "open_ended_question_ids.csv"

UNIFORMITY_SYSTEM_PROMPT = (
    "You compare extracted final-answer strings from several audio models "
    "on the same question. Decide whether they are all identical in meaning.\n"
    "\n"
    "Treat strings as identical when they name the same specific concept, "
    "even if wording differs. Examples of identical: \"a dog\", \"dog\", "
    "\"there's a dog\", \"I heard a dog\".\n"
    "\n"
    "Do not treat them as identical when one is a strictly broader category "
    "(hypernym) or a different concept. Examples of not identical: "
    "\"a dog\" vs \"an animal\"; \"a dog\" vs \"a chair\".\n"
    "\n"
    "Ignore capitalization, punctuation, and light extra phrasing. "
    "Do not use any information except the listed answer strings.\n"
    "\n"
    "Reason briefly if needed, then end your reply with a single final line "
    "containing only Yes or No. Yes means every listed string refers to the "
    "same specific concept. No means at least one does not."
)

_VERDICT_PHRASE_RE = re.compile(
    r"(?:final\s+)?(?:verdict|answer|identical)\s*[:\-]?\s*[\"']?(yes|no)[\"']?\b",
    re.IGNORECASE,
)
_BARE_VERDICT_RE = re.compile(r"^[\"']?(yes|no)[\"']?[.\s]*$", re.IGNORECASE)
_CHOICE_LETTER_RE = re.compile(r"^[(\[]?[A-Da-d][)\].:]+\s*")
_YES_CHOICE_RE = re.compile(r"^(yes|true|correct)\b", re.IGNORECASE)
_NO_CHOICE_RE = re.compile(r"^(no|false|incorrect)\b", re.IGNORECASE)
# Polar 2-choice stems: AUX + determiner/pronoun. User-requested core
# plus a few matching variants.
YES_NO_QUESTION_PREFIXES = (
    "is this",
    "is the",
    "is that",
    "is there",
    "does this",
    "does the",
    "does that",
    "did this",
    "did the",
    "are the",
    "are they",
    "are these",
    "was the",
    "was this",
)
_YES_NO_STEM_RE = re.compile(
    r"(?:^|[.?!]\s+)(?:"
    + "|".join(re.escape(prefix) for prefix in YES_NO_QUESTION_PREFIXES)
    + r")\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


def _norm_verdict(token: str) -> str:
    return "Yes" if token.lower() == "yes" else "No"


def _normalize_choice(choice: str) -> str:
    text = str(choice or "").strip().strip("\"'")
    return _CHOICE_LETTER_RE.sub("", text).strip()


def _choice_polarity(choice: str) -> str | None:
    """Return ``yes``, ``no``, or None if the choice is not a yes/no string."""
    text = _normalize_choice(choice)
    if not text:
        return None
    if _YES_CHOICE_RE.match(text):
        return "yes"
    if _NO_CHOICE_RE.match(text):
        return "no"
    return None


def _choices_are_yes_no(choices: list | None) -> bool:
    labels = [_choice_polarity(c) for c in (choices or []) if str(c).strip()]
    if len(labels) < 2 or any(label is None for label in labels):
        return False
    return "yes" in labels and "no" in labels


def _question_has_polar_stem(question: str | None) -> bool:
    text = str(question or "").strip()
    if not text:
        return False
    return _YES_NO_STEM_RE.search(text) is not None


def is_yes_no_question(
    choices: list | None, question: str | None = None
) -> bool:
    """True for yes/no-style items that collapse freeform variety.

    1. Every MCQ option is a Yes/No (or True/False) polarity string,
       including prefixed forms such as ``Yes, he complied.``
    2. Or there are exactly two choices and the stem is polar
       (``is this``, ``is the``, ``does this``, ``does the``, plus
       matching variants like ``is that``, ``did the``, ``are they``).
    """
    if _choices_are_yes_no(choices):
        return True
    nonempty = [c for c in (choices or []) if str(c).strip()]
    return len(nonempty) == 2 and _question_has_polar_stem(question)


def visible_judge_text(text: str) -> str:
    raw = str(text or "")
    match = THINK_BLOCK_RE.search(raw)
    if not match:
        return raw.strip()
    remainder = (raw[: match.start()] + raw[match.end() :]).strip()
    return remainder or raw.strip()


def parse_uniformity_verdict(text: str) -> str | None:
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


def extract_reason(text: str, verdict: str | None) -> str:
    """Drop the trailing Yes/No line from the judge reply."""
    region = visible_judge_text(text)
    if not region:
        return ""
    lines = region.splitlines()
    if verdict and lines:
        last = lines[-1].strip().strip("\"'")
        if last.lower() == verdict.lower():
            return "\n".join(lines[:-1]).strip()
    return region.strip()


def source_run_dir(results_root: Path, source_run_id: str) -> Path:
    return Path(results_root).expanduser().resolve() / SOURCE_EXPERIMENT / source_run_id


def variety_run_dir(results_root: Path, source_run_id: str) -> Path:
    return (
        Path(results_root).expanduser().resolve()
        / ANSWER_VARIETY_EXPERIMENT
        / source_run_id
    )


def evaluated_path_for(run_dir: Path) -> Path:
    return Path(run_dir) / "predictions.evaluated.jsonl"


def configure_view_difficulty(results_dir: Path, audio_dir: Path | None = None) -> None:
    vd.CONFIG["results_dir"] = Path(results_dir).expanduser().resolve()
    if audio_dir is not None:
        vd.CONFIG["audio_dir"] = Path(audio_dir).expanduser().resolve()
    elif "audio_dir" not in vd.CONFIG:
        vd.CONFIG["audio_dir"] = vd.DEFAULT_AUDIO_DIR


def load_source_bundle(results_dir: Path, source_run_id: str) -> dict[str, Any]:
    configure_view_difficulty(results_dir)
    bundle = vd.load_run_bundle(source_run_id)
    if bundle.get("mode") != "freeform":
        raise SystemExit(
            f"Source run {source_run_id} is mode={bundle.get('mode')!r}, "
            "expected freeform. Aborting."
        )
    return bundle


def _shot_index(shot: dict) -> int:
    raw = shot.get("shot_index")
    if raw is None:
        return 10**9
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 10**9


def compact_shot_judges(shot: dict, record: dict | None = None) -> dict[str, Any]:
    """Primary + per-judge pass/fail for one generation."""
    judges_raw = shot.get("judges") if isinstance(shot.get("judges"), dict) else {}
    primary = str(
        (record or {}).get("primary_judge")
        or shot.get("primary_judge")
        or ""
    ).strip()
    if not primary:
        for label, entry in judges_raw.items():
            if isinstance(entry, dict) and entry.get("primary"):
                primary = str(label)
                break
        if not primary and judges_raw:
            primary = str(next(iter(judges_raw)))
    compact: dict[str, dict[str, Any]] = {}
    for label, entry in judges_raw.items():
        if not isinstance(entry, dict):
            continue
        correct = entry.get("correct")
        verdict = str(entry.get("verdict") or "").strip().lower()
        if verdict not in {"pass", "fail"}:
            if correct is True:
                verdict = "pass"
            elif correct is False:
                verdict = "fail"
            else:
                verdict = ""
        compact[str(label)] = {
            "correct": correct,
            "verdict": verdict or None,
            "primary": str(label) == primary,
        }
    correct = shot.get("correct")
    if correct is None and primary and primary in compact:
        correct = compact[primary].get("correct")
    return {
        "correct": correct,
        "primary_judge": primary or None,
        "judges": compact,
    }


def collect_shot_answers(record: dict | None, model_label: str) -> list[dict[str, Any]]:
    """Extract ``answer_prediction`` from every shot; skip empty strings."""
    if not record:
        return []
    shots = list(record.get("shots") or [])
    shots.sort(key=_shot_index)
    out: list[dict[str, Any]] = []
    if shots:
        for shot in shots:
            text = str(shot.get("answer_prediction") or "").strip()
            if not text:
                continue
            idx = shot.get("shot_index")
            try:
                shot_index = int(idx) if idx is not None else len(out)
            except (TypeError, ValueError):
                shot_index = len(out)
            out.append(
                {
                    "model": model_label,
                    "shot_index": shot_index,
                    "answer_prediction": text,
                    **compact_shot_judges(shot, record),
                }
            )
        return out
    text = str(record.get("answer_prediction") or "").strip()
    if text:
        out.append(
            {
                "model": model_label,
                "shot_index": 0,
                "answer_prediction": text,
                **compact_shot_judges(record, record),
            }
        )
    return out


def question_ids_from_bundle(bundle: dict[str, Any], source_dir: Path) -> list[str]:
    ids_path = source_dir / "question_ids.json"
    if ids_path.is_file():
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(x) for x in payload]
        if isinstance(payload, dict) and isinstance(payload.get("ids"), list):
            return [str(x) for x in payload["ids"]]
    difficulty = bundle.get("difficulty") or []
    if difficulty:
        return [str(row["id"]) for row in difficulty if row.get("id")]
    seen: list[str] = []
    found: set[str] = set()
    for per_model in (bundle.get("predictions") or {}).values():
        for qid in per_model:
            if qid not in found:
                found.add(qid)
                seen.append(qid)
    return seen


_META_KEYS = (
    "id",
    "question",
    "choices",
    "answer",
    "audio_path",
    "modality",
    "category",
    "sub-category",
    "language",
    "source",
)


def sample_fields(bundle: dict[str, Any], qid: str) -> dict[str, Any]:
    out = dict((bundle.get("by_id") or {}).get(qid) or {})
    for label in bundle.get("model_labels") or []:
        pred = (bundle.get("predictions") or {}).get(label, {}).get(qid)
        if not pred:
            continue
        for key in _META_KEYS:
            if out.get(key) in (None, "", []):
                value = pred.get(key)
                if value not in (None, "", []):
                    out[key] = value
        if out.get("question"):
            break
    return out


def build_uniformity_items(
    results_dir: Path,
    source_run_id: str,
    *,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = load_source_bundle(results_dir, source_run_id)
    source_dir = source_run_dir(results_dir, source_run_id)
    ids = question_ids_from_bundle(bundle, source_dir)
    if limit and limit > 0:
        ids = ids[: int(limit)]
    model_labels = list(bundle.get("model_labels") or [])
    items: list[dict[str, Any]] = []
    for qid in ids:
        answers: list[dict[str, Any]] = []
        for label in model_labels:
            pred = (bundle.get("predictions") or {}).get(label, {}).get(qid)
            answers.extend(collect_shot_answers(pred, label))
        meta = sample_fields(bundle, qid)
        items.append(
            {
                "id": qid,
                "question": meta.get("question") or "",
                "choices": meta.get("choices") or [],
                "answer": meta.get("answer") or "",
                "audio_path": meta.get("audio_path") or "",
                "modality": meta.get("modality") or "unknown",
                "category": meta.get("category") or "unknown",
                "sub-category": meta.get("sub-category"),
                "language": meta.get("language"),
                "source": meta.get("source"),
                "answers": answers,
                "n_answers": len(answers),
                "model_labels": model_labels,
                "yes_no": is_yes_no_question(
                    meta.get("choices") or [],
                    meta.get("question") or "",
                ),
                "source_judges": bundle.get("judges") or [],
                "source_primary_judge": bundle.get("primary_judge"),
            }
        )
    return items, bundle


def format_answer_list(answers: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, entry in enumerate(answers, start=1):
        text = str(entry.get("answer_prediction") or "").strip() or "(empty)"
        lines.append(f"{i}. {text}")
    return "\n".join(lines) if lines else "(no extracted answers)"


def create_uniformity_user_prompt(answers: list[dict[str, Any]]) -> str:
    return (
        "Are these answer strings all identical?\n\n"
        f"{format_answer_list(answers)}"
    )


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
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY to call Gemini.")
    return genai.Client(api_key=api_key)


class GeminiUniformityScorer:
    """Text-only Gemini judge: numbered answer strings, Yes/No identity."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_GEMINI_MODEL,
        api_max_retries: int = 20,
        api_retry_interval: float = 1.0,
        qps: float = 4.0,
        timeout: float = 180.0,
        max_output_tokens: int = 2048,
        thinking_level: str = "medium",
        system_instruction: str = UNIFORMITY_SYSTEM_PROMPT,
    ):
        from google.genai import types

        self.types = types
        self.client = _gemini_client()
        self.model_name = model_name
        self.api_max_retries = api_max_retries
        self.api_retry_interval = api_retry_interval
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level
        self.system_instruction = system_instruction
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

    async def _complete(self, user_prompt: str) -> str:
        thinking = None
        if self.thinking_level:
            thinking = self.types.ThinkingConfig(thinking_level=self.thinking_level)
        config = self.types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            max_output_tokens=int(self.max_output_tokens),
            thinking_config=thinking,
        )
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=[user_prompt],
            config=config,
        )
        return gemini_response_text(response)

    async def call(self, log_id: str, user_prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self.api_max_retries + 1):
            try:
                await self._throttle()
                text = await asyncio.wait_for(
                    self._complete(user_prompt),
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


def is_fully_graded(item: dict) -> bool:
    if not item.get("id"):
        return False
    if item.get("verdict") not in {"Yes", "No"}:
        return False
    return bool(str(item.get("raw_response") or "").strip())


def load_completed_ids(evaluated_path: Path) -> set[str]:
    if not evaluated_path.is_file() or evaluated_path.stat().st_size == 0:
        return set()
    return {
        str(item["id"])
        for item in load_jsonl(evaluated_path)
        if is_fully_graded(item)
    }


def prune_incomplete_evaluations(evaluated_path: Path) -> int:
    if not evaluated_path.is_file() or evaluated_path.stat().st_size == 0:
        return 0
    records = load_jsonl(evaluated_path)
    keep = [item for item in records if is_fully_graded(item)]
    removed = len(records) - len(keep)
    if removed <= 0:
        return 0
    by_id: dict[str, dict] = {}
    for item in keep:
        by_id[str(item["id"])] = item
    write_jsonl(evaluated_path, list(by_id.values()), mode="w")
    return removed


def evaluated_record_from_verdict(
    item: dict,
    *,
    raw_response: str,
    verdict: str | None,
) -> dict[str, Any]:
    yes = verdict == "Yes"
    return {
        **item,
        "verdict": verdict,
        "uniform": yes if verdict is not None else None,
        "reason": extract_reason(raw_response, verdict),
        "raw_response": raw_response,
        "scoring": "answer_uniformity",
    }


def write_variety_manifest(
    run_dir: Path,
    *,
    source_run_id: str,
    question_ids: list[str],
    judge_model_id: str,
    limit: int,
    model_labels: list[str],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    path = run_dir / "manifest.json"
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    payload = {
        **existing,
        "experiment": ANSWER_VARIETY_EXPERIMENT,
        "eval_kind": "answer_uniformity",
        "source_run_id": source_run_id,
        "source_experiment": SOURCE_EXPERIMENT,
        "models": model_labels,
        "judge_model_id": judge_model_id,
        "limit": limit,
        "n_questions": len(question_ids),
        "scoring": "answer_uniformity",
        "shot": "all",
        "updated_at": now,
    }
    if "created_at" not in payload:
        payload["created_at"] = now
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


def record_is_yes_no(item: dict) -> bool:
    return is_yes_no_question(
        item.get("choices") or [],
        item.get("question") or "",
    )


def load_id_csv(path: Path) -> list[str]:
    """Read an ``id``-column CSV, preserving order and dropping blanks."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        has_header = "id" in sample.splitlines()[0].lower() if sample.strip() else False
        if has_header:
            reader = csv.DictReader(handle)
            raw_ids = [str(row.get("id") or "").strip() for row in reader]
        else:
            raw_ids = [line.strip() for line in handle if line.strip()]
    for qid in raw_ids:
        if qid and qid.lower() != "id" and qid not in seen:
            seen.add(qid)
            ids.append(qid)
    return ids


def write_id_csv(path: Path, ids: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id"])
        for qid in ids:
            writer.writerow([qid])
    return path


def load_gemini_binary_ids(path: Path | None = None) -> set[str]:
    return set(load_id_csv(path or GEMINI_BINARY_IDS_PATH))


def item_is_closed_form(
    item: dict, gemini_binary_ids: set[str] | None = None
) -> bool:
    """Heuristic yes/no, or Gemini-implied binary from the extra ID list."""
    if record_is_yes_no(item):
        return True
    ids = (
        gemini_binary_ids
        if gemini_binary_ids is not None
        else load_gemini_binary_ids()
    )
    return str(item.get("id") or "") in ids


def _rate(n_yes: int, n: int) -> float | None:
    return (n_yes / n) if n else None


def write_variety_scores(run_dir: Path, evaluated_path: Path, *, judge_model_id: str) -> dict:
    records = load_jsonl(evaluated_path) if evaluated_path.is_file() else []
    n = len(records)
    n_uniform = sum(1 for item in records if item.get("verdict") == "Yes")
    n_varied = sum(1 for item in records if item.get("verdict") == "No")
    n_unparsed = sum(
        1 for item in records if item.get("verdict") not in {"Yes", "No"}
    )
    gemini_binary_ids = load_gemini_binary_ids()
    yn = [item for item in records if item_is_closed_form(item, gemini_binary_ids)]
    open_items = [
        item for item in records if not item_is_closed_form(item, gemini_binary_ids)
    ]
    n_yn_uniform = sum(1 for item in yn if item.get("verdict") == "Yes")
    n_open_uniform = sum(1 for item in open_items if item.get("verdict") == "Yes")
    summary = {
        "judge_model_id": judge_model_id,
        "scoring": "answer_uniformity",
        "n_questions": n,
        "n_uniform": n_uniform,
        "n_varied": n_varied,
        "n_unparsed": n_unparsed,
        "uniform_rate": _rate(n_uniform, n),
        "n_yes_no": len(yn),
        "n_open": len(open_items),
        "n_yes_no_uniform": n_yn_uniform,
        "n_open_uniform": n_open_uniform,
        "uniform_rate_yes_no": _rate(n_yn_uniform, len(yn)),
        "uniform_rate_open": _rate(n_open_uniform, len(open_items)),
        "evaluated_path": str(evaluated_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "scores.json", summary)
    return summary


def append_evaluated(evaluated_path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    write_jsonl(evaluated_path, records, mode="a")


def discover_variety_runs(results_dir: Path) -> list[dict[str, Any]]:
    root = Path(results_dir).expanduser().resolve() / ANSWER_VARIETY_EXPERIMENT
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if not evaluated_path_for(child).is_file() and not (child / "manifest.json").is_file():
            continue
        manifest: dict[str, Any] = {}
        scores: dict[str, Any] = {}
        try:
            if (child / "manifest.json").is_file():
                manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
            if (child / "scores.json").is_file():
                scores = json.loads((child / "scores.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
        n = scores.get("n_questions") or manifest.get("n_questions")
        runs.append(
            {
                "id": child.name,
                "path": str(child),
                "source_run_id": manifest.get("source_run_id") or child.name,
                "judge_model_id": scores.get("judge_model_id")
                or manifest.get("judge_model_id"),
                "n_questions": n,
                "n_uniform": scores.get("n_uniform"),
                "n_varied": scores.get("n_varied"),
                "uniform_rate": scores.get("uniform_rate"),
                "n_yes_no": scores.get("n_yes_no"),
                "uniform_rate_open": scores.get("uniform_rate_open"),
                "models": manifest.get("models") or [],
            }
        )
    return runs


def enrich_answers_from_source(item: dict, source_bundle: dict[str, Any]) -> dict:
    """Attach source-run judge verdicts onto stored answer rows if missing."""
    qid = str(item.get("id") or "")
    answers = list(item.get("answers") or [])
    if not qid or not answers:
        return item
    preds = source_bundle.get("predictions") or {}
    for ans in answers:
        if ans.get("judges"):
            continue
        label = str(ans.get("model") or "")
        record = (preds.get(label) or {}).get(qid)
        if not record:
            continue
        shot_index = ans.get("shot_index")
        try:
            want = int(shot_index) if shot_index is not None else 0
        except (TypeError, ValueError):
            want = 0
        matched = None
        for shot in record.get("shots") or []:
            if _shot_index(shot) == want:
                matched = shot
                break
        if matched is None:
            matched = record
        ans.update(compact_shot_judges(matched, record))
    item["answers"] = answers
    if not item.get("source_judges"):
        item["source_judges"] = source_bundle.get("judges") or []
        item["source_primary_judge"] = source_bundle.get("primary_judge")
    return item


def _try_load_source_bundle(results_dir: Path, source_run_id: str) -> dict[str, Any] | None:
    if not source_run_id:
        return None
    try:
        configure_view_difficulty(results_dir)
        return vd.load_run_bundle(source_run_id)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load source run %s: %s", source_run_id, exc)
        return None


def load_variety_bundle(results_dir: Path, run_id: str) -> dict[str, Any]:
    run_dir = variety_run_dir(results_dir, run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    manifest: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    if (run_dir / "manifest.json").is_file():
        try:
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    if (run_dir / "scores.json").is_file():
        try:
            scores = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            scores = {}
    records = load_jsonl(evaluated_path_for(run_dir))
    source_run_id = str(manifest.get("source_run_id") or run_id)
    source_bundle = _try_load_source_bundle(results_dir, source_run_id)
    gemini_binary_ids = load_gemini_binary_ids()
    by_id: dict[str, dict] = {}
    for item in records:
        qid = item.get("id")
        if not qid:
            continue
        heuristic = record_is_yes_no(item)
        gemini_binary = str(qid) in gemini_binary_ids
        item["yes_no_heuristic"] = heuristic
        item["binary_implied"] = gemini_binary
        item["yes_no"] = heuristic or gemini_binary
        if source_bundle:
            enrich_answers_from_source(item, source_bundle)
        by_id[str(qid)] = item

    def _sort_key(item: dict) -> tuple:
        verdict = item.get("verdict")
        varied_rank = 0 if verdict == "No" else (1 if verdict not in {"Yes", "No"} else 2)
        return (varied_rank, str(item.get("id") or ""))

    ordered = sorted(by_id.values(), key=_sort_key)
    scored = list(by_id.values())
    yn = [item for item in scored if item.get("yes_no")]
    open_items = [item for item in scored if not item.get("yes_no")]
    n_yn_uniform = sum(1 for item in yn if item.get("verdict") == "Yes")
    n_open_uniform = sum(1 for item in open_items if item.get("verdict") == "Yes")
    scores = {
        **scores,
        "n_yes_no": len(yn),
        "n_open": len(open_items),
        "n_yes_no_uniform": n_yn_uniform,
        "n_open_uniform": n_open_uniform,
        "uniform_rate_yes_no": _rate(n_yn_uniform, len(yn)),
        "uniform_rate_open": _rate(n_open_uniform, len(open_items)),
    }
    questions = [
        {
            "id": item.get("id"),
            "question": item.get("question") or "",
            "verdict": item.get("verdict"),
            "uniform": item.get("uniform"),
            "n_answers": item.get("n_answers") or len(item.get("answers") or []),
            "yes_no": bool(item.get("yes_no")),
            "yes_no_heuristic": bool(item.get("yes_no_heuristic")),
            "binary_implied": bool(item.get("binary_implied")),
            "modality": item.get("modality"),
            "category": item.get("category"),
        }
        for item in ordered
    ]
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": manifest,
        "scores": scores,
        "by_id": by_id,
        "questions": questions,
        "model_labels": manifest.get("models") or [],
        "judge_model_id": scores.get("judge_model_id") or manifest.get("judge_model_id"),
        "source_run_id": source_run_id,
        "source_judges": (source_bundle or {}).get("judges") or [],
        "source_primary_judge": (source_bundle or {}).get("primary_judge"),
    }


async def grade_items_with_gemini(
    items: list[dict[str, Any]],
    evaluated_path: Path,
    *,
    model_id: str = DEFAULT_GEMINI_MODEL,
    max_workers: int = 8,
    qps: float = 4.0,
    timeout: float = 180.0,
    retries: int = 20,
    retry_interval: float = 1.0,
    max_output_tokens: int = 2048,
    thinking_level: str = "medium",
) -> dict[str, Any]:
    evaluated_path.parent.mkdir(parents=True, exist_ok=True)
    removed = prune_incomplete_evaluations(evaluated_path)
    if removed:
        print(f"[gemini] pruned {removed} incomplete rows")
    completed = load_completed_ids(evaluated_path)
    pending = [item for item in items if item["id"] not in completed]
    print(
        f"[gemini] judge={model_id} pending={len(pending)} "
        f"completed={len(completed)} -> {evaluated_path}"
    )
    if not pending:
        return {"status": "already_done", "n_ok": 0, "n_fail": 0, "n_pending": 0}

    scorer = GeminiUniformityScorer(
        model_name=model_id,
        api_max_retries=retries,
        api_retry_interval=retry_interval,
        qps=qps,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
        thinking_level=thinking_level,
    )
    semaphore = asyncio.Semaphore(max_workers)
    write_lock = asyncio.Lock()
    n_ok = 0
    n_fail = 0
    n_unparsed = 0

    async def _one(item: dict) -> None:
        nonlocal n_ok, n_fail, n_unparsed
        answers = item.get("answers") or []
        if len(answers) <= 1:
            verdict = "Yes"
            raw = (
                "Fewer than two extracted answer strings; treated as identical.\n"
                "Yes"
            )
            async with write_lock:
                append_evaluated(
                    evaluated_path,
                    [evaluated_record_from_verdict(item, raw_response=raw, verdict=verdict)],
                )
                n_ok += 1
            return
        user_prompt = create_uniformity_user_prompt(answers)
        try:
            async with semaphore:
                raw = await scorer.call(item["id"], user_prompt)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"[gemini] failed {item['id']}: {exc}")
            return
        verdict = parse_uniformity_verdict(raw)
        if verdict is None:
            n_unparsed += 1
            print(f"[gemini] unparsed verdict for {item['id']}: {raw[:180]!r}")
        async with write_lock:
            append_evaluated(
                evaluated_path,
                [evaluated_record_from_verdict(item, raw_response=raw, verdict=verdict)],
            )
            n_ok += 1
            if n_ok % 10 == 0 or n_ok == len(pending):
                print(f"[gemini] scored {n_ok}/{len(pending)}")

    await asyncio.gather(*[_one(item) for item in pending])
    return {
        "status": "ok",
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_unparsed": n_unparsed,
        "n_pending": len(pending),
    }
