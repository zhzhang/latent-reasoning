"""Shared MMAR data helpers."""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from tkinter import N
from typing import Callable

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
AF3_THINK_SUFFIX = "Please think and reason about the input audio before you respond."
AF_NEXT_THINK_SUFFIX = (
    "Reason step by step with timestamps, then give the final answer."
)
MUSIC_FLAMINGO_THINK_SUFFIX = (
    "Output the thinking process in <think> </think> and final answer in "
    "<answer> </answer>"
)
ANSWER_MARKERS = (
    r"therefore[, ]+the answer is[:\s]*",
    r"the answer is[:\s]*",
    r"final answer[:\s]*",
)
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
THINK_CLOSE = "</think>"
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Prefill on the assistant/model turn so these checkpoints enter native CoT.
ASSISTANT_THINK_OPEN = "<think>\n"
PREFIX_ASSISTANT_THINK_LABELS = frozenset(
    {
        "af-next-think",
        "music-flamingo",
        "step-audio-2-mini-think",
        "mimo-audio-7b",
    }
)
_EMPTY_THINK_TAIL_RE = re.compile(r"<think>\s*</think>\s*$", re.IGNORECASE)


def ensure_assistant_think_open(label: str, prompt: str) -> str:
    """Make the assistant turn start with an open ``<think>`` for native CoT.

    Strips a trailing empty ``<think></think>`` (the Qwen-style *disable*
    pattern) then appends ``ASSISTANT_THINK_OPEN`` when it is not already
    the last content.
    """
    if label not in PREFIX_ASSISTANT_THINK_LABELS:
        return prompt
    text = _EMPTY_THINK_TAIL_RE.sub("", prompt)
    stripped = text.rstrip()
    if stripped.lower().endswith("<think>"):
        return text if text.endswith("\n") else f"{text}\n"
    return f"{text}{ASSISTANT_THINK_OPEN}"


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


def load_question_ids_csv(path: Path | str) -> list[str]:
    """Load an ``id`` column (or a single-column file) of question ids."""
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"question-ids csv not found: {csv_path}")
    ids: list[str] = []
    with open(csv_path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            text = line.strip().strip("\ufeff")
            if not text:
                continue
            first = text.split(",", 1)[0].strip().strip('"')
            if not first:
                continue
            if index == 0 and first.lower() in {"id", "ids", "question_id", "qid"}:
                continue
            ids.append(first)
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(ids))


# Fixed shuffle for ``n_questions`` grading. Not a kwarg at call sites: larger
# N continues down this same permutation (N=10 is a prefix of N=50).
GRADE_SAMPLE_SEED = 0


def shuffled_question_ids(
    ids: list[str] | tuple[str, ...],
    *,
    seed: int = GRADE_SAMPLE_SEED,
) -> list[str]:
    """Unique ids, sorted then shuffled.

    The permutation is keyed only by:
    - the unique stripped non-empty id strings (input order is ignored)
    - ``seed`` (default ``GRADE_SAMPLE_SEED`` = 0)

    Sorting first makes the permutation independent of file order, so every
    model that shares the same id set samples the same questions.
    """
    unique: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        qid = str(raw or "").strip()
        if not qid or qid in seen:
            continue
        seen.add(qid)
        unique.append(qid)
    unique.sort()
    rng = random.Random(seed)
    rng.shuffle(unique)
    return unique


def select_grade_question_ids(
    ids: list[str] | tuple[str, ...],
    n_questions: int | None,
    *,
    seed: int = GRADE_SAMPLE_SEED,
) -> list[str] | None:
    """First ``n_questions`` ids in the fixed shuffled order, or ``None`` for all.

    ``n_questions`` None or < 0 grades every question. Larger N is a prefix of
    the same shuffle.
    """
    if n_questions is None or int(n_questions) < 0:
        return None
    return shuffled_question_ids(ids, seed=seed)[: int(n_questions)]


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


def build_mmar_freeform_prompt(item, think_suffix: str | None = None):
    """Prompt the model with the question only (no multiple-choice options)."""
    question = str(item["question"]).strip()
    prompt = (
        "Listen to the audio and answer the question. "
        "Reason step by step before answering, then give a concise, final answer in a single line in exactly this format:\n"
        "Answer: <your answer>\n"
        f"This question is: {question}\n"
    )
    if think_suffix:
        prompt += f"\n{think_suffix}"
    return prompt


def _last_think_close_span(text: str) -> tuple[int, int] | None:
    """Byte span of the last ``</think>`` (case-insensitive), if any."""
    idx = text.lower().rfind(THINK_CLOSE)
    if idx < 0:
        return None
    return idx, idx + len(THINK_CLOSE)


def after_last_think_close(text: str) -> str:
    """Drop everything through the last ``</think>``; else return ``text``."""
    span = _last_think_close_span(text)
    if span is None:
        return text
    return text[span[1] :].strip()


def split_last_think_close(text: str) -> tuple[str, str] | None:
    """Split on the last ``</think>``. ``None`` when the closing tag is absent."""
    span = _last_think_close_span(text)
    if span is None:
        return None
    start, end = span
    return text[:start].strip(), text[end:].strip()


def _strip_think_tags(text: str) -> str:
    """Keep think-block inner text; drop leftover ``<think>`` / ``</think>`` tags."""
    cleaned = THINK_BLOCK_RE.sub(lambda match: match.group(1), text)
    return THINK_TAG_RE.sub("", cleaned).strip()


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
    text = after_last_think_close((raw_text or "").strip())
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
    """Parse outputs that may wrap CoT in ``<think>...</think>`` tags.

    Answer extraction never includes text before the last closing ``</think>``.
    Existing choice / marker / last-line rules then run on the remainder.
    """
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    split = split_last_think_close(text)
    if split is not None:
        prefix, remainder = split
        thinking_prediction = _strip_think_tags(prefix)
        if remainder:
            _, answer_prediction = parse_choice_output(remainder, choices)
            return thinking_prediction, answer_prediction
        return thinking_prediction, ""

    return parse_choice_output(text, choices)


def parse_answer_tagged_output(raw_text, choices=None):
    """Extract ``(thinking, answer)`` from the last ``<answer>`` block.

    Returns ``None`` when tags are missing or empty so callers can try other
    extractors. Thinking is the last ``<think>`` prefix when present, else the
    text before the answer tags. ``choices`` is used only to resolve the inner
    text to a listed option.
    """
    text = (raw_text or "").strip()
    if not text:
        return None
    matches = list(ANSWER_BLOCK_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    answer_prediction = match.group(1).strip()
    if not answer_prediction:
        return None
    prefix = text[: match.start()].strip()
    split = split_last_think_close(prefix)
    if split is not None:
        thinking_prediction = _strip_think_tags(split[0])
    else:
        thinking_prediction = _strip_think_tags(prefix)
    if choices:
        matched = _match_choice_in_text(answer_prediction, choices)
        if matched:
            answer_prediction = matched
    return thinking_prediction, answer_prediction


def parse_music_flamingo_output(raw_text, choices=None, *, fallback=None):
    """Music Flamingo: ``<answer>`` tags first, then ``fallback`` extractors.

    Creators instruct the model to wrap the final answer in ``<answer>`` tags.
    If that extractor fails, fall back to think-tagged (MC) or freeform rules.
    """
    tagged = parse_answer_tagged_output(raw_text, choices)
    if tagged is not None:
        return tagged
    parser = fallback or parse_think_tagged_output
    return parser(raw_text, choices)


def parse_freeform_output(raw_text, choices=None):
    """Split free-form model text into (thinking, answer) without choice matching.

    ``choices`` is accepted for API compatibility with choice parsers but ignored.
    Answer extraction never includes text before the last closing ``</think>``.
    """
    del choices  # unused — free-form answers are not constrained to options
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    split = split_last_think_close(text)
    if split is not None:
        prefix, remainder = split
        thinking_prediction = _strip_think_tags(prefix)
        if remainder:
            return thinking_prediction, remainder
        return thinking_prediction, ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for marker in ANSWER_MARKERS:
        match = re.search(marker + r"(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        answer_prediction = match.group(1).strip().splitlines()[0].strip()
        answer_index = text.lower().rfind(answer_prediction.lower())
        thinking = text[:answer_index].strip() if answer_index > 0 else ""
        return thinking, answer_prediction

    if len(lines) > 1:
        return "\n".join(lines[:-1]).strip(), lines[-1]
    return "", text


def string_match(answer, prediction, choices):
    """MMAR answer match: GT tokens present, no exclusive tokens from wrong choices."""

    def tokenize(text):
        return set(re.findall(r"\b\w+\b", str(text).lower()))

    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    if not prediction_tokens:
        return False

    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    return answer_tokens.issubset(prediction_tokens) and prediction_tokens.isdisjoint(
        incorrect_tokens
    )


def score_answer_prediction(item: dict, answer_prediction: str) -> bool:
    return string_match(
        item.get("answer", ""),
        after_last_think_close(answer_prediction or ""),
        item.get("choices") or [],
    )


def aggregate_n_shot_record(
    item: dict,
    shot_outputs: list[dict],
    *,
    score_fn: Callable[[dict, str], bool | None] | None = None,
    pending_grade: bool = False,
) -> dict:
    """Build one prediction record from ``n`` independent generation outputs.

    Args:
        score_fn: Optional ``(item, answer_prediction) -> bool | None``. Defaults
            to string-match scoring. Pass a no-op / ``None``-returning fn together
            with ``pending_grade=True`` when an external judge will score later.
        pending_grade: When True, leave correctness fields for a later grading pass
            (``correct`` / rates stay ``None`` until graded).
    """
    if score_fn is None and not pending_grade:
        score_fn = score_answer_prediction

    shots = []
    for shot_index, output in enumerate(shot_outputs):
        answer_prediction = output.get("answer_prediction", "")
        if pending_grade:
            correct = None
        else:
            correct = score_fn(item, answer_prediction)
        shot = {
            "shot_index": shot_index,
            "model_output": output.get("model_output"),
            "raw_tokens": output.get("raw_tokens"),
            "thinking_prediction": output.get("thinking_prediction"),
            "answer_prediction": answer_prediction,
            "correct": correct,
        }
        if pending_grade:
            shot["pending_grade"] = True
        shots.append(shot)

    n_shots = len(shots)
    if pending_grade:
        n_shot_correct = None
        shot_success_rate = None
        any_correct = None
        primary = shots[0] if shots else {}
    else:
        n_shot_correct = sum(1 for shot in shots if shot["correct"])
        shot_success_rate = (n_shot_correct / n_shots) if n_shots else 0.0
        any_correct = n_shot_correct > 0
        primary = next((shot for shot in shots if shot["correct"]), shots[0])
    record = {
        **item,
        "model_output": primary.get("model_output"),
        "raw_tokens": primary.get("raw_tokens"),
        "thinking_prediction": primary.get("thinking_prediction"),
        "answer_prediction": primary.get("answer_prediction"),
        "n_shots": n_shots,
        "shots": shots,
        "correct": any_correct,
        "n_shot_correct": n_shot_correct,
        "shot_success_rate": shot_success_rate,
    }
    if pending_grade:
        record["pending_grade"] = True
    return record


def recompute_n_shot_scores(record: dict) -> dict:
    """Recompute aggregate correctness fields from per-shot ``correct`` flags."""
    shots = list(record.get("shots") or [])
    n_shots = len(shots)
    n_shot_correct = sum(1 for shot in shots if shot.get("correct"))
    primary = next((shot for shot in shots if shot.get("correct")), shots[0] if shots else {})
    record["n_shots"] = n_shots
    record["n_shot_correct"] = n_shot_correct
    record["shot_success_rate"] = (n_shot_correct / n_shots) if n_shots else 0.0
    record["correct"] = n_shot_correct > 0
    if primary:
        record["model_output"] = primary.get("model_output")
        record["raw_tokens"] = primary.get("raw_tokens")
        record["thinking_prediction"] = primary.get("thinking_prediction")
        record["answer_prediction"] = primary.get("answer_prediction")
    record.pop("pending_grade", None)
    return record


STRING_MATCH_JUDGE_LABEL = "string-match"


def judge_label(model_id: str | None) -> str:
    """Slug for a judge HF id: last path segment, lowercased."""
    text = str(model_id or "").strip()
    if not text:
        return ""
    return text.rstrip("/").split("/")[-1].lower()


def _shot_has_legacy_judge(shot: dict) -> bool:
    return bool(shot.get("grader") or shot.get("grader_output")) or (
        shot.get("correct") is not None and not shot.get("pending_grade")
    )


def ensure_judge_schema(
    record: dict,
    *,
    fallback_label: str | None = None,
    fallback_model_id: str | None = None,
) -> dict:
    """Idempotent in-place migration to the multi-judge shot schema.

    Freeform: lift flat ``grader`` / ``grader_output`` / ``correct`` into
    ``shot["judges"][label]``. MC: synthesize a ``string-match`` judge from
    each shot's ``correct`` flag when no judges map exists yet.
    """
    shots = list(record.get("shots") or [])
    scoring = str(record.get("scoring") or "").lower()
    is_freeform = (
        "freeform" in scoring
        or "qwen_freeform" in scoring
        or bool(record.get("grader"))
        or any(
            shot.get("grader") or shot.get("grader_output") or shot.get("pending_grade")
            for shot in shots
        )
    )

    for shot in shots:
        judges = shot.get("judges")
        if not isinstance(judges, dict):
            judges = {}
            shot["judges"] = judges

        if judges:
            # Normalize: ensure generation key exists for LLM judge entries.
            for label, entry in list(judges.items()):
                if not isinstance(entry, dict):
                    continue
                if "generation" not in entry:
                    entry["generation"] = entry.get("output") or ""
                if "verdict" not in entry and entry.get("correct") is not None:
                    entry["verdict"] = "pass" if entry.get("correct") else "fail"
            continue

        if is_freeform and _shot_has_legacy_judge(shot):
            model_id = (
                shot.get("grader")
                or record.get("grader")
                or fallback_model_id
                or ""
            )
            label = (
                judge_label(model_id)
                or fallback_label
                or "qwen2.5-3b-instruct"
            )
            if shot.get("correct") is not None or shot.get("grader_output") is not None:
                legacy_out = shot.get("grader_output")
                judges[label] = {
                    "correct": shot.get("correct"),
                    "output": legacy_out,
                    "generation": legacy_out if legacy_out is not None else "",
                    "model_id": model_id or None,
                }
        elif not is_freeform and shot.get("correct") is not None:
            judges[STRING_MATCH_JUDGE_LABEL] = {
                "correct": bool(shot.get("correct")),
                "output": "MATCH" if shot.get("correct") else "NO_MATCH",
                "generation": "",
                "model_id": None,
            }

    # Preserve ordered judge list when already present.
    if not record.get("judges"):
        ordered: list[str] = []
        for shot in shots:
            for label in (shot.get("judges") or {}):
                if label not in ordered:
                    ordered.append(label)
        if ordered:
            record["judges"] = ordered

    if not record.get("primary_judge") and record.get("judges"):
        record["primary_judge"] = record["judges"][0]

    return record


def recompute_multi_judge_scores(
    record: dict,
    primary_label: str | None = None,
) -> dict:
    """Fill ``per_judge`` aggregates and mirror the primary judge onto canonical fields."""
    ensure_judge_schema(record)
    shots = list(record.get("shots") or [])
    n_shots = len(shots)

    ordered: list[str] = []
    for label in record.get("judges") or []:
        if label and label not in ordered:
            ordered.append(str(label))
    for shot in shots:
        for label in (shot.get("judges") or {}):
            if label not in ordered:
                ordered.append(str(label))

    primary = (
        primary_label
        or record.get("primary_judge")
        or (ordered[0] if ordered else None)
    )
    if primary and primary not in ordered:
        ordered.insert(0, primary)

    per_judge: dict[str, dict] = {}
    for label in ordered:
        n_correct = 0
        n_scored = 0
        for shot in shots:
            entry = (shot.get("judges") or {}).get(label)
            if not entry or entry.get("correct") is None:
                continue
            n_scored += 1
            n_correct += int(bool(entry.get("correct")))
        if n_scored == 0:
            per_judge[label] = {
                "n_shots": 0,
                "n_shot_correct": None,
                "shot_success_rate": None,
                "correct": None,
            }
        else:
            per_judge[label] = {
                "n_shots": n_scored,
                "n_shot_correct": n_correct,
                "shot_success_rate": n_correct / n_scored,
                "correct": n_correct > 0,
            }

    record["judges"] = ordered
    record["primary_judge"] = primary
    record["per_judge"] = per_judge
    record["n_shots"] = n_shots

    if primary and primary in per_judge and per_judge[primary]["n_shot_correct"] is not None:
        stats = per_judge[primary]
        record["n_shot_correct"] = stats["n_shot_correct"]
        record["shot_success_rate"] = stats["shot_success_rate"]
        record["correct"] = stats["correct"]
        # Mirror primary onto legacy flat shot fields (LLM judges only).
        primary_model_id = None
        for shot in shots:
            entry = (shot.get("judges") or {}).get(primary) or {}
            if "correct" in entry:
                shot["correct"] = entry.get("correct")
            if primary == STRING_MATCH_JUDGE_LABEL:
                # Keep MC string-match clean — no legacy grader_* stamps.
                shot.pop("grader", None)
                shot.pop("grader_output", None)
            else:
                # Prefer short 0/1 (or legacy Pass/Fail) label; fall back to full generation.
                short = entry.get("output")
                if short is None and entry.get("generation") is not None:
                    short = entry.get("generation")
                if short is not None:
                    shot["grader_output"] = short
                if entry.get("model_id"):
                    shot["grader"] = entry.get("model_id")
                    primary_model_id = entry.get("model_id")
                else:
                    shot["grader"] = primary
                    primary_model_id = primary_model_id or primary
            shot.pop("pending_grade", None)
        if primary == STRING_MATCH_JUDGE_LABEL:
            record.pop("grader", None)
        elif primary_model_id:
            record["grader"] = primary_model_id
        else:
            record["grader"] = primary
        record.pop("pending_grade", None)
    elif any(shot.get("pending_grade") for shot in shots) or record.get("pending_grade"):
        # Still awaiting at least the primary judge.
        record["n_shot_correct"] = None
        record["shot_success_rate"] = None
        record["correct"] = None
    else:
        # Fall back to legacy flat correct flags.
        recompute_n_shot_scores(record)

    # Prefer a correct primary shot for the record-level answer preview.
    primary_shot = None
    if primary:
        for shot in shots:
            entry = (shot.get("judges") or {}).get(primary) or {}
            if entry.get("correct"):
                primary_shot = shot
                break
    if primary_shot is None and shots:
        primary_shot = next((s for s in shots if s.get("correct")), shots[0])
    if primary_shot:
        record["model_output"] = primary_shot.get("model_output")
        record["raw_tokens"] = primary_shot.get("raw_tokens")
        record["thinking_prediction"] = primary_shot.get("thinking_prediction")
        record["answer_prediction"] = primary_shot.get("answer_prediction")

    return record


# ---------------------------------------------------------------------------
# Resume / run ids
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


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

