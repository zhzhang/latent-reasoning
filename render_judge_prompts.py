"""Render the judge prompts ``grade_shot_batch`` would send to each judge.

Uses the same wraps as ``grader.py``: ``JUDGE_FORMATS`` text for gold / API,
string audio templates for vLLM / Omni, chat-message dumps for
``vllm_chat`` / ``hf_chat``. Chat backends dump the ``LLM.chat`` messages;
vLLM applies the Jinja chat template later.

Usage::

    uv run python render_judge_prompts.py
    uv run python render_judge_prompts.py --judges qwen3-omni-instruct,gemma-4-e4b
    uv run python render_judge_prompts.py --prompt with_gt,free
    uv run python render_judge_prompts.py --question "What instrument is playing?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grader import (
    JUDGE_FORMATS,
    JUDGE_SPECS,
    _apply_grade_sampling_knobs,
    get_judge_format,
    parse_grade_prompt_list,
    parse_judge_list,
    render_judge_prompt,
)
from mmar_common import load_jsonl
from mmar_models import MODEL_SPECS, chat_kwargs_for

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_META = REPO_ROOT / "data" / "mmar" / "MMAR-meta.jsonl"
DEFAULT_AUDIO_PATH = "/cache/data/mmar/audio/example.wav"
DEFAULT_QUESTION = "What instrument is playing?"
DEFAULT_ANSWER = "piano"
DEFAULT_PREDICTION = "a grand piano"

_RULE = "=" * 80
_THIN = "-" * 80


def _dummy_job(question: str, answer: str, prediction: str) -> dict[str, Any]:
    return {
        "id": "example",
        "question": question,
        "answer": answer,
        "prediction": prediction,
        "choices": ["piano", "guitar", "violin", "drums"],
        "audio_path": DEFAULT_AUDIO_PATH,
    }


def _load_job(
    *,
    meta: Path | None,
    question_id: str | None,
    question: str | None,
    answer: str | None,
    prediction: str | None,
) -> dict[str, Any]:
    if meta is None and question_id is None and question is None:
        if DEFAULT_META.is_file():
            meta = DEFAULT_META
        else:
            return _dummy_job(
                DEFAULT_QUESTION,
                answer or DEFAULT_ANSWER,
                prediction or DEFAULT_PREDICTION,
            )

    if meta is not None:
        items = load_jsonl(meta)
        if not items:
            raise SystemExit(f"No MMAR rows in {meta}")
        if question_id:
            by_id = {str(item.get("id")): item for item in items}
            item = by_id.get(question_id)
            if item is None:
                raise SystemExit(f"Question id {question_id!r} not in {meta}")
        else:
            item = items[0]
        job = dict(item)
        job.setdefault("audio_path", job.get("audio_path") or DEFAULT_AUDIO_PATH)
        if question:
            job["question"] = question
        if answer:
            job["answer"] = answer
        else:
            job.setdefault("answer", DEFAULT_ANSWER)
        job["prediction"] = prediction or job.get("answer") or DEFAULT_PREDICTION
        return job

    if question_id:
        raise SystemExit("--question-id requires --meta")
    return _dummy_job(
        question or DEFAULT_QUESTION,
        answer or DEFAULT_ANSWER,
        prediction or DEFAULT_PREDICTION,
    )


def _judge_meta(label: str) -> dict[str, Any]:
    from mmar_api import API_SPECS, JUDGE_SAMPLING, resolve_api_judge_label

    if label in MODEL_SPECS:
        spec = MODEL_SPECS[label]
        sampling = _apply_grade_sampling_knobs(spec.get("sampling") or {})
        return {
            "model_id": spec["model_id"],
            "backend": spec.get("backend"),
            "sampling": sampling,
            "chat_kwargs": chat_kwargs_for(label),
        }
    if label in JUDGE_SPECS:
        spec = JUDGE_SPECS[label]
        return {
            "model_id": spec["model_id"],
            "backend": "vllm",
            "sampling": _apply_grade_sampling_knobs(spec.get("sampling") or {}),
            "chat_kwargs": {},
        }
    api = resolve_api_judge_label(label)
    if api:
        spec = API_SPECS[api]
        return {
            "model_id": spec["model_id"],
            "backend": spec.get("backend"),
            "sampling": dict(spec.get("sampling") or JUDGE_SAMPLING),
            "chat_kwargs": {},
        }
    return {
        "model_id": label,
        "backend": "vllm",
        "sampling": None,
        "chat_kwargs": {},
    }


def _print_combo(
    label: str,
    prompt_name: str,
    job: dict[str, Any],
) -> None:
    fmt = get_judge_format(prompt_name)
    meta = _judge_meta(label)
    print(_RULE)
    print(f"{label}  prompt={prompt_name}")
    print(
        f"  model_id={meta['model_id']}  backend={meta.get('backend')}  "
        f"audio_included={fmt.audio_included}  include_gold={fmt.include_gold}"
    )
    if meta.get("sampling"):
        print(f"  sampling={meta['sampling']}")
    if meta.get("chat_kwargs"):
        print(f"  chat_kwargs={meta['chat_kwargs']}")
    print(_THIN)
    try:
        text = render_judge_prompt(label, job, prompt=prompt_name)
    except ValueError as exc:
        print(f"(skipped: {exc})")
        print()
        return
    print(text.rstrip("\n"))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print the prompts grade_shot_batch would send to each judge "
            "with filled question / gold / prediction slots."
        )
    )
    parser.add_argument(
        "--judges",
        default="all",
        help="Comma-separated labels or 'all' (suite + dedicated + API).",
    )
    parser.add_argument(
        "--prompt",
        default="all",
        help="Comma-separated JUDGE_FORMATS keys, or 'all'.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Override the question text (default: first MMAR row, else a dummy).",
    )
    parser.add_argument(
        "--answer",
        default=None,
        help="Override the gold answer.",
    )
    parser.add_argument(
        "--prediction",
        default=None,
        help="Override the test-taker response (default: gold answer, else a dummy).",
    )
    parser.add_argument(
        "--question-id",
        default=None,
        help="MMAR question id to load from --meta.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help=f"MMAR-meta.jsonl (default: {DEFAULT_META} if it exists).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object per judge×format instead of the text dump.",
    )
    args = parser.parse_args()

    try:
        labels = parse_judge_list(args.judges)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    prompt_names = parse_grade_prompt_list(args.prompt, include_gold=None)
    job = _load_job(
        meta=args.meta.expanduser().resolve() if args.meta else None,
        question_id=args.question_id,
        question=args.question,
        answer=args.answer,
        prediction=args.prediction,
    )
    qid = job.get("id")
    print(
        f"grade_shot_batch wraps: n_judges={len(labels)}  "
        f"n_prompts={len(prompt_names)}  sample_id={qid!s}  "
        f"question={job.get('question')!r}  "
        f"answer={job.get('answer')!r}  "
        f"prediction={job.get('prediction')!r}",
        flush=True,
    )
    print()

    if args.json:
        rows = []
        for label in labels:
            meta = _judge_meta(label)
            for prompt_name in prompt_names:
                fmt = JUDGE_FORMATS[prompt_name]
                row: dict[str, Any] = {
                    "label": label,
                    "prompt_name": prompt_name,
                    "include_gold": fmt.include_gold,
                    "audio_included": fmt.audio_included,
                    "model_id": meta["model_id"],
                    "backend": meta.get("backend"),
                    "sampling": meta.get("sampling"),
                    "chat_kwargs": meta.get("chat_kwargs") or {},
                    "question_id": qid,
                    "question": job.get("question"),
                    "answer": job.get("answer"),
                    "prediction": job.get("prediction"),
                }
                try:
                    row["prompt"] = render_judge_prompt(
                        label, job, prompt=prompt_name
                    )
                except ValueError as exc:
                    row["prompt"] = None
                    row["skipped"] = str(exc)
                rows.append(row)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    for label in labels:
        for prompt_name in prompt_names:
            _print_combo(label, prompt_name, job)


if __name__ == "__main__":
    main()
