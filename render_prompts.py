"""Render the freeform prompts ``run_experiment`` would send to each test-taker.

Uses the same builders as ``generate_batch`` with ``run_experiment`` defaults:
``prompt_mode=freeform`` and per-model ``MODEL_SPECS`` sampling (no CLI
overrides). Chat backends dump the ``LLM.chat`` messages; vLLM applies the
Jinja chat template later.

Usage::

    uv run python render_prompts.py
    uv run python render_prompts.py --models qwen3-omni,gemma-4-e4b
    uv run python render_prompts.py --question "What instrument is playing?"
    uv run python render_prompts.py --meta data/mmar/MMAR-meta.jsonl --question-id GJ6r_T6ckc4_00-00-00_00-00-06
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mmar_common import load_jsonl
from mmar_models import (
    MODEL_SPECS,
    chat_kwargs_for,
    parse_model_list,
    render_prompt,
    resolve_sampling,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_META = REPO_ROOT / "data" / "mmar" / "MMAR-meta.jsonl"
DEFAULT_AUDIO_PATH = "/cache/data/mmar/audio/example.wav"
DEFAULT_QUESTION = "What instrument is playing?"

# Matches ``run_experiment._run_model_eval`` (always freeform; no sampling overrides).
_EXPERIMENT_ARGS = SimpleNamespace(
    prompt_mode="freeform",
    temperature=None,
    top_p=None,
    max_new_tokens=None,
    greedy_non_thinking=False,
)

_RULE = "=" * 80
_THIN = "-" * 80


def _dummy_sample(question: str) -> dict[str, Any]:
    return {
        "id": "example",
        "question": question,
        "choices": ["piano", "guitar", "violin", "drums"],
        "audio_path": DEFAULT_AUDIO_PATH,
    }


def _load_sample(
    *,
    meta: Path | None,
    question_id: str | None,
    question: str | None,
) -> dict[str, Any]:
    if meta is None and question_id is None and question is None:
        if DEFAULT_META.is_file():
            meta = DEFAULT_META
        else:
            return _dummy_sample(DEFAULT_QUESTION)

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
        sample = dict(item)
        sample.setdefault("audio_path", sample.get("audio_path") or DEFAULT_AUDIO_PATH)
        if question:
            sample["question"] = question
        return sample

    if question_id:
        raise SystemExit("--question-id requires --meta")
    return _dummy_sample(question or DEFAULT_QUESTION)


def _thinking_tag(spec: dict[str, Any]) -> str:
    native = bool(spec.get("native_thinking"))
    if "enable_thinking" in spec:
        return f"native_thinking={native}  enable_thinking={bool(spec['enable_thinking'])}"
    return f"native_thinking={native}"


def _print_model(label: str, sample: dict[str, Any]) -> None:
    spec = MODEL_SPECS[label]
    sampling = resolve_sampling(label, _EXPERIMENT_ARGS)
    chat_kwargs = chat_kwargs_for(label)
    prompt = render_prompt(label, sample, _EXPERIMENT_ARGS)

    print(_RULE)
    print(label)
    print(
        f"  model_id={spec['model_id']}  backend={spec.get('backend')}"
    )
    print(f"  {_thinking_tag(spec)}")
    print(f"  sampling={sampling}")
    if chat_kwargs:
        print(f"  chat_kwargs={chat_kwargs}")
    print(_THIN)
    print(prompt.rstrip("\n"))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print the freeform prompts run_experiment would send to each "
            "test-taker with default settings."
        )
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated labels or 'all' (the run_experiment default).",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Override the question text (default: first MMAR row, else a dummy).",
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
        help="Emit one JSON object per model instead of the text dump.",
    )
    args = parser.parse_args()

    labels = parse_model_list(args.models)
    sample = _load_sample(
        meta=args.meta.expanduser().resolve() if args.meta else None,
        question_id=args.question_id,
        question=args.question,
    )
    qid = sample.get("id")
    print(
        f"run_experiment defaults: prompt_mode=freeform  "
        f"n_models={len(labels)}  sample_id={qid!s}  "
        f"question={sample.get('question')!r}",
        flush=True,
    )
    print()

    if args.json:
        rows = []
        for label in labels:
            spec = MODEL_SPECS[label]
            rows.append(
                {
                    "label": label,
                    "model_id": spec["model_id"],
                    "backend": spec.get("backend"),
                    "native_thinking": bool(spec.get("native_thinking")),
                    "enable_thinking": spec.get("enable_thinking"),
                    "sampling": resolve_sampling(label, _EXPERIMENT_ARGS),
                    "chat_kwargs": chat_kwargs_for(label),
                    "prompt": render_prompt(label, sample, _EXPERIMENT_ARGS),
                    "question_id": qid,
                    "question": sample.get("question"),
                }
            )
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    for label in labels:
        _print_model(label, sample)


if __name__ == "__main__":
    main()
