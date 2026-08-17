"""Local MMAR test-taker via OpenAI / Gemini audio APIs.

Writes under ``outputs/exp-mmar-question-difficulty-api/<run_id>/`` so Modal
downloads into ``outputs/exp-mmar-question-difficulty/`` cannot overwrite them.

Usage::

    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...

    uv run python run_experiment_api.py \\
      --models gpt-audio-mini,gemini-3.7-flash \\
      --mode freeform \\
      --n-shots 5 \\
      --question-ids-csv answer-variety/open_ended_question_ids.csv

    # Full-MMAR native MCQ sanity check (1 greedy pass; Gemini keeps T=1.0)
    uv run python run_experiment_api.py \\
      --models all --mode mc --n-shots 1 --num-samples -1 \\
      --greedy-non-thinking --question-ids-csv none
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmar_common import (
    aggregate_n_shot_record,
    build_mmar_freeform_prompt,
    build_mmar_prompt,
    load_completed_ids,
    load_jsonl,
    load_question_ids_csv,
    make_run_id,
    parse_choice_output,
    parse_freeform_output,
    parse_think_tagged_output,
    resolve_path,
    write_json,
    write_jsonl,
)
from mmar_groundedness import gemini_response_text
from view_mmar import DEFAULT_AUDIO_DIR, DEFAULT_DATA_DIR, ensure_mmar_audio

REPO_ROOT = Path(__file__).resolve().parent
API_EXPERIMENT = "exp-mmar-question-difficulty-api"
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs"
DEFAULT_META = DEFAULT_DATA_DIR / "MMAR-meta.jsonl"
DEFAULT_IDS_CSV = REPO_ROOT / "answer-variety" / "open_ended_question_ids.csv"
DEFAULT_N_SHOTS = 5
DEFAULT_SEED = 42
DEFAULT_MODE = "freeform"

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
logger = logging.getLogger(__name__)


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


def _shot_seed(seed: int, question_id: str, shot_index: int) -> int:
    digest = hashlib.md5(f"{seed}:{question_id}:{shot_index}".encode()).hexdigest()
    return seed + (int(digest[:8], 16) % 1_000_000)


def _normalize_mode(mode: str | None) -> str:
    value = str(mode or DEFAULT_MODE).strip().lower()
    if value in {"freeform", "free_form", "free-form", "open"}:
        return "freeform"
    if value in {"mc", "multiple_choice", "multiple-choice", "mcq", "choice"}:
        return "mc"
    raise SystemExit(f"Unknown mode {mode!r}; expected 'mc' or 'freeform'")


def resolve_api_sampling(label: str, args: argparse.Namespace) -> dict[str, Any]:
    spec = API_SPECS[label]
    out = dict(spec["sampling"])
    if getattr(args, "greedy_non_thinking", False) and not spec.get("native_thinking"):
        out["temperature"] = 0.0
    if args.temperature is not None:
        out["temperature"] = float(args.temperature)
    if args.max_new_tokens is not None:
        out["max_tokens"] = int(args.max_new_tokens)
    return out


def _optional_csv(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "skip", "-"}:
        return None
    return Path(text).expanduser().resolve()


def _build_api_prompt(item: dict, mode: str) -> str:
    if mode == "freeform":
        return build_mmar_freeform_prompt(item)
    return build_mmar_prompt(item)


def _load_all_items(meta_path: Path, data_root: Path) -> list[dict]:
    items: list[dict] = []
    for item in load_jsonl(meta_path):
        audio_path = resolve_path(data_root, item["audio_path"])
        if not Path(audio_path).is_file():
            print(f"Skipping {item.get('id')}: missing audio at {audio_path}")
            continue
        items.append({**item, "audio_path": audio_path})
    return items


def _load_selected_items(
    meta_path: Path,
    data_root: Path,
    question_ids: list[str],
) -> list[dict]:
    by_id = {str(item["id"]): item for item in load_jsonl(meta_path)}
    items: list[dict] = []
    for qid in question_ids:
        item = by_id.get(qid)
        if item is None:
            print(f"Skipping {qid}: not in MMAR meta")
            continue
        audio_path = resolve_path(data_root, item["audio_path"])
        if not Path(audio_path).is_file():
            print(f"Skipping {qid}: missing audio at {audio_path}")
            continue
        items.append({**item, "audio_path": audio_path})
    return items


def _output_from_text(raw: str, item: dict, *, mode: str, native_thinking: bool) -> dict:
    if mode == "mc":
        parse_fn = parse_think_tagged_output if native_thinking else parse_choice_output
        thinking, answer = parse_fn(raw, item.get("choices") or [])
    else:
        thinking, answer = parse_freeform_output(raw)
    return {
        "model_output": raw,
        "raw_tokens": None,
        "thinking_prediction": thinking,
        "answer_prediction": answer,
    }


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

    async def complete(self, prompt: str, audio_path: str, seed: int) -> str:
        audio_b64 = base64.b64encode(Path(audio_path).read_bytes()).decode("ascii")
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
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": audio_b64,
                                            "format": "wav",
                                        },
                                    },
                                ],
                            }
                        ],
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
                return text
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

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
                now = time.monotonic()
            self._next_time = now + self._interval

    async def complete(self, prompt: str, audio_path: str, seed: int) -> str:
        del seed  # Gemini generate_content has no per-request seed here.
        audio_bytes = Path(audio_path).read_bytes()
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
                        contents=[
                            self.types.Part.from_bytes(
                                data=audio_bytes, mime_type="audio/wav"
                            ),
                            prompt,
                        ],
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
                return text
            logger.warning("gemini attempt %s empty response", attempt)
            await asyncio.sleep(self.retry_interval)
        raise RuntimeError(f"Gemini retries exhausted: {last_exc}")


def _make_taker(label: str, args: argparse.Namespace):
    spec = API_SPECS[label]
    sampling = resolve_api_sampling(label, args)
    kwargs = dict(
        model_id=spec["model_id"],
        temperature=float(sampling["temperature"]),
        max_tokens=int(sampling["max_tokens"]),
        qps=args.qps,
        timeout=args.timeout,
        retries=args.retries,
        retry_interval=args.retry_interval,
    )
    if spec["backend"] == "openai":
        return OpenAIAudioTaker(**kwargs)
    if spec["backend"] == "gemini":
        return GeminiAudioTaker(**kwargs)
    raise SystemExit(f"Unknown backend for {label}")


async def _run_label(
    *,
    label: str,
    items: list[dict],
    run_dir: Path,
    args: argparse.Namespace,
    question_ids: list[str],
) -> dict:
    spec = API_SPECS[label]
    prompt_mode = _normalize_mode(args.mode)
    pending_grade = prompt_mode == "freeform"
    native_thinking = bool(spec.get("native_thinking"))
    model_dir = run_dir / "models" / label
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = model_dir / "predictions.jsonl"
    completed = {str(x) for x in load_completed_ids(predictions_path)}
    pending = [item for item in items if str(item["id"]) not in completed]
    print(
        f"[{label}] model={spec['model_id']} "
        f"{len(items)} selected, {len(completed)} done, {len(pending)} pending "
        f"(n_shots={args.n_shots}, mode={prompt_mode})"
    )
    if not pending:
        return {
            "status": "already_complete",
            "model_label": label,
            "n_predictions": len(completed),
            "predictions_path": str(predictions_path),
        }

    taker = _make_taker(label, args)
    semaphore = asyncio.Semaphore(args.max_workers)
    write_lock = asyncio.Lock()
    written = 0

    async def _one(item: dict) -> None:
        nonlocal written
        prompt = _build_api_prompt(item, prompt_mode)
        shot_outputs: list[dict] = []
        for shot_index in range(args.n_shots):
            seed = _shot_seed(args.seed, str(item["id"]), shot_index)
            async with semaphore:
                text = await taker.complete(prompt, item["audio_path"], seed)
            shot_outputs.append(
                _output_from_text(
                    text,
                    item,
                    mode=prompt_mode,
                    native_thinking=native_thinking,
                )
            )
        record = aggregate_n_shot_record(
            item, shot_outputs, pending_grade=pending_grade
        )
        async with write_lock:
            write_jsonl(predictions_path, [record], mode="a")
            written += 1
            if written % args.print_every == 0 or written == len(pending):
                print(
                    f"[{label}] {written}/{len(pending)} id={item['id']} "
                    f"answer={record.get('answer_prediction')!r}"
                )

    await asyncio.gather(*(_one(item) for item in pending))
    total = len(load_completed_ids(predictions_path) & set(question_ids))
    return {
        "status": "ok",
        "model_label": label,
        "n_written": written,
        "n_predictions": total,
        "predictions_path": str(predictions_path),
        "backend": spec["backend"],
        "mode": prompt_mode,
    }


async def async_main(args: argparse.Namespace) -> dict:
    prompt_mode = _normalize_mode(args.mode)
    labels = parse_api_model_list(args.models)
    run_id = args.run_id or make_run_id()
    results_dir = Path(args.results_dir).expanduser().resolve()
    run_dir = results_dir / API_EXPERIMENT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.meta).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    if not args.skip_audio_download:
        audio_dir = ensure_mmar_audio(audio_dir)

    csv_path = _optional_csv(args.question_ids_csv)
    if csv_path is not None:
        wanted = load_question_ids_csv(csv_path)
        items = _load_selected_items(meta_path, data_root, wanted)
    else:
        items = _load_all_items(meta_path, data_root)
        if args.num_samples >= 0 and args.num_samples < len(items):
            rng = random.Random(args.seed)
            items = [items[i] for i in rng.sample(range(len(items)), args.num_samples)]
    question_ids = [str(item["id"]) for item in items]
    write_json(
        run_dir / "question_ids.json",
        {
            "n": len(question_ids),
            "ids": question_ids,
            "question_ids_csv": str(csv_path) if csv_path else None,
            "num_samples": args.num_samples,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    sampling = {label: resolve_api_sampling(label, args) for label in labels}

    manifest_path = run_dir / "manifest.json"
    existing: dict = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    prior = [str(x) for x in (existing.get("models") or [])]
    merged = list(dict.fromkeys([*prior, *labels]))
    now = datetime.now(timezone.utc).isoformat()
    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "experiment": API_EXPERIMENT,
            "mode": prompt_mode,
            "models": merged,
            "n_shots": args.n_shots,
            "seed": args.seed,
            "model_sampling": sampling,
            "sampling_overrides": {
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "greedy_non_thinking": bool(args.greedy_non_thinking),
            },
            "scoring": (
                "pending_local_api"
                if prompt_mode == "freeform"
                else "mean_shot_success_rate_string_match"
            ),
            "inference": "api",
            "n_questions": len(question_ids),
            "question_ids_csv": str(csv_path) if csv_path else None,
            "model_specs": {
                label: {
                    "model_id": API_SPECS[label]["model_id"],
                    "backend": API_SPECS[label]["backend"],
                    "native_thinking": bool(API_SPECS[label].get("native_thinking")),
                    "sampling": API_SPECS[label]["sampling"],
                }
                for label in ALL_API_LABELS
            },
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
        },
    )

    results = []
    for label in labels:
        results.append(
            await _run_label(
                label=label,
                items=items,
                run_dir=run_dir,
                args=args,
                question_ids=question_ids,
            )
        )
    print("Done:", {"run_id": run_id, "models": results})
    return {"run_id": run_id, "mode": prompt_mode, "models": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="gpt-audio-mini,gemini-3.7-flash",
        help="Comma-separated labels or 'all'",
    )
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--n-shots", type=int, default=DEFAULT_N_SHOTS)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--greedy-non-thinking",
        action="store_true",
        help="Force temperature=0 on models without native thinking/reasoning.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--question-ids-csv",
        default=str(DEFAULT_IDS_CSV),
        help="CSV of question ids, or 'none' for the full MMAR meta set",
    )
    parser.add_argument("--qps", type=float, default=4.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-interval", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--skip-audio-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
