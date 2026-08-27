"""Flag MMAR questions that cannot be answered without the MC choice list.

Sends question text only (no gold, no choices, no audio) to Claude Haiku.
Flags wording that deictically points at a hidden option list
(``which of the following options``, ``which option depicts…``). Does not
flag questions that are merely underspecified or that use ``following`` to
refer to the audio itself.

Writes:

- ``judgments.jsonl`` — full judge rows (resumable)
- ``flagged_question_ids.csv`` — IDs that depend on the choice list
- ``summary.json``

Usage::

    export ANTHROPIC_API_KEY=...

    uv run python flag_mc_dependent_questions.py
    uv run python flag_mc_dependent_questions.py --limit 20
    uv run python flag_mc_dependent_questions.py --force
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from mmar_common import load_jsonl, write_json, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_META = REPO_ROOT / "data" / "mmar" / "MMAR-meta.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "mc-dependent"
MMAR_META_URL = "https://raw.githubusercontent.com/ddlBoJack/MMAR/main/MMAR-meta.jsonl"

DEFAULT_MODEL = "claude-haiku-4-5"
JUDGMENTS_NAME = "judgments.jsonl"
FLAGGED_IDS_NAME = "flagged_question_ids.csv"
SUMMARY_NAME = "summary.json"

SYSTEM_PROMPT = (
    "You classify audio-understanding questions from question text alone.\n"
    "Decide whether the question cannot be answered without a hidden "
    "multiple-choice list that is not written in the question.\n"
    "\n"
    "Flag (Yes) only when the wording deictically refers to answer choices "
    "that are not listed in the question itself. Typical patterns:\n"
    "- \"which of the following options\"\n"
    "- \"which of the following\" plus a category (instruments, scenarios, "
    "objects, clips) when those items are not named in the question\n"
    "- \"which option depicts\" / \"which option describes\"\n"
    "- \"choose from the following\" / \"from the options below\"\n"
    "\n"
    "Do not flag (No) when:\n"
    "- \"the following audio / clip / passage / segment / works\" refers to "
    "the audio itself, not to answer choices\n"
    "- \"these two people / sounds / clips\" refers to things heard in the "
    "audio\n"
    "- the question lists its alternatives inline, e.g. \"among the following "
    "four techniques: glissando, vibrato, scat, and falsetto\"\n"
    "- the question is merely underspecified or has many plausible answers, "
    "e.g. \"what is the most likely scenario?\", \"what instrument is this?\"\n"
    "- \"option\" refers to a choice a speaker in the audio is making\n"
    "\n"
    "Judge only the question wording. Do not assume a multiple-choice list.\n"
    "\n"
    "Reason briefly if needed, then end your reply with a single final line "
    "containing only Yes or No. Yes means the question depends on a hidden "
    "choice list. No means it does not."
)

_VERDICT_PHRASE_RE = re.compile(
    r"(?:final\s+)?(?:verdict|answer|flag)\s*[:\-]?\s*[\"']?(yes|no)[\"']?\b",
    re.IGNORECASE,
)
_BARE_VERDICT_RE = re.compile(r"^[\"']?(yes|no)[\"']?[.\s]*$", re.IGNORECASE)


def _norm_verdict(text: str) -> str:
    return "Yes" if text.strip().lower() == "yes" else "No"


def parse_verdict(text: str) -> str | None:
    region = str(text or "").strip()
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
    region = str(text or "").strip()
    if not region:
        return ""
    lines = region.splitlines()
    if verdict and lines:
        last = lines[-1].strip().strip("\"'")
        if last.lower() == verdict.lower():
            return "\n".join(lines[:-1]).strip()
    return region


def ensure_mmar_meta(meta_path: Path) -> Path:
    if meta_path.is_file() and meta_path.stat().st_size > 0:
        return meta_path
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    print(f"[mc-dep] downloading MMAR-meta.jsonl from {MMAR_META_URL}")
    urllib.request.urlretrieve(MMAR_META_URL, tmp)
    tmp.replace(meta_path)
    return meta_path


def load_questions(meta_path: Path, *, limit: int) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in load_jsonl(meta_path):
        qid = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        if not qid or not question or qid in seen:
            continue
        seen.add(qid)
        items.append({"id": qid, "question": question})
        if limit > 0 and len(items) >= limit:
            break
    if not items:
        raise SystemExit(f"No MMAR questions in {meta_path}")
    return items


def create_user_prompt(question: str) -> str:
    text = str(question or "").strip() or "(empty question)"
    return (
        "Does this question depend on a hidden multiple-choice list?\n\n"
        f"Question:\n{text}"
    )


def message_text(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            piece = getattr(block, "text", None)
            if piece:
                parts.append(str(piece))
    return "\n".join(parts).strip()


def is_fully_graded(item: dict) -> bool:
    if not item.get("id"):
        return False
    if item.get("verdict") not in {"Yes", "No"}:
        return False
    return bool(str(item.get("raw_response") or "").strip())


def load_completed_ids(path: Path) -> set[str]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    return {str(item["id"]) for item in load_jsonl(path) if is_fully_graded(item)}


def prune_incomplete(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    records = load_jsonl(path)
    keep = [item for item in records if is_fully_graded(item)]
    removed = len(records) - len(keep)
    if removed <= 0:
        return 0
    by_id = {str(item["id"]): item for item in keep}
    write_jsonl(path, list(by_id.values()), mode="w")
    return removed


def write_id_csv(path: Path, ids: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id"])
        for qid in ids:
            writer.writerow([qid])
    return path


def judgment_record(item: dict, *, raw_response: str, verdict: str | None) -> dict[str, Any]:
    return {
        "id": item["id"],
        "question": item.get("question") or "",
        "verdict": verdict,
        "flagged": True if verdict == "Yes" else (False if verdict == "No" else None),
        "reason": extract_reason(raw_response, verdict),
        "raw_response": raw_response,
        "scoring": "mc_dependent_question_text",
    }


class HaikuClassifier:
    def __init__(
        self,
        *,
        model_id: str,
        api_max_retries: int,
        api_retry_interval: float,
        qps: float,
        timeout: float,
        max_tokens: int,
    ):
        from anthropic import AsyncAnthropic

        if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
            raise SystemExit("Set ANTHROPIC_API_KEY to call Claude.")
        self.client = AsyncAnthropic()
        self.model_id = model_id
        self.api_max_retries = api_max_retries
        self.api_retry_interval = api_retry_interval
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._next_time = 0.0
        self._interval = 1.0 / max(float(qps), 0.1)
        self._lock = asyncio.Lock()

    def _is_rate_limit(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in (429, 529):
            return True
        text = str(exc).lower()
        return "rate" in text or "overloaded" in text or "429" in text

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
                now = time.monotonic()
            self._next_time = now + self._interval

    async def _complete(self, user_prompt: str) -> str:
        message = await self.client.messages.create(
            model=self.model_id,
            max_tokens=int(self.max_tokens),
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message_text(message)

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
                logging.warning("%s (api_attempt=%s) timeout: %s", log_id, attempt, exc)
                await asyncio.sleep(self.api_retry_interval)
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if self._is_rate_limit(exc):
                    logging.warning(
                        "%s (api_attempt=%s) rate limit: %s", log_id, attempt, exc
                    )
                    await asyncio.sleep(self.api_retry_interval * attempt)
                    continue
                raise
            if text:
                return text
            logging.warning("%s (api_attempt=%s) empty Claude response", log_id, attempt)
            await asyncio.sleep(self.api_retry_interval)
        raise RuntimeError(
            f"{log_id} Claude retries exhausted ({self.api_max_retries}): {last_exc}"
        )


async def classify(
    items: list[dict[str, str]],
    judgments_path: Path,
    *,
    model_id: str,
    max_workers: int,
    qps: float,
    timeout: float,
    retries: int,
    retry_interval: float,
    max_tokens: int,
    force: bool,
) -> dict[str, Any]:
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    if force and judgments_path.is_file():
        judgments_path.unlink()
        print(f"[mc-dep] --force: cleared {judgments_path}")
    removed = prune_incomplete(judgments_path)
    if removed:
        print(f"[mc-dep] pruned {removed} incomplete rows")
    completed = load_completed_ids(judgments_path)
    pending = [item for item in items if item["id"] not in completed]
    print(
        f"[mc-dep] judge={model_id} pending={len(pending)} "
        f"completed={len(completed)} -> {judgments_path}"
    )
    if not pending:
        return {"status": "already_done", "n_ok": 0, "n_fail": 0, "n_pending": 0}

    classifier = HaikuClassifier(
        model_id=model_id,
        api_max_retries=retries,
        api_retry_interval=retry_interval,
        qps=qps,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    semaphore = asyncio.Semaphore(max_workers)
    write_lock = asyncio.Lock()
    n_ok = 0
    n_fail = 0
    n_unparsed = 0

    async def _one(item: dict[str, str]) -> None:
        nonlocal n_ok, n_fail, n_unparsed
        user_prompt = create_user_prompt(item["question"])
        try:
            async with semaphore:
                raw = await classifier.call(item["id"], user_prompt)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"[mc-dep] failed {item['id']}: {exc}")
            return
        verdict = parse_verdict(raw)
        if verdict is None:
            n_unparsed += 1
            print(f"[mc-dep] unparsed verdict for {item['id']}: {raw[:180]!r}")
        async with write_lock:
            write_jsonl(
                judgments_path,
                [judgment_record(item, raw_response=raw, verdict=verdict)],
                mode="a",
            )
            n_ok += 1
            if n_ok % 50 == 0 or n_ok == len(pending):
                print(f"[mc-dep] scored {n_ok}/{len(pending)}")

    await asyncio.gather(*[_one(item) for item in pending])
    return {
        "status": "ok",
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_unparsed": n_unparsed,
        "n_pending": len(pending),
    }


def write_outputs(
    judgments_path: Path,
    *,
    flagged_ids_path: Path,
    summary_path: Path,
    judge_model_id: str,
    n_questions: int,
) -> dict[str, Any]:
    records = load_jsonl(judgments_path) if judgments_path.is_file() else []
    by_id: dict[str, dict] = {}
    for item in records:
        qid = str(item.get("id") or "")
        if qid:
            by_id[qid] = item
    ordered = list(by_id.values())
    flagged = [str(item["id"]) for item in ordered if item.get("verdict") == "Yes"]
    n_no = sum(1 for item in ordered if item.get("verdict") == "No")
    n_unparsed = sum(1 for item in ordered if item.get("verdict") not in {"Yes", "No"})
    write_id_csv(flagged_ids_path, flagged)
    summary = {
        "judge_model_id": judge_model_id,
        "scoring": "mc_dependent_question_text",
        "n_questions": n_questions,
        "n_judged": len(ordered),
        "n_flagged": len(flagged),
        "n_not_flagged": n_no,
        "n_unparsed": n_unparsed,
        "flagged_ids": flagged,
        "judgments_path": str(judgments_path),
        "flagged_ids_path": str(flagged_ids_path),
    }
    write_json(summary_path, summary)
    return summary


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    meta_path = ensure_mmar_meta(Path(args.meta).expanduser().resolve())
    items = load_questions(meta_path, limit=args.limit)
    print(f"[mc-dep] {len(items)} questions from {meta_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = out_dir / JUDGMENTS_NAME
    flagged_ids_path = out_dir / FLAGGED_IDS_NAME
    summary_path = out_dir / SUMMARY_NAME

    grade = await classify(
        items,
        judgments_path,
        model_id=args.model,
        max_workers=args.max_workers,
        qps=args.qps,
        timeout=args.timeout,
        retries=args.retries,
        retry_interval=args.retry_interval,
        max_tokens=args.max_tokens,
        force=args.force,
    )
    summary = write_outputs(
        judgments_path,
        flagged_ids_path=flagged_ids_path,
        summary_path=summary_path,
        judge_model_id=args.model,
        n_questions=len(items),
    )
    flagged = summary["flagged_ids"]
    print(f"[mc-dep] flagged {len(flagged)}/{summary['n_judged']} -> {flagged_ids_path}")
    if flagged:
        by_id = {
            str(item["id"]): item
            for item in load_jsonl(judgments_path)
            if item.get("id")
        }
        print("[mc-dep] flagged IDs:")
        for qid in flagged:
            question = (by_id.get(qid) or {}).get("question") or ""
            print(f"  {qid}")
            if question:
                print(f"    {question}")
    else:
        print("[mc-dep] no questions flagged")
    return {**grade, "summary": summary}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max questions to classify (0 = all).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-judge every question, ignoring existing judgments.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-interval", type=float, default=1.0)
    parser.add_argument("--qps", type=float, default=8.0)
    parser.add_argument("--max-workers", "-j", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
