"""Shared OpenAI / Gemini Batch API helpers (50% cheaper async processing).

Both providers accept JSONL request files, process within a 24h window, and
return JSONL (or inline) results keyed by a per-request id.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

DEFAULT_POLL_INTERVAL_S = 30.0
OPENAI_BATCH_ENDPOINT = "/v1/chat/completions"
OPENAI_COMPLETION_WINDOW = "24h"
# Stay under provider caps (OpenAI 200MB / 50k reqs; Gemini file 2GB).
OPENAI_MAX_REQUESTS_PER_FILE = 40_000
GEMINI_MAX_REQUESTS_PER_FILE = 40_000
OPENAI_MAX_FILE_BYTES = 180 * 1024 * 1024
GEMINI_MAX_FILE_BYTES = 1800 * 1024 * 1024

_OPENAI_DONE = frozenset(
    {"completed", "failed", "expired", "cancelled", "cancelling"}
)
_GEMINI_DONE = frozenset(
    {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _chunk_by_count_and_size(
    rows: list[dict[str, Any]],
    *,
    max_count: int,
    max_bytes: int,
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False).encode("utf-8")
        line_bytes = len(encoded) + 1
        if current and (
            len(current) >= max_count or current_bytes + line_bytes > max_bytes
        ):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(row)
        current_bytes += line_bytes
    if current:
        chunks.append(current)
    return chunks


def openai_chat_request(
    *,
    custom_id: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
    seed: int | None = None,
    modalities: list[str] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = float(temperature)
    if max_tokens is not None:
        body["max_tokens"] = int(max_tokens)
    if seed is not None:
        body["seed"] = int(seed)
    if modalities is not None:
        body["modalities"] = list(modalities)
    if extra_body:
        body.update(extra_body)
    return {
        "custom_id": str(custom_id),
        "method": "POST",
        "url": OPENAI_BATCH_ENDPOINT,
        "body": body,
    }


def openai_batch_chat_text(result_row: dict[str, Any]) -> str | None:
    """Extract assistant text from one OpenAI batch output line."""
    if result_row.get("error"):
        return None
    response = result_row.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    text = message.get("content")
    if text is None:
        return None
    return str(text).strip() or None


def run_openai_chat_batch(
    requests: list[dict[str, Any]],
    *,
    work_dir: Path,
    display_name: str = "openai-batch",
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    endpoint: str = OPENAI_BATCH_ENDPOINT,
) -> dict[str, dict[str, Any]]:
    """Upload JSONL request(s), create batch job(s), poll, return rows by custom_id."""
    from openai import OpenAI

    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        raise RuntimeError("OPENAI_API_KEY is not set")
    if not requests:
        return {}

    client = OpenAI()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    chunks = _chunk_by_count_and_size(
        requests,
        max_count=OPENAI_MAX_REQUESTS_PER_FILE,
        max_bytes=OPENAI_MAX_FILE_BYTES,
    )
    by_id: dict[str, dict[str, Any]] = {}
    for chunk_index, chunk in enumerate(chunks):
        input_path = work_dir / f"{display_name}.chunk{chunk_index:03d}.input.jsonl"
        output_path = work_dir / f"{display_name}.chunk{chunk_index:03d}.output.jsonl"
        meta_path = work_dir / f"{display_name}.chunk{chunk_index:03d}.meta.json"
        _write_jsonl(input_path, chunk)
        print(
            f"[openai-batch] uploading {input_path.name} "
            f"({len(chunk)} requests, {input_path.stat().st_size} bytes)"
        )
        with open(input_path, "rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint=endpoint,
            completion_window=OPENAI_COMPLETION_WINDOW,
            metadata={
                "description": display_name,
                "chunk": str(chunk_index),
            },
        )
        meta = {
            "batch_id": batch.id,
            "input_file_id": uploaded.id,
            "n_requests": len(chunk),
            "status": batch.status,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[openai-batch] created {batch.id} status={batch.status}")

        while True:
            batch = client.batches.retrieve(batch.id)
            counts = getattr(batch, "request_counts", None)
            print(
                f"[openai-batch] {batch.id} status={batch.status}"
                + (f" counts={counts}" if counts is not None else "")
            )
            meta["status"] = batch.status
            meta["output_file_id"] = getattr(batch, "output_file_id", None)
            meta["error_file_id"] = getattr(batch, "error_file_id", None)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            if batch.status in _OPENAI_DONE:
                break
            time.sleep(poll_interval_s)

        if batch.status != "completed":
            raise RuntimeError(
                f"OpenAI batch {batch.id} finished with status={batch.status!r}"
            )
        if not batch.output_file_id:
            raise RuntimeError(f"OpenAI batch {batch.id} completed without output_file_id")

        content = client.files.content(batch.output_file_id)
        raw = content.read()
        output_path.write_bytes(raw)
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = str(row.get("custom_id") or "")
            if custom_id:
                by_id[custom_id] = row
        print(
            f"[openai-batch] {batch.id} done: wrote {output_path} "
            f"({len(by_id)} cumulative results)"
        )
    return by_id


def gemini_generate_request(
    *,
    key: str,
    contents: list[dict[str, Any]],
    system_instruction: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    thinking_level: str | None = None,
) -> dict[str, Any]:
    """One Gemini Batch JSONL line: ``{key, request}``."""
    request: dict[str, Any] = {"contents": contents}
    if system_instruction:
        request["system_instruction"] = {"parts": [{"text": system_instruction}]}
    generation_config: dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = float(temperature)
    if max_output_tokens is not None:
        generation_config["max_output_tokens"] = int(max_output_tokens)
    if thinking_level:
        generation_config["thinking_config"] = {"thinking_level": thinking_level}
    if generation_config:
        request["generation_config"] = generation_config
    return {"key": str(key), "request": request}


def gemini_text_part(text: str) -> dict[str, Any]:
    return {"text": text}


def gemini_inline_audio_part(audio_bytes: bytes, *, mime_type: str = "audio/wav") -> dict[str, Any]:
    import base64

    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": base64.b64encode(audio_bytes).decode("ascii"),
        }
    }


def gemini_user_contents(*parts: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "parts": list(parts)}]


def gemini_response_text_from_dict(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, dict):
        text = response.get("text")
        if text:
            return str(text).strip()
        parts: list[str] = []
        for candidate in response.get("candidates") or []:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                piece = part.get("text")
                if piece:
                    parts.append(str(piece))
        return "\n".join(parts).strip()
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    parts = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            piece = getattr(part, "text", None)
            if piece:
                parts.append(str(piece))
    return "\n".join(parts).strip()


def run_gemini_generate_batch(
    requests: list[dict[str, Any]],
    *,
    model: str,
    work_dir: Path,
    display_name: str = "gemini-batch",
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> dict[str, str]:
    """Upload JSONL generate requests, create batch job(s), return text by key."""
    from google import genai
    from google.genai import types

    api_key = (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    ).strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    if not requests:
        return {}

    client = genai.Client(api_key=api_key)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    chunks = _chunk_by_count_and_size(
        requests,
        max_count=GEMINI_MAX_REQUESTS_PER_FILE,
        max_bytes=GEMINI_MAX_FILE_BYTES,
    )
    by_key: dict[str, str] = {}
    for chunk_index, chunk in enumerate(chunks):
        input_path = work_dir / f"{display_name}.chunk{chunk_index:03d}.input.jsonl"
        output_path = work_dir / f"{display_name}.chunk{chunk_index:03d}.output.jsonl"
        meta_path = work_dir / f"{display_name}.chunk{chunk_index:03d}.meta.json"
        _write_jsonl(input_path, chunk)
        print(
            f"[gemini-batch] uploading {input_path.name} "
            f"({len(chunk)} requests, {input_path.stat().st_size} bytes)"
        )
        uploaded = client.files.upload(
            file=str(input_path),
            config=types.UploadFileConfig(
                display_name=f"{display_name}-{chunk_index}",
                mime_type="jsonl",
            ),
        )
        batch_job = client.batches.create(
            model=model,
            src=uploaded.name,
            config={"display_name": f"{display_name}-{chunk_index}"},
        )
        job_name = batch_job.name
        meta = {
            "batch_name": job_name,
            "input_file": uploaded.name,
            "n_requests": len(chunk),
            "state": getattr(getattr(batch_job, "state", None), "name", None),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[gemini-batch] created {job_name} state={meta['state']}")

        while True:
            batch_job = client.batches.get(name=job_name)
            state = getattr(getattr(batch_job, "state", None), "name", None)
            print(f"[gemini-batch] {job_name} state={state}")
            meta["state"] = state
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            if state in _GEMINI_DONE:
                break
            time.sleep(poll_interval_s)

        if state != "JOB_STATE_SUCCEEDED":
            error = getattr(batch_job, "error", None)
            raise RuntimeError(
                f"Gemini batch {job_name} finished with state={state!r} error={error!r}"
            )

        dest = getattr(batch_job, "dest", None)
        file_name = getattr(dest, "file_name", None) if dest is not None else None
        if file_name:
            raw = client.files.download(file=file_name)
            if isinstance(raw, bytes):
                text = raw.decode("utf-8")
            else:
                text = str(raw)
            output_path.write_text(text, encoding="utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("key") or "")
                if not key:
                    continue
                if row.get("error"):
                    print(f"[gemini-batch] error for {key}: {row.get('error')}")
                    continue
                response = row.get("response")
                extracted = gemini_response_text_from_dict(response)
                if extracted:
                    by_key[key] = extracted
        elif dest is not None and getattr(dest, "inlined_responses", None):
            lines: list[str] = []
            for index, inline in enumerate(dest.inlined_responses):
                key = str(chunk[index]["key"]) if index < len(chunk) else f"inline-{index}"
                if getattr(inline, "error", None):
                    print(f"[gemini-batch] error for {key}: {inline.error}")
                    continue
                response = getattr(inline, "response", None)
                extracted = gemini_response_text_from_dict(response)
                if extracted:
                    by_key[key] = extracted
                lines.append(
                    json.dumps(
                        {
                            "key": key,
                            "response": extracted,
                            "error": str(getattr(inline, "error", "") or "") or None,
                        },
                        ensure_ascii=False,
                    )
                )
            output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            raise RuntimeError(
                f"Gemini batch {job_name} succeeded without dest file or inlined responses"
            )
        print(
            f"[gemini-batch] {job_name} done: wrote {output_path} "
            f"({len(by_key)} cumulative results)"
        )
    return by_key
