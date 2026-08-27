"""Show which MMAR test-takers expose ``enable_thinking`` in tokenizer config.

Primary file is ``tokenizer_config.json``. Some checkpoints keep the Jinja
chat template in ``chat_template.json`` or ``chat_template.jinja`` instead
(Qwen3-Omni, Gemma 4), so those are checked too.

API test-takers are skipped because they have no Hugging Face tokenizer.

Usage::

    uv run python check_enable_thinking.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from mmar_models import MODEL_SPECS

TEMPLATE_FILES = (
    "tokenizer_config.json",
    "chat_template.json",
    "chat_template.jinja",
)


def _walk_enable_thinking(obj: Any, prefix: str = "") -> list[str]:
    """Return JSON paths / template hits where ``enable_thinking`` appears."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if key == "enable_thinking":
                hits.append(f"{path}={value!r}")
            hits.extend(_walk_enable_thinking(value, path))
        return hits
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            hits.extend(_walk_enable_thinking(value, f"{prefix}[{i}]"))
        return hits
    if isinstance(obj, str) and "enable_thinking" in obj:
        hits.append(f"{prefix or '<string>'} (jinja mentions enable_thinking)")
    return hits


def _file_hits(path: str, filename: str) -> list[str]:
    if filename.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return [f"{filename}: {hit}" for hit in _walk_enable_thinking(data)]
    text = open(path, encoding="utf-8").read()
    if "enable_thinking" in text:
        return [f"{filename}: jinja mentions enable_thinking"]
    return []


def _check_repo(repo_id: str) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    try:
        files = set(list_repo_files(repo_id, token=token))
    except RepositoryNotFoundError as exc:
        return {"ok": False, "error": f"repo not found: {exc}"}
    except GatedRepoError as exc:
        return {"ok": False, "error": f"gated (need HF_TOKEN): {exc}"}
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return {"ok": False, "error": f"HTTP {status}: {exc}"}
    except Exception as exc:  # noqa: BLE001 — Hub client raises a wide set
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    present = [name for name in TEMPLATE_FILES if name in files]
    if not present:
        return {"ok": False, "error": "no tokenizer_config.json / chat_template.* on Hub"}

    hits: list[str] = []
    in_tokenizer_config = False
    for name in present:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=name, token=token)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"download {name}: {type(exc).__name__}: {exc}"}
        file_hits = _file_hits(path, name)
        hits.extend(file_hits)
        if name == "tokenizer_config.json" and file_hits:
            in_tokenizer_config = True
    return {
        "ok": True,
        "hits": hits,
        "present": present,
        "in_tokenizer_config": in_tokenizer_config,
    }


def main() -> None:
    jobs: list[tuple[str, str, str]] = []
    for label, spec in MODEL_SPECS.items():
        model_id = spec["model_id"]
        jobs.append((label, "model_id", model_id))
        tokenizer_id = spec.get("tokenizer_id")
        if tokenizer_id and tokenizer_id != model_id:
            jobs.append((label, "tokenizer_id", tokenizer_id))

    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_map = {pool.submit(_check_repo, job[2]): job for job in jobs}
        for future in as_completed(future_map):
            results[future_map[future]] = future.result()

    in_tok: list[str] = []
    elsewhere: list[str] = []
    no: list[str] = []
    missing: list[str] = []

    print(f"{'label':<28} {'repo_id':<52} enable_thinking")
    print("-" * 120)
    for job in jobs:
        label, field, repo_id = job
        info = results[job]
        tag = f"{label} ({field}={repo_id})" if field != "model_id" else f"{label} ({repo_id})"
        if not info["ok"]:
            status = f"NO FILE / {info['error']}"
            missing.append(f"{tag}: {info['error']}")
        elif info["hits"]:
            where = "tokenizer_config.json" if info["in_tokenizer_config"] else "sibling template file"
            status = f"YES ({where}) — " + "; ".join(info["hits"])
            (in_tok if info["in_tokenizer_config"] else elsewhere).append(tag)
        else:
            status = f"no  (checked {', '.join(info['present'])})"
            no.append(tag)
        print(f"{label:<28} {repo_id:<52} {status}")

    print("\nIn tokenizer_config.json:")
    print("\n".join(f"  - {line}" for line in in_tok) if in_tok else "  (none)")
    print("\nOnly in chat_template.json / chat_template.jinja:")
    print("\n".join(f"  - {line}" for line in elsewhere) if elsewhere else "  (none)")
    print("\nNo enable_thinking in tokenizer_config or sibling templates:")
    print("\n".join(f"  - {line}" for line in no) if no else "  (none)")
    if missing:
        print("\nMissing or unreadable:")
        print("\n".join(f"  - {line}" for line in missing))


if __name__ == "__main__":
    main()
