"""List OpenAI models available to the current API key.

Usage::

    export OPENAI_API_KEY=...
    uv run python list_openai_models.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

from openai import OpenAI


def main() -> None:
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        raise SystemExit("Set OPENAI_API_KEY to list models.")

    client = OpenAI()
    models = sorted(client.models.list(), key=lambda m: m.id)

    if not models:
        print("No models returned for this key.")
        return

    by_owner: dict[str, list[str]] = defaultdict(list)
    for model in models:
        by_owner[getattr(model, "owned_by", "unknown")].append(model.id)

    print(f"{len(models)} models available\n")
    for owner, ids in sorted(by_owner.items()):
        print(f"{owner} ({len(ids)})")
        for model_id in ids:
            print(f"  {model_id}")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
