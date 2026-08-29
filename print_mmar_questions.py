"""Print up to 100 question texts from each MMAR category."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_META = REPO_ROOT / "data" / "mmar" / "MMAR-meta.jsonl"
PER_CATEGORY = 100


def main() -> None:
    by_category: dict[str, list[str]] = defaultdict(list)
    with DEFAULT_META.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question") or "").strip()
            if not question:
                continue
            category = str(row.get("category") or "(uncategorized)")
            if len(by_category[category]) < PER_CATEGORY:
                by_category[category].append(question)

    for category in sorted(by_category):
        questions = by_category[category]
        print(f"=== {category} ({len(questions)}) ===")
        for question in questions:
            print(question)
        print()


if __name__ == "__main__":
    main()
