"""SQLite helpers for the audio-caption benchmark browser."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS examples (
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  caption TEXT NOT NULL,
  audio_path TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  UNIQUE(source, source_id)
);

CREATE TABLE IF NOT EXISTS benchmark_items (
  id INTEGER PRIMARY KEY,
  example_id INTEGER NOT NULL UNIQUE REFERENCES examples(id) ON DELETE CASCADE,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

SOURCES = ("wavcaps", "audiocaps", "clotho")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = db_path.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def count_examples(conn: sqlite3.Connection, source: str | None = None) -> int:
    if source:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM examples WHERE source = ?", (source,)
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) AS n FROM examples").fetchone()
    return int(row["n"])


def upsert_example(
    conn: sqlite3.Connection,
    *,
    source: str,
    source_id: str,
    caption: str,
    audio_path: str,
    metadata: dict[str, Any],
) -> int:
    conn.execute(
        """
        INSERT INTO examples (source, source_id, caption, audio_path, metadata_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source, source_id) DO UPDATE SET
          caption = excluded.caption,
          audio_path = excluded.audio_path,
          metadata_json = excluded.metadata_json
        """,
        (source, source_id, caption, audio_path, json.dumps(metadata, ensure_ascii=False)),
    )
    row = conn.execute(
        "SELECT id FROM examples WHERE source = ? AND source_id = ?",
        (source, source_id),
    ).fetchone()
    return int(row["id"])


def delete_examples_for_source(conn: sqlite3.Connection, source: str) -> None:
    conn.execute("DELETE FROM examples WHERE source = ?", (source,))


def list_examples(
    conn: sqlite3.Connection,
    *,
    source: str | None = None,
    annotated: str | None = None,
) -> list[dict[str, Any]]:
    """annotated: None/'all', 'yes', or 'no'."""
    clauses: list[str] = []
    params: list[Any] = []
    if source and source != "all":
        clauses.append("e.source = ?")
        params.append(source)
    if annotated == "yes":
        clauses.append(
            "EXISTS (SELECT 1 FROM benchmark_items b WHERE b.example_id = e.id "
            "AND length(trim(b.question)) > 0 AND length(trim(b.answer)) > 0)"
        )
    elif annotated == "no":
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM benchmark_items b WHERE b.example_id = e.id "
            "AND length(trim(b.question)) > 0 AND length(trim(b.answer)) > 0)"
        )
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT e.id, e.source, e.source_id, e.caption, e.audio_path,
               b.question, b.answer
        FROM examples e
        LEFT JOIN benchmark_items b ON b.example_id = e.id
        {where}
        ORDER BY e.source, e.id
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def get_example(conn: sqlite3.Connection, example_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT e.id, e.source, e.source_id, e.caption, e.audio_path, e.metadata_json,
               b.question, b.answer, b.updated_at
        FROM examples e
        LEFT JOIN benchmark_items b ON b.example_id = e.id
        WHERE e.id = ?
        """,
        (example_id,),
    ).fetchone()
    if row is None:
        return None
    item = dict(row)
    try:
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
    except json.JSONDecodeError:
        item["metadata"] = {"raw": item.pop("metadata_json", "")}
    return item


def upsert_benchmark(
    conn: sqlite3.Connection,
    *,
    example_id: int,
    question: str,
    answer: str,
) -> dict[str, Any]:
    question = question.strip()
    answer = answer.strip()
    if not question and not answer:
        conn.execute("DELETE FROM benchmark_items WHERE example_id = ?", (example_id,))
        conn.commit()
        return {"example_id": example_id, "question": "", "answer": "", "cleared": True}

    now = utc_now()
    conn.execute(
        """
        INSERT INTO benchmark_items (example_id, question, answer, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(example_id) DO UPDATE SET
          question = excluded.question,
          answer = excluded.answer,
          updated_at = excluded.updated_at
        """,
        (example_id, question, answer, now),
    )
    conn.commit()
    return {
        "example_id": example_id,
        "question": question,
        "answer": answer,
        "updated_at": now,
        "cleared": False,
    }


def clear_benchmark(conn: sqlite3.Connection, example_id: int) -> None:
    conn.execute("DELETE FROM benchmark_items WHERE example_id = ?", (example_id,))
    conn.commit()


def export_benchmark(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT e.source, e.source_id, e.caption, e.audio_path, e.metadata_json,
               b.question, b.answer, b.updated_at
        FROM benchmark_items b
        JOIN examples e ON e.id = b.example_id
        WHERE length(trim(b.question)) > 0 AND length(trim(b.answer)) > 0
        ORDER BY e.source, e.id
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            metadata = json.loads(item.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        out.append(
            {
                "source": item["source"],
                "source_id": item["source_id"],
                "audio_path": item["audio_path"],
                "caption": item["caption"],
                "question": item["question"],
                "answer": item["answer"],
                "updated_at": item["updated_at"],
                "metadata": metadata,
            }
        )
    return out


def get_state(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return str(row["value"])


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO app_state (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()
