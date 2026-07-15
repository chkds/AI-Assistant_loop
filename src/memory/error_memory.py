"""Error / correction memory (JSONL + optional lessons for broker)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import resolve_path


def _path() -> Path:
    p = resolve_path("data/error_memory.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def record_error(
    *,
    error_type: str,
    description: str,
    context: str = "",
    correction: str = "",
    session_id: str | None = None,
) -> dict[str, Any]:
    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": error_type,
        "description": description,
        "context": context[:2000],
        "correction": correction[:2000],
        "session_id": session_id,
    }
    with _path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def recent_lessons(limit: int = 5, query: str | None = None) -> list[dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if query:
        q = query.lower()
        scored = []
        for r in rows:
            blob = f"{r.get('description','')} {r.get('correction','')}".lower()
            score = sum(1 for w in q.split() if w and w in blob)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for s, r in scored if s > 0][:limit] or rows[-limit:]
    return rows[-limit:]
