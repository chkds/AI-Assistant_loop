"""Context broker — assemble prompt context within token budget."""

from __future__ import annotations

from typing import Any

from src import load_routing
from src.ingest.chunker.multimodal import count_tokens
from src.memory.error_memory import recent_lessons


def assemble_context(state: dict[str, Any], budget_tokens: int | None = None) -> str:
    routing = load_routing()
    budget = budget_tokens or int(
        routing.get("defaults", {}).get("token_budget", 32000)
        * routing.get("defaults", {}).get("safe_context_ratio", 0.35)
    )
    stale = set(state.get("stale_evidence_ids") or [])
    parts: list[str] = []

    parts.append("## Goal\n" + str(state.get("goal") or state.get("query") or ""))
    if state.get("compress_summary"):
        parts.append("## CompactState\n" + str(state["compress_summary"]))
    if state.get("plan"):
        parts.append("## Plan\n" + json_dumps(state["plan"]))

    lessons = recent_lessons(limit=2, query=str(state.get("query") or ""))
    if lessons:
        lesson_txt = "\n".join(
            f"- [{x.get('type')}] {x.get('description')} => {x.get('correction')}" for x in lessons
        )
        parts.append("## PastLessons\n" + lesson_txt)

    # Pinned bodies first (explicit paper injection)
    evidence = list(state.get("evidence") or [])
    evidence.sort(key=lambda e: (0 if e.get("pinned") else 1))
    for e in evidence:
        if e.get("id") in stale or e.get("stale"):
            continue
        body = e.get("text") or e.get("parent_text") or ""
        if not body and e.get("snippet"):
            # title/snippet alone — mark weak
            body = f"[WEAK snippet only] {e.get('title','')}\n{e.get('snippet','')}"
        pin = " pinned" if e.get("pinned") else ""
        block = (
            f"### Evidence {e.get('id')} ({e.get('source_type')}{pin})\n"
            f"src={e.get('url') or e.get('doc_id') or ''}\n{body}"
        )
        parts.append(block)

    if state.get("last_observation"):
        parts.append("## LastObservation\n" + json_dumps(state["last_observation"]))

    # fit budget
    out: list[str] = []
    used = 0
    for p in parts:
        t = count_tokens(p)
        if out and used + t > budget:
            break
        out.append(p)
        used += t
    return "\n\n".join(out)


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)
