"""Context projection / result merge (collaboration foundation F3).

Workers (today: same runtime; tomorrow: subprocess) should only see the
projection, not the full message history.
"""

from __future__ import annotations

from typing import Any

from src.agent.work_items import current_work_item, get_item, mark_item


def project_context(
    state: dict[str, Any],
    work_item: dict[str, Any] | None = None,
    *,
    max_evidence: int = 8,
    max_chars_per_evidence: int = 1200,
) -> dict[str, Any]:
    """Build a narrow view for the current (or given) work item."""
    item = work_item or current_work_item(state)
    if item is None:
        return {
            "goal": state.get("goal") or state.get("query") or "",
            "evidence_slice": [],
            "artifacts": list(state.get("artifacts") or [])[:10],
            "budget": {
                "max_steps": None,
                "allowed_tools": None,
                "timeout_sec": None,
            },
            "work_item": None,
        }

    wanted = set(item.get("evidence_ids") or [])
    # Also include evidence from completed dependencies (one-to-many / backfill)
    dep_ids = set(item.get("depends_on") or [])
    for other in state.get("work_items") or []:
        if other.get("id") in dep_ids and other.get("status") == "done":
            wanted.update(other.get("evidence_ids") or [])

    slice_: list[dict[str, Any]] = []
    stale = set(state.get("stale_evidence_ids") or [])
    for e in state.get("evidence") or []:
        if e.get("id") in stale or e.get("stale"):
            continue
        if wanted and e.get("id") not in wanted:
            # If item has no evidence_ids yet, fall back to recent non-stale
            continue
        text = str(e.get("text") or e.get("snippet") or "")
        slice_.append(
            {
                "id": e.get("id"),
                "source_type": e.get("source_type"),
                "has_body": e.get("has_body"),
                "text": text[:max_chars_per_evidence],
                "url": e.get("url"),
            }
        )
        if len(slice_) >= max_evidence:
            break

    if not wanted:
        # First pass on item: take latest body-ish evidence up to max
        for e in reversed(state.get("evidence") or []):
            if e.get("id") in stale or e.get("stale"):
                continue
            if e.get("source_type") in {"web_snippet"} and not e.get("has_body"):
                continue
            text = str(e.get("text") or "")
            slice_.insert(
                0,
                {
                    "id": e.get("id"),
                    "source_type": e.get("source_type"),
                    "has_body": e.get("has_body"),
                    "text": text[:max_chars_per_evidence],
                    "url": e.get("url"),
                },
            )
            if len(slice_) >= max_evidence:
                break

    return {
        "goal": item.get("goal") or state.get("goal") or "",
        "evidence_slice": slice_,
        "artifacts": list(state.get("artifacts") or [])[:10],
        "budget": {
            "max_steps": None,
            "task_type": item.get("task_type"),
            "expect": item.get("expect"),
            "acceptance": item.get("acceptance"),
        },
        "work_item": {
            "id": item.get("id"),
            "title": item.get("title"),
            "task_type": item.get("task_type"),
            "status": item.get("status"),
            "depends_on": list(item.get("depends_on") or []),
        },
    }


def merge_result(
    state: dict[str, Any],
    work_item_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Merge a worker Result into parent state (append-only where possible)."""
    status = result.get("status") or "ok"
    item_status = {
        "ok": "done",
        "failed": "failed",
        "needs_review": "needs_revise",
        "needs_input": "needs_revise",
    }.get(status, "done")

    evid_delta = result.get("evidence_delta") or []
    new_ids: list[str] = []
    for e in evid_delta:
        if not isinstance(e, dict):
            continue
        eid = e.get("id")
        if eid:
            new_ids.append(str(eid))
        state.setdefault("evidence", []).append(e)

    for uri in result.get("artifact_uris") or []:
        state.setdefault("artifacts", []).append({"uri": uri, "work_item_id": work_item_id})

    outputs = result.get("outputs") or {}
    if outputs.get("final_answer"):
        # Accumulate per-item answers; orchestrator may compose later
        state.setdefault("item_answers", {})[work_item_id] = outputs["final_answer"]

    item = get_item(state, work_item_id)
    prev_ids = list((item or {}).get("evidence_ids") or [])
    mark_item(
        state,
        work_item_id,
        status=item_status,
        result_ref=result.get("result_ref") or outputs.get("artifact"),
        evidence_ids=prev_ids + new_ids,
        feedback=result.get("feedback"),
    )
    return state


def projection_to_broker_text(projection: dict[str, Any]) -> str:
    """Flatten projection for existing LLM planner prompts."""
    parts = [
        f"WorkItem: {projection.get('work_item')}",
        f"Goal: {projection.get('goal')}",
        f"Acceptance/Expect: {projection.get('budget')}",
        "Evidence:",
    ]
    for e in projection.get("evidence_slice") or []:
        parts.append(
            f"- [{e.get('source_type')}] id={e.get('id')} has_body={e.get('has_body')}\n"
            f"  {(e.get('text') or '')[:800]}"
        )
    return "\n".join(parts)
