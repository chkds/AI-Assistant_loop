"""Tool I/O adapter — normalize registry results into observation + evidence.

Functional chain:
  plan.next_action → registry.call → apply_tool_io → state.tool_trace
                                                 → state.evidence (harvest)
                                                 → state.last_observation
                                                 → observe / reflect
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable

from src.agent.evidence import harvest_tool_result


def truncate_result(result: dict[str, Any], limit: int = 4000) -> dict[str, Any]:
    raw = json.dumps(result, ensure_ascii=False, default=str)
    if len(raw) <= limit:
        return result
    return {"truncated": True, "preview": raw[:limit]}


def apply_tool_io(
    state: dict[str, Any],
    *,
    tool: str,
    arguments: dict[str, Any] | None,
    result: dict[str, Any],
    evidence_kind: str | None,
    add_evidence: Callable[[dict[str, Any], dict[str, Any]], None],
) -> dict[str, Any]:
    """
    Side-effect: appends tool_trace entries and harvested evidence onto state.
    Returns observation dict for last_observation / events.
    """
    args = dict(arguments or {})
    evidence_ids: list[str] = []
    harvested = harvest_tool_result(result, evidence_kind=evidence_kind)
    for item in harvested:
        if not item.get("id"):
            item = {**item, "id": uuid.uuid4().hex[:10]}
        add_evidence(state, item)
        # id may be assigned inside add_evidence
        evid = (state.get("evidence") or [])[-1]
        evidence_ids.append(str(evid.get("id")))

    preview = truncate_result(result if isinstance(result, dict) else {"raw": result})
    record: dict[str, Any] = {
        "tool": tool,
        "arguments": args,
        "ok": bool(result.get("ok")) if isinstance(result, dict) else False,
        "evidence_kind": evidence_kind,
        "evidence_ids": evidence_ids,
        "evidence_count": len(evidence_ids),
        "error": (result.get("error") if isinstance(result, dict) else None),
        "source_type": (result.get("source_type") if isinstance(result, dict) else None),
        "result_preview": preview,
    }
    state.setdefault("tool_trace", []).append(record)
    return {
        "type": "tool",
        "tool": tool,
        "ok": record["ok"],
        "evidence_ids": evidence_ids,
        "evidence_kind": evidence_kind,
        "result": preview,
        "error": record["error"],
    }
