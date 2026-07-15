"""Work-item scheduling units (collaboration foundation F1).

Orchestrator writes these; workers (same runtime today) consume the current item.
Subagents later map assignee_role → process; structure stays the same.
"""

from __future__ import annotations

import uuid
from typing import Any


WORK_ITEM_STATUSES = {
    "pending",
    "ready",
    "running",
    "done",
    "failed",
    "blocked",
    "needs_revise",
}

EXPECT_KINDS = {"final_answer", "artifact", "evidence", "patch"}


def new_work_item(
    *,
    title: str,
    task_type: str,
    goal: str,
    depends_on: list[str] | None = None,
    acceptance: str = "",
    assignee_role: str | None = None,
    expect: str = "final_answer",
    status: str = "pending",
    item_id: str | None = None,
) -> dict[str, Any]:
    exp = expect if expect in EXPECT_KINDS else "final_answer"
    st = status if status in WORK_ITEM_STATUSES else "pending"
    return {
        "id": item_id or uuid.uuid4().hex[:10],
        "title": title,
        "status": st,
        "task_type": task_type,
        "depends_on": list(depends_on or []),
        "acceptance": acceptance,
        "assignee_role": assignee_role or task_type,
        "expect": exp,
        "goal": goal,
        "result_ref": None,
        "feedback": None,
        "evidence_ids": [],
    }


def get_item(state: dict[str, Any], item_id: str | None) -> dict[str, Any] | None:
    if not item_id:
        return None
    for it in state.get("work_items") or []:
        if it.get("id") == item_id:
            return it
    return None


def current_work_item(state: dict[str, Any]) -> dict[str, Any] | None:
    return get_item(state, state.get("current_work_item_id"))


def dependencies_satisfied(item: dict[str, Any], items: list[dict[str, Any]]) -> bool:
    by_id = {i["id"]: i for i in items}
    for dep in item.get("depends_on") or []:
        d = by_id.get(dep)
        if d is None or d.get("status") != "done":
            return False
    return True


def refresh_ready(state: dict[str, Any]) -> dict[str, Any]:
    """pending/blocked → ready when depends_on are done; blocked otherwise."""
    items = list(state.get("work_items") or [])
    for it in items:
        if it.get("status") in {"done", "failed", "running", "needs_revise"}:
            continue
        if dependencies_satisfied(it, items):
            if it.get("status") in {"pending", "blocked"}:
                it["status"] = "ready"
        else:
            if it.get("status") == "ready":
                it["status"] = "blocked"
            elif it.get("status") == "pending":
                it["status"] = "blocked"
    state["work_items"] = items
    return state


def select_next_work_item(state: dict[str, Any]) -> dict[str, Any] | None:
    """Pick next ready/needs_revise item (not one-to-one; may revisit earlier)."""
    refresh_ready(state)
    items = state.get("work_items") or []
    # Prefer needs_revise (feedback / supplement), then ready in order
    for it in items:
        if it.get("status") == "needs_revise":
            return it
    for it in items:
        if it.get("status") == "ready":
            return it
    return None


def all_terminal(state: dict[str, Any]) -> bool:
    items = state.get("work_items") or []
    if not items:
        return False
    return all(i.get("status") in {"done", "failed"} for i in items)


def mark_item(
    state: dict[str, Any],
    item_id: str,
    *,
    status: str | None = None,
    result_ref: str | None = None,
    feedback: dict[str, Any] | None = None,
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    for it in state.get("work_items") or []:
        if it.get("id") != item_id:
            continue
        if status:
            it["status"] = status
        if result_ref is not None:
            it["result_ref"] = result_ref
        if feedback is not None:
            it["feedback"] = feedback
        if evidence_ids is not None:
            it["evidence_ids"] = list(evidence_ids)
        break
    return state
