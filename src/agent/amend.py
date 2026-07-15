"""Amend classification and resume/rollback policy."""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.checkpoint import latest_plan_checkpoint, list_checkpoints, rollback_to
from src.agent.orchestrator import apply_item_feedback
from src.agent.session import SessionStore
from src.llm.client import get_chat_client
from src.memory.error_memory import record_error


AMEND_TYPES = {
    "preference",
    "enrichment",
    "goal_shift",
    "planning_error",
    "execution_error",
    "file_patch",
}


def _rule_classify(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("文件", "改了", "file", "edited", "修改了脚本", "改动了")):
        return "file_patch"
    if any(k in t for k in ("理解错", "目标改", "不是这个", "换个方向", "goal", "重新规划")):
        return "goal_shift"
    if any(k in t for k in ("步骤错", "计划错", "规划错", "plan wrong")):
        return "planning_error"
    if any(k in t for k in ("工具错", "执行错", "检索错", "抓取错", "execution")):
        return "execution_error"
    if any(k in t for k in ("语气", "风格", "简短", "详细一点", "preference")):
        return "preference"
    return "enrichment"


def classify_amend(text: str, use_llm: bool = True) -> dict[str, Any]:
    rule = _rule_classify(text)
    if not use_llm:
        return {"type": rule, "reason": "rule", "rollback_node": _default_node(rule)}
    try:
        client = get_chat_client("lite")
        prompt = (
            "Classify user amendment into exactly one of: "
            "preference, enrichment, goal_shift, planning_error, execution_error, file_patch.\n"
            "Return JSON: {\"type\":\"...\",\"reason\":\"...\"}\n"
            f"User text:\n{text}"
        )
        raw = client.chat([{"role": "user", "content": prompt}])
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0) if m else "{}")
        typ = data.get("type") if data.get("type") in AMEND_TYPES else rule
        return {
            "type": typ,
            "reason": data.get("reason") or "llm",
            "rollback_node": _default_node(typ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"type": rule, "reason": f"rule_fallback:{exc}", "rollback_node": _default_node(rule)}


def _default_node(typ: str) -> str | None:
    return {
        "preference": None,
        "enrichment": None,
        "goal_shift": "classify",
        "planning_error": "plan",
        "execution_error": "plan",
        "file_patch": None,
    }.get(typ)


def apply_amend(
    store: SessionStore,
    text: str,
    tags: list[str] | None = None,
    *,
    target_work_item_id: str | None = None,
) -> dict[str, Any]:
    state = store.load_state()
    classification = classify_amend(text)
    typ = classification["type"]
    target_id = target_work_item_id or state.get("current_work_item_id")
    # Keep event type="amend"; classification lives in amend_type (avoid clobbering type)
    entry = {
        "text": text,
        "tags": tags or [],
        "target_work_item_id": target_id,
        "amend_type": typ,
        "reason": classification.get("reason"),
        "rollback_node": classification.get("rollback_node"),
        "type": typ,  # retained on state.amendments for backward compat
    }
    state.setdefault("amendments", []).append(entry)
    # merge into goal
    state["goal"] = (state.get("goal") or state.get("query") or "") + f"\n[amend:{typ}] {text}"
    store.append_event(
        {
            "type": "amend",
            "actor": "human",
            "text": text,
            "tags": tags or [],
            "target_work_item_id": target_id,
            "amend_type": typ,
            "reason": classification.get("reason"),
            "rollback_node": classification.get("rollback_node"),
        },
        state=state,
    )
    record_error(
        error_type=typ if typ in {"planning_error", "execution_error", "goal_shift"} else "preference",
        description=f"user amend: {text[:500]}",
        correction=f"classified={typ}; rollback={classification.get('rollback_node')}; target={target_id}",
        context=str(state.get("query") or "")[:500],
        session_id=store.session_id,
    )

    # Addressable feedback on work_items (F5) — still may rollback checkpoints
    if target_id and typ == "goal_shift":
        apply_item_feedback(
            state,
            target_work_item_id=target_id,
            decision="replan",
            reasons=[text[:300]],
            amendment_type=typ,
        )
    elif target_id and typ in {"planning_error", "execution_error", "enrichment"}:
        apply_item_feedback(
            state,
            target_work_item_id=target_id,
            decision="revise",
            reasons=[text[:300]],
            amendment_type=typ,
        )

    if typ == "file_patch":
        state["status"] = "awaiting_confirm"
        state["pending_file_confirm"] = {"reason": "amend_file_patch", "text": text}
        store.save_state(state)
        return {"state": state, "action": "await_file_confirm", "classification": classification}

    node = classification.get("rollback_node")
    if node:
        # find latest checkpoint for that node
        target = None
        for cid in reversed(list_checkpoints(store)):
            if cid.endswith(f"_{node}"):
                target = cid
                break
        if target is None and node == "plan":
            target = latest_plan_checkpoint(store)
        if target:
            state = rollback_to(store, state, target)
            state["goal"] = (state.get("goal") or "") + f"\n[amend:{typ}] {text}"
            state.setdefault("amendments", []).append(entry)
            if target_id and typ in {"planning_error", "execution_error", "enrichment"}:
                apply_item_feedback(
                    state,
                    target_work_item_id=target_id,
                    decision="revise",
                    reasons=[text[:300]],
                    amendment_type=typ,
                )
            state["status"] = "interrupted"
            state["resume_from"] = state.get("resume_from") or classification.get("rollback_node") or "plan"
            state["resume_hint"] = state["resume_from"]
            store.save_state(state)
            return {
                "state": state,
                "action": "rollback",
                "to": target,
                "classification": classification,
                "target_work_item_id": target_id,
            }

    # preference / enrichment: continue from plan without rollback
    state["status"] = "interrupted"
    state["resume_from"] = "plan"
    state["resume_hint"] = "plan"
    store.save_state(state)
    return {
        "state": state,
        "action": "resume_plan",
        "classification": classification,
        "target_work_item_id": target_id,
    }
