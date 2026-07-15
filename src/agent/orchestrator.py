"""Orchestrator — understand, decompose, schedule, request supplements (F4).

Does not execute tools. TaskSpec remains worker policy; this module owns the
work graph. Relationships are not 1:1: one user goal → many work_items;
later items may reopen earlier ones (needs_revise / depends_on).
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.work_items import (
    all_terminal,
    mark_item,
    new_work_item,
    refresh_ready,
    select_next_work_item,
)
from src.llm.client import get_chat_client


def _llm_json(prompt: str) -> dict[str, Any]:
    try:
        client = get_chat_client("lite")
        raw = client.chat(
            [
                {
                    "role": "system",
                    "content": "Reply with a single JSON object only. No markdown fences.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {}


def _deep_macro_commodity_chain(query: str) -> list[dict[str, Any]] | None:
    """Multi-facet research chain for geopolitics / trade / US data ↔ Au/Cu futures.

    Returns None when the query is not this class of work (caller falls through).
    """
    q = (query or "").strip()
    ql = q.lower()
    driver_hits = sum(
        1
        for k in (
            "地域冲突",
            "地缘",
            "冲突",
            "战争",
            "大宗商品",
            "贸易",
            "经济指标",
            "非农",
            "美联储",
            "cpi",
            "fomc",
            "利率",
            "geopolit",
            "conflict",
            "commodity",
        )
        if k in q or k in ql
    )
    metal_hits = sum(
        1
        for k in ("黄金", "金价", "铜", "铜价", "期货", "gold", "copper", "futures", "xau", "hg")
        if k in q or k in ql
    )
    link_hits = sum(
        1 for k in ("关联", "对照", "核查", "影响", "correlation", "relate") if k in q or k in ql
    )
    if metal_hits < 1 or driver_hits < 1:
        return None
    # Prefer deep chain when user asks to relate events ↔ prices
    if link_hits < 1 and driver_hits + metal_hits < 3:
        return None

    facets = [
        (
            "Geopolitics past year",
            "梳理近一年主要地域/地缘冲突与升级节点（时间线、涉事方、对供应链/风险偏好的直接影响）",
            "evidence",
            "Timeline + named conflicts with dates; prefer sources from last 12–18 months",
        ),
        (
            "Commodity trade flows",
            "梳理近一年大宗商品贸易格局变化（能源/金属相关制裁、出口限制、航运与库存信号）",
            "evidence",
            "Trade/policy facts with dates; link to industrial metals where possible",
        ),
        (
            "US macro releases",
            "梳理近一年美国重要经济指标与政策发布（非农、CPI、GDP、FOMC/利率决议等）及市场解读要点",
            "evidence",
            "Release calendar hits with dates; note gold/USD and risk-asset reactions if cited",
        ),
        (
            "Au/Cu futures moves",
            "核查近一年黄金、铜期货价格的主要波动阶段（高低点、幅度、时间窗口），并记录同期叙事",
            "evidence",
            "Price windows with dates/levels from reputable market sources; no invented quotes",
        ),
    ]
    items: list[dict[str, Any]] = []
    facet_ids: list[str] = []
    for title, goal, expect, acceptance in facets:
        wi = new_work_item(
            title=title,
            task_type="web_research",
            goal=goal,
            acceptance=acceptance,
            expect=expect,
            assignee_role="research",
            status="ready",
            depends_on=[],
        )
        items.append(wi)
        facet_ids.append(wi["id"])

    items.append(
        new_work_item(
            title="Correlate events vs Au/Cu",
            task_type="web_research",
            goal=(
                "对照前述冲突、贸易与美国指标证据，归纳黄金/铜期货波动与事件的关联（同向/滞后/无关），"
                "标明证据 id 与时间对齐；证据不足则指出缺口而非编造。"
            ),
            depends_on=facet_ids,
            acceptance="Cites prior work_item evidence; separates correlation narrative from proven causation",
            expect="final_answer",
            assignee_role="research",
            status="blocked",
        )
    )
    return items


def _rule_decompose(query: str, hint_task_type: str | None = None) -> list[dict[str, Any]]:
    """Deterministic decompose — no LLM. Avoids scheduling every pack."""
    q = (query or "").strip()
    ql = q.lower()
    items: list[dict[str, Any]] = []

    needs_matlab = any(
        k in ql or k in q
        for k in (
            "matlab",
            ".m ",
            "simulink",
            "写个matlab",
            "matlab代码",
            "matlab 代码",
            "run matlab",
        )
    )
    needs_web = any(
        k in ql or k in q
        for k in (
            "搜索",
            "联网",
            "最新",
            "news",
            "web",
            "检索一下",
            "anysearch",
            "近一年",
            "核查",
            "对照",
            "期货",
        )
    )
    needs_kb = any(
        k in ql or k in q
        for k in ("论文", "知识库", "kb", "bufort", "文献", "paper", "corpus")
    )

    deep = _deep_macro_commodity_chain(q)
    if deep is not None and not needs_matlab:
        return deep

    # MATLAB only when the query itself needs coding — never because classify hinted it
    if needs_matlab and not needs_kb and not needs_web:
        items.append(
            new_work_item(
                title="MATLAB code assist",
                task_type="matlab_assist",
                goal=q,
                acceptance="Runnable code or clear execution log; self-check if possible",
                expect="artifact",
                assignee_role="code_matlab",
                status="ready",
            )
        )
        return items

    if needs_kb or hint_task_type == "research_qa":
        items.append(
            new_work_item(
                title="Local KB research",
                task_type="research_qa",
                goal=q,
                acceptance="Answer cites kb_body evidence",
                expect="final_answer",
                assignee_role="research",
                status="ready",
            )
        )

    if needs_web or hint_task_type == "web_research":
        dep = [items[0]["id"]] if items else []
        items.append(
            new_work_item(
                title="Web / AnySearch research",
                task_type="web_research",
                goal=q,
                depends_on=dep,
                acceptance="Body evidence (mcp/web_body) before final",
                expect="evidence",
                assignee_role="research",
                status="pending" if dep else "ready",
            )
        )

    if needs_matlab:
        dep = [items[-1]["id"]] if items else []
        items.append(
            new_work_item(
                title="MATLAB coding assist",
                task_type="matlab_assist",
                goal=f"Implement/verify related to: {q}",
                depends_on=dep,
                acceptance="Code path or eval log; failures reopen this item",
                expect="artifact",
                assignee_role="code_matlab",
                status="pending" if dep else "ready",
            )
        )

    if not items:
        tt = hint_task_type if hint_task_type and hint_task_type != "matlab_assist" else "general_qa"
        items.append(
            new_work_item(
                title="Primary task",
                task_type=tt,
                goal=q,
                acceptance="Adequate answer for the user goal",
                expect="final_answer",
                assignee_role=tt,
                status="ready",
            )
        )
    return items


def decompose(
    query: str,
    *,
    hint_task_type: str | None = None,
    available_task_types: list[str] | None = None,
    use_llm: bool = False,
) -> list[dict[str, Any]]:
    """Build work_items for a user instruction. Prefer rules; optional LLM refine."""
    items = _rule_decompose(query, hint_task_type=hint_task_type)
    allowed = set(available_task_types or [])
    if allowed:
        for it in items:
            if it["task_type"] not in allowed:
                it["task_type"] = "general_qa" if "general_qa" in allowed else next(iter(allowed))

    if not use_llm:
        return items

    data = _llm_json(
        "Decompose the user goal into 1-4 work items. "
        "MATLAB/code only if coding is required. Do NOT invent finance/matlab for pure QA.\n"
        f"Allowed task_types: {available_task_types or []}\n"
        "Return JSON: {\"items\":[{\"title\":str,\"task_type\":str,\"goal\":str,"
        "\"depends_on_titles\":[str],\"acceptance\":str,\"expect\":str}]}\n"
        f"User goal:\n{query}"
    )
    raw_items = data.get("items") if isinstance(data.get("items"), list) else None
    if not raw_items:
        return items

    built: list[dict[str, Any]] = []
    title_to_id: dict[str, str] = {}
    for raw in raw_items[:4]:
        title = str(raw.get("title") or "step")
        tt = str(raw.get("task_type") or "general_qa")
        if allowed and tt not in allowed:
            tt = "general_qa" if "general_qa" in allowed else items[0]["task_type"]
        wi = new_work_item(
            title=title,
            task_type=tt,
            goal=str(raw.get("goal") or query),
            acceptance=str(raw.get("acceptance") or ""),
            expect=str(raw.get("expect") or "final_answer"),
            assignee_role=tt,
            status="pending",
        )
        title_to_id[title] = wi["id"]
        built.append(wi)
    for raw, wi in zip(raw_items[:4], built):
        deps = []
        for t in raw.get("depends_on_titles") or []:
            if t in title_to_id and title_to_id[t] != wi["id"]:
                deps.append(title_to_id[t])
        wi["depends_on"] = deps
    # first without deps → ready
    for wi in built:
        wi["status"] = "ready" if not wi["depends_on"] else "blocked"
    return built or items


def bind_current_item(state: dict[str, Any]) -> dict[str, Any]:
    """Select or keep current work item; bind task_type for worker TaskSpec."""
    refresh_ready(state)
    cur_id = state.get("current_work_item_id")
    cur = None
    if cur_id:
        for it in state.get("work_items") or []:
            if it.get("id") == cur_id and it.get("status") in {"ready", "running", "needs_revise"}:
                cur = it
                break
    if cur is None:
        cur = select_next_work_item(state)
    if cur is None:
        state["current_work_item_id"] = None
        return state
    cur["status"] = "running"
    state["current_work_item_id"] = cur["id"]
    state["task_type"] = cur.get("task_type") or state.get("task_type") or "general_qa"
    state["goal"] = cur.get("goal") or state.get("goal")
    return state


def request_supplement(
    state: dict[str, Any],
    *,
    from_item_id: str,
    target_item_id: str,
    reason: str,
) -> dict[str, Any]:
    """Later stage asks an earlier item to revise / add evidence (not 1:1 forward-only)."""
    mark_item(
        state,
        target_item_id,
        status="needs_revise",
        feedback={
            "from_item_id": from_item_id,
            "decision": "revise",
            "reasons": [reason],
            "kind": "supplement_request",
        },
    )
    # Current item waits on target again
    for it in state.get("work_items") or []:
        if it.get("id") == from_item_id:
            deps = list(it.get("depends_on") or [])
            if target_item_id not in deps:
                deps.append(target_item_id)
            it["depends_on"] = deps
            it["status"] = "blocked"
            break
    state["current_work_item_id"] = target_item_id
    refresh_ready(state)
    return state


def apply_item_feedback(
    state: dict[str, Any],
    *,
    target_work_item_id: str,
    decision: str,
    reasons: list[str] | None = None,
    amendment_type: str | None = None,
) -> dict[str, Any]:
    """Addressable feedback from Reflect / HITL (F5)."""
    reasons = reasons or []
    if decision == "accept":
        mark_item(state, target_work_item_id, status="done", feedback={"decision": decision, "reasons": reasons})
    elif decision in {"revise", "needs_input"}:
        mark_item(
            state,
            target_work_item_id,
            status="needs_revise",
            feedback={
                "decision": decision,
                "reasons": reasons,
                "amendment_type": amendment_type,
            },
        )
        state["current_work_item_id"] = target_work_item_id
    elif decision == "replan":
        # Clear graph; caller should re-decompose
        state["work_items"] = []
        state["current_work_item_id"] = None
        state["needs_recompose"] = True
    elif decision == "abort":
        mark_item(state, target_work_item_id, status="failed", feedback={"decision": decision, "reasons": reasons})
    refresh_ready(state)
    return state


def compose_final_answer(state: dict[str, Any]) -> str:
    answers = state.get("item_answers") or {}
    parts = []
    for it in state.get("work_items") or []:
        aid = it.get("id")
        if aid in answers:
            parts.append(f"### {it.get('title')}\n{answers[aid]}")
    if parts:
        return "\n\n".join(parts)
    return str(state.get("final_answer") or "")


def orchestrate_step(state: dict[str, Any], *, available_task_types: list[str] | None = None) -> dict[str, Any]:
    """Ensure work_items exist and bind the current one. Idempotent."""
    if state.get("needs_recompose") or not state.get("work_items"):
        state["work_items"] = decompose(
            str(state.get("query") or state.get("goal") or ""),
            hint_task_type=state.get("task_type"),
            available_task_types=available_task_types,
            use_llm=False,
        )
        state["needs_recompose"] = False
        state["current_work_item_id"] = None
    state = bind_current_item(state)
    if all_terminal(state) and not select_next_work_item(state):
        state["final_answer"] = compose_final_answer(state) or state.get("final_answer")
        state["status"] = "done"
    return state
