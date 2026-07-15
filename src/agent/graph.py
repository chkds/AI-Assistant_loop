"""Phase-2 cyclic agent graph with checkpoints and interrupt checks."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from src.agent.checkpoint import save_checkpoint
from src.agent.context_projection import merge_result, project_context, projection_to_broker_text
from src.agent.file_guard import take_snapshot
from src.agent.interrupt import InterruptedError, check_interrupt
from src.agent.orchestrator import (
    apply_item_feedback,
    compose_final_answer,
    orchestrate_step,
    request_supplement,
)
from src.agent.reflect import decide_reflect, should_compress
from src.agent.session import SessionStore
from src.agent.tool_io import apply_tool_io
from src.agent.work_items import current_work_item, mark_item, select_next_work_item
from src.control.retrieval_gate import RetrievalGate
from src.ingest.chunker.multimodal import count_tokens
from src.llm.client import get_chat_client
from src.memory.broker import assemble_context
from src.memory.error_memory import record_error
from src.memory.pinned import extract_pinned_refs, inject_pinned_evidence
from src.tools.registry import ToolRegistry, get_registry
from src.tools.search_stack import call_with_search_fallback
from src.tasks.registry import get_task_registry
from src import load_agent


def _llm_json(prompt: str, tier: str = "lite") -> dict[str, Any]:
    client = get_chat_client(tier)
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
        return {"raw": raw}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"raw": raw}


def _add_evidence(state: dict[str, Any], item: dict[str, Any]) -> None:
    item = {**item, "id": item.get("id") or uuid.uuid4().hex[:10]}
    state.setdefault("evidence", []).append(item)


class AgentRuntime:
    """Binds a SessionStore into graph node functions."""

    def __init__(
        self,
        store: SessionStore,
        state: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.store = store
        self.registry = registry or get_registry()
        self.task_registry = get_task_registry()
        self.agent_cfg = load_agent()
        self.max_steps = int(self.agent_cfg.get("max_steps", 12))
        self.task_spec = None
        if state and state.get("task_type"):
            self.task_spec = self.task_registry.get(state["task_type"])
            if self.task_spec:
                base = registry or get_registry()
                self.registry = base.filter_by_allowlist(self.task_spec.allowed_tools)
                if self.task_spec.max_steps:
                    self.max_steps = int(self.task_spec.max_steps)

    def _save(self, state: dict[str, Any], node: str | None = None) -> dict[str, Any]:
        if node and node in set(self.agent_cfg.get("stable_nodes") or []):
            save_checkpoint(self.store, state, node)
        else:
            self.store.save_state(state)
        return state

    def _evt(self, state: dict[str, Any], event: dict[str, Any], *, actor: str = "worker") -> None:
        self.store.append_event({**event, "actor": actor}, state=state)

    def _rebind_task(self, state: dict[str, Any]) -> None:
        """Bind TaskSpec/registry to current work_item (worker policy, not orchestrator)."""
        item = current_work_item(state)
        task_type = (item or {}).get("task_type") or state.get("task_type") or "general_qa"
        domain = state.get("domain") or "general"
        spec = self.task_registry.resolve(task_type=task_type, domain=domain)
        self.task_spec = spec
        state["task_type"] = spec.task_type
        state["domain"] = spec.domain
        if item:
            state["knowledge_mode"] = spec.default_knowledge_mode
            state["goal"] = item.get("goal") or state.get("goal")
        else:
            state["knowledge_mode"] = state.get("knowledge_mode") or spec.default_knowledge_mode
        state["task_planner_block"] = spec.planner_block()
        self.registry = get_registry().filter_by_allowlist(spec.allowed_tools)
        if spec.max_steps:
            self.max_steps = int(spec.max_steps)

    def classify(self, state: dict[str, Any]) -> dict[str, Any]:
        check_interrupt(self.store, state)
        state["status"] = "running"
        state["step_count"] = int(state.get("step_count") or 0) + 1
        query = state.get("goal") or state.get("query") or ""
        task_names = self.task_registry.names()
        data = _llm_json(
            "Classify a soft hint for orchestration (do NOT force matlab/finance unless clearly needed). "
            "Return JSON keys: task_type "
            f"(one of {task_names} or general_qa), domain, complexity "
            "(simple|standard|hard), knowledge_mode (none|pinned|retrieve), "
            f"needs_tools (list of tool names). Task:\n{query}"
        )
        task_type = data.get("task_type") or "general_qa"
        domain = data.get("domain") or "general"
        # Soft hint only — orchestrator owns the schedule
        state["task_type"] = task_type
        state["domain"] = domain
        state["complexity"] = data.get("complexity") or "simple"
        state["classify_hint"] = data
        state["tier"] = "lite"
        self._evt(
            state,
            {"type": "node", "node": "classify", "data": data, "task_type": task_type},
            actor="orchestrator",
        )
        return self._save(state, "classify")

    def orchestrate(self, state: dict[str, Any]) -> dict[str, Any]:
        """Decompose / bind current work_item. MATLAB only if coding item exists."""
        check_interrupt(self.store, state)
        state["step_count"] = int(state.get("step_count") or 0) + 1
        state = orchestrate_step(state, available_task_types=self.task_registry.names())
        self._rebind_task(state)
        item = current_work_item(state)
        self._evt(
            state,
            {
                "type": "node",
                "node": "orchestrate",
                "work_items": [
                    {"id": i.get("id"), "title": i.get("title"), "task_type": i.get("task_type"), "status": i.get("status")}
                    for i in (state.get("work_items") or [])
                ],
                "current": (item or {}).get("id"),
                "task_type": state.get("task_type"),
            },
            actor="orchestrator",
        )
        return self._save(state, "orchestrate")

    def gate(self, state: dict[str, Any]) -> dict[str, Any]:
        check_interrupt(self.store, state)
        state["step_count"] = int(state.get("step_count") or 0) + 1
        if self.task_spec is None:
            self._rebind_task(state)
        # Merge pinned refs from query / API
        q = str(state.get("goal") or state.get("query") or "")
        found = extract_pinned_refs(q)
        if found:
            pins = list(state.get("pinned_docs") or [])
            for r in found:
                if r not in pins:
                    pins.append(r)
            state["pinned_docs"] = pins
        pins = list(state.get("pinned_docs") or [])
        # Pins win over TaskSpec default retrieve (rebind may have set knowledge_mode)
        if pins:
            force = "pinned"
        elif self.task_spec and self.task_spec.default_knowledge_mode == "none":
            force = "none"
        else:
            force = state.get("knowledge_mode")
        gate = RetrievalGate()
        decision = gate.decide(q, pinned_docs=pins, force_mode=force)
        state["knowledge_mode"] = decision.knowledge_mode
        state["gate_reason"] = decision.reason
        state["gate_pinned_docs"] = list(decision.pinned_docs or pins)
        self._evt(
            state,
            {
                "type": "node",
                "node": "gate",
                "mode": decision.knowledge_mode,
                "pinned_docs": state.get("pinned_docs") or [],
            },
        )
        return self._save(state, "gate")

    def broker(self, state: dict[str, Any]) -> dict[str, Any]:
        check_interrupt(self.store, state)
        state["step_count"] = int(state.get("step_count") or 0) + 1
        mode = state.get("knowledge_mode")
        if mode == "pinned" and state.get("pinned_docs"):
            from src.control.retrieval_gate import RetrievalGate

            budget = RetrievalGate().safe_budget
            state = inject_pinned_evidence(state, max_tokens=budget, add_evidence=_add_evidence)
            self._evt(
                state,
                {
                    "type": "node",
                    "node": "broker",
                    "mode": "pinned",
                    "pinned_ids": state.get("broker_pinned_ids") or [],
                },
            )
        elif mode == "retrieve" and "kb_retrieve" in self.registry.names():
            has_kb = any(
                e.get("source_type") == "kb_body" and e.get("has_body")
                for e in state.get("evidence") or []
                if e.get("id") not in set(state.get("stale_evidence_ids") or [])
            )
            if not has_kb:
                args = {"query": state.get("goal") or state.get("query")}
                result = self.registry.call("kb_retrieve", args)
                spec = self.registry.get("kb_retrieve")
                apply_tool_io(
                    state,
                    tool="kb_retrieve",
                    arguments=args,
                    result=result,
                    evidence_kind=spec.evidence_kind if spec else "kb_body",
                    add_evidence=_add_evidence,
                )
        projection = project_context(state)
        state["projection"] = projection
        # Worker sees projection + light assemble_context (not full unrelated item noise)
        base = assemble_context(state)
        state["broker_context"] = projection_to_broker_text(projection) + "\n\n---\n" + base[:8000]
        return self._save(state)

    def plan(self, state: dict[str, Any]) -> dict[str, Any]:
        check_interrupt(self.store, state)
        state["step_count"] = int(state.get("step_count") or 0) + 1
        if self.task_spec is None or state.get("current_work_item_id"):
            self._rebind_task(state)
        tools = self.registry.schemas()
        ctx = state.get("broker_context") or assemble_context(state)
        task_block = state.get("task_planner_block") or ""
        item = current_work_item(state)
        item_line = ""
        if item:
            item_line = (
                f"Current work_item id={item.get('id')} title={item.get('title')} "
                f"expect={item.get('expect')} acceptance={item.get('acceptance')}\n"
            )
        data = _llm_json(
            "You are a planner for the CURRENT work_item only (orchestrator scheduled it).\n"
            f"{item_line}"
            f"TaskSpec:\n{task_block}\n"
            f"Available tools (JSON schemas): {json.dumps(tools, ensure_ascii=False)}\n"
            "Evidence must eventually include body text (kb_body, web_body, or mcp with has_body), "
            "not title/snippet alone. After tavily_search you MUST plan fetch_page. "
            "MCP tools (names mcp_*) return body via their tool output when successful.\n"
            "Only call tools listed in Available tools. Do not invent MATLAB runs unless this "
            "work_item is matlab_assist / code.\n"
            "Return JSON: {\"action\":\"tool|respond\", \"tool\":str|null, "
            "\"arguments\":{}, \"rationale\":str, \"done_hint\":bool}\n"
            f"Context:\n{ctx[:12000]}"
        )
        # Enforce allowlist at plan time
        if data.get("action") == "tool" and data.get("tool"):
            if data["tool"] not in self.registry.names():
                data = {
                    "action": "respond",
                    "tool": None,
                    "arguments": {},
                    "rationale": f"blocked_tool:{data.get('tool')}",
                }
        state["next_action"] = data
        plans = list(state.get("plan") or [])
        plans.append(data)
        state["plan"] = plans[-20:]
        self._evt(state, {"type": "node", "node": "plan", "action": data})
        return self._save(state, "plan")

    def act(self, state: dict[str, Any]) -> dict[str, Any]:
        check_interrupt(self.store, state)
        state["step_count"] = int(state.get("step_count") or 0) + 1
        action = state.get("next_action") or {}
        kind = action.get("action") or "respond"
        obs: dict[str, Any]
        if kind == "tool":
            tool = str(action.get("tool") or "")
            args = action.get("arguments") or {}
            stack = list(self.task_spec.search_stack) if self.task_spec else []
            used_tool, result = call_with_search_fallback(
                registry=self.registry,
                tool=tool,
                arguments=args,
                search_stack=stack,
            )
            if used_tool != tool:
                state.setdefault("search_fallbacks", []).append(
                    {"from": tool, "to": used_tool}
                )
                tool = used_tool
            spec = self.registry.get(tool)
            obs = apply_tool_io(
                state,
                tool=tool,
                arguments=args,
                result=result,
                evidence_kind=spec.evidence_kind if spec else None,
                add_evidence=_add_evidence,
            )
            # Attach new evidence ids to current work_item
            item = current_work_item(state)
            if item and obs.get("evidence_ids"):
                ids = list(item.get("evidence_ids") or [])
                for eid in obs["evidence_ids"]:
                    if eid not in ids:
                        ids.append(eid)
                mark_item(state, item["id"], evidence_ids=ids)
        else:
            ctx = state.get("broker_context") or assemble_context(state)
            client = get_chat_client("lite")
            answer = client.chat(
                [
                    {
                        "role": "system",
                        "content": "Answer using provided evidence bodies. Cite sources. "
                        "If evidence is only titles/snippets, say evidence is insufficient.",
                    },
                    {"role": "user", "content": ctx[:14000]},
                ]
            )
            state["final_answer"] = answer
            obs = {"type": "respond", "answer_preview": answer[:500]}
        state["last_observation"] = obs
        self._evt(
            state,
            {
                "type": "node",
                "node": "act",
                "obs_type": obs.get("type"),
                "tool": obs.get("tool"),
                "ok": obs.get("ok"),
                "evidence_ids": obs.get("evidence_ids"),
                "evidence_kind": obs.get("evidence_kind"),
            },
        )
        # act is not a stable checkpoint
        self.store.save_state(state)
        return state

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        check_interrupt(self.store, state)
        state["step_count"] = int(state.get("step_count") or 0) + 1
        obs = state.get("last_observation") or {}
        self._evt(
            state,
            {
                "type": "node",
                "node": "observe",
                "obs_type": obs.get("type"),
                "tool": obs.get("tool"),
                "evidence_ids": obs.get("evidence_ids"),
                "evidence_count": len(state.get("evidence") or []),
            },
        )
        return self._save(state, "observe")

    def reflect(self, state: dict[str, Any]) -> dict[str, Any]:
        check_interrupt(self.store, state)
        state["step_count"] = int(state.get("step_count") or 0) + 1
        cfg = self.agent_cfg.get("reflect", {})
        reject_title = bool(cfg.get("reject_title_only_evidence", True))
        min_body = int(cfg.get("require_body_chars", 200))
        require_body = True
        if self.task_spec is None and state.get("task_type"):
            self.task_spec = self.task_registry.get(state["task_type"])
        if self.task_spec and self.task_spec.require_body_evidence:
            reject_title = True
            require_body = True
        if self.task_spec and not self.task_spec.require_body_evidence:
            require_body = False
            reject_title = False

        rules = list(self.task_spec.reflect_rules) if self.task_spec else None
        preliminary = decide_reflect(
            state,
            reject_title_only=reject_title,
            min_body_chars=min_body,
            require_body=require_body,
            max_steps=self.max_steps,
            reflect_rules=rules,
        )
        decision = preliminary["decision"]
        reason = preliminary["reason"]
        target_id = state.get("current_work_item_id")

        if decision == "need_fetch":
            record_error(
                error_type="execution_error",
                description="Attempted final answer with snippet-only web evidence",
                correction="Must fetch_page for body before done",
                session_id=self.store.session_id,
            )
        elif decision == "llm_judge":
            active = [
                e
                for e in state.get("evidence") or []
                if e.get("id") not in set(state.get("stale_evidence_ids") or []) and not e.get("stale")
            ]
            data = _llm_json(
                "Judge if the CURRENT work_item answer is adequate given evidence. "
                "Return JSON {\"done\":bool,\"need_fetch\":bool,\"replan\":bool,"
                "\"supplement_prior\":bool,\"reason\":str}. "
                "supplement_prior=true if an earlier research item should add evidence.\n"
                f"Work item: {current_work_item(state)}\n"
                f"Answer:\n{state.get('final_answer','')[:3000]}\n"
                f"Evidence types: {[e.get('source_type') for e in active]}"
            )
            if data.get("need_fetch"):
                decision = "need_fetch"
                reason = data.get("reason") or "llm_need_fetch"
            elif data.get("supplement_prior") and target_id:
                # Reopen a prior dependency (1:N / back-edge)
                deps = list((current_work_item(state) or {}).get("depends_on") or [])
                if deps:
                    state = request_supplement(
                        state,
                        from_item_id=target_id,
                        target_item_id=deps[-1],
                        reason=str(data.get("reason") or "supplement_prior"),
                    )
                    decision = "next_item"
                    reason = "supplement_prior"
                    target_id = deps[-1]
                    # Keep needs_revise on the reopened prior (do not clobber to running)
                    self._rebind_task(state)
                else:
                    decision = "replan"
                    reason = data.get("reason") or "llm_replan"
            elif data.get("replan"):
                decision = "replan"
                reason = data.get("reason") or "llm_replan"
                if target_id:
                    apply_item_feedback(
                        state,
                        target_work_item_id=target_id,
                        decision="revise",
                        reasons=[reason],
                        amendment_type="planning_error",
                    )
            elif data.get("done"):
                decision = "done"
                reason = data.get("reason") or "llm_done"
            else:
                decision = "continue"
                reason = data.get("reason") or "llm_continue"
        elif decision == "done" and reason == "max_steps" and not state.get("final_answer"):
            state["final_answer"] = str((state.get("last_observation") or {}))

        # Complete current work_item and advance schedule (not session-end yet)
        if decision == "done" and target_id and state.get("work_items"):
            state = merge_result(
                state,
                target_id,
                {
                    "status": "ok",
                    "outputs": {"final_answer": state.get("final_answer") or ""},
                    "evidence_delta": [],
                },
            )
            nxt = select_next_work_item(state)
            if nxt is not None:
                state["current_work_item_id"] = nxt["id"]
                nxt["status"] = "running"
                state["final_answer"] = ""
                state["next_action"] = None
                self._rebind_task(state)
                decision = "next_item"
                reason = f"advance_to:{nxt['id']}"
            else:
                state["final_answer"] = compose_final_answer(state) or state.get("final_answer")
                state["status"] = "done"
        elif decision == "done" and not state.get("work_items"):
            state["status"] = "done"

        state["reflect_decision"] = {
            "decision": decision,
            "reason": reason,
            "target_work_item_id": target_id,
        }
        self._evt(
            state,
            {
                "type": "node",
                "node": "reflect",
                "decision": decision,
                "reason": reason,
                "target_work_item_id": target_id,
            },
            actor="orchestrator",
        )
        return self._save(state, "reflect")

    def compress(self, state: dict[str, Any]) -> dict[str, Any]:
        check_interrupt(self.store, state)
        state["step_count"] = int(state.get("step_count") or 0) + 1
        every = int(self.agent_cfg.get("compress_every_n_steps", 4))
        ratio = float(self.agent_cfg.get("compress_token_ratio", 0.70))
        ctx = assemble_context(state)
        token_limit = int(32000 * 0.35 * ratio)
        need = should_compress(
            state,
            every_n=every,
            token_count=count_tokens(ctx),
            token_limit=token_limit,
        )
        if need:
            data = _llm_json(
                "Compress agent state. Return JSON "
                "{\"summary\":str,\"open_questions\":[str],\"key_evidence_ids\":[str]}.\n"
                f"Context:\n{ctx[:10000]}"
            )
            state["compress_summary"] = data.get("summary") or str(data)[:2000]
            # drop verbose messages
            state["messages"] = []
        return self._save(state, "compress")


def route_after_reflect(state: dict[str, Any]) -> Literal["compress", "plan", "done", "act", "broker"]:
    d = (state.get("reflect_decision") or {}).get("decision")
    if d == "done":
        return "done"
    if d == "next_item":
        # Re-project context for the newly bound work_item
        return "broker"
    if d == "need_fetch":
        # force next action to fetch if we have snippet urls
        urls = []
        for e in state.get("evidence") or []:
            if e.get("source_type") == "web_snippet" and e.get("url"):
                urls.append(e["url"])
        if urls:
            state["next_action"] = {
                "action": "tool",
                "tool": "fetch_page",
                "arguments": {"urls": urls[:2]},
                "rationale": "reflect_need_fetch",
            }
            return "act"
        return "plan"
    if d == "replan":
        return "plan"
    return "compress"


def route_after_compress(state: dict[str, Any]) -> Literal["plan", "done"]:
    if state.get("status") == "done":
        return "done"
    if int(state.get("step_count") or 0) >= int(load_agent().get("max_steps", 12)):
        state["status"] = "done"
        return "done"
    return "plan"


def build_agent_graph(store: SessionStore, state: dict[str, Any] | None = None):
    rt = AgentRuntime(store, state)

    def done_node(state: dict[str, Any]) -> dict[str, Any]:
        state["status"] = "done"
        if state.get("work_items") and not state.get("final_answer"):
            state["final_answer"] = compose_final_answer(state)
        take_snapshot(store, state)
        store.append_event({"type": "done", "actor": "orchestrator"}, state=state)
        store.save_state(state)
        return state

    graph = StateGraph(dict)
    graph.add_node("classify", rt.classify)
    graph.add_node("orchestrate", rt.orchestrate)
    graph.add_node("gate", rt.gate)
    graph.add_node("broker", rt.broker)
    graph.add_node("plan", rt.plan)
    graph.add_node("act", rt.act)
    graph.add_node("observe", rt.observe)
    graph.add_node("reflect", rt.reflect)
    graph.add_node("compress", rt.compress)
    graph.add_node("done", done_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "orchestrate")
    graph.add_edge("orchestrate", "gate")
    graph.add_edge("gate", "broker")
    graph.add_edge("broker", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "observe")
    graph.add_edge("observe", "reflect")
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "compress": "compress",
            "plan": "plan",
            "done": "done",
            "act": "act",
            "broker": "broker",
        },
    )
    graph.add_conditional_edges(
        "compress",
        route_after_compress,
        {"plan": "plan", "done": "done"},
    )
    graph.add_edge("done", END)
    return graph.compile()


def run_session(
    store: SessionStore,
    state: dict[str, Any] | None = None,
    *,
    start_node: str | None = None,
) -> dict[str, Any]:
    """Run or resume. start_node used after rollback (re-enter from classify/gate/plan...)."""
    if state is None:
        state = store.load_state()
    store.set_interrupt(False)
    state["interrupt_flag"] = False
    state["status"] = "running"
    take_snapshot(store, state)
    store.save_state(state)

    app = build_agent_graph(store, state)
    # LangGraph invoke always starts at entry; for resume from mid-graph we
    # partially replay by setting resume_from and using a wrapper entry.
    resume = start_node or state.get("resume_from")
    try:
        if resume and resume != "classify":
            state = _resume_from(store, state, resume)
        else:
            state = dict(app.invoke(state))
    except InterruptedError:
        state = store.load_state()
        state["status"] = "interrupted"
        store.save_state(state)
    except Exception as exc:  # noqa: BLE001
        state = store.load_state()
        state["status"] = "failed"
        state["last_observation"] = {"type": "error", "error": str(exc)}
        store.append_event({"type": "failed", "error": str(exc)})
        store.save_state(state)
        record_error(
            error_type="execution_error",
            description=str(exc),
            session_id=store.session_id,
        )
    return store.load_state()


def _resume_from(store: SessionStore, state: dict[str, Any], node: str) -> dict[str, Any]:
    """Continue loop from a mid node without re-classify when possible."""
    rt = AgentRuntime(store, state)
    # Map resume node to starting function sequence in a manual loop
    start_idx = {
        "classify": 0,
        "orchestrate": 1,
        "gate": 2,
        "broker": 3,
        "plan": 4,
        "observe": 6,
        "reflect": 7,
        "compress": 8,
    }.get(node, 4)

    if start_idx <= 0:
        state = rt.classify(state)
        start_idx = 1
    if start_idx <= 1:
        state = rt.orchestrate(state)
        start_idx = 2
    if start_idx <= 2:
        state = rt.gate(state)
        start_idx = 3
    if start_idx <= 3:
        state = rt.broker(state)
        start_idx = 4

    while state.get("status") not in {"done", "interrupted", "failed", "awaiting_confirm"}:
        check_interrupt(store, state)
        if int(state.get("step_count") or 0) >= rt.max_steps:
            state["status"] = "done"
            break
        state = rt.plan(state)
        state = rt.act(state)
        state = rt.observe(state)
        state = rt.reflect(state)
        d = (state.get("reflect_decision") or {}).get("decision")
        if d == "done":
            state["status"] = "done"
            take_snapshot(store, state)
            break
        if d == "next_item":
            state = rt.broker(state)
            continue
        if d == "need_fetch":
            urls = [
                e.get("url")
                for e in state.get("evidence") or []
                if e.get("source_type") == "web_snippet" and e.get("url")
            ]
            if urls:
                state["next_action"] = {
                    "action": "tool",
                    "tool": "fetch_page",
                    "arguments": {"urls": urls[:2]},
                }
                state = rt.act(state)
                state = rt.observe(state)
                continue
        state = rt.compress(state)
    store.save_state(state)
    return state


# Back-compat echo API
def run_echo(query: str) -> dict[str, Any]:
    return dict(build_echo_graph_legacy().invoke({"query": query}))


def build_echo_graph_legacy():
    from typing import TypedDict

    class AgentState(TypedDict, total=False):
        query: str
        knowledge_mode: str
        tier: str
        echo: str
        status: str

    def classify_stub(state: AgentState) -> AgentState:
        q = state.get("query", "")
        return {
            **state,
            "knowledge_mode": "retrieve" if q.strip() else "none",
            "tier": "lite",
            "status": "classified",
        }

    def echo_node(state: AgentState) -> AgentState:
        return {
            **state,
            "echo": f"[echo] mode={state.get('knowledge_mode')} query={state.get('query')!r}",
            "status": "echoed",
        }

    def done_node(state: AgentState) -> AgentState:
        return {**state, "status": "done"}

    g = StateGraph(AgentState)
    g.add_node("classify", classify_stub)
    g.add_node("echo", echo_node)
    g.add_node("done", done_node)
    g.set_entry_point("classify")
    g.add_edge("classify", "echo")
    g.add_edge("echo", "done")
    g.add_edge("done", END)
    return g.compile()
