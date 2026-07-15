"""E2E collaboration journey smoke — script I/O, no mental arithmetic.

Proves (SessionStore + AgentRuntime.reflect; LLM judge stubbed):
  1) Plain QA → zero matlab_assist work_items
  2) Multi work_item: depends_on → reflect next_item → item_answers + session done
  3) Reflect supplement_prior → request_supplement reopens prior (needs_revise)
  4) amend(target_work_item_id=…) writes type=amend event with target field

Usage:
  E:\\application\\miniforge3\\envs\\copilot-agent\\python.exe scripts\\smoke_work_items_journey.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import graph as graph_mod  # noqa: E402
from src.agent.amend import apply_amend  # noqa: E402
from src.agent.graph import AgentRuntime  # noqa: E402
from src.agent.orchestrator import decompose, orchestrate_step  # noqa: E402
from src.agent.session import SessionStore  # noqa: E402
from src.agent.work_items import current_work_item  # noqa: E402
from src.util.console_io import configure_stdout_utf8, dump_json, safe_print  # noqa: E402

configure_stdout_utf8()

# Two-item chain: web then matlab (no 论文 → avoids research_qa third item)
CHAIN_QUERY = "搜索最新新闻并写个 matlab 验证脚本"
ALLOWED = ["web_research", "matlab_assist", "research_qa", "general_qa"]


def _dump(label: str, obj: Any) -> None:
    dump_json(label, obj)


def _item_snap(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": i.get("id"),
            "title": i.get("title"),
            "task_type": i.get("task_type"),
            "status": i.get("status"),
            "depends_on": i.get("depends_on") or [],
        }
        for i in state.get("work_items") or []
    ]


def _read_events(store: SessionStore) -> list[dict[str, Any]]:
    if not store.events_path.exists():
        return []
    out = []
    for line in store.events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _patch_llm(responses: list[dict[str, Any]]) -> Callable[[], None]:
    queue = list(responses)
    orig = graph_mod._llm_json

    def fake(_prompt: str, tier: str = "lite") -> dict[str, Any]:  # noqa: ARG001
        if not queue:
            return {
                "done": True,
                "need_fetch": False,
                "replan": False,
                "supplement_prior": False,
                "reason": "stub_default_done",
            }
        return queue.pop(0)

    graph_mod._llm_json = fake  # type: ignore[assignment]

    def restore() -> None:
        graph_mod._llm_json = orig

    return restore


def _done_stub(reason: str) -> dict[str, Any]:
    return {
        "done": True,
        "need_fetch": False,
        "replan": False,
        "supplement_prior": False,
        "reason": reason,
    }


def _inject_body(state: dict[str, Any], eid: str = "ev_body") -> None:
    state["evidence"] = list(state.get("evidence") or [])
    state["evidence"].append(
        {
            "id": eid,
            "source_type": "web_body",
            "has_body": True,
            "text": ("synthetic body for smoke journey. " * 40).strip(),
            "url": "https://example.invalid/smoke",
        }
    )
    cur = current_work_item(state)
    if cur:
        ids = list(cur.get("evidence_ids") or [])
        if eid not in ids:
            ids.append(eid)
        cur["evidence_ids"] = ids


def check_plain_qa_no_matlab() -> dict[str, Any]:
    safe_print("\n=== CHECK 1: plain QA has no matlab_assist ===")
    query = "什么是门控检索？请用三句话解释。"
    items = decompose(query, hint_task_type="matlab_assist")
    types = [i["task_type"] for i in items]
    _dump("decompose_plain_qa", {"query": query, "types": types, "items": _item_snap({"work_items": items})})
    code_items = decompose("用 matlab 写一个求矩阵特征值的脚本")
    code_types = [i["task_type"] for i in code_items]
    _dump("decompose_coding", {"types": code_types})
    ok = "matlab_assist" not in types and "matlab_assist" in code_types
    safe_print("RESULT check1:", "PASS" if ok else "FAIL")
    return {"ok": ok, "qa_types": types, "code_types": code_types}


def check_multi_item_serial() -> dict[str, Any]:
    safe_print("\n=== CHECK 2: multi work_item serial (depend → next_item → compose) ===")
    restore = _patch_llm([_done_stub("stub_done_1"), _done_stub("stub_done_2")])
    store = SessionStore()
    state = store.new_state(CHAIN_QUERY)
    state = orchestrate_step(state, available_task_types=ALLOWED)
    store.save_state(state)
    snap0 = _item_snap(state)
    _dump(
        "after_orchestrate",
        {"session": store.session_id, "current": state.get("current_work_item_id"), "items": snap0},
    )

    items = state.get("work_items") or []
    if len(items) != 2:
        restore()
        safe_print("RESULT check2: FAIL (expected exactly 2 work_items, got", len(items), ")")
        return {"ok": False, "reason": "unexpected_item_count", "items": snap0}

    first_id = items[0]["id"]
    second_id = items[1]["id"]
    if second_id not in (items[1].get("id") for _ in [0]) or first_id not in (items[1].get("depends_on") or []):
        # second must depend on first
        if first_id not in (items[1].get("depends_on") or []):
            restore()
            safe_print("RESULT check2: FAIL (missing depends_on)")
            return {"ok": False, "reason": "no_depends_on", "items": snap0}

    rt = AgentRuntime(store, state)
    rt._rebind_task(state)
    _inject_body(state, "ev_1")
    state["next_action"] = {"action": "respond", "arguments": {}}
    state["final_answer"] = f"ANSWER_FOR_{first_id}"
    state["knowledge_mode"] = "none"
    state = rt.reflect(state)
    rd1 = dict(state.get("reflect_decision") or {})
    snap1 = _item_snap(state)
    _dump("after_reflect_item1", {"reflect": rd1, "current": state.get("current_work_item_id"), "items": snap1})

    ok_advance = rd1.get("decision") == "next_item" and state.get("current_work_item_id") == second_id
    ok_first_done = next(i for i in state["work_items"] if i["id"] == first_id).get("status") == "done"

    _inject_body(state, "ev_2")
    state["next_action"] = {"action": "respond", "arguments": {}}
    state["final_answer"] = f"ANSWER_FOR_{second_id}"
    rt._rebind_task(state)
    state = rt.reflect(state)
    rd2 = dict(state.get("reflect_decision") or {})
    snap2 = _item_snap(state)
    answers = dict(state.get("item_answers") or {})
    _dump(
        "after_reflect_item2",
        {
            "reflect": rd2,
            "status": state.get("status"),
            "items": snap2,
            "item_answers": answers,
            "final_answer_preview": (state.get("final_answer") or "")[:400],
        },
    )

    events = _read_events(store)
    reflect_ev = [
        e
        for e in events
        if e.get("type") == "node" and e.get("node") == "reflect"
    ]
    _dump(
        "reflect_node_events",
        [
            {
                "decision": e.get("decision"),
                "target_work_item_id": e.get("target_work_item_id"),
                "work_item_id": e.get("work_item_id"),
                "actor": e.get("actor"),
            }
            for e in reflect_ev
        ],
    )

    all_done = all(i.get("status") == "done" for i in state["work_items"])
    ok_answers = first_id in answers and second_id in answers
    ok_session = state.get("status") == "done" and rd2.get("decision") == "done"
    ok = ok_advance and ok_first_done and all_done and ok_answers and ok_session
    store.save_state(state)
    safe_print("session_dir:", store.root)
    safe_print("RESULT check2:", "PASS" if ok else "FAIL")
    restore()
    return {
        "ok": ok,
        "session_id": store.session_id,
        "ok_advance": ok_advance,
        "ok_first_done": ok_first_done,
        "all_done": all_done,
        "ok_answers": ok_answers,
        "ok_session": ok_session,
        "items_final": snap2,
        "item_answers": answers,
        "reflect": [rd1, rd2],
    }


def check_supplement_reopen() -> dict[str, Any]:
    safe_print("\n=== CHECK 3: reflect supplement → reopen prior (needs_revise) ===")
    restore = _patch_llm(
        [
            _done_stub("stub_done_prior"),
            {
                "done": False,
                "need_fetch": False,
                "replan": False,
                "supplement_prior": True,
                "reason": "need_more_numbers_from_prior",
            },
        ]
    )
    store = SessionStore()
    state = store.new_state(CHAIN_QUERY)
    state = orchestrate_step(state, available_task_types=ALLOWED)
    items = state.get("work_items") or []
    if len(items) < 2:
        restore()
        safe_print("RESULT check3: FAIL (need chain)")
        return {"ok": False, "reason": "no_chain"}

    prior_id = items[0]["id"]
    later_id = items[1]["id"]
    rt = AgentRuntime(store, state)
    rt._rebind_task(state)
    _inject_body(state, "ev_p")
    state["next_action"] = {"action": "respond", "arguments": {}}
    state["final_answer"] = "prior answer"
    state["knowledge_mode"] = "none"
    state = rt.reflect(state)
    _dump("after_prior_done", {"reflect": state.get("reflect_decision"), "items": _item_snap(state)})

    state["next_action"] = {"action": "respond", "arguments": {}}
    state["final_answer"] = "incomplete without prior numbers"
    _inject_body(state, "ev_l")
    state = rt.reflect(state)
    snap = _item_snap(state)
    rd = dict(state.get("reflect_decision") or {})
    prior_st = next(i for i in state["work_items"] if i["id"] == prior_id)
    later_st = next(i for i in state["work_items"] if i["id"] == later_id)
    _dump(
        "after_supplement",
        {
            "reflect": rd,
            "current": state.get("current_work_item_id"),
            "prior_status": prior_st.get("status"),
            "later_status": later_st.get("status"),
            "prior_feedback": prior_st.get("feedback"),
            "items": snap,
        },
    )

    ok = (
        rd.get("decision") == "next_item"
        and rd.get("reason") == "supplement_prior"
        and prior_st.get("status") == "needs_revise"
        and later_st.get("status") == "blocked"
        and state.get("current_work_item_id") == prior_id
        and (prior_st.get("feedback") or {}).get("kind") == "supplement_request"
    )
    store.save_state(state)
    safe_print("session_dir:", store.root)
    safe_print("RESULT check3:", "PASS" if ok else "FAIL")
    restore()
    return {
        "ok": ok,
        "session_id": store.session_id,
        "reflect": rd,
        "prior_status": prior_st.get("status"),
        "later_status": later_st.get("status"),
        "items": snap,
    }


def check_amend_target() -> dict[str, Any]:
    safe_print("\n=== CHECK 4: amend with target_work_item_id ===")
    store = SessionStore()
    state = store.new_state("普通问答会话用于 amend")
    state = orchestrate_step(state, available_task_types=["general_qa", "matlab_assist"])
    item = current_work_item(state)
    assert item is not None
    target = item["id"]
    store.save_state(state)

    result = apply_amend(
        store,
        "步骤错了，请重做当前分解项",
        tags=["smoke"],
        target_work_item_id=target,
        # classify_amend may call LLM; force rule path via monkeypatch below if needed
    )
    # If LLM classified oddly, still check event wiring; prefer planning_error via rule
    state = store.load_state()
    events = _read_events(store)
    amend_ev = [e for e in events if e.get("type") == "amend"]
    _dump(
        "amend_io",
        {
            "action": result.get("action"),
            "result_target": result.get("target_work_item_id"),
            "classification": result.get("classification"),
            "item_status": next(i.get("status") for i in state["work_items"] if i["id"] == target),
            "amend_events": [
                {
                    "type": e.get("type"),
                    "amend_type": e.get("amend_type"),
                    "target_work_item_id": e.get("target_work_item_id"),
                    "actor": e.get("actor"),
                    "work_item_id": e.get("work_item_id"),
                    "text": (e.get("text") or "")[:80],
                }
                for e in amend_ev
            ],
        },
    )

    ok_target = result.get("target_work_item_id") == target
    ok_event = any(
        e.get("type") == "amend"
        and e.get("target_work_item_id") == target
        and e.get("actor") == "human"
        and e.get("amend_type")
        for e in amend_ev
    )
    item_st = next(i for i in state["work_items"] if i["id"] == target)
    ok = ok_target and ok_event and item_st.get("status") == "needs_revise"
    safe_print("session_dir:", store.root)
    safe_print("RESULT check4:", "PASS" if ok else "FAIL")
    return {
        "ok": ok,
        "session_id": store.session_id,
        "target": target,
        "item_status": item_st.get("status"),
        "classification": result.get("classification"),
        "ok_target": ok_target,
        "ok_event": ok_event,
    }


def main() -> int:
    safe_print("=== smoke_work_items_journey ===")
    safe_print("cwd:", ROOT)
    results: dict[str, Any] = {}
    try:
        results["check1"] = check_plain_qa_no_matlab()
        results["check2"] = check_multi_item_serial()
        results["check3"] = check_supplement_reopen()
        results["check4"] = check_amend_target()
    except Exception as exc:  # noqa: BLE001
        safe_print("EXCEPTION:", exc)
        traceback.print_exc()
        results["exception"] = str(exc)
        _dump("RESULTS_PARTIAL", results)
        safe_print("RESULT: FAIL")
        return 2

    summary = {k: bool(v.get("ok")) for k, v in results.items() if isinstance(v, dict)}
    _dump("SUMMARY", summary)
    all_ok = all(summary.values()) and len(summary) == 4
    safe_print("\nRESULT:", "SUCCESS" if all_ok else "FAIL")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
