"""Pinned → broker injection smoke (script I/O, no LLM).

Usage:
  python scripts/smoke_pinned_broker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.graph import AgentRuntime, _add_evidence  # noqa: E402
from src.agent.session import SessionStore  # noqa: E402
from src.memory.pinned import extract_pinned_refs, load_pinned_documents  # noqa: E402
from src.util.console_io import configure_stdout_utf8, dump_json, safe_print  # noqa: E402

configure_stdout_utf8()


def main() -> int:
    safe_print("=== smoke_pinned_broker ===")
    q = "钉住 Bufort，解释其 GNN 传播建模思路"
    refs = extract_pinned_refs(q)
    dump_json("extracted_refs", refs)
    docs = load_pinned_documents(refs or ["Bufort"], max_tokens=3000)
    dump_json(
        "loaded",
        [
            {
                "doc_id": d.get("doc_id"),
                "has_body": d.get("has_body"),
                "chars": len(d.get("text") or ""),
                "error": d.get("error"),
            }
            for d in docs
        ],
    )
    if not docs or docs[0].get("error") == "not_found" or not docs[0].get("has_body"):
        safe_print("RESULT: FAIL (Bufort full.md not loadable)")
        return 2

    store = SessionStore()
    state = store.new_state(q)
    state["pinned_docs"] = refs or ["Bufort"]
    state["task_type"] = "research_qa"
    rt = AgentRuntime(store, state)
    rt._rebind_task(state)
    state = rt.gate(state)
    dump_json("gate", {"mode": state.get("knowledge_mode"), "reason": state.get("gate_reason"), "pins": state.get("pinned_docs")})
    state = rt.broker(state)
    pinned_ev = [e for e in state.get("evidence") or [] if e.get("pinned")]
    dump_json(
        "broker",
        {
            "pinned_evidence": len(pinned_ev),
            "broker_has_pinned": "pinned" in (state.get("broker_context") or ""),
            "preview": (state.get("broker_context") or "")[:400],
        },
    )
    ok = (
        state.get("knowledge_mode") == "pinned"
        and len(pinned_ev) >= 1
        and pinned_ev[0].get("has_body")
        and "pinned" in (state.get("broker_context") or "")
    )
    store.save_state(state)
    safe_print("session:", store.session_id)
    safe_print("RESULT:", "SUCCESS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
