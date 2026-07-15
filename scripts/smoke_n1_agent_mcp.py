"""N1 agent-path smoke: AnySearch + MATLAB MCP → evidence via act/observe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_yaml  # noqa: E402
from src.agent.graph import AgentRuntime  # noqa: E402
from src.agent.session import SessionStore  # noqa: E402
from src.tools.mcp_adapter import close_all_mcp_clients  # noqa: E402
from src.tools.registry import get_registry  # noqa: E402


def _run_tool(rt: AgentRuntime, state: dict, tool: str, arguments: dict) -> dict:
    state["next_action"] = {"action": "tool", "tool": tool, "arguments": arguments}
    state = rt.act(state)
    state = rt.observe(state)
    return state


def main() -> int:
    print("=== N1 agent MCP smoke ===")
    load_yaml.cache_clear()
    reg = get_registry(reload=True)
    names = set(reg.names())
    print("mcp tools:", sorted(n for n in names if n.startswith("mcp_")))

    ok_any = "mcp_anysearch_search" in names
    ok_mat = "mcp_matlab_evaluate_matlab_code" in names
    if not ok_any:
        print("FAIL: anysearch not registered")
        return 2
    if not ok_mat:
        print("FAIL: matlab lazy tools not registered")
        return 2

    store = SessionStore()
    state = store.new_state("N1 smoke anysearch+matlab")
    state["task_type"] = "web_research"
    rt = AgentRuntime(store, state, registry=reg)

    state = _run_tool(
        rt,
        state,
        "mcp_anysearch_search",
        {"query": "graph neural network wireless channel", "max_results": 2},
    )
    any_ev = [e for e in state.get("evidence") or [] if e.get("source_type") == "mcp"]
    print("after anysearch: evidence_mcp=", len(any_ev), "ok=", (state.get("last_observation") or {}).get("ok"))

    # Switch allowlist for matlab tools
    state["task_type"] = "matlab_assist"
    rt = AgentRuntime(store, state, registry=reg)
    state = _run_tool(
        rt,
        state,
        "mcp_matlab_evaluate_matlab_code",
        {"code": "disp(1+1);", "project_path": str(ROOT)},
    )
    print(
        "after matlab:",
        "obs_ok=",
        (state.get("last_observation") or {}).get("ok"),
        "preview=",
        str(((state.get("last_observation") or {}).get("result") or {}).get("text") or "")[:120],
    )

    mcp_ev = [e for e in state.get("evidence") or [] if e.get("source_type") == "mcp"]
    out = {
        "session_id": store.session_id,
        "evidence_mcp": len(mcp_ev),
        "tool_trace": len(state.get("tool_trace") or []),
        "anysearch_ok": bool(any_ev),
        "matlab_text": str((state.get("last_observation") or {}).get("result") or {})[:200],
    }
    print(json.dumps({k: out[k] for k in out if k != "matlab_text"}, ensure_ascii=False, indent=2))
    success = out["anysearch_ok"] and out["evidence_mcp"] >= 1 and (state.get("last_observation") or {}).get("ok")
    print("RESULT:", "SUCCESS" if success else "FAIL")
    close_all_mcp_clients()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
