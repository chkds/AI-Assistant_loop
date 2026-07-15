"""N1 smoke: MCP tool I/O → harvest → evidence (no full agent LLM loop)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_yaml  # noqa: E402
from src.agent.graph import AgentRuntime, _add_evidence  # noqa: E402
from src.agent.session import SessionStore  # noqa: E402
from src.agent.tool_io import apply_tool_io  # noqa: E402
from src.tools.mcp_adapter import close_all_mcp_clients  # noqa: E402
from src.tools.registry import get_registry  # noqa: E402


def main() -> int:
    print("=== N1 MCP I/O chain smoke ===")
    load_yaml.cache_clear()
    reg = get_registry(reload=True)
    names = reg.names()
    mcp_names = [n for n in names if n.startswith("mcp_")]
    print("registry tools:", names)
    print("mcp tools:", mcp_names)
    if not any(n.startswith("mcp_anysearch_") for n in mcp_names):
        print("FAIL: anysearch MCP tools not registered (check mcp_servers.yaml enabled)")
        return 2

    search_tool = "mcp_anysearch_search"
    if search_tool not in names:
        # pick first anysearch tool
        search_tool = next(n for n in mcp_names if n.startswith("mcp_anysearch_"))
    print("calling", search_tool)

    args = {"query": "graph neural network radio propagation", "max_results": 2}
    result = reg.call(search_tool, args)
    print("call ok:", result.get("ok"), "source_type:", result.get("source_type"))
    if not result.get("ok"):
        print("FAIL result:", json.dumps({k: result.get(k) for k in result if k != "raw"}, ensure_ascii=False)[:800])
        close_all_mcp_clients()
        return 2

    store = SessionStore()
    state = store.new_state("n1 mcp io smoke")
    obs = apply_tool_io(
        state,
        tool=search_tool,
        arguments=args,
        result=result,
        evidence_kind=reg.get(search_tool).evidence_kind if reg.get(search_tool) else "mcp",
        add_evidence=_add_evidence,
    )
    state["last_observation"] = obs
    state["next_action"] = {"action": "tool", "tool": search_tool, "arguments": args}
    store.save_state(state)
    store.append_event({"type": "smoke", "node": "tool_io", **{k: obs.get(k) for k in ("tool", "ok", "evidence_ids")}})

    # Also exercise AgentRuntime.act path with injected registry (no LLM if next is tool)
    state2 = store.load_state()
    # fresh next tool call via runtime.act
    text = "m" * 200
    # Prefer live result already harvested; verify act path with a second call only if needed
    rt = AgentRuntime(store, state2, registry=reg)
    state2["next_action"] = {
        "action": "tool",
        "tool": search_tool,
        "arguments": {"query": "AnySearch MCP", "max_results": 1},
    }
    state2 = rt.act(state2)
    state2 = rt.observe(state2)

    mcp_ev = [e for e in (state2.get("evidence") or []) if e.get("source_type") == "mcp"]
    out = {
        "session_id": store.session_id,
        "mcp_tools": mcp_names,
        "evidence_mcp": len(mcp_ev),
        "tool_trace": len(state2.get("tool_trace") or []),
        "last_obs_type": (state2.get("last_observation") or {}).get("type"),
        "has_body_any": any(e.get("has_body") for e in mcp_ev),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    ok = out["evidence_mcp"] >= 1 and out["tool_trace"] >= 1
    print("RESULT:", "SUCCESS" if ok else "FAIL")
    close_all_mcp_clients()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
