"""One-shot probe: connect to local matlab-mcp-core-server and exercise MATLAB."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.mcp_adapter import (  # noqa: E402
    _client_from_server_entry,
    close_all_mcp_clients,
    normalize_mcp_result,
    tool_specs_from_client,
)


ENTRY = {
    "id": "matlab",
    "enabled": True,
    "transport": "stdio",
    "keep_alive": True,
    "command": str(ROOT / "mcp" / "matlab-mcp-core-server-win64.exe"),
    "args": [
        "--matlab-display-mode=nodesktop",
        "--initialize-matlab-on-startup=true",
        "--matlab-root=E:/application/matlab/R2023a",
        "--initial-working-folder=E:/project/AI-Assistant_loop",
    ],
    "timeout_sec": 180,
}


def main() -> int:
    print("=== MATLAB MCP probe ===")
    print("binary:", ENTRY["command"])
    print("args:", ENTRY["args"])
    client = None
    try:
        print("\n[1] Starting MCP stdio session (may launch MATLAB)...")
        client = _client_from_server_entry(ENTRY)
        tools = client.list_tools()
        print(f"[1] OK — {len(tools)} tools:")
        for t in tools:
            print(f"    - {t.name}: {(t.description or '')[:80]}")

        print("\n[2] Calling detect_matlab_toolboxes (if present)...")
        names = {t.name for t in tools}
        if "detect_matlab_toolboxes" in names:
            raw = client.call_tool("detect_matlab_toolboxes", {})
            out = normalize_mcp_result(raw)
            print(json.dumps({k: out[k] for k in out if k != "raw"}, ensure_ascii=False, indent=2)[:2000])
        else:
            print("    (tool not listed; skipping)")

        print("\n[3] Calling evaluate_matlab_code: version / 1+1 ...")
        if "evaluate_matlab_code" in names:
            raw = client.call_tool(
                "evaluate_matlab_code",
                {
                    "code": "disp(version); disp(1+1);",
                    "project_path": str(ROOT),
                },
            )
            out = normalize_mcp_result(raw)
            print(json.dumps({k: out[k] for k in out if k != "raw"}, ensure_ascii=False, indent=2)[:3000])
            ok = bool(out.get("ok"))
        else:
            print("    evaluate_matlab_code missing")
            ok = False

        specs = tool_specs_from_client(client, server_id="matlab")
        print("\n[4] Agent-facing names:", [s.name for s in specs])
        print("\nRESULT:", "SUCCESS" if ok else "PARTIAL/FAIL")
        return 0 if ok else 2
    except Exception as exc:  # noqa: BLE001
        print("\nRESULT: FAILED")
        print(type(exc).__name__ + ":", exc)
        traceback.print_exc()
        return 1
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        close_all_mcp_clients()


if __name__ == "__main__":
    raise SystemExit(main())
