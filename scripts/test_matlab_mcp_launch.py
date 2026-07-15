"""Visible MATLAB MCP launch test.

Uses matlab-display-mode=desktop so you can see the MATLAB window.
Also reports matlab.exe / MCP process presence before and after.

Usage:
  E:\\application\\miniforge3\\envs\\copilot-agent\\python.exe scripts\\test_matlab_mcp_launch.py
  E:\\application\\miniforge3\\envs\\copilot-agent\\python.exe scripts\\test_matlab_mcp_launch.py --hold 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.mcp_adapter import (  # noqa: E402
    _client_from_server_entry,
    close_all_mcp_clients,
    normalize_mcp_result,
)

MCP_BIN = ROOT / "mcp" / "matlab-mcp-core-server-win64.exe"
MATLAB_ROOT = Path(r"E:\application\matlab\R2023a")


def _ps_list(name_substr: str) -> list[dict]:
    """Return running processes whose ImageName contains name_substr (case-insensitive)."""
    # PowerShell is more reliable than tasklist encoding on Chinese Windows.
    cmd = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.Name -like '*{name_substr}*' }} | "
        "Select-Object ProcessId,Name,CreationDate,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", cmd],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        print(f"  (process query failed: {exc})")
        return []
    if not raw:
        return []
    data = json.loads(raw)
    if isinstance(data, dict):
        return [data]
    return list(data)


def _print_procs(label: str, procs: list[dict]) -> None:
    print(f"\n--- {label}: {len(procs)} process(es) ---")
    for p in procs:
        cmd = (p.get("CommandLine") or "")[:160]
        print(f"  pid={p.get('ProcessId')} name={p.get('Name')} created={p.get('CreationDate')}")
        if cmd:
            print(f"    cmd: {cmd}")


def build_entry(*, desktop: bool, init_on_startup: bool) -> dict:
    mode = "desktop" if desktop else "nodesktop"
    args = [
        f"--matlab-display-mode={mode}",
        f"--matlab-root={MATLAB_ROOT.as_posix()}",
        f"--initial-working-folder={ROOT.as_posix()}",
    ]
    if init_on_startup:
        args.append("--initialize-matlab-on-startup=true")
    return {
        "id": "matlab",
        "transport": "stdio",
        "keep_alive": True,
        "command": str(MCP_BIN),
        "args": args,
        "timeout_sec": 240,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Visible test: MCP starts MATLAB")
    ap.add_argument(
        "--hold",
        type=int,
        default=20,
        help="Seconds to keep session open after success so you can see the window (default 20)",
    )
    ap.add_argument(
        "--nodesktop",
        action="store_true",
        help="Use nodesktop (no MATLAB GUI). Default is desktop so the window is visible.",
    )
    ap.add_argument(
        "--lazy-matlab",
        action="store_true",
        help="Do not pass --initialize-matlab-on-startup (MATLAB starts on first tool call).",
    )
    args = ap.parse_args()

    print("=== Visible MATLAB MCP launch test ===")
    print(f"MCP binary : {MCP_BIN}")
    print(f"exists     : {MCP_BIN.exists()}")
    print(f"MATLAB root: {MATLAB_ROOT}")
    print(f"exists     : {MATLAB_ROOT.exists()}")
    print(f"display    : {'nodesktop' if args.nodesktop else 'desktop (GUI should appear)'}")
    print(f"init on start: {not args.lazy_matlab}")
    print(f"hold after : {args.hold}s")

    if not MCP_BIN.exists():
        print("FAIL: MCP binary missing")
        return 1
    if not (MATLAB_ROOT / "bin" / "matlab.exe").exists():
        print(f"FAIL: matlab.exe not found under {MATLAB_ROOT / 'bin'}")
        return 1

    before_matlab = _ps_list("matlab")
    before_mcp = _ps_list("matlab-mcp")
    _print_procs("BEFORE matlab*", before_matlab)
    _print_procs("BEFORE matlab-mcp*", before_mcp)
    before_pids = {p.get("ProcessId") for p in before_matlab}

    entry = build_entry(desktop=not args.nodesktop, init_on_startup=not args.lazy_matlab)
    print("\nMCP args:", entry["args"])

    client = None
    try:
        t0 = time.time()
        print("\n[1] Connecting MCP (stdio). Watch for a MATLAB desktop window...")
        client = _client_from_server_entry(entry)
        tools = client.list_tools()
        print(f"[1] MCP connected in {time.time() - t0:.1f}s — tools={len(tools)}")

        mid_matlab = _ps_list("matlab")
        mid_mcp = _ps_list("matlab-mcp")
        _print_procs("AFTER MCP CONNECT matlab*", mid_matlab)
        _print_procs("AFTER MCP CONNECT matlab-mcp*", mid_mcp)
        new_pids = {p.get("ProcessId") for p in mid_matlab} - before_pids
        if new_pids:
            print(f"\n>>> New MATLAB-related PID(s): {sorted(new_pids)}")
        else:
            print("\n>>> No new matlab* PID yet (may start on first tool call).")

        print("\n[2] evaluate_matlab_code → force/confirm MATLAB engine...")
        raw = client.call_tool(
            "evaluate_matlab_code",
            {
                # Visible cue even in nodesktop: creates a figure briefly when desktop mode
                "code": (
                    "fprintf('MCP_LAUNCH_OK version=%s\\n', version);\n"
                    "disp(1+1);\n"
                    "if usejava('desktop')\n"
                    "  f = figure('Name','MCP Launch Test','NumberTitle','off');\n"
                    "  title('MATLAB started by MCP — you can close this figure');\n"
                    "  drawnow;\n"
                    "end\n"
                ),
                "project_path": str(ROOT),
            },
        )
        out = normalize_mcp_result(raw)
        print(json.dumps({k: out[k] for k in out if k != "raw"}, ensure_ascii=False, indent=2)[:2500])

        after_matlab = _ps_list("matlab")
        _print_procs("AFTER evaluate matlab*", after_matlab)
        new_pids = {p.get("ProcessId") for p in after_matlab} - before_pids
        ok_text = bool(out.get("ok")) and ("MCP_LAUNCH_OK" in (out.get("text") or "") or "2" in (out.get("text") or ""))
        ok_proc = bool(new_pids) or any(
            str(p.get("Name", "")).lower().startswith("matlab") for p in after_matlab
        )

        print("\n=== Verdict ===")
        print(f"tool_ok     : {out.get('ok')}  text_ok={ok_text}")
        print(f"process_ok  : {ok_proc}  new_pids={sorted(new_pids) if new_pids else []}")
        if not args.nodesktop:
            print("UI hint     : Look for a MATLAB desktop window / figure titled 'MCP Launch Test'.")
        else:
            print("UI hint     : nodesktop mode — no desktop UI expected; rely on process list + tool output.")

        if args.hold > 0:
            print(f"\nHolding session open for {args.hold}s (Ctrl+C to stop early)...")
            time.sleep(args.hold)

        success = bool(out.get("ok")) and ok_proc
        print("\nRESULT:", "SUCCESS — MATLAB was started via MCP" if success else "FAIL / inconclusive")
        return 0 if success else 2
    except Exception as exc:  # noqa: BLE001
        print("\nRESULT: FAILED")
        print(type(exc).__name__ + ":", exc)
        traceback.print_exc()
        return 1
    finally:
        print("\nClosing MCP session (MATLAB may exit shortly after)...")
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        close_all_mcp_clients()
        time.sleep(2)
        _print_procs("AFTER CLOSE matlab*", _ps_list("matlab"))


if __name__ == "__main__":
    raise SystemExit(main())
