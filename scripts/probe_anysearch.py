"""Probe AnySearch search (official skill CLI + optional project MCP adapter).

Reads API key from setting/API-key/AnySearch-API.txt when present.
If the key is rejected, falls back to anonymous access and reports clearly.

Usage:
  E:\\application\\miniforge3\\envs\\copilot-agent\\python.exe scripts\\probe_anysearch.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENDPOINT = "https://api.anysearch.com/mcp"
KEY_FILE = ROOT / "setting" / "API-key" / "AnySearch-API.txt"


def read_key() -> str | None:
    if not KEY_FILE.exists():
        return None
    text = KEY_FILE.read_text(encoding="utf-8-sig").strip()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return text or None


def rpc(method: str, params: dict, api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    r = requests.post(ENDPOINT, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def tool_text(data: dict) -> tuple[str, bool]:
    result = data.get("result") or {}
    is_err = bool(result.get("isError"))
    parts = []
    for item in result.get("content") or []:
        if item.get("type") == "text":
            parts.append(item.get("text") or "")
    return "\n".join(parts), is_err


def main() -> int:
    print("=== AnySearch probe ===")
    print("endpoint:", ENDPOINT)
    print("key file:", KEY_FILE, "exists=", KEY_FILE.exists())
    key = read_key()
    if key:
        print(f"key loaded: len={len(key)} prefix={key[:6]}...")
    else:
        print("key loaded: none (anonymous)")

    mode = "anonymous"
    active_key: str | None = None
    if key:
        data = rpc("tools/call", {"name": "search", "arguments": {"query": "ping", "max_results": 1}}, key)
        text, is_err = tool_text(data)
        if is_err or "invalid_api_key" in text.lower():
            print("AUTH: provided API key REJECTED → falling back to anonymous")
            print("      tip: create a new key at https://anysearch.com/console/api-keys")
            mode = "anonymous"
            active_key = None
        else:
            print("AUTH: API key accepted")
            mode = "authenticated"
            active_key = key
    else:
        print("AUTH: no key file → anonymous")

    checks: list[tuple[str, bool, str]] = []

    # 1) general search
    data = rpc(
        "tools/call",
        {"name": "search", "arguments": {"query": "radio propagation GNN channel estimation", "max_results": 3}},
        active_key,
    )
    text, is_err = tool_text(data)
    ok = (not is_err) and ("Search Results" in text or "### 1." in text)
    checks.append(("general_search", ok, text[:500]))
    print("\n[1] general search:", "OK" if ok else "FAIL")
    print(text[:800])

    # 2) get_sub_domains + vertical
    data = rpc("tools/call", {"name": "get_sub_domains", "arguments": {"domain": "academic"}}, active_key)
    text, is_err = tool_text(data)
    ok = (not is_err) and ("academic" in text.lower())
    checks.append(("get_sub_domains", ok, text[:400]))
    print("\n[2] get_sub_domains academic:", "OK" if ok else "FAIL")
    print(text[:600])

    data = rpc(
        "tools/call",
        {
            "name": "search",
            "arguments": {
                "query": "graph neural network wireless",
                "domain": "academic",
                "sub_domain": "academic.preprint",
                "max_results": 2,
            },
        },
        active_key,
    )
    text, is_err = tool_text(data)
    ok = not is_err and len(text) > 50
    checks.append(("vertical_search", ok, text[:400]))
    print("\n[3] vertical academic.preprint:", "OK" if ok else "FAIL")
    print(text[:700])

    # 3) batch
    data = rpc(
        "tools/call",
        {
            "name": "batch_search",
            "arguments": {
                "queries": [
                    {"query": "Bufort GNN", "max_results": 2},
                    {"query": "5G channel estimation deep learning", "max_results": 2},
                ]
            },
        },
        active_key,
    )
    text, is_err = tool_text(data)
    ok = not is_err and len(text) > 50
    checks.append(("batch_search", ok, text[:400]))
    print("\n[4] batch_search:", "OK" if ok else "FAIL")
    print(text[:700])

    # 4) extract
    data = rpc(
        "tools/call",
        {"name": "extract", "arguments": {"url": "https://example.com"}},
        active_key,
    )
    text, is_err = tool_text(data)
    ok = not is_err and ("Example Domain" in text or "example" in text.lower())
    checks.append(("extract", ok, text[:400]))
    print("\n[5] extract example.com:", "OK" if ok else "FAIL")
    print(text[:500])

    # 5) project MCP adapter (streamable_http)
    print("\n[6] project Streamable HTTP MCP adapter...")
    try:
        from src.tools.mcp_adapter import SessionBackedMcpClient, close_all_mcp_clients, normalize_mcp_result
        from src.tools.mcp_session import PersistentMcpSession, open_streamable_http_session

        headers = {"Authorization": f"Bearer {active_key}"} if active_key else {}

        def factory():
            return open_streamable_http_session(ENDPOINT, headers=headers or None, timeout_sec=60)

        client = SessionBackedMcpClient(PersistentMcpSession(factory, start_timeout=60))
        try:
            tools = client.list_tools()
            names = [t.name for t in tools]
            print("  tools:", names)
            # tool name may be search
            call_name = "search" if "search" in names else names[0]
            raw = client.call_tool(call_name, {"query": "AnySearch MCP test", "max_results": 2})
            out = normalize_mcp_result(raw)
            ok = bool(out.get("ok")) and len(out.get("text") or "") > 20
            checks.append(("mcp_adapter_search", ok, (out.get("text") or out.get("error") or "")[:400]))
            print("  adapter search:", "OK" if ok else "FAIL")
            print(" ", (out.get("text") or "")[:500])
        finally:
            client.close()
            close_all_mcp_clients()
    except Exception as exc:  # noqa: BLE001
        checks.append(("mcp_adapter_search", False, str(exc)))
        print("  adapter FAIL:", type(exc).__name__, exc)

    print("\n=== Summary ===")
    print("mode:", mode)
    all_ok = True
    for name, ok, _ in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        all_ok = all_ok and ok
    print("RESULT:", "SUCCESS" if all_ok else "PARTIAL/FAIL")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
