"""MCP server → ToolSpec adapter.

Transports (priority):
- stdio: local servers (e.g. MathWorks matlab-mcp-server)
- streamable_http (aliases: http, https): mainstream remote MCP
- fake: unit tests / offline demos

Legacy HTTP+SSE is intentionally not supported.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from src import load_yaml, read_api_key, resolve_path
from src.tools.mcp_session import (
    PersistentMcpSession,
    open_stdio_session,
    open_streamable_http_session,
)
from src.tools.mcp_types import McpClient, McpToolInfo
from src.tools.protocol import ToolSpec

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"[^a-zA-Z0-9_]+")
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_ACTIVE_CLIENTS: list[Any] = []


def mcp_name(server_id: str, tool_name: str) -> str:
    """Stable agent-facing name: mcp_<server>_<tool> (alnum + underscore)."""
    s = _SAFE.sub("_", server_id.strip()).strip("_").lower() or "server"
    t = _SAFE.sub("_", tool_name.strip()).strip("_").lower() or "tool"
    return f"mcp_{s}_{t}"


def expand_env(value: str) -> str:
    """Replace ${VAR} from process environment."""

    def repl(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), "")

    return _ENV.sub(repl, value)


def expand_mapping(data: dict[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (data or {}).items():
        out[str(k)] = expand_env(str(v))
    return out


def build_auth_headers(entry: dict[str, Any]) -> dict[str, str]:
    """Merge static headers + bearer/token_file auth for Streamable HTTP."""
    headers = expand_mapping(entry.get("headers"))
    auth = entry.get("auth") or {}
    if not auth:
        return headers

    auth_type = str(auth.get("type") or "bearer").lower()
    header_name = str(auth.get("header_name") or "Authorization")
    token = auth.get("token")
    if auth.get("token_file"):
        token = read_api_key(auth["token_file"])
    if token is not None:
        token = expand_env(str(token).strip())
    if not token:
        raise ValueError(f"MCP server {entry.get('id')!r}: auth configured but token empty")

    if auth_type in {"bearer", "token"}:
        headers[header_name] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    elif auth_type in {"header", "raw"}:
        headers[header_name] = token
    elif auth_type == "api_key":
        headers[str(auth.get("header_name") or "X-API-Key")] = token
    else:
        raise ValueError(f"unsupported MCP auth.type: {auth_type}")
    return headers


def normalize_transport(name: str) -> str:
    t = (name or "fake").strip().lower()
    if t in {"http", "https", "streamable-http", "streamable_http", "streamablehttp"}:
        return "streamable_http"
    if t == "stdio":
        return "stdio"
    if t == "fake":
        return "fake"
    if t in {"sse", "http+sse", "http_sse"}:
        raise ValueError(
            "Legacy HTTP+SSE transport is not supported; use transport: streamable_http "
            "(or stdio for local servers like matlab-mcp-server)."
        )
    return t


@dataclass
class FakeMcpClient:
    """In-memory MCP client for unit tests and config transport=fake."""

    tools: list[McpToolInfo] = field(default_factory=list)
    call_handler: Callable[[str, dict[str, Any]], Any] | None = None
    responses: dict[str, Any] = field(default_factory=dict)

    def list_tools(self) -> list[McpToolInfo]:
        return list(self.tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.call_handler is not None:
            return self.call_handler(name, arguments)
        if name in self.responses:
            return self.responses[name]
        return {"content": [{"type": "text", "text": f"fake:{name}"}], "isError": False}

    def close(self) -> None:
        return None


@dataclass
class SessionBackedMcpClient:
    """Wrap PersistentMcpSession as McpClient."""

    session: PersistentMcpSession

    def list_tools(self) -> list[McpToolInfo]:
        return self.session.list_tools()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return self.session.call_tool(name, arguments)

    def close(self) -> None:
        self.session.close()


class LazyMcpClient:
    """Connect on first list_tools/call_tool — avoids starting MATLAB at registry load."""

    def __init__(self, factory: Callable[[], McpClient]):
        self._factory = factory
        self._inner: McpClient | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> McpClient:
        with self._lock:
            if self._inner is None:
                self._inner = self._factory()
            return self._inner

    def list_tools(self) -> list[McpToolInfo]:
        return self._ensure().list_tools()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return self._ensure().call_tool(name, arguments)

    def close(self) -> None:
        with self._lock:
            if self._inner is not None:
                close = getattr(self._inner, "close", None)
                if callable(close):
                    close()
                self._inner = None


def tool_specs_from_infos(
    client: McpClient,
    infos: list[McpToolInfo],
    *,
    server_id: str,
    allowlist: list[str] | None = None,
    timeout_sec: float = 60.0,
    evidence_kind: str | None = "mcp",
    requires_body: bool = False,
) -> list[ToolSpec]:
    """Bind predeclared tool infos (lazy manifest) to a client."""
    allowed = set(allowlist) if allowlist else None
    specs: list[ToolSpec] = []
    for info in infos:
        if allowed is not None and info.name not in allowed:
            continue
        tool_name = info.name

        def _make_handler(tn: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
            def handler(arguments: dict[str, Any]) -> dict[str, Any]:
                return normalize_mcp_result(client.call_tool(tn, arguments or {}))

            return handler

        specs.append(
            ToolSpec(
                name=mcp_name(server_id, tool_name),
                description=f"[mcp:{server_id}] {info.description or tool_name}",
                args_schema=info.input_schema or {},
                handler=_make_handler(tool_name),
                timeout_sec=timeout_sec,
                permissions=["mcp", f"mcp:{server_id}"],
                evidence_kind=evidence_kind,
                requires_body=requires_body,
                enabled=True,
            )
        )
    return specs


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text" and "text" in block:
                    parts.append(str(block["text"]))
                elif "text" in block:
                    parts.append(str(block["text"]))
                elif btype == "image":
                    parts.append(f"[image mime={block.get('mimeType') or block.get('mime_type') or 'unknown'}]")
                elif btype == "resource" or btype == "resource_link":
                    parts.append(f"[resource {block.get('uri') or block.get('name') or ''}]".strip())
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)


def normalize_mcp_result(payload: Any) -> dict[str, Any]:
    """Map MCP CallToolResult-like payloads to ToolResult-compatible dicts."""
    if payload is None:
        return {"ok": False, "error": "empty_mcp_result", "source_type": "mcp"}

    if isinstance(payload, dict) and "ok" in payload and "content" not in payload and "isError" not in payload:
        out = dict(payload)
        out.setdefault("source_type", "mcp")
        return out

    if isinstance(payload, dict):
        is_error = bool(payload.get("isError") or payload.get("is_error"))
        text = _content_to_text(payload.get("content"))
        structured = payload.get("structuredContent") or payload.get("structured_content")
        if is_error:
            return {
                "ok": False,
                "error": text or str(payload.get("error") or "mcp_tool_error"),
                "source_type": "mcp",
                "raw": payload,
            }
        out = {
            "ok": True,
            "text": text,
            "has_body": len(text.strip()) >= 200,
            "source_type": "mcp",
            "raw": payload,
        }
        if structured is not None:
            out["structured"] = structured
        return out

    text = str(payload)
    return {
        "ok": True,
        "text": text,
        "has_body": len(text.strip()) >= 200,
        "source_type": "mcp",
    }


def tool_specs_from_client(
    client: McpClient,
    *,
    server_id: str,
    allowlist: list[str] | None = None,
    timeout_sec: float = 60.0,
    evidence_kind: str | None = "mcp",
    requires_body: bool = False,
) -> list[ToolSpec]:
    """Convert MCP list_tools() into ToolSpec plugins bound to client.call_tool."""
    return tool_specs_from_infos(
        client,
        client.list_tools(),
        server_id=server_id,
        allowlist=allowlist,
        timeout_sec=timeout_sec,
        evidence_kind=evidence_kind,
        requires_body=requires_body,
    )

def close_all_mcp_clients() -> None:
    global _ACTIVE_CLIENTS
    for c in _ACTIVE_CLIENTS:
        try:
            close = getattr(c, "close", None)
            if callable(close):
                close()
        except Exception:  # noqa: BLE001
            pass
    _ACTIVE_CLIENTS = []


def _track(client: Any) -> Any:
    _ACTIVE_CLIENTS.append(client)
    return client


def _client_from_server_entry(entry: dict[str, Any]) -> McpClient:
    transport = normalize_transport(str(entry.get("transport") or "fake"))
    timeout = float(entry.get("timeout_sec") or 60.0)
    keep_alive = bool(entry.get("keep_alive", True))

    if transport == "fake":
        tools = [
            McpToolInfo(
                name=t["name"],
                description=t.get("description") or t["name"],
                input_schema=t.get("input_schema") or {},
            )
            for t in entry.get("fake_tools") or []
        ]
        return FakeMcpClient(tools=tools, responses=dict(entry.get("fake_responses") or {}))

    if transport == "stdio":
        command = entry.get("command")
        if not command:
            raise ValueError("stdio MCP server requires 'command'")
        command = expand_env(str(command))
        args = [expand_env(str(a)) for a in (entry.get("args") or [])]
        env = expand_mapping(entry.get("env")) or None
        # Inherit process env and overlay (WINDIR etc. for MATLAB on Windows)
        if env is not None:
            merged = dict(os.environ)
            merged.update(env)
            env = merged

        def factory():
            return open_stdio_session(command, args=args, env=env)

        if keep_alive:
            return SessionBackedMcpClient(PersistentMcpSession(factory, start_timeout=timeout))
        # One-shot: still use persistent wrapper for a single discover+call cycle lifetime
        return SessionBackedMcpClient(PersistentMcpSession(factory, start_timeout=timeout))

    if transport == "streamable_http":
        url = entry.get("url") or entry.get("endpoint")
        if not url:
            raise ValueError("streamable_http MCP server requires 'url'")
        url = expand_env(str(url))
        headers = build_auth_headers(entry)

        def factory():
            return open_streamable_http_session(url, headers=headers or None, timeout_sec=timeout)

        return SessionBackedMcpClient(PersistentMcpSession(factory, start_timeout=timeout))

    raise ValueError(f"unsupported MCP transport: {transport}")


def load_mcp_config() -> dict[str, Any]:
    try:
        return load_yaml("mcp_servers.yaml")
    except Exception:
        return {"servers": []}


def discover_mcp_tool_specs(config: dict[str, Any] | None = None) -> list[ToolSpec]:
    """Load enabled MCP servers from config and return ToolSpec list."""
    cfg = config if config is not None else load_mcp_config()
    fail_hard = bool((cfg.get("registry") or {}).get("fail_on_load_error", False))
    # Only close tracked live clients when loading the real config (reload path).
    if config is None:
        close_all_mcp_clients()

    specs: list[ToolSpec] = []
    for entry in cfg.get("servers") or []:
        if not entry.get("enabled", False):
            continue
        server_id = entry.get("id") or entry.get("name") or "mcp"
        try:
            timeout_sec = float(entry.get("timeout_sec") or 60.0)
            evidence_kind = entry.get("evidence_kind", "mcp")
            requires_body = bool(entry.get("requires_body", False))
            allowlist = entry.get("tool_allowlist") or None
            lazy = bool(entry.get("lazy", False))
            manifest = entry.get("tools_manifest") or []

            if lazy:
                if not manifest:
                    raise ValueError(
                        f"MCP server {server_id!r} has lazy=true but no tools_manifest; "
                        "declare tools_manifest to avoid connecting at registry load"
                    )
                # Defer process/HTTP session until first tool call.
                entry_eager = {**entry, "lazy": False}
                client = LazyMcpClient(lambda e=entry_eager: _client_from_server_entry(e))
                _track(client)
                infos = [
                    McpToolInfo(
                        name=t["name"],
                        description=t.get("description") or t["name"],
                        input_schema=t.get("input_schema") or {},
                    )
                    for t in manifest
                ]
                specs.extend(
                    tool_specs_from_infos(
                        client,
                        infos,
                        server_id=str(server_id),
                        allowlist=allowlist,
                        timeout_sec=timeout_sec,
                        evidence_kind=evidence_kind,
                        requires_body=requires_body,
                    )
                )
            else:
                client = _client_from_server_entry(entry)
                _track(client)
                specs.extend(
                    tool_specs_from_client(
                        client,
                        server_id=str(server_id),
                        allowlist=allowlist,
                        timeout_sec=timeout_sec,
                        evidence_kind=evidence_kind,
                        requires_body=requires_body,
                    )
                )
        except Exception as exc:
            if fail_hard:
                raise
            logger.warning("skipping MCP server %s: %s", server_id, exc)
            continue
    return specs

def resolve_binary_hint(path_like: str) -> str:
    """Expand env and resolve relative paths for documentation / config checks."""
    expanded = expand_env(path_like)
    p = resolve_path(expanded) if not os.path.isabs(expanded) else expanded
    return str(p)
