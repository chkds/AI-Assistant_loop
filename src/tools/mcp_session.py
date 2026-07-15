"""Persistent MCP client session (stdio / Streamable HTTP).

Keeps one server connection alive across list_tools + call_tool — required for
local servers like matlab-mcp-server (avoid restarting MATLAB every call).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from src.tools.mcp_types import McpToolInfo


SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]


def call_tool_result_to_dict(result: Any) -> dict[str, Any]:
    """Normalize mcp SDK CallToolResult (or dict) to a plain dict."""
    if isinstance(result, dict):
        return result
    content = []
    for block in getattr(result, "content", None) or []:
        if hasattr(block, "model_dump"):
            content.append(block.model_dump())
        elif hasattr(block, "text"):
            content.append({"type": "text", "text": block.text})
        else:
            content.append({"type": "text", "text": str(block)})
    out: dict[str, Any] = {
        "content": content,
        "isError": bool(getattr(result, "isError", False)),
    }
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        if hasattr(structured, "model_dump"):
            structured = structured.model_dump()
        out["structuredContent"] = structured
    return out


def tool_infos_from_list_result(result: Any) -> list[McpToolInfo]:
    tools = getattr(result, "tools", None) or result
    out: list[McpToolInfo] = []
    for t in tools:
        schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None) or {}
        if hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        name = t.name if hasattr(t, "name") else t["name"]
        if hasattr(t, "description"):
            desc = t.description or name
        elif isinstance(t, dict):
            desc = t.get("description") or name
        else:
            desc = name
        out.append(
            McpToolInfo(
                name=name,
                description=desc,
                input_schema=dict(schema) if isinstance(schema, dict) else {},
            )
        )
    return out


class PersistentMcpSession:
    """Thread-backed MCP session: one connection, many sync list/call ops."""

    def __init__(self, session_factory: SessionFactory, *, start_timeout: float = 90.0):
        self._factory = session_factory
        self._start_timeout = start_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: Any | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._thread_main, name="mcp-session", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=start_timeout):
            self.close()
            raise TimeoutError(f"MCP session failed to start within {start_timeout}s")
        if self._error is not None:
            raise RuntimeError(f"MCP session failed to start: {self._error}") from self._error

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._async_main())
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()

    async def _async_main(self) -> None:
        self._loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        self._stop = stop
        try:
            async with self._factory() as session:
                self._session = session
                self._ready.set()
                await stop.wait()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()
            raise
        finally:
            self._session = None

    def _submit(self, coro_factory: Callable[[Any], Any], timeout: float | None = None) -> Any:
        if self._closed or self._loop is None or self._session is None:
            raise RuntimeError("MCP session is not running")
        fut: concurrent.futures.Future[Any] = concurrent.futures.Future()

        def _schedule() -> None:
            async def _run() -> None:
                try:
                    fut.set_result(await coro_factory(self._session))
                except BaseException as exc:  # noqa: BLE001
                    fut.set_exception(exc)

            asyncio.create_task(_run())

        self._loop.call_soon_threadsafe(_schedule)
        return fut.result(timeout=timeout or self._start_timeout)

    def list_tools(self) -> list[McpToolInfo]:
        async def _op(session: Any) -> list[McpToolInfo]:
            return tool_infos_from_list_result(await session.list_tools())

        return self._submit(_op)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async def _op(session: Any) -> dict[str, Any]:
            return call_tool_result_to_dict(await session.call_tool(name, arguments or {}))

        return self._submit(_op)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        stop = self._stop
        if loop is not None and loop.is_running() and stop is not None:

            def _set_stop() -> None:
                stop.set()

            try:
                loop.call_soon_threadsafe(_set_stop)
            except RuntimeError:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=15.0)


@asynccontextmanager
async def open_stdio_session(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[Any]:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=command, args=list(args or []), env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def open_streamable_http_session(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_sec: float = 60.0,
) -> AsyncIterator[Any]:
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    # Auth/headers live on the httpx client (current MCP SDK contract).
    client = create_mcp_http_client(
        headers=headers or None,
        timeout=httpx.Timeout(timeout_sec, read=timeout_sec * 5),
    )
    async with client:
        async with streamable_http_client(url, http_client=client) as (read, write, _get_sid):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
