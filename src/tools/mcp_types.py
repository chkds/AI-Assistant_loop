"""Shared MCP adapter types (no SDK imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class McpToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class McpClient(Protocol):
    def list_tools(self) -> list[McpToolInfo]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...

    def close(self) -> None: ...  # optional for Fake; Persistent implements
