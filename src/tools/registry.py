"""Tool registry — discovers ToolSpec plugins; decoupled from agent graph."""

from __future__ import annotations

import logging
from typing import Any

from src import load_tools_cfg, load_yaml
from src.tools.allowlist import tool_name_allowed
from src.tools.discovery import discover_tool_specs, load_enabled_config
from src.tools.mcp_adapter import close_all_mcp_clients, discover_mcp_tool_specs, load_mcp_config
from src.tools.protocol import ToolResult, ToolSpec

logger = logging.getLogger(__name__)


def _resolve_fail_hard(
    tools_cfg: dict[str, Any] | None,
    mcp_cfg: dict[str, Any] | None,
    override: bool | None,
) -> bool:
    if override is not None:
        return bool(override)
    tools_flag = bool(((tools_cfg or {}).get("registry") or {}).get("fail_on_load_error", False))
    mcp_flag = bool(((mcp_cfg or {}).get("registry") or {}).get("fail_on_load_error", False))
    return tools_flag or mcp_flag


class ToolRegistry:
    def __init__(
        self,
        specs: list[ToolSpec] | None = None,
        *,
        fail_on_load_error: bool | None = None,
    ):
        enabled_cfg: dict[str, Any] = {}
        mcp_cfg: dict[str, Any] = {}
        if specs is None:
            try:
                enabled_cfg = load_enabled_config()
            except Exception as exc:  # noqa: BLE001
                fail_hard = _resolve_fail_hard(None, None, fail_on_load_error)
                if fail_hard:
                    raise
                logger.warning("failed to load tools_enabled.yaml: %s", exc)
                enabled_cfg = {}
            try:
                mcp_cfg = load_mcp_config()
            except Exception as exc:  # noqa: BLE001
                fail_hard = _resolve_fail_hard(enabled_cfg, None, fail_on_load_error)
                if fail_hard:
                    raise
                logger.warning("failed to load mcp_servers.yaml: %s", exc)
                mcp_cfg = {}

            fail_hard = _resolve_fail_hard(enabled_cfg, mcp_cfg, fail_on_load_error)
            specs = []
            try:
                specs = list(discover_tool_specs(enabled_cfg))
            except Exception as exc:  # noqa: BLE001
                if fail_hard:
                    raise
                logger.warning("tool plugin discovery failed: %s", exc)
                specs = []
            try:
                specs.extend(discover_mcp_tool_specs(mcp_cfg if mcp_cfg else None))
            except Exception as exc:  # noqa: BLE001
                if fail_hard:
                    raise
                logger.warning("MCP tool discovery failed: %s", exc)

        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)
        cfg = {}
        try:
            cfg = (enabled_cfg or load_enabled_config()).get("registry") or {}
        except Exception:  # noqa: BLE001
            pass
        legacy = load_tools_cfg().get("registry") or {}
        self.max_calls = int(cfg.get("max_tool_calls_per_task") or legacy.get("max_tool_calls_per_task") or 20)
        self._call_count = 0

    def register(self, spec: ToolSpec) -> None:
        spec.validate()
        if not spec.enabled:
            return
        self._specs[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._specs.pop(name, None)

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs.keys())

    def schemas(self) -> list[dict[str, Any]]:
        return [s.planner_schema() for s in self._specs.values()]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._call_count >= self.max_calls:
            return ToolResult.fail("max_tool_calls_exceeded").to_dict()
        spec = self._specs.get(name)
        if not spec:
            return ToolResult.fail(f"unknown_tool: {name}").to_dict()
        self._call_count += 1
        return spec.invoke(arguments).to_dict()

    def filter_by_allowlist(self, allowed: list[str] | None) -> "ToolRegistry":
        """Return a new registry with only allowed tools (empty/None = all).

        Patterns:
        - exact name: kb_retrieve
        - prefix glob: mcp_matlab_*
        - server scope: mcp:matlab  → all tools named mcp_matlab_*
        """
        if not allowed:
            return self
        specs = [s for n, s in self._specs.items() if tool_name_allowed(n, allowed)]
        return ToolRegistry(specs=specs)


_REGISTRY: ToolRegistry | None = None


def get_registry(reload: bool = False) -> ToolRegistry:
    global _REGISTRY
    if _REGISTRY is None or reload:
        if reload:
            close_all_mcp_clients()
            load_yaml.cache_clear()
        _REGISTRY = ToolRegistry()
    return _REGISTRY
