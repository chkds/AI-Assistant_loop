"""Load ToolSpec plugins from config/tools_enabled.yaml."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from src import load_yaml
from src.tools.protocol import ToolSpec

logger = logging.getLogger(__name__)


def load_enabled_config() -> dict[str, Any]:
    return load_yaml("tools_enabled.yaml")


def discover_tool_specs(config: dict[str, Any] | None = None) -> list[ToolSpec]:
    cfg = config or load_enabled_config()
    fail_hard = bool((cfg.get("registry") or {}).get("fail_on_load_error", False))
    specs: list[ToolSpec] = []
    for entry in cfg.get("tools") or []:
        if not entry.get("enabled", True):
            continue
        module = entry.get("module") or "?"
        try:
            mod = importlib.import_module(entry["module"])
            factory = getattr(mod, entry.get("factory", "get_tool_spec"))
            spec = factory()
            if not isinstance(spec, ToolSpec):
                raise TypeError(f"{entry['module']} factory did not return ToolSpec")
            if entry.get("name") and spec.name != entry["name"]:
                raise ValueError(f"spec name {spec.name} != config name {entry['name']}")
            spec.validate()
            specs.append(spec)
        except Exception as exc:
            if fail_hard:
                raise
            logger.warning("skipping tool plugin %s: %s", module, exc)
            continue
    return specs