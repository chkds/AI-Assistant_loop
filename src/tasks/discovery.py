"""Discover TaskSpec plugins (kept separate from tool discovery to avoid cycles)."""

from __future__ import annotations

import importlib
from typing import Any

from src import load_yaml
from src.tasks.protocol import TaskSpec


def load_enabled_config() -> dict[str, Any]:
    return load_yaml("tools_enabled.yaml")


def discover_task_specs(config: dict[str, Any] | None = None) -> list[TaskSpec]:
    cfg = config or load_enabled_config()
    fail_hard = bool((cfg.get("registry") or {}).get("fail_on_load_error", False))
    specs: list[TaskSpec] = []
    for entry in cfg.get("tasks") or []:
        if not entry.get("enabled", True):
            continue
        try:
            mod = importlib.import_module(entry["module"])
            factory = getattr(mod, entry.get("factory", "get_task_spec"))
            spec = factory()
            if not isinstance(spec, TaskSpec):
                raise TypeError(f"{entry['module']} factory did not return TaskSpec")
            spec.validate()
            specs.append(spec)
        except Exception:
            if fail_hard:
                raise
            continue
    return specs
