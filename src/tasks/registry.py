"""Task registry — discover TaskSpec packs from config."""

from __future__ import annotations

from src.tasks.discovery import discover_task_specs, load_enabled_config
from src.tasks.protocol import TaskSpec


class TaskRegistry:
    def __init__(self, specs: list[TaskSpec] | None = None):
        if specs is None:
            try:
                specs = discover_task_specs(load_enabled_config())
            except Exception:
                specs = []
        self._specs: dict[str, TaskSpec] = {}
        for s in specs:
            self.register(s)

    def register(self, spec: TaskSpec) -> None:
        spec.validate()
        if spec.enabled:
            self._specs[spec.task_type] = spec

    def get(self, task_type: str) -> TaskSpec | None:
        return self._specs.get(task_type)

    def by_domain(self, domain: str) -> TaskSpec | None:
        for s in self._specs.values():
            if s.domain == domain:
                return s
        return None

    def resolve(self, *, task_type: str | None = None, domain: str | None = None) -> TaskSpec:
        if task_type and task_type in self._specs:
            return self._specs[task_type]
        if domain:
            hit = self.by_domain(domain)
            if hit:
                return hit
        return self._specs.get("general_qa") or TaskSpec(
            task_type="general_qa",
            domain="general",
            default_knowledge_mode="none",
            require_body_evidence=False,
        )

    def names(self) -> list[str]:
        return sorted(self._specs.keys())


_TASKS: TaskRegistry | None = None


def get_task_registry(reload: bool = False) -> TaskRegistry:
    global _TASKS
    if _TASKS is None or reload:
        _TASKS = TaskRegistry()
    return _TASKS
