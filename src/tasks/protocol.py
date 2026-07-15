"""Task / skill protocol — domain packs without editing the core graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.tools.allowlist import tool_name_allowed


@dataclass
class TaskSpec:
    """Describes how a task type should be planned and reflected."""

    task_type: str
    domain: str
    description: str = ""
    default_knowledge_mode: str = "retrieve"  # none|pinned|retrieve
    allowed_tools: list[str] = field(default_factory=list)
    planner_hints: str = ""
    reflect_rules: list[str] = field(default_factory=list)
    require_body_evidence: bool = True
    max_steps: int | None = None
    enabled: bool = True
    # Prefer earlier tools on failure (e.g. AnySearch → Tavily)
    search_stack: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.task_type or not self.domain:
            raise ValueError("task_type and domain are required")
        if self.default_knowledge_mode not in {"none", "pinned", "retrieve", "agentic"}:
            raise ValueError(f"bad knowledge_mode: {self.default_knowledge_mode}")

    def allows_tool(self, name: str) -> bool:
        return tool_name_allowed(name, self.allowed_tools)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def planner_block(self) -> str:
        tools = ", ".join(self.allowed_tools) if self.allowed_tools else "(all registered)"
        rules = "; ".join(self.reflect_rules) if self.reflect_rules else ""
        stack = ", ".join(self.search_stack) if self.search_stack else "(none)"
        return (
            f"TaskType={self.task_type} domain={self.domain}\n"
            f"knowledge_mode_default={self.default_knowledge_mode}\n"
            f"allowed_tools={tools}\n"
            f"search_stack_primary_first={stack}\n"
            f"hints={self.planner_hints}\n"
            f"reflect_rules={rules}\n"
            f"require_body_evidence={self.require_body_evidence}"
        )
