"""Tool plugin protocol — decoupled from Registry and agent graph.

New tools should:
1. Implement/export a ToolSpec (or be described in config/tools_enabled.yaml)
2. Pass unit tests in isolation
3. Then be registered into ToolRegistry / agent loop
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolResult:
    """Normalized tool outcome. Always JSON-serializable fields."""

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"ok": self.ok, **self.data}
        if self.error:
            out["error"] = self.error
        return out

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ToolResult":
        data = dict(payload)
        ok = bool(data.pop("ok", False))
        error = data.pop("error", None)
        return cls(ok=ok, data=data, error=str(error) if error else None)

    @classmethod
    def fail(cls, message: str) -> "ToolResult":
        return cls(ok=False, error=message)


@dataclass
class ToolSpec:
    """Declarative tool contract for discovery, planning, and invocation."""

    name: str
    description: str
    args_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any] | ToolResult]
    timeout_sec: float = 60.0
    permissions: list[str] = field(default_factory=list)
    # Evidence contract hints for Reflect (e.g. requires body text)
    evidence_kind: str | None = None  # kb_body | web_snippet | web_body | none
    requires_body: bool = False
    enabled: bool = True

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"invalid tool name: {self.name!r}")
        if not self.description.strip():
            raise ValueError(f"tool {self.name}: description required")
        if not isinstance(self.args_schema, dict):
            raise ValueError(f"tool {self.name}: args_schema must be dict")
        if not callable(self.handler):
            raise ValueError(f"tool {self.name}: handler must be callable")

    def invoke(self, arguments: dict[str, Any] | None = None) -> ToolResult:
        self.validate()
        args = arguments or {}
        try:
            if self.timeout_sec and self.timeout_sec > 0:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(self.handler, args)
                    try:
                        raw = fut.result(timeout=float(self.timeout_sec))
                    except FuturesTimeout:
                        return ToolResult.fail(f"timeout after {self.timeout_sec}s")
            else:
                raw = self.handler(args)
        except Exception as exc:  # noqa: BLE001 — boundary: tool failures become ToolResult
            return ToolResult.fail(str(exc))
        if isinstance(raw, ToolResult):
            return raw
        if isinstance(raw, Mapping):
            return ToolResult.from_mapping(raw)
        return ToolResult.fail(f"handler returned unsupported type: {type(raw)}")
    def planner_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "args": self.args_schema,
            "evidence_kind": self.evidence_kind,
            "requires_body": self.requires_body,
        }

    def meta_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("handler", None)
        return d


@runtime_checkable
class ToolPlugin(Protocol):
    """Optional module-level plugin: export get_tool_spec()."""

    def get_tool_spec(self) -> ToolSpec: ...
