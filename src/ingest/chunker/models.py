"""Chunk data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str
    content_type: str  # text|table|formula|figure|section
    domain: str = "research"
    parent_id: str | None = None
    continuation_id: str | None = None
    related_assets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data["text"],
            content_type=data["content_type"],
            domain=data.get("domain", "research"),
            parent_id=data.get("parent_id"),
            continuation_id=data.get("continuation_id"),
            related_assets=list(data.get("related_assets") or []),
            metadata=dict(data.get("metadata") or {}),
        )

    def embed_text(self) -> str:
        headers = self.metadata.get("headers_path") or ""
        if headers:
            return f"{headers}\n{self.text}".strip()
        return self.text.strip()
