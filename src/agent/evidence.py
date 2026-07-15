"""Harvest tool results into evidence items (pure, unit-testable).

Chain: registry.call → harvest_tool_result → state.evidence → decide_reflect
"""

from __future__ import annotations

from typing import Any


def harvest_tool_result(
    result: dict[str, Any] | None,
    *,
    evidence_kind: str | None,
) -> list[dict[str, Any]]:
    """Return evidence items to append; empty if none / ok=False / kind=none."""
    if not isinstance(result, dict):
        return []
    if not result.get("ok"):
        return []
    if evidence_kind == "none":
        return []

    kind = evidence_kind
    source_type = result.get("source_type")

    if kind == "kb_body":
        hits = result.get("hits")
        if isinstance(hits, list):
            return [dict(h) for h in hits if isinstance(h, dict)]
        return []

    if kind == "web_snippet":
        rows = result.get("results")
        if isinstance(rows, list):
            return [dict(r) for r in rows if isinstance(r, dict)]
        return []

    if kind == "web_body":
        pages = result.get("pages")
        if isinstance(pages, list) and pages:
            return [dict(p) for p in pages if isinstance(p, dict)]
        return [_as_evidence_item(result, default_source="web_body")]

    if kind == "mcp" or source_type == "mcp":
        return [_as_mcp_item(result)]

    # Unknown kind: still harvest common builtin shapes
    if isinstance(result.get("hits"), list):
        return [dict(h) for h in result["hits"] if isinstance(h, dict)]
    if isinstance(result.get("results"), list):
        return [dict(r) for r in result["results"] if isinstance(r, dict)]

    # Future plugins: opaque result with text + source_type
    if result.get("text") is not None and source_type:
        return [_as_evidence_item(result, default_source=str(source_type))]

    return []


def _as_mcp_item(result: dict[str, Any]) -> dict[str, Any]:
    text = str(result.get("text") or "")
    has_body = bool(result.get("has_body")) or len(text.strip()) >= 200
    item: dict[str, Any] = {
        "source_type": "mcp",
        "text": text,
        "has_body": has_body,
    }
    if "structured" in result:
        item["structured"] = result["structured"]
    return item


def _as_evidence_item(result: dict[str, Any], *, default_source: str) -> dict[str, Any]:
    skip = {"ok", "error", "hits", "results", "pages", "raw"}
    item = {k: v for k, v in result.items() if k not in skip}
    item.setdefault("source_type", default_source)
    if "text" in result:
        item["text"] = result["text"]
    if "has_body" in result:
        item["has_body"] = result["has_body"]
    elif "text" in item:
        item["has_body"] = len(str(item.get("text") or "").strip()) >= 200
    return item
