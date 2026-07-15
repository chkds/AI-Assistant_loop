"""Shared tool allowlist matching for Registry and TaskSpec."""

from __future__ import annotations


def tool_name_allowed(name: str, allowed: list[str] | None) -> bool:
    """Match exact names, prefix globs (foo_*), or mcp:<server> scopes."""
    if not allowed:
        return True
    for pattern in allowed:
        if pattern == name:
            return True
        if pattern.endswith("*") and name.startswith(pattern[:-1]):
            return True
        if pattern.startswith("mcp:") and name.startswith(f"mcp_{pattern[4:]}_"):
            return True
    return False
