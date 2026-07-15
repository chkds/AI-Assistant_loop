"""Search primary/backup: try tools in TaskSpec.search_stack order."""

from __future__ import annotations

from typing import Any, Callable


SEARCHISH = {
    "tavily_search",
    "mcp_anysearch_search",
    "mcp_anysearch_batch_search",
}


def is_search_tool(name: str) -> bool:
    n = (name or "").lower()
    if n in SEARCHISH:
        return True
    return "search" in n and (n.startswith("mcp_") or n.startswith("tavily"))


def next_fallback(failed_tool: str, stack: list[str], available: set[str]) -> str | None:
    """Return next tool in stack after failed_tool that is available."""
    if not stack:
        return None
    # normalize stack entries that may be mcp:anysearch patterns
    resolved: list[str] = []
    for entry in stack:
        if entry in available:
            resolved.append(entry)
            continue
        if entry.startswith("mcp:") and entry.count(":") == 1:
            prefix = "mcp_" + entry.split(":", 1)[1] + "_"
            # prefer *_search
            cands = sorted(n for n in available if n.startswith(prefix) and "search" in n)
            if cands:
                resolved.append(cands[0])
                continue
            cands = sorted(n for n in available if n.startswith(prefix))
            if cands:
                resolved.append(cands[0])
        elif entry.endswith("_*"):
            prefix = entry[:-1]
            cands = sorted(n for n in available if n.startswith(prefix) and "search" in n)
            if cands:
                resolved.append(cands[0])
    if failed_tool not in resolved:
        # still try first available not equal failed
        for t in resolved:
            if t != failed_tool:
                return t
        return None
    idx = resolved.index(failed_tool)
    for t in resolved[idx + 1 :]:
        if t != failed_tool:
            return t
    return None


def call_with_search_fallback(
    *,
    registry,
    tool: str,
    arguments: dict[str, Any],
    search_stack: list[str] | None,
    call: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call tool; on search failure try next in search_stack once/chain."""
    do_call = call or (lambda n, a: registry.call(n, a))
    available = set(registry.names())
    result = do_call(tool, arguments)
    if result.get("ok") or not is_search_tool(tool):
        return tool, result
    stack = list(search_stack or [])
    tried = {tool}
    current = tool
    while True:
        nxt = next_fallback(current, stack, available)
        if not nxt or nxt in tried:
            break
        tried.add(nxt)
        # remap common arg: query
        args = dict(arguments)
        result = do_call(nxt, args)
        current = nxt
        if result.get("ok"):
            result = {**result, "fallback_from": tool, "fallback_tool": nxt}
            return nxt, result
    return tool, result
