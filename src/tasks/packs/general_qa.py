"""Task pack: general reasoning with optional tools."""

from __future__ import annotations

from src.tasks.protocol import TaskSpec


def get_task_spec() -> TaskSpec:
    return TaskSpec(
        task_type="general_qa",
        domain="general",
        description="General Q&A; tools optional.",
        default_knowledge_mode="none",
        allowed_tools=[
            "kb_retrieve",
            "tavily_search",
            "fetch_page",
            "run_local_script",
            "mcp:anysearch",
        ],
        planner_hints=(
            "Use tools only when needed for facts; otherwise respond directly. "
            "When searching: prefer mcp_anysearch_search, fallback tavily_search."
        ),
        reflect_rules=["If tools used for facts, require body evidence"],
        require_body_evidence=False,
        search_stack=["mcp_anysearch_search", "mcp:anysearch", "tavily_search"],
    )
