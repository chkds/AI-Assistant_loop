"""Task pack stub: finance QA (tools to be added after unit tests)."""

from __future__ import annotations

from src.tasks.protocol import TaskSpec


def get_task_spec() -> TaskSpec:
    return TaskSpec(
        task_type="finance_qa",
        domain="finance",
        description="Finance Q&A scaffold — prefer MCP/search tools once unit-tested.",
        default_knowledge_mode="none",
        allowed_tools=["mcp:anysearch", "kb_retrieve", "fetch_page", "tavily_search"],
        planner_hints=(
            "Use mcp_anysearch_search with finance vertical via get_sub_domains when needed. "
            "Do not invent prices; cite tool evidence."
        ),
        reflect_rules=[
            "If tools used for facts, require body evidence",
            "If only web_snippet exists, decision=need_fetch",
        ],
        require_body_evidence=True,
    )
