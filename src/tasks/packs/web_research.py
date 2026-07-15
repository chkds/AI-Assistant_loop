"""Task pack: web + optional KB research."""

from __future__ import annotations

from src.tasks.protocol import TaskSpec


def get_task_spec() -> TaskSpec:
    return TaskSpec(
        task_type="web_research",
        domain="general",
        description="Research using web search plus page body fetch; KB optional.",
        default_knowledge_mode="none",
        allowed_tools=[
            "tavily_search",
            "fetch_page",
            "kb_retrieve",
            "run_local_script",
            "mcp:anysearch",  # Streamable HTTP MCP when enabled in mcp_servers.yaml
        ],
        planner_hints=(
            "Search stack (primary first): mcp_anysearch_search, then tavily_search. "
            "Prefer AnySearch; if it fails the runtime may fall back to Tavily. "
            "After tavily_search you MUST plan fetch_page for body. "
            "MCP search text with has_body can count as body evidence; "
            "never treat short snippets alone as final evidence."
        ),
        reflect_rules=[
            "Require body evidence: web_body, kb_body, or mcp with has_body",
            "If only web_snippet exists, decision=need_fetch",
        ],
        require_body_evidence=True,
        search_stack=["mcp_anysearch_search", "mcp:anysearch", "tavily_search"],
    )
