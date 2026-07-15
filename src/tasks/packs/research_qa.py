"""Task pack: local paper QA."""

from __future__ import annotations

from src.tasks.protocol import TaskSpec


def get_task_spec() -> TaskSpec:
    return TaskSpec(
        task_type="research_qa",
        domain="research",
        description="Answer questions using the local MinerU paper knowledge base.",
        default_knowledge_mode="retrieve",
        allowed_tools=["kb_retrieve", "run_local_script"],
        planner_hints=(
            "Prefer kb_retrieve for corpus QA. "
            "If user pins a paper (钉住/pinned:/@Doc), knowledge_mode=pinned and broker injects full.md — "
            "do not vector-search that turn. Do not use web search unless user explicitly asks."
        ),
        reflect_rules=[
            "Final answer requires kb_body evidence with has_body=true",
            "Reject title-only evidence",
        ],
        require_body_evidence=True,
    )
