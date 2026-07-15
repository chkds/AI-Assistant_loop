"""Minimal Retrieval Gate — decide whether / how to use the knowledge base."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from src import load_routing
from src.ingest.chunker.multimodal import count_tokens


@dataclass
class GateDecision:
    knowledge_mode: str  # none | pinned | retrieve
    reason: str
    pinned_docs: list[str] = field(default_factory=list)
    token_budget: int = 32000


class RetrievalGate:
    def __init__(self, config: dict | None = None):
        self.cfg = config or load_routing()
        defaults = self.cfg.get("defaults", {})
        self.default_mode = defaults.get("knowledge_mode", "retrieve")
        self.token_budget = int(defaults.get("token_budget", 32000))
        self.safe_ratio = float(defaults.get("safe_context_ratio", 0.35))

    @property
    def safe_budget(self) -> int:
        return int(self.token_budget * self.safe_ratio)

    def decide(
        self,
        query: str,
        pinned_docs: Sequence[str] | None = None,
        pinned_texts: Sequence[str] | None = None,
        force_mode: str | None = None,
    ) -> GateDecision:
        if force_mode:
            return GateDecision(
                knowledge_mode=force_mode,
                reason=f"forced:{force_mode}",
                pinned_docs=list(pinned_docs or []),
                token_budget=self.token_budget,
            )

        pinned_docs = list(pinned_docs or [])
        pinned_texts = list(pinned_texts or [])

        if not query.strip() and not pinned_docs:
            return GateDecision("none", "empty_query", token_budget=self.token_budget)

        if pinned_docs or pinned_texts:
            total = sum(count_tokens(t) for t in pinned_texts) if pinned_texts else 0
            # if texts not provided, assume pinned paths should be loaded by caller; prefer pinned
            if not pinned_texts or total <= self.safe_budget:
                return GateDecision(
                    "pinned",
                    f"pinned_within_budget tokens={total} budget={self.safe_budget}",
                    pinned_docs=pinned_docs,
                    token_budget=self.token_budget,
                )
            return GateDecision(
                "retrieve",
                f"pinned_overflow tokens={total} budget={self.safe_budget}; fall_back_retrieve",
                pinned_docs=pinned_docs,
                token_budget=self.token_budget,
            )

        # Phase 1 default: retrieve for non-empty knowledge-seeking queries
        return GateDecision(
            self.default_mode,
            "default_retrieve",
            token_budget=self.token_budget,
        )


def expand_parent(hits: list[dict[str, Any]], store) -> list[dict[str, Any]]:
    """Small-to-big: attach parent section text when child is hit."""
    expanded = []
    seen_parents: set[str] = set()
    for hit in hits:
        item = dict(hit)
        parent_id = hit.get("parent_id")
        if parent_id and parent_id not in seen_parents:
            parent = store.get_parent(parent_id)
            if parent:
                item["parent_text"] = parent.get("text")
                item["parent_headers"] = parent.get("headers_path")
                seen_parents.add(parent_id)
        expanded.append(item)
    return expanded
