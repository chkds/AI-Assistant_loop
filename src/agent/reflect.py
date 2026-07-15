"""Reflect decisions — pure logic, unit-tested without LLM/network."""

from __future__ import annotations

from typing import Any


BODY_SOURCE_TYPES = {"kb_body", "web_body", "table", "formula", "figure", "text", "mcp"}


def active_evidence(state: dict[str, Any]) -> list[dict[str, Any]]:
    stale = set(state.get("stale_evidence_ids") or [])
    return [
        e
        for e in state.get("evidence") or []
        if e.get("id") not in stale and not e.get("stale")
    ]


def has_body_evidence(evidence: list[dict[str, Any]], min_body_chars: int = 200) -> bool:
    for e in evidence:
        if e.get("source_type") not in BODY_SOURCE_TYPES:
            continue
        if e.get("has_body") or len((e.get("text") or "").strip()) >= min_body_chars:
            return True
    return False


def only_snippet_evidence(evidence: list[dict[str, Any]], min_body_chars: int = 200) -> bool:
    if not evidence:
        return False
    if has_body_evidence(evidence, min_body_chars):
        return False
    return all(
        e.get("source_type") == "web_snippet" or not e.get("has_body") for e in evidence
    )


def apply_reflect_rules(
    state: dict[str, Any],
    decision: dict[str, str],
    rules: list[str] | None,
    *,
    min_body_chars: int = 200,
) -> dict[str, str]:
    """Enforce TaskSpec.reflect_rules as light keyword policies on top of decide_reflect."""
    if not rules:
        return decision
    evidence = active_evidence(state)
    joined = " | ".join(rules).lower()
    out = dict(decision)

    if ("need_fetch" in joined or "snippet" in joined) and only_snippet_evidence(evidence, min_body_chars):
        if (state.get("next_action") or {}).get("action") == "respond" and state.get("final_answer"):
            return {"decision": "need_fetch", "reason": "reflect_rule_snippet_only"}

    if "body" in joined and ("require" in joined or "至少" in joined or "must" in joined):
        if (state.get("next_action") or {}).get("action") == "respond" and state.get("final_answer"):
            if not has_body_evidence(evidence, min_body_chars):
                # Prefer continue (replan/act) over done
                if out.get("decision") in {"llm_judge", "done"}:
                    return {"decision": "continue", "reason": "reflect_rule_missing_body"}

    return out


def decide_reflect(
    state: dict[str, Any],
    *,
    reject_title_only: bool = True,
    min_body_chars: int = 200,
    require_body: bool = True,
    max_steps: int = 12,
    reflect_rules: list[str] | None = None,
) -> dict[str, str]:
    """
    Return {decision, reason} where decision in:
    continue | done | need_fetch | replan | llm_judge
    """
    evidence = active_evidence(state)
    has_body = has_body_evidence(evidence, min_body_chars)
    snippets_only = only_snippet_evidence(evidence, min_body_chars)
    action = (state.get("next_action") or {}).get("action")
    step_count = int(state.get("step_count") or 0)

    if action == "respond" and state.get("final_answer"):
        if reject_title_only and snippets_only:
            base = {"decision": "need_fetch", "reason": "title_or_snippet_only"}
        elif require_body and reject_title_only and not has_body and state.get("knowledge_mode") == "retrieve":
            base = {"decision": "continue", "reason": "missing_body_evidence"}
        else:
            base = {"decision": "llm_judge", "reason": "needs_llm_quality_check"}
        return apply_reflect_rules(state, base, reflect_rules, min_body_chars=min_body_chars)

    if step_count >= max_steps:
        return {"decision": "done", "reason": "max_steps"}

    return {"decision": "continue", "reason": "default_continue"}


def should_compress(
    state: dict[str, Any],
    *,
    every_n: int = 4,
    token_count: int = 0,
    token_limit: int = 8000,
) -> bool:
    step = int(state.get("step_count") or 0)
    if every_n > 0 and step > 0 and step % every_n == 0:
        return True
    return token_count > token_limit
