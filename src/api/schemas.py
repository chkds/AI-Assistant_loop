"""API request/response contracts for frontend ↔ backend (HITL-ready).

These models define the stable wire format. WebUI can bind to them without
reaching into SessionStore internals.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ----- control requests -----


class RunBody(BaseModel):
    query: str
    watch: list[str] = Field(default_factory=list)
    task_type: str | None = None
    # Default async for HITL; sync=True blocks until done (CLI/tests)
    sync: bool = False
    pinned_docs: list[str] = Field(default_factory=list)


class AmendBody(BaseModel):
    text: str
    tags: list[str] = Field(default_factory=list)
    target_work_item_id: str | None = None


class ResumeBody(BaseModel):
    from_node: str | None = None
    force: bool = False
    sync: bool = False


class ConfirmBody(BaseModel):
    accept: list[str] | None = None
    reject: list[str] | None = None


# ----- catalog (tools / tasks for UI planner panels) -----


class ToolSchemaOut(BaseModel):
    name: str
    description: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    evidence_kind: str | None = None
    requires_body: bool = False


class TaskSpecOut(BaseModel):
    task_type: str
    domain: str
    description: str = ""
    default_knowledge_mode: str = "retrieve"
    allowed_tools: list[str] = Field(default_factory=list)
    require_body_evidence: bool = True
    search_stack: list[str] = Field(default_factory=list)


class CatalogOut(BaseModel):
    tools: list[ToolSchemaOut]
    tasks: list[TaskSpecOut]
    mcp_enabled_servers: list[str] = Field(default_factory=list)


# ----- session views -----


class SessionSummary(BaseModel):
    session_id: str
    status: str | None = None
    query: str | None = None
    goal: str | None = None
    task_type: str | None = None
    step_count: int | None = None
    updated_at: str | None = None
    has_final_answer: bool = False


class SessionListOut(BaseModel):
    sessions: list[SessionSummary]


class EvidenceItemOut(BaseModel):
    id: str | None = None
    source_type: str | None = None
    has_body: bool | None = None
    text_preview: str | None = None
    url: str | None = None
    stale: bool = False


class ToolTraceItemOut(BaseModel):
    tool: str | None = None
    ok: bool | None = None
    evidence_kind: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_count: int | None = None
    error: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)


class WorkItemOut(BaseModel):
    id: str | None = None
    title: str | None = None
    status: str | None = None
    task_type: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    goal: str | None = None
    expect: str | None = None
    acceptance: str | None = None
    result_ref: str | None = None
    feedback: dict[str, Any] | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class TrajectoryOut(BaseModel):
    """Single-payload view for a HITL timeline / debug panel."""

    session_id: str
    status: str | None = None
    query: str | None = None
    goal: str | None = None
    task_type: str | None = None
    knowledge_mode: str | None = None
    step_count: int | None = None
    plan: list[dict[str, Any]] = Field(default_factory=list)
    tool_trace: list[ToolTraceItemOut] = Field(default_factory=list)
    evidence: list[EvidenceItemOut] = Field(default_factory=list)
    last_observation: dict[str, Any] | None = None
    reflect_decision: dict[str, Any] | None = None
    final_answer: str | None = None
    resume_from: str | None = None
    resume_hint: str | None = None
    checkpoints: list[str] = Field(default_factory=list)
    pending: dict[str, Any] | None = None
    # Collaboration / HITL human-readable
    work_items: list[WorkItemOut] = Field(default_factory=list)
    current_work_item_id: str | None = None
    item_answers: dict[str, str] = Field(default_factory=dict)
    job_running: bool = False


class EventsOut(BaseModel):
    session_id: str
    events: list[dict[str, Any]]
    next_offset: int


class ControlOut(BaseModel):
    session_id: str
    status: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class BootstrapOut(BaseModel):
    ok: bool
    health: dict[str, Any] = Field(default_factory=dict)
    corpus: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    status: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
