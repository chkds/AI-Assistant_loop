"""FastAPI health + agent session control surface (HITL-ready contracts)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.agent.amend import apply_amend
from src.agent.checkpoint import list_checkpoints
from src.agent.file_guard import confirm_files, require_confirm_if_changed
from src.agent.graph import run_session
from src.agent.interrupt import request_interrupt
from src.agent.session import SessionStore, get_session, sessions_root
from src.api.bootstrap import corpus_status, start_bootstrap_run
from src.api.jobs import is_running, start_session_job
from src.api.schemas import (
    AmendBody,
    BootstrapOut,
    CatalogOut,
    ConfirmBody,
    ControlOut,
    EventsOut,
    EvidenceItemOut,
    ResumeBody,
    RunBody,
    SessionListOut,
    SessionSummary,
    TaskSpecOut,
    ToolSchemaOut,
    ToolTraceItemOut,
    TrajectoryOut,
    WorkItemOut,
)
from src.memory.qdrant_store import make_qdrant_client
from src.tasks.registry import get_task_registry
from src.tools.mcp_adapter import load_mcp_config
from src.tools.registry import get_registry

app = FastAPI(
    title="AI Assistant Loop",
    version="0.3.0",
    description=(
        "Agent control + catalog APIs for CLI and HITL WebUI. "
        "Trajectory/events expose tool I/O and evidence for the frontend."
    ),
)

_WEB_HITL = Path(__file__).resolve().parents[2] / "web" / "hitl"
if _WEB_HITL.is_dir():
    app.mount("/ui/assets", StaticFiles(directory=str(_WEB_HITL)), name="hitl_assets")


@app.get("/ui")
@app.get("/ui/")
def hitl_ui() -> FileResponse:
    index = _WEB_HITL / "index.html"
    if not index.exists():
        raise HTTPException(404, "HITL UI not found (web/hitl/index.html)")
    return FileResponse(index)


def _ping_qdrant() -> dict[str, Any]:
    try:
        client, info = make_qdrant_client(timeout=2.0)
        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        close = getattr(client, "close", None)
        if callable(close):
            close()
        return {"ok": True, **info, "collections": names}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _text_preview(text: str | None, n: int = 240) -> str | None:
    if text is None:
        return None
    t = str(text)
    return t if len(t) <= n else t[:n] + "…"


def _evidence_out(items: list[dict[str, Any]] | None) -> list[EvidenceItemOut]:
    out: list[EvidenceItemOut] = []
    for e in items or []:
        out.append(
            EvidenceItemOut(
                id=e.get("id"),
                source_type=e.get("source_type"),
                has_body=e.get("has_body"),
                text_preview=_text_preview(e.get("text") or e.get("snippet")),
                url=e.get("url"),
                stale=bool(e.get("stale")),
            )
        )
    return out


def _tool_trace_out(items: list[dict[str, Any]] | None) -> list[ToolTraceItemOut]:
    out: list[ToolTraceItemOut] = []
    for t in items or []:
        out.append(
            ToolTraceItemOut(
                tool=t.get("tool"),
                ok=t.get("ok") if "ok" in t else t.get("result_ok"),
                evidence_kind=t.get("evidence_kind"),
                evidence_ids=list(t.get("evidence_ids") or []),
                evidence_count=t.get("evidence_count"),
                error=t.get("error"),
                arguments=dict(t.get("arguments") or {}),
            )
        )
    return out


def _work_items_out(state: dict[str, Any]) -> list[WorkItemOut]:
    out: list[WorkItemOut] = []
    for it in state.get("work_items") or []:
        out.append(
            WorkItemOut(
                id=it.get("id"),
                title=it.get("title"),
                status=it.get("status"),
                task_type=it.get("task_type"),
                depends_on=list(it.get("depends_on") or []),
                goal=it.get("goal"),
                expect=it.get("expect"),
                acceptance=it.get("acceptance"),
                result_ref=it.get("result_ref"),
                feedback=it.get("feedback") if isinstance(it.get("feedback"), dict) else None,
                evidence_ids=list(it.get("evidence_ids") or []),
            )
        )
    return out


def _trajectory(store: SessionStore, state: dict[str, Any]) -> TrajectoryOut:
    answers = state.get("item_answers") or {}
    return TrajectoryOut(
        session_id=store.session_id,
        status=state.get("status"),
        query=state.get("query"),
        goal=state.get("goal"),
        task_type=state.get("task_type"),
        knowledge_mode=state.get("knowledge_mode"),
        step_count=state.get("step_count"),
        plan=list(state.get("plan") or []),
        tool_trace=_tool_trace_out(state.get("tool_trace")),
        evidence=_evidence_out(state.get("evidence")),
        last_observation=state.get("last_observation"),
        reflect_decision=state.get("reflect_decision"),
        final_answer=state.get("final_answer"),
        resume_from=state.get("resume_from"),
        resume_hint=state.get("resume_hint"),
        checkpoints=list_checkpoints(store),
        pending=store.load_pending(),
        work_items=_work_items_out(state),
        current_work_item_id=state.get("current_work_item_id"),
        item_answers={str(k): str(v) for k, v in answers.items()},
        job_running=is_running(store.session_id),
    )


def _read_events(store: SessionStore, offset: int = 0, limit: int = 200) -> EventsOut:
    path = store.events_path
    events: list[dict[str, Any]] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        slice_ = lines[offset : offset + limit]
        for line in slice_:
            if not line.strip():
                continue
            try:
                events.append(__import__("json").loads(line))
            except Exception:  # noqa: BLE001
                events.append({"raw": line})
        next_off = offset + len(slice_)
    else:
        next_off = offset
    return EventsOut(session_id=store.session_id, events=events, next_offset=next_off)


@app.get("/health")
def health() -> dict[str, Any]:
    q = _ping_qdrant()
    corp = corpus_status()
    return {
        "status": "ok",
        "qdrant": q,
        "corpus": corp,
        "ready_for_bootstrap": bool(q.get("ok") and corp.get("ok")),
    }


@app.get("/agent/catalog", response_model=CatalogOut)
def agent_catalog() -> CatalogOut:
    """Tool + task catalog for frontend planner / allowlist UI."""
    reg = get_registry()
    tasks = get_task_registry()
    mcp_cfg = load_mcp_config()
    enabled_servers = [
        str(s.get("id") or s.get("name"))
        for s in (mcp_cfg.get("servers") or [])
        if s.get("enabled")
    ]
    return CatalogOut(
        tools=[ToolSchemaOut(**s) for s in reg.schemas()],
        tasks=[
            TaskSpecOut(
                task_type=t.task_type,
                domain=t.domain,
                description=t.description,
                default_knowledge_mode=t.default_knowledge_mode,
                allowed_tools=list(t.allowed_tools),
                require_body_evidence=t.require_body_evidence,
                search_stack=list(getattr(t, "search_stack", None) or []),
            )
            for t in (tasks.get(n) for n in tasks.names())
            if t is not None
        ],
        mcp_enabled_servers=enabled_servers,
    )


@app.get("/agent/sessions", response_model=SessionListOut)
def agent_list_sessions() -> SessionListOut:
    root = sessions_root()
    rows: list[SessionSummary] = []
    for p in sorted(root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        state_path = p / "state.json"
        if not state_path.exists():
            continue
        try:
            import json

            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            SessionSummary(
                session_id=p.name,
                status=state.get("status"),
                query=state.get("query"),
                goal=state.get("goal"),
                task_type=state.get("task_type"),
                step_count=state.get("step_count"),
                updated_at=state.get("updated_at"),
                has_final_answer=bool(state.get("final_answer")),
            )
        )
    return SessionListOut(sessions=rows)


@app.post("/agent/run", response_model=ControlOut)
def agent_run(body: RunBody) -> ControlOut:
    store = SessionStore()
    state = store.new_state(body.query)
    if body.task_type:
        state["task_type"] = body.task_type
    if body.watch:
        state["file_watch_paths"] = list(body.watch)
    if body.pinned_docs:
        state["pinned_docs"] = list(body.pinned_docs)
        state["knowledge_mode"] = "pinned"
    state["status"] = "running"
    store.save_state(state)

    if body.sync:
        state = run_session(store, state)
        return ControlOut(
            session_id=store.session_id,
            status=state.get("status"),
            detail={
                "async": False,
                "final_answer": state.get("final_answer"),
                "reflect": state.get("reflect_decision"),
                "step_count": state.get("step_count"),
                "task_type": state.get("task_type"),
                "evidence_count": len(state.get("evidence") or []),
                "tool_trace_count": len(state.get("tool_trace") or []),
            },
        )

    job = start_session_job(store.session_id)
    return ControlOut(
        session_id=store.session_id,
        status="running",
        detail={
            "async": True,
            **job,
            "task_type": state.get("task_type"),
            "hint": "Poll GET /agent/sessions/{id} or SSE .../events/stream",
        },
    )


@app.post("/agent/bootstrap", response_model=BootstrapOut)
def agent_bootstrap(sync: bool = Query(False)) -> BootstrapOut:
    """First-success path: report corpus readiness and start fixed research_qa run."""
    q = _ping_qdrant()
    corp = corpus_status()
    if not q.get("ok"):
        return BootstrapOut(
            ok=False,
            health={"qdrant": q},
            corpus=corp,
            message="Qdrant unavailable",
        )
    if not corp.get("ok"):
        return BootstrapOut(
            ok=False,
            health={"qdrant": q},
            corpus=corp,
            message=corp.get("hint") or "Corpus empty — ingest default doc first",
        )
    started = start_bootstrap_run(sync=sync)
    return BootstrapOut(
        ok=True,
        health={"qdrant": q},
        corpus=corp,
        session_id=started.get("session_id"),
        status=started.get("status"),
        detail=started,
        message="bootstrap session started",
    )


@app.get("/agent/sessions/{session_id}", response_model=TrajectoryOut)
def agent_status(session_id: str) -> TrajectoryOut:
    try:
        store = get_session(session_id)
        state = store.load_state()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return _trajectory(store, state)


@app.get("/agent/sessions/{session_id}/trajectory", response_model=TrajectoryOut)
def agent_trajectory(session_id: str) -> TrajectoryOut:
    """Alias of session status — explicit name for frontend timeline views."""
    return agent_status(session_id)


@app.get("/agent/sessions/{session_id}/events", response_model=EventsOut)
def agent_events(
    session_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
) -> EventsOut:
    try:
        store = get_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return _read_events(store, offset=offset, limit=limit)


@app.get("/agent/sessions/{session_id}/events/stream")
async def agent_events_stream(session_id: str, offset: int = Query(0, ge=0)):
    """SSE: emit new events.jsonl lines; ends when session leaves running and job idle."""
    try:
        store = get_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    async def gen():
        off = offset
        idle_rounds = 0
        while True:
            batch = _read_events(store, offset=off, limit=50)
            for ev in batch.events:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            off = batch.next_offset
            try:
                st = store.load_state()
                status = st.get("status")
            except Exception:  # noqa: BLE001
                status = "failed"
            running = is_running(session_id) or status == "running"
            if not batch.events:
                idle_rounds += 1
            else:
                idle_rounds = 0
            if not running and idle_rounds >= 2:
                yield f"data: {json.dumps({'type': 'stream_end', 'status': status}, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(0.8)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/agent/sessions/{session_id}/interrupt", response_model=ControlOut)
def agent_interrupt(session_id: str) -> ControlOut:
    store = get_session(session_id)
    state = request_interrupt(store)
    return ControlOut(
        session_id=session_id,
        status=state.get("status"),
        detail={"resume_hint": state.get("resume_hint"), "resume_from": state.get("resume_from")},
    )


@app.post("/agent/sessions/{session_id}/amend", response_model=ControlOut)
def agent_amend(session_id: str, body: AmendBody) -> ControlOut:
    store = get_session(session_id)
    st = store.load_state()
    if st.get("status") == "running":
        request_interrupt(store)
    result = apply_amend(
        store,
        body.text,
        body.tags,
        target_work_item_id=body.target_work_item_id,
    )
    return ControlOut(
        session_id=session_id,
        status=result["state"].get("status"),
        detail={
            "action": result.get("action"),
            "classification": result.get("classification"),
            "to": result.get("to"),
            "target_work_item_id": result.get("target_work_item_id"),
            "resume_from": result["state"].get("resume_from"),
        },
    )


@app.post("/agent/sessions/{session_id}/resume", response_model=ControlOut)
def agent_resume(session_id: str, body: ResumeBody) -> ControlOut:
    store = get_session(session_id)
    state = store.load_state()
    gate = require_confirm_if_changed(store, state)
    if gate.get("needs_confirm") and not body.force:
        return ControlOut(
            session_id=session_id,
            status="awaiting_confirm",
            detail={"changes": gate.get("changes")},
        )
    if is_running(session_id):
        return ControlOut(
            session_id=session_id,
            status="running",
            detail={"async": True, "started": False, "reason": "already_running"},
        )
    start = body.from_node or state.get("resume_from") or state.get("resume_hint") or "plan"
    if body.sync:
        state = run_session(store, state, start_node=start)
        return ControlOut(
            session_id=session_id,
            status=state.get("status"),
            detail={
                "async": False,
                "final_answer": state.get("final_answer"),
                "step_count": state.get("step_count"),
                "evidence_count": len(state.get("evidence") or []),
            },
        )
    state["status"] = "running"
    store.save_state(state)
    job = start_session_job(session_id, start_node=start)
    return ControlOut(
        session_id=session_id,
        status="running",
        detail={"async": True, "start_node": start, **job},
    )


@app.post("/agent/sessions/{session_id}/confirm-files", response_model=ControlOut)
def agent_confirm_files(session_id: str, body: ConfirmBody) -> ControlOut:
    store = get_session(session_id)
    result = confirm_files(store, accepted=body.accept, rejected=body.reject)
    return ControlOut(
        session_id=session_id,
        status=result["state"].get("status"),
        detail={"resume_node": result.get("resume_node")},
    )


def main() -> None:
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
