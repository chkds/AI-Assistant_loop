"""Cooperative interrupt flag helpers."""

from __future__ import annotations

from src.agent.session import SessionStore


class InterruptedError(RuntimeError):
    pass


def check_interrupt(store: SessionStore, state: dict) -> None:
    if store.is_interrupted() or state.get("interrupt_flag"):
        state["interrupt_flag"] = True
        state["status"] = "interrupted"
        raise InterruptedError("session interrupted by user")


def request_interrupt(store: SessionStore, state: dict | None = None) -> dict:
    store.set_interrupt(True)
    if state is None:
        state = store.load_state()
    state["interrupt_flag"] = True
    state["status"] = "interrupted"
    # act mid-flight → resume hint is prior plan
    hint = "plan"
    if state.get("current_step_id") and "_act" not in str(state.get("current_step_id")):
        node = str(state.get("current_step_id")).split("_", 1)[-1]
        if node in {"classify", "gate", "plan", "observe", "reflect", "compress"}:
            hint = node
    state["resume_hint"] = hint
    store.append_event({"type": "interrupt", "resume_hint": hint})
    store.save_state(state)
    return state
