"""Checkpoint save / rollback."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src import load_agent
from src.agent.session import SessionStore

STABLE = {"classify", "gate", "plan", "observe", "reflect", "compress", "done"}


def stable_nodes() -> set[str]:
    cfg = load_agent().get("stable_nodes") or list(STABLE)
    return set(cfg)


def save_checkpoint(store: SessionStore, state: dict[str, Any], node: str) -> str:
    step_id = f"{state.get('step_count', 0):04d}_{node}"
    path = store.root / "checkpoints" / f"{step_id}.json"
    snap = deepcopy(state)
    snap["checkpoint_node"] = node
    snap["checkpoint_id"] = step_id
    path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    state["current_step_id"] = step_id
    state.setdefault("checkpoints", [])
    if step_id not in state["checkpoints"]:
        state["checkpoints"].append(step_id)
    store.append_event({"type": "checkpoint", "step_id": step_id, "node": node})
    store.save_state(state)
    return step_id


def list_checkpoints(store: SessionStore) -> list[str]:
    d = store.root / "checkpoints"
    return sorted([p.stem for p in d.glob("*.json")])


def load_checkpoint(store: SessionStore, step_id: str) -> dict[str, Any]:
    path = store.root / "checkpoints" / f"{step_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {step_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def latest_plan_checkpoint(store: SessionStore) -> str | None:
    cps = [c for c in list_checkpoints(store) if c.endswith("_plan")]
    return cps[-1] if cps else None


def rollback_to(store: SessionStore, state: dict[str, Any], step_id: str) -> dict[str, Any]:
    """Restore checkpoint and mark later evidence/tools stale."""
    snap = load_checkpoint(store, step_id)
    # Preserve amendments and session meta
    amendments = list(state.get("amendments") or [])
    watch = list(state.get("file_watch_paths") or [])
    restored = deepcopy(snap)
    restored["amendments"] = amendments
    restored["file_watch_paths"] = watch
    restored["session_id"] = store.session_id
    restored["interrupt_flag"] = False
    restored["status"] = "running"
    restored["resume_from"] = snap.get("checkpoint_node") or "plan"

    # Mark evidence created after this step as stale (by absence from snap)
    keep_ids = {e.get("id") for e in (snap.get("evidence") or []) if e.get("id")}
    stale = list(state.get("stale_evidence_ids") or [])
    for e in state.get("evidence") or []:
        eid = e.get("id")
        if eid and eid not in keep_ids and eid not in stale:
            stale.append(eid)
    restored["stale_evidence_ids"] = stale
    # Keep full tool_trace in events only; active tool_trace from snap
    store.append_event(
        {
            "type": "rollback",
            "to_step_id": step_id,
            "to_node": restored.get("resume_from"),
            "stale_count": len(stale),
        }
    )
    store.set_interrupt(False)
    store.save_state(restored)
    return restored
