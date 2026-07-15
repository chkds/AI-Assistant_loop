"""Session persistence and status machine."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import load_paths, load_session_cfg, resolve_path

STATUSES = {
    "created",
    "running",
    "interrupted",
    "awaiting_confirm",
    "failed",
    "done",
    "cancelled",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sessions_root() -> Path:
    paths = load_paths()
    rel = paths.get("sessions_dir") or load_session_cfg().get("sessions_dir") or "data/sessions"
    root = resolve_path(rel)
    root.mkdir(parents=True, exist_ok=True)
    return root


class SessionStore:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.root = sessions_root() / self.session_id
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "checkpoints").mkdir(exist_ok=True)
        (self.root / "artifacts").mkdir(exist_ok=True)
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self.snapshot_path = self.root / "workspace_snapshot.json"
        self.pending_path = self.root / "pending_confirm.json"
        self.interrupt_path = self.root / "INTERRUPT"

    def new_state(self, query: str) -> dict[str, Any]:
        state: dict[str, Any] = {
            "session_id": self.session_id,
            "query": query,
            "goal": query,
            "status": "created",
            "tier": "lite",
            "knowledge_mode": "retrieve",
            "plan": [],
            "current_step_id": None,
            "resume_from": None,
            "resume_hint": None,
            "interrupt_flag": False,
            "messages": [],
            "evidence": [],
            "stale_evidence_ids": [],
            "artifacts": [],
            "tool_trace": [],
            "amendments": [],
            "error_lessons": [],
            "compress_summary": "",
            "final_answer": "",
            "step_count": 0,
            "file_watch_paths": [],
            "pending_file_confirm": None,
            "reflect_decision": None,
            "last_observation": None,
            "next_action": None,
            # Collaboration foundation (F1/F6): schedulable units + session tree
            "work_items": [],
            "current_work_item_id": None,
            "child_session_ids": [],
            "item_answers": {},
            "needs_recompose": False,
            "pinned_docs": [],
            "pinned_loaded": False,
            "broker_pinned_ids": [],
            "created_at": _now(),
            "updated_at": _now(),
        }
        self.save_state(state)
        self.append_event({"type": "created", "query": query, "actor": "orchestrator"})
        return state

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            raise FileNotFoundError(f"session state missing: {self.state_path}")
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_event(self, event: dict[str, Any], state: dict[str, Any] | None = None) -> None:
        """Append timeline event. Optional state injects work_item_id for routing (F2)."""
        enriched = {**event, "ts": _now(), "session_id": self.session_id}
        if state is not None:
            wid = state.get("current_work_item_id")
            if wid and "work_item_id" not in enriched:
                enriched["work_item_id"] = wid
            if "actor" not in enriched:
                enriched["actor"] = "worker"
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    def set_interrupt(self, value: bool = True) -> None:
        if value:
            self.interrupt_path.write_text("1", encoding="utf-8")
        elif self.interrupt_path.exists():
            self.interrupt_path.unlink()

    def is_interrupted(self) -> bool:
        return self.interrupt_path.exists()

    def save_pending(self, payload: dict[str, Any]) -> None:
        self.pending_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_pending(self) -> dict[str, Any] | None:
        if not self.pending_path.exists():
            return None
        return json.loads(self.pending_path.read_text(encoding="utf-8"))

    def clear_pending(self) -> None:
        if self.pending_path.exists():
            self.pending_path.unlink()

    @staticmethod
    def list_sessions() -> list[str]:
        root = sessions_root()
        return sorted([p.name for p in root.iterdir() if p.is_dir()])


def get_session(session_id: str) -> SessionStore:
    return SessionStore(session_id=session_id)
