"""Background agent jobs for async /agent/run (HITL non-blocking)."""

from __future__ import annotations

import threading
import traceback
from typing import Any

from src.agent.graph import run_session
from src.agent.interrupt import InterruptedError
from src.agent.session import get_session

_lock = threading.Lock()
_threads: dict[str, threading.Thread] = {}


def is_running(session_id: str) -> bool:
    with _lock:
        t = _threads.get(session_id)
        return bool(t and t.is_alive())


def _worker(session_id: str, start_node: str | None) -> None:
    try:
        store = get_session(session_id)
        state = store.load_state()
        run_session(store, state, start_node=start_node)
    except InterruptedError:
        try:
            store = get_session(session_id)
            state = store.load_state()
            state["status"] = "interrupted"
            store.append_event({"type": "interrupt", "actor": "worker"}, state=state)
            store.save_state(state)
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        try:
            store = get_session(session_id)
            state = store.load_state()
            state["status"] = "failed"
            state["last_observation"] = {
                "type": "error",
                "error": str(exc),
                "traceback": traceback.format_exc()[-2000:],
            }
            store.append_event(
                {"type": "failed", "error": str(exc), "actor": "worker"},
                state=state,
            )
            store.save_state(state)
        except Exception:  # noqa: BLE001
            pass
    finally:
        with _lock:
            _threads.pop(session_id, None)


def start_session_job(session_id: str, *, start_node: str | None = None) -> dict[str, Any]:
    """Start run_session in a daemon thread. Returns immediately."""
    with _lock:
        existing = _threads.get(session_id)
        if existing and existing.is_alive():
            return {"started": False, "reason": "already_running", "async": True}
        t = threading.Thread(
            target=_worker,
            args=(session_id, start_node),
            name=f"agent-job-{session_id}",
            daemon=True,
        )
        _threads[session_id] = t
        t.start()
    return {"started": True, "async": True, "start_node": start_node}
