"""CLI: run / stop / amend / resume / confirm-files / status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_yaml
from src.agent.amend import apply_amend
from src.agent.file_guard import confirm_files, require_confirm_if_changed
from src.agent.graph import run_session
from src.agent.interrupt import request_interrupt
from src.agent.session import SessionStore, get_session


def _print(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_run(args: argparse.Namespace) -> None:
    load_yaml.cache_clear()
    store = SessionStore()
    state = store.new_state(args.query)
    # optional watch paths
    if args.watch:
        state["file_watch_paths"] = list(args.watch)
        store.save_state(state)
    try:
        state = run_session(store, state)
    except KeyboardInterrupt:
        state = request_interrupt(store)
    _print(
        {
            "session_id": store.session_id,
            "status": state.get("status"),
            "final_answer": (state.get("final_answer") or "")[:2000],
            "reflect": state.get("reflect_decision"),
            "steps": state.get("step_count"),
            "evidence_count": len(state.get("evidence") or []),
        }
    )


def cmd_stop(args: argparse.Namespace) -> None:
    store = get_session(args.session)
    state = request_interrupt(store)
    _print({"session_id": store.session_id, "status": state.get("status"), "resume_hint": state.get("resume_hint")})


def cmd_amend(args: argparse.Namespace) -> None:
    store = get_session(args.session)
    # ensure interrupted first
    try:
        st = store.load_state()
        if st.get("status") == "running":
            request_interrupt(store)
    except Exception:
        pass
    result = apply_amend(store, args.text)
    _print(
        {
            "session_id": store.session_id,
            "action": result.get("action"),
            "classification": result.get("classification"),
            "to": result.get("to"),
            "status": result["state"].get("status"),
            "resume_from": result["state"].get("resume_from"),
        }
    )


def cmd_resume(args: argparse.Namespace) -> None:
    store = get_session(args.session)
    state = store.load_state()
    gate = require_confirm_if_changed(store, state)
    if gate.get("needs_confirm") and not args.force:
        _print(
            {
                "session_id": store.session_id,
                "status": "awaiting_confirm",
                "message": "File changes detected; run confirm-files first",
                "changes": gate.get("changes"),
            }
        )
        return
    start = args.from_node or state.get("resume_from") or state.get("resume_hint") or "plan"
    state["resume_from"] = start
    state = run_session(store, state, start_node=start)
    _print(
        {
            "session_id": store.session_id,
            "status": state.get("status"),
            "final_answer": (state.get("final_answer") or "")[:2000],
            "steps": state.get("step_count"),
        }
    )


def cmd_confirm_files(args: argparse.Namespace) -> None:
    store = get_session(args.session)
    accepted = args.accept.split(",") if args.accept else None
    rejected = args.reject.split(",") if args.reject else None
    result = confirm_files(store, accepted=accepted, rejected=rejected)
    _print(
        {
            "session_id": store.session_id,
            "resume_node": result.get("resume_node"),
            "status": result["state"].get("status"),
        }
    )


def cmd_status(args: argparse.Namespace) -> None:
    store = get_session(args.session)
    state = store.load_state()
    from src.agent.checkpoint import list_checkpoints

    _print(
        {
            "session_id": store.session_id,
            "status": state.get("status"),
            "goal": state.get("goal"),
            "resume_hint": state.get("resume_hint"),
            "resume_from": state.get("resume_from"),
            "checkpoints": list_checkpoints(store),
            "pending": store.load_pending(),
            "step_count": state.get("step_count"),
            "evidence": [
                {
                    "id": e.get("id"),
                    "source_type": e.get("source_type"),
                    "has_body": e.get("has_body"),
                }
                for e in (state.get("evidence") or [])[:20]
            ],
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase-2 agent control")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("query")
    p_run.add_argument("--watch", nargs="*", default=[])
    p_run.set_defaults(func=cmd_run)

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--session", required=True)
    p_stop.set_defaults(func=cmd_stop)

    p_amend = sub.add_parser("amend")
    p_amend.add_argument("--session", required=True)
    p_amend.add_argument("--text", required=True)
    p_amend.set_defaults(func=cmd_amend)

    p_resume = sub.add_parser("resume")
    p_resume.add_argument("--session", required=True)
    p_resume.add_argument("--from-node", default=None)
    p_resume.add_argument("--force", action="store_true", help="Skip file confirm gate")
    p_resume.set_defaults(func=cmd_resume)

    p_cf = sub.add_parser("confirm-files")
    p_cf.add_argument("--session", required=True)
    p_cf.add_argument("--accept", default=None, help="Comma-separated paths (default all)")
    p_cf.add_argument("--reject", default=None)
    p_cf.set_defaults(func=cmd_confirm_files)

    p_st = sub.add_parser("status")
    p_st.add_argument("--session", required=True)
    p_st.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
