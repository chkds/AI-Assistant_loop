"""First-success bootstrap CLI: health/corpus check → optional ingest hint → async session.

Usage:
  E:\\application\\miniforge3\\envs\\copilot-agent\\python.exe scripts\\bootstrap_first_run.py
  E:\\application\\miniforge3\\envs\\copilot-agent\\python.exe scripts\\bootstrap_first_run.py --sync
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.bootstrap import corpus_status, start_bootstrap_run  # noqa: E402
from src.api.jobs import is_running  # noqa: E402
from src.agent.session import get_session  # noqa: E402
from src.memory.qdrant_store import make_qdrant_client  # noqa: E402
from src.util.console_io import configure_stdout_utf8, dump_json, safe_print  # noqa: E402

configure_stdout_utf8()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true", help="Block until agent finishes")
    ap.add_argument("--wait-sec", type=int, default=120, help="Max wait when async")
    args = ap.parse_args()

    safe_print("=== bootstrap_first_run ===")
    try:
        client, info = make_qdrant_client(timeout=3.0)
        close = getattr(client, "close", None)
        if callable(close):
            close()
        q_ok = True
        q = info
    except Exception as exc:  # noqa: BLE001
        q_ok = False
        q = {"error": str(exc)}

    corp = corpus_status()
    dump_json("health", {"qdrant_ok": q_ok, "qdrant": q, "corpus": corp})
    if not q_ok:
        safe_print("RESULT: FAIL (qdrant)")
        return 2
    if not corp.get("ok"):
        safe_print("RESULT: FAIL (corpus empty)")
        safe_print("HINT:", corp.get("hint"))
        return 3

    started = start_bootstrap_run(sync=args.sync)
    dump_json("started", started)
    sid = started["session_id"]
    if args.sync:
        safe_print("RESULT:", "SUCCESS" if started.get("status") == "done" else "FAIL")
        return 0 if started.get("status") == "done" else 2

    deadline = time.time() + max(5, args.wait_sec)
    while time.time() < deadline:
        st = get_session(sid).load_state()
        running = is_running(sid) or st.get("status") == "running"
        safe_print("poll", sid, "status=", st.get("status"), "job=", is_running(sid), "steps=", st.get("step_count"))
        if not running and st.get("status") in {"done", "failed", "interrupted", "awaiting_confirm"}:
            dump_json(
                "final",
                {
                    "status": st.get("status"),
                    "final_answer_preview": (st.get("final_answer") or "")[:500],
                    "work_items": [
                        {"id": i.get("id"), "title": i.get("title"), "status": i.get("status")}
                        for i in (st.get("work_items") or [])
                    ],
                    "evidence": len(st.get("evidence") or []),
                },
            )
            ok = st.get("status") == "done" and bool(st.get("final_answer") or st.get("item_answers"))
            safe_print("RESULT:", "SUCCESS" if ok else "FAIL")
            return 0 if ok else 2
        time.sleep(2)

    safe_print("RESULT: FAIL (timeout)")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
