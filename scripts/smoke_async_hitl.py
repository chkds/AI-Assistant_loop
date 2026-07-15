"""Short smoke: async /agent/run returns immediately → interrupt → status not running.

Also pokes SSE stream for one event batch (script I/O).

Usage:
  # terminal A:
  #   uvicorn src.api.main:app --host 127.0.0.1 --port 8000
  # terminal B:
  #   python scripts/smoke_async_hitl.py

  # Or in-process TestClient mode (default, no server):
  python scripts/smoke_async_hitl.py --inprocess
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.util.console_io import configure_stdout_utf8, dump_json, safe_print  # noqa: E402

configure_stdout_utf8()


def _inprocess() -> int:
    from fastapi.testclient import TestClient

    from src.api.main import app

    def slow_run(store, state, start_node=None):  # noqa: ARG001
        # Cooperative interrupt: check flag like real loop
        from src.agent.interrupt import InterruptedError, check_interrupt

        for _ in range(40):
            check_interrupt(store, state)
            time.sleep(0.05)
            state = store.load_state()
            state["step_count"] = int(state.get("step_count") or 0) + 1
            store.save_state(state)
        state["status"] = "done"
        state["final_answer"] = "should_not_reach_if_interrupted"
        store.save_state(state)
        return state

    client = TestClient(app)
    with patch("src.api.jobs.run_session", side_effect=slow_run):
        t0 = time.time()
        r = client.post("/agent/run", json={"query": "async interrupt smoke", "sync": False})
        elapsed = time.time() - t0
        dump_json("run_response", {"elapsed": elapsed, "body": r.json(), "status_code": r.status_code})
        assert r.status_code == 200
        data = r.json()
        ok_async = data.get("status") == "running" and data.get("detail", {}).get("async") is True and elapsed < 0.35
        sid = data["session_id"]

        time.sleep(0.15)
        ir = client.post(f"/agent/sessions/{sid}/interrupt", json={})
        dump_json("interrupt_response", ir.json())
        # API sets status=interrupted immediately (cooperative)
        final_status = ir.json().get("status")
        tr = client.get(f"/agent/sessions/{sid}/trajectory")
        traj = tr.json()
        if traj.get("status") and traj.get("status") != "running":
            final_status = traj.get("status")
        # Wait briefly for worker thread to exit
        for _ in range(40):
            tr = client.get(f"/agent/sessions/{sid}/trajectory")
            body = tr.json()
            final_status = body.get("status")
            if final_status != "running":
                break
            time.sleep(0.05)
        dump_json("after_interrupt", {"status": final_status, "job_running": traj.get("job_running")})

        # SSE: read a few lines via streaming (may end quickly)
        sse_ok = False
        try:
            with client.stream("GET", f"/agent/sessions/{sid}/events/stream?offset=0") as resp:
                n = 0
                for line in resp.iter_lines():
                    if line:
                        n += 1
                        safe_print("sse:", line[:200])
                    if n >= 3:
                        sse_ok = True
                        break
                if n > 0:
                    sse_ok = True
        except Exception as exc:  # noqa: BLE001
            safe_print("sse_error", exc)

        ok_stop = final_status is not None and final_status != "running"
        dump_json("summary", {"ok_async": ok_async, "ok_stop": ok_stop, "sse_ok": sse_ok, "final_status": final_status})
        ok = ok_async and ok_stop
        safe_print("RESULT:", "SUCCESS" if ok else "FAIL")
        return 0 if ok else 2


def _http(base: str) -> int:
    import urllib.request

    import json

    t0 = time.time()
    req = urllib.request.Request(
        base + "/agent/run",
        data=json.dumps({"query": "async interrupt smoke live", "sync": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    dump_json("run_response", {"elapsed": elapsed, "body": data})
    ok_async = data.get("status") == "running" and elapsed < 1.0
    sid = data["session_id"]
    time.sleep(0.5)
    req2 = urllib.request.Request(base + f"/agent/sessions/{sid}/interrupt", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req2, timeout=10) as resp:
        dump_json("interrupt", json.loads(resp.read().decode("utf-8")))
    final = None
    for _ in range(40):
        with urllib.request.urlopen(base + f"/agent/sessions/{sid}/trajectory", timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        final = body.get("status")
        if final != "running":
            break
        time.sleep(0.5)
    dump_json("final", {"status": final})
    ok = ok_async and final != "running"
    safe_print("RESULT:", "SUCCESS" if ok else "FAIL")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inprocess", action="store_true", default=True)
    ap.add_argument("--http", default="", help="Base URL e.g. http://127.0.0.1:8000")
    args = ap.parse_args()
    safe_print("=== smoke_async_hitl ===")
    if args.http:
        return _http(args.http.rstrip("/"))
    return _inprocess()


if __name__ == "__main__":
    raise SystemExit(main())
