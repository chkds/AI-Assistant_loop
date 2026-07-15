"""First-success bootstrap: health + default corpus readiness + fixed QA session."""

from __future__ import annotations

from typing import Any

from src import load_yaml
from src.agent.session import SessionStore
from src.api.jobs import start_session_job
from src.memory.qdrant_store import make_qdrant_client

DEFAULT_DOC = "Bufort"
DEFAULT_QUERY = "用知识库解释 Bufort 的 GNN 传播建模思路"


def _expected_docs() -> list[str]:
    try:
        cfg = load_yaml("corpus.yaml")
        docs = list(cfg.get("default_embed_docs") or [])
        return docs or [DEFAULT_DOC]
    except Exception:  # noqa: BLE001
        return [DEFAULT_DOC]


def corpus_status() -> dict[str, Any]:
    expected = _expected_docs()
    try:
        client, info = make_qdrant_client(timeout=3.0)
        name = str(info.get("collection") or "research_papers")
        cols = client.get_collections()
        names = [c.name for c in cols.collections]
        count = None
        sample_docs: list[str] = []
        if name in names:
            try:
                count = client.count(collection_name=name, exact=False).count
            except Exception:  # noqa: BLE001
                count = None
            # sample distinct doc_id from a few points (best-effort)
            try:
                points, _ = client.scroll(collection_name=name, limit=64, with_payload=True)
                seen: set[str] = set()
                for p in points or []:
                    pl = p.payload or {}
                    did = pl.get("doc_id")
                    if did and did not in seen:
                        seen.add(str(did))
                        sample_docs.append(str(did))
                    if len(sample_docs) >= 12:
                        break
            except Exception:  # noqa: BLE001
                pass
        close = getattr(client, "close", None)
        if callable(close):
            close()

        missing = []
        for exp in expected:
            if not any(exp.lower() in d.lower() for d in sample_docs):
                # if we couldn't sample, don't mark missing solely on empty sample when count>0
                if sample_docs:
                    missing.append(exp)
        ready = bool(name in names and (count is None or int(count) > 0))
        if sample_docs and missing:
            ready = False
        return {
            "ok": ready,
            "collection": name,
            "collections": names,
            "point_count": count,
            "default_doc": expected[0] if expected else DEFAULT_DOC,
            "expected_docs": expected,
            "sampled_doc_ids": sample_docs,
            "missing_expected": missing,
            "hint": None
            if ready
            else f"Run: python scripts/ingest_one.py --doc {expected[0]} --embed",
            "mode": info.get("mode"),
            "path": info.get("path"),
            "url": info.get("url"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "default_doc": expected[0] if expected else DEFAULT_DOC,
            "expected_docs": expected,
            "hint": f"Start Qdrant local / ingest {expected[0] if expected else DEFAULT_DOC}",
        }


def start_bootstrap_run(*, sync: bool = False, pinned_docs: list[str] | None = None) -> dict[str, Any]:
    """Create a research_qa session with the fixed first-success query."""
    store = SessionStore()
    state = store.new_state(DEFAULT_QUERY)
    state["task_type"] = "research_qa"
    state["status"] = "running"
    if pinned_docs:
        state["pinned_docs"] = list(pinned_docs)
        state["knowledge_mode"] = "pinned"
    store.save_state(state)
    if sync:
        from src.agent.graph import run_session

        state = run_session(store, state)
        return {
            "session_id": store.session_id,
            "status": state.get("status"),
            "async": False,
            "query": DEFAULT_QUERY,
            "final_answer": state.get("final_answer"),
            "pinned_docs": state.get("pinned_docs") or [],
        }
    job = start_session_job(store.session_id)
    return {
        "session_id": store.session_id,
        "status": "running",
        "async": True,
        "query": DEFAULT_QUERY,
        **job,
    }
