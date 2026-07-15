"""Offline retrieval eval against local Qdrant (no LLM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_yaml  # noqa: E402
from src.control.retrieval_gate import RetrievalGate  # noqa: E402
from src.ingest.embedder.base import get_text_embedder  # noqa: E402
from src.memory.qdrant_store import QdrantChunkStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eval_retrieve.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    queries = cfg.get("queries") or []
    top_k = int(cfg.get("top_k") or 5)

    gate = RetrievalGate()
    embedder = get_text_embedder()
    store = QdrantChunkStore(vector_size=embedder.dimensions)

    rows = []
    passed = 0
    for q in queries:
        qid = q.get("id") or q.get("query")
        query = q["query"]
        mode = gate.decide(query).knowledge_mode
        vec = embedder.embed_query(query)
        hits = store.search(vec, top_k=top_k)
        blob = " ".join(
            f"{h.get('doc_id','')} {h.get('text','')}" for h in hits
        ).lower()
        expects = [s.lower() for s in (q.get("expect_doc_substrings") or [])]
        hit_ok = all(exp in blob for exp in expects) if expects else bool(hits)
        passed += int(hit_ok)
        rows.append(
            {
                "id": qid,
                "mode": mode,
                "ok": hit_ok,
                "n_hits": len(hits),
                "top_doc": (hits[0].get("doc_id") if hits else None),
            }
        )
        print(f"[{'PASS' if hit_ok else 'FAIL'}] {qid} hits={len(hits)} top={rows[-1]['top_doc']}")

    summary = {"total": len(rows), "passed": passed, "failed": len(rows) - passed}
    print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    return 0 if passed == len(rows) and rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
