"""Smoke test: RetrievalGate + embed query + Qdrant search (+ parent expand)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_routing
from src.control.retrieval_gate import RetrievalGate, expand_parent
from src.ingest.embedder.base import get_text_embedder
from src.memory.qdrant_store import QdrantChunkStore


def _safe_print(msg: str) -> None:
    """Avoid Windows console GBK crashes on paper unicode."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Smoke retrieve against Qdrant")
    parser.add_argument("query", nargs="?", default="radio propagation GNN")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--pinned", nargs="*", default=[])
    parser.add_argument("--force-mode", choices=["none", "pinned", "retrieve"], default=None)
    args = parser.parse_args(argv)

    routing = load_routing()
    top_k = args.top_k or int(routing.get("retrieval", {}).get("top_k", 5))
    expand = bool(routing.get("retrieval", {}).get("expand_parent", True))

    gate = RetrievalGate()
    decision = gate.decide(args.query, pinned_docs=args.pinned, force_mode=args.force_mode)
    _safe_print("Gate: " + json.dumps(decision.__dict__, ensure_ascii=False))

    if decision.knowledge_mode == "none":
        _safe_print("No retrieval (mode=none)")
        return

    if decision.knowledge_mode == "pinned":
        _safe_print(
            "Pinned mode - skip vector search; caller should inject docs: "
            + str(list(args.pinned))
        )
        return

    embedder = get_text_embedder()
    store = QdrantChunkStore(vector_size=embedder.dimensions)
    vector = embedder.embed_query(args.query)
    hits = store.search(
        vector,
        top_k=top_k,
        content_types=["text", "table", "formula", "figure"],
    )
    if expand:
        hits = expand_parent(hits, store)

    _safe_print(f"Hits: {len(hits)} | model={embedder.model} dims={embedder.dimensions}")
    for i, h in enumerate(hits, 1):
        preview = (h.get("text") or "")[:240].replace("\n", " ")
        _safe_print(
            f"\n[{i}] score={h.get('score'):.4f} doc={h.get('doc_id')}\n"
            f"    type={h.get('content_type')} section={h.get('headers_path')}\n"
            f"    {preview}..."
        )
        if h.get("parent_text"):
            pprev = h["parent_text"][:160].replace("\n", " ")
            _safe_print(f"    parent: {pprev}...")


if __name__ == "__main__":
    main()
