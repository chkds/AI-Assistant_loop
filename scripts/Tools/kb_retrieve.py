"""KB retrieve: Qdrant top-k with parent expand (body chunks, cost-capped)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_tools_cfg, load_yaml
from src.control.retrieval_gate import expand_parent
from src.ingest.chunker.multimodal import count_tokens
from src.ingest.embedder.base import get_text_embedder
from src.memory.qdrant_store import QdrantChunkStore


def kb_retrieve(query: str, top_k: int | None = None) -> dict:
    load_yaml.cache_clear()
    cfg = load_tools_cfg().get("kb_retrieve", {})
    k = int(top_k or cfg.get("top_k", 3))
    max_tokens = int(cfg.get("max_inject_tokens", 5000))
    content_types = list(cfg.get("content_types") or ["text", "table", "formula", "figure"])
    expand = bool(cfg.get("expand_parent", True))

    embedder = get_text_embedder()
    store = QdrantChunkStore(vector_size=embedder.dimensions)
    vector = embedder.embed_query(query)
    hits = store.search(vector, top_k=k, content_types=content_types)
    if expand:
        hits = expand_parent(hits, store)

    selected = []
    used = 0
    for h in hits:
        body = (h.get("text") or "") + "\n" + (h.get("parent_text") or "")
        t = count_tokens(body)
        if selected and used + t > max_tokens:
            break
        selected.append(
            {
                "source_type": "kb_body",
                "doc_id": h.get("doc_id"),
                "chunk_id": h.get("chunk_id"),
                "headers_path": h.get("headers_path"),
                "content_type": h.get("content_type"),
                "score": h.get("score"),
                "text": h.get("text") or "",
                "parent_text": h.get("parent_text") or "",
                "has_body": len((h.get("text") or "").strip()) >= 50,
            }
        )
        used += t

    return {
        "ok": True,
        "query": query,
        "top_k": k,
        "hits": selected,
        "tokens_approx": used,
        "truncated": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KB retrieve with body chunks")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(kb_retrieve(args.query, args.top_k), ensure_ascii=False, indent=2))


def get_tool_spec():
    from src.tools.protocol import ToolSpec

    def handler(args: dict) -> dict:
        return kb_retrieve(str(args.get("query") or ""), args.get("top_k"))

    return ToolSpec(
        name="kb_retrieve",
        description="Retrieve body chunks from local paper KB (Qdrant). Returns text bodies, not titles only.",
        args_schema={"query": "str", "top_k": "int?"},
        handler=handler,
        timeout_sec=120.0,
        permissions=["kb_read"],
        evidence_kind="kb_body",
        requires_body=True,
    )


if __name__ == "__main__":
    main()
