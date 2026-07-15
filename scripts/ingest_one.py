"""Ingest a single MinerU paper: load → chunk → (optional) embed → store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_one.py` without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_paths, resolve_path
from src.ingest.chunker.multimodal import MultiModalChunker
from src.ingest.mineru_loader import find_document_by_substr, load_document, list_doc_dirs


def save_chunks_jsonl(chunks, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def resolve_doc(doc_arg: str):
    path = Path(doc_arg)
    if path.exists() and path.is_dir():
        return load_document(path)
    # exact folder name under raw/pdf2md
    for d in list_doc_dirs():
        if d.name == doc_arg:
            return load_document(d)
    return find_document_by_substr(doc_arg)


def ingest_one(
    doc_arg: str,
    *,
    embed: bool = False,
    recreate_collection: bool = False,
    skip_parents_embed: bool = True,
) -> dict:
    doc = resolve_doc(doc_arg)
    chunker = MultiModalChunker()
    chunks = chunker.process(doc)

    paths = load_paths()
    out_dir = resolve_path(paths.get("chunks_dir", "data/chunks"))
    # sanitize filename
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in doc.doc_id)[:120]
    out_path = out_dir / f"{safe_name}.jsonl"
    save_chunks_jsonl(chunks, out_path)

    stats = {
        "doc_id": doc.doc_id,
        "blocks": len(doc.blocks),
        "type_counts": doc.type_counts(),
        "assets": len(doc.asset_map),
        "chunks": len(chunks),
        "chunk_types": {},
        "jsonl": str(out_path),
        "embedded": 0,
    }
    for c in chunks:
        stats["chunk_types"][c.content_type] = stats["chunk_types"].get(c.content_type, 0) + 1

    if embed:
        from src.ingest.embedder.base import get_text_embedder
        from src.memory.qdrant_store import QdrantChunkStore

        # Prefer embedding children (+ figures/tables/formulas); parents optional
        to_embed = [c for c in chunks if not (skip_parents_embed and c.content_type == "section")]
        if not to_embed:
            to_embed = list(chunks)

        # Phase 1 default: text-embedding-v4 for all chunks (figure captions as text).
        # Vision embedder (plus) is available via get_vision_embedder() for a separate collection.
        embedder = get_text_embedder()
        texts = [c.embed_text() for c in to_embed]
        vectors = embedder.embed_texts(texts)
        store = QdrantChunkStore(vector_size=embedder.dimensions)
        store.ensure_collection(recreate=recreate_collection)
        n = store.upsert_chunks(to_embed, vectors)
        stats["embedded"] = n
        stats["embed_model"] = embedder.model
        stats["embed_dims"] = embedder.dimensions
        if skip_parents_embed:
            parents = [c for c in chunks if c.content_type == "section"]
            if parents:
                pvecs = embedder.embed_texts([c.embed_text() for c in parents])
                store.upsert_chunks(parents, pvecs)
                stats["embedded"] += len(parents)

    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest one MinerU paper")
    parser.add_argument("--doc", required=True, help="Folder name, path, or substring")
    parser.add_argument("--embed", action="store_true", help="Call Qwen embedding API and upsert to Qdrant")
    parser.add_argument("--recreate-collection", action="store_true")
    args = parser.parse_args(argv)
    stats = ingest_one(args.doc, embed=args.embed, recreate_collection=args.recreate_collection)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
