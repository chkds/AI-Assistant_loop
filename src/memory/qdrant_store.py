"""Qdrant vector store for research paper chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src import load_models, load_paths, resolve_path
from src.ingest.chunker.models import Chunk


def make_qdrant_client(paths: dict | None = None, timeout: float | None = None) -> tuple[QdrantClient, dict[str, Any]]:
    """Create Qdrant client from config. Prefer local path when mode=local (no Docker)."""
    paths = paths or load_paths()
    qcfg = paths.get("qdrant", {})
    mode = str(qcfg.get("mode", "local")).lower()
    info: dict[str, Any] = {"mode": mode, "collection": qcfg.get("collection", "research_papers")}
    kwargs: dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = timeout

    if mode == "local":
        local_path = resolve_path(qcfg.get("path", "data/qdrant_local"))
        local_path.mkdir(parents=True, exist_ok=True)
        info["path"] = str(local_path)
        client = QdrantClient(path=str(local_path), **kwargs)
        return client, info

    url = qcfg.get("url", "http://localhost:6333")
    info["url"] = url
    client = QdrantClient(url=url, **kwargs)
    return client, info


class QdrantChunkStore:
    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        vector_size: int | None = None,
        client: QdrantClient | None = None,
    ):
        paths = load_paths()
        qcfg = paths.get("qdrant", {})
        self.collection = collection or qcfg.get("collection", "research_papers")
        models = load_models()
        emb = models.get("embedding", {})
        text_dims = (emb.get("text") or {}).get("dimensions")
        self.vector_size = vector_size or int(text_dims or emb.get("dimensions") or 1024)
        if client is not None:
            self.client = client
            self.client_info = {"mode": "injected"}
        elif url:
            self.client = QdrantClient(url=url)
            self.client_info = {"mode": "server", "url": url}
        else:
            self.client, self.client_info = make_qdrant_client(paths)

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.vector_size, distance=qm.Distance.COSINE),
            )

    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self.ensure_collection()
        points: list[qm.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload: dict[str, Any] = {
                "chunk_id": chunk.id,
                "text": chunk.text,
                "content_type": chunk.content_type,
                "domain": chunk.domain,
                "parent_id": chunk.parent_id,
                "continuation_id": chunk.continuation_id,
                "related_assets": chunk.related_assets,
                "doc_id": chunk.metadata.get("doc_id"),
                "source_dir": chunk.metadata.get("source_dir"),
                "headers_path": chunk.metadata.get("headers_path"),
                "role": chunk.metadata.get("role"),
                "page_idx": chunk.metadata.get("page_idx"),
                "metadata": chunk.metadata,
            }
            points.append(
                qm.PointStruct(
                    id=self._point_id(chunk.id),
                    vector=list(vector),
                    payload=payload,
                )
            )
        batch = 64
        for i in range(0, len(points), batch):
            self.client.upsert(collection_name=self.collection, points=points[i : i + batch])
        return len(points)

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        content_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = None
        if content_types:
            query_filter = qm.Filter(
                must=[qm.FieldCondition(key="content_type", match=qm.MatchAny(any=content_types))]
            )
        hits = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        results = []
        for h in hits:
            payload = h.payload or {}
            results.append(
                {
                    "score": h.score,
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text"),
                    "content_type": payload.get("content_type"),
                    "parent_id": payload.get("parent_id"),
                    "doc_id": payload.get("doc_id"),
                    "headers_path": payload.get("headers_path"),
                    "source_dir": payload.get("source_dir"),
                    "related_assets": payload.get("related_assets") or [],
                    "payload": payload,
                }
            )
        return results

    def get_by_chunk_id(self, chunk_id: str) -> dict[str, Any] | None:
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=[self._point_id(chunk_id)],
            with_payload=True,
        )
        if not points:
            return None
        return points[0].payload

    def get_parent(self, parent_id: str) -> dict[str, Any] | None:
        if not parent_id:
            return None
        hits = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=parent_id))]
            ),
            limit=1,
            with_payload=True,
        )[0]
        if not hits:
            return None
        return hits[0].payload

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        import uuid

        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
