"""Embedder protocol and factory — API now, local later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from src import load_models


class Embedder(ABC):
    """Unified embedding interface for text (and optional multimodal)."""

    modality: str = "text"
    dimensions: int = 0
    model: str = ""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_images(
        self,
        image_paths: Sequence[str | Path],
        texts: Sequence[str] | None = None,
    ) -> list[list[float]]:
        """Optional multimodal path. Default: not supported on text-only backends."""
        raise NotImplementedError(f"{type(self).__name__} does not support image embedding")


def get_text_embedder(config: dict | None = None) -> Embedder:
    models = config or load_models()
    emb = models.get("embedding", {})
    backend = str(emb.get("backend", "api")).lower()
    if backend == "api":
        from src.ingest.embedder.qwen_api import QwenTextEmbedder

        return QwenTextEmbedder(models)
    if backend == "local":
        from src.ingest.embedder.local import LocalTextEmbedder

        return LocalTextEmbedder(models)
    raise ValueError(f"Unknown embedding.backend: {backend!r} (expected api|local)")


def get_vision_embedder(config: dict | None = None) -> Embedder:
    models = config or load_models()
    emb = models.get("embedding", {})
    backend = str(emb.get("backend", "api")).lower()
    if backend == "api":
        from src.ingest.embedder.qwen_vision_api import QwenVisionEmbedder

        return QwenVisionEmbedder(models)
    if backend == "local":
        from src.ingest.embedder.local import LocalVisionEmbedder

        return LocalVisionEmbedder(models)
    raise ValueError(f"Unknown embedding.backend: {backend!r} (expected api|local)")
