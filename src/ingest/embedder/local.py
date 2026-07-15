"""Local (on-prem) embedder stubs — switch via embedding.backend: local."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src import load_models
from src.ingest.embedder.base import Embedder


class LocalTextEmbedder(Embedder):
    modality = "text"

    def __init__(self, config: dict | None = None):
        models = config or load_models()
        local = (models.get("embedding") or {}).get("local") or {}
        text = local.get("text") or {}
        self.model = text.get("model_name") or ""
        self.model_path = text.get("model_path")
        self.device = text.get("device", "cuda")
        self.dimensions = int(
            ((models.get("embedding") or {}).get("text") or {}).get("dimensions") or 1024
        )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Local text embedding is not configured. "
            "Set embedding.local.text.model_path and implement LocalTextEmbedder, "
            "or set embedding.backend: api."
        )


class LocalVisionEmbedder(Embedder):
    modality = "vision"

    def __init__(self, config: dict | None = None):
        models = config or load_models()
        local = (models.get("embedding") or {}).get("local") or {}
        vision = local.get("vision") or {}
        self.model = vision.get("model_name") or ""
        self.model_path = vision.get("model_path")
        self.device = vision.get("device", "cuda")
        self.dimensions = int(
            ((models.get("embedding") or {}).get("vision") or {}).get("dimensions") or 1152
        )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Local vision embedding is not configured. "
            "Set embedding.local.vision.model_path and implement LocalVisionEmbedder, "
            "or set embedding.backend: api."
        )

    def embed_images(
        self,
        image_paths: Sequence[str | Path],
        texts: Sequence[str] | None = None,
    ) -> list[list[float]]:
        raise NotImplementedError(
            "Local vision embedding is not configured. "
            "Set embedding.local.vision.model_path and implement LocalVisionEmbedder, "
            "or set embedding.backend: api."
        )
