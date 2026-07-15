"""Qwen multimodal embedding via DashScope native HTTP API (plus preferred, flash fallback)."""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any, Sequence

import httpx

from src import load_models, read_api_key
from src.ingest.embedder.base import Embedder


def _image_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class QwenVisionEmbedder(Embedder):
    modality = "vision"

    def __init__(self, config: dict | None = None):
        models = config or load_models()
        emb = models.get("embedding", {})
        vision = emb.get("vision", {})
        self.model = vision.get("model", "tongyi-embedding-vision-plus-2026-03-06")
        self.fallback_model = vision.get(
            "fallback_model", "tongyi-embedding-vision-flash-2026-03-06"
        )
        self.dimensions = int(vision.get("dimensions", 1152))
        self.mode = str(vision.get("mode", "fused")).lower()
        self.endpoint = vision.get(
            "endpoint",
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
        )
        self.max_retries = int(emb.get("max_retries", 3))
        self.retry_backoff = float(emb.get("retry_backoff_sec", 1.5))
        self.api_key = read_api_key(emb.get("api_key_file"))

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text-only via the vision model (independent text items)."""
        out: list[list[float]] = []
        for text in texts:
            contents = [{"text": text if text.strip() else " "}]
            out.append(self._embed_contents(contents))
        return out

    def embed_images(
        self,
        image_paths: Sequence[str | Path],
        texts: Sequence[str] | None = None,
    ) -> list[list[float]]:
        texts = list(texts) if texts is not None else [""] * len(image_paths)
        if len(texts) != len(image_paths):
            raise ValueError("texts and image_paths length mismatch")
        out: list[list[float]] = []
        for path, text in zip(image_paths, texts, strict=True):
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"image not found: {p}")
            uri = _image_data_uri(p)
            if self.mode == "fused" and text.strip():
                contents = [{"image": uri, "text": text}]
            elif text.strip():
                # independent: return fused preference still packs one call with image only;
                # caption embedded separately is caller's choice — here image+optional text fused
                contents = [{"image": uri}, {"text": text}]
            else:
                contents = [{"image": uri}]
            # For independent multi-item response, take first vector; fused yields one
            out.append(self._embed_contents(contents))
        return out

    def _embed_contents(self, contents: list[dict[str, Any]]) -> list[float]:
        last_err: Exception | None = None
        for model in (self.model, self.fallback_model):
            if not model:
                continue
            # flash default dim differs; only send dimension when using plus family or configured
            params: dict[str, Any] = {"dimension": self.dimensions}
            payload = {
                "model": model,
                "input": {"contents": contents},
                "parameters": params,
            }
            for attempt in range(self.max_retries):
                try:
                    with httpx.Client(timeout=120.0) as client:
                        resp = client.post(
                            self.endpoint,
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            json=payload,
                        )
                    data = resp.json()
                    if resp.status_code >= 400:
                        raise RuntimeError(f"HTTP {resp.status_code}: {data}")
                    embeddings = (
                        (data.get("output") or {}).get("embeddings")
                        or data.get("embeddings")
                        or []
                    )
                    if not embeddings:
                        raise RuntimeError(f"empty embeddings: {data}")
                    # Prefer single fused vector; else first item
                    item = embeddings[0]
                    vec = item.get("embedding") or item.get("vector")
                    if not vec:
                        raise RuntimeError(f"missing embedding field: {item}")
                    self.model = model
                    return list(vec)
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(self.retry_backoff * (attempt + 1))
        raise RuntimeError(f"Vision embedding failed after retries: {last_err}")
