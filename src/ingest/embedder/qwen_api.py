"""Qwen text embedding via DashScope OpenAI-compatible API (v4 preferred, v3 fallback)."""

from __future__ import annotations

import time
from typing import Sequence

from openai import OpenAI

from src import load_models, read_api_key
from src.ingest.embedder.base import Embedder


class QwenTextEmbedder(Embedder):
    modality = "text"

    def __init__(self, config: dict | None = None):
        models = config or load_models()
        emb = models.get("embedding", {})
        text = emb.get("text", {})
        # Backward-compatible flat keys
        self.model = text.get("model") or emb.get("model") or "text-embedding-v4"
        self.fallback_model = text.get("fallback_model") or "text-embedding-v3"
        self.dimensions = int(text.get("dimensions") or emb.get("dimensions") or 1024)
        self.batch_size = int(text.get("batch_size") or emb.get("batch_size") or 10)
        self.max_retries = int(emb.get("max_retries", 3))
        self.retry_backoff = float(emb.get("retry_backoff_sec", 1.5))
        base_url = text.get("base_url") or emb.get("base_url") or (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        api_key = read_api_key(emb.get("api_key_file"))
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            cleaned = [t if t.strip() else " " for t in batch]
            vectors = self._embed_batch(cleaned)
            out.extend(vectors)
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for model in (self.model, self.fallback_model):
            if not model:
                continue
            for attempt in range(self.max_retries):
                try:
                    # DashScope OpenAI-compatible body (official):
                    # model, input (str|list), dimensions, encoding_format="float"
                    resp = self.client.embeddings.create(
                        model=model,
                        input=batch if len(batch) > 1 else batch[0],
                        dimensions=self.dimensions,
                        encoding_format="float",
                    )
                    data = sorted(resp.data, key=lambda d: d.index)
                    self.model = model  # record which one succeeded
                    return [list(d.embedding) for d in data]
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(self.retry_backoff * (attempt + 1))
        raise RuntimeError(f"Text embedding failed after retries: {last_err}")


# Back-compat alias used by older scripts
QwenEmbedder = QwenTextEmbedder
