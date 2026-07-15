"""ComiRouter chat client — OpenAI-compatible JSON over HTTP.

Important: this relay WAF-blocks the official OpenAI Python SDK because of
Stainless / OpenAI User-Agent headers. We call the HTTP API directly with:

    Authorization: Bearer <TOKEN>

(Token = contents of api_key_file, never the filename.)
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import httpx

from src import load_models, read_api_key


class MultimodalNotConfiguredError(RuntimeError):
    """Raised when a VLM/multimodal chat model is required but not configured."""


class ChatClient:
    """OpenAI-compatible /v1/chat/completions via httpx (Bearer auth)."""

    def __init__(self, config: dict | None = None, tier: str | None = None):
        models = config or load_models()
        llm = models.get("llm") or {}
        backend = str(llm.get("backend", "api")).lower()
        if backend == "local":
            raise NotImplementedError(
                "llm.backend=local is reserved. Set model_path under llm.local "
                "and implement LocalChatClient, or use backend: api."
            )
        if backend != "api":
            raise ValueError(f"Unknown llm.backend: {backend!r}")

        self.base_url = str(llm.get("base_url", "https://comirouter.com/v1")).rstrip("/")
        # Path where the secret string is stored (read contents → Bearer token).
        self.api_key_file = llm.get(
            "api_key_file", "setting/API-key/comirouter-sales-API-key.txt"
        )
        self.default_model = llm.get("default_model", "deepseek-v4-flash")
        self.max_retries = int(llm.get("max_retries", 3))
        self.retry_backoff = float(llm.get("retry_backoff_sec", 1.5))
        self.pause_policy = llm.get("pause_policy") or {}
        self.tiers = models.get("llm_tiers") or {}
        self.timeout = float(llm.get("timeout_sec", 120))

        # Secret string from file — used only as Authorization: Bearer <TOKEN>
        self._token = read_api_key(self.api_key_file)
        self.tier = tier
        self.model = self.resolve_model(tier)

    def resolve_model(self, tier: str | None = None) -> str:
        if not tier:
            return self.default_model
        entry = self.tiers.get(tier) or {}
        model_id = entry.get("model_id")
        status = entry.get("status")
        if status == "paused_needs_multimodal_chat" or not model_id:
            raise MultimodalNotConfiguredError(
                f"Tier {tier!r} is not configured for text chat. "
                "DeepSeek flash on ComiRouter is text-only. "
                "Pause: please provide a multimodal chat model id available on "
                "ComiRouter (or another API key) before image/VLM tasks."
            )
        return str(model_id)

    def _headers(self) -> dict[str, str]:
        # Required by ComiRouter. Keep headers minimal to avoid WAF blocks
        # (OpenAI SDK Stainless headers return 403 on this relay).
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        use_model = model or self.model
        payload: dict[str, Any] = {
            "model": use_model,
            "messages": list(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        url = f"{self.base_url}/chat/completions"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, headers=self._headers(), json=payload)
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"empty choices: {data}")
                message = choices[0].get("message") or {}
                return str(message.get("content") or "")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(self.retry_backoff * (attempt + 1))
        raise RuntimeError(f"Chat completion failed after retries: {last_err}")


def get_chat_client(tier: str | None = "lite", config: dict | None = None) -> ChatClient:
    return ChatClient(config=config, tier=tier)


def chat_completion(
    user_content: str,
    *,
    system: str | None = None,
    tier: str | None = "lite",
    model: str | None = None,
) -> str:
    """Convenience one-shot chat (test default: deepseek-v4-flash via lite tier)."""
    client = get_chat_client(tier=tier)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    return client.chat(messages, model=model)
