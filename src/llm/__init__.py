"""LLM package — ComiRouter / OpenAI-compatible chat clients."""

from src.llm.client import ChatClient, get_chat_client, chat_completion

__all__ = ["ChatClient", "get_chat_client", "chat_completion"]
