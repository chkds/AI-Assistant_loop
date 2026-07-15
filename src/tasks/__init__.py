"""Tasks package — keep imports shallow to avoid cycles with tools."""

from src.tasks.protocol import TaskSpec

__all__ = ["TaskSpec"]
