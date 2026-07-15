"""Shared configuration and path helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@lru_cache(maxsize=16)
def load_yaml(name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "config" / name
    if not path.suffix:
        path = path.with_suffix(".yaml")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_paths() -> dict[str, Any]:
    return load_yaml("paths.yaml")


def load_chunking() -> dict[str, Any]:
    return load_yaml("chunking.yaml")


def load_models() -> dict[str, Any]:
    return load_yaml("models.yaml")


def load_routing() -> dict[str, Any]:
    return load_yaml("routing.yaml")


def load_agent() -> dict[str, Any]:
    return load_yaml("agent.yaml")


def load_tools_cfg() -> dict[str, Any]:
    return load_yaml("tools.yaml")


def load_session_cfg() -> dict[str, Any]:
    return load_yaml("session.yaml")


def read_api_key(key_file: str | Path | None = None) -> str:
    models = load_models()
    rel = key_file or models.get("embedding", {}).get("api_key_file")
    if not rel:
        raise FileNotFoundError("No api_key_file configured")
    path = resolve_path(rel)
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    # Prefer first non-empty, non-comment line
    for line in text.splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith("#"):
            return candidate
    if not text:
        raise ValueError(f"API key file is empty: {path}")
    return text
