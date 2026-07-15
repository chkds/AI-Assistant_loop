"""Console I/O helpers — Windows GBK-safe printing for smoke scripts."""

from __future__ import annotations

import json
import sys
from typing import Any


def configure_stdout_utf8() -> None:
    """Best-effort reconfigure stdout/stderr to UTF-8 (Python 3.7+)."""
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


def safe_print(*args: Any, **kwargs: Any) -> None:
    """print that never raises UnicodeEncodeError on legacy consoles."""
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    file = kwargs.get("file", sys.stdout)
    text = sep.join(str(a) for a in args) + end
    try:
        file.write(text)
        flush = kwargs.get("flush", False)
        if flush:
            file.flush()
    except UnicodeEncodeError:
        enc = getattr(file, "encoding", None) or "utf-8"
        data = text.encode(enc, errors="replace")
        buf = getattr(file, "buffer", None)
        if buf is not None:
            buf.write(data)
            if kwargs.get("flush", False):
                buf.flush()
        else:
            file.write(data.decode(enc, errors="replace"))


def dump_json(label: str, obj: Any) -> None:
    safe_print(f"\n--- {label} ---")
    safe_print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
