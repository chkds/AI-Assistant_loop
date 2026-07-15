"""Smoke test ComiRouter chat (default: deepseek-v4-flash)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_yaml
from src.llm.client import MultimodalNotConfiguredError, get_chat_client


def main(argv: list[str] | None = None) -> None:
    load_yaml.cache_clear()
    parser = argparse.ArgumentParser(description="Smoke test ComiRouter LLM")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Reply with exactly: ok",
    )
    parser.add_argument("--tier", default="lite", help="lite|standard|heavy|vlm")
    parser.add_argument("--model", default=None, help="Override model id")
    args = parser.parse_args(argv)

    try:
        client = get_chat_client(tier=args.tier)
        model = args.model or client.model
        text = client.chat(
            [{"role": "user", "content": args.prompt}],
            model=model,
        )
        out = {
            "ok": True,
            "provider": "comirouter",
            "tier": args.tier,
            "model": model,
            "reply_preview": (text or "")[:500],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except MultimodalNotConfiguredError as exc:
        print(json.dumps({"ok": False, "paused": True, "reason": str(exc)}, ensure_ascii=False, indent=2))
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
