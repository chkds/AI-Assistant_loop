"""Fetch URL and extract main body text (not title/nav only)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_tools_cfg, load_yaml
from src.ingest.chunker.multimodal import count_tokens


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self._parts: list[str] = []
        self._skip_tags = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in self._skip_tags:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._skip_tags and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        raw = "\n".join(self._parts)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _extract_body(html: str) -> str:
    # Prefer trafilatura if installed
    try:
        import trafilatura

        extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
        if extracted and len(extracted.strip()) > 100:
            return extracted.strip()
    except Exception:
        pass
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    return parser.text()


def fetch_page(url: str) -> dict:
    load_yaml.cache_clear()
    cfg = load_tools_cfg().get("fetch_page", {})
    timeout = float(cfg.get("timeout_sec", 30))
    max_tokens = int(cfg.get("max_body_tokens", 4000))
    ua = cfg.get("user_agent", "AI-Assistant-Loop/0.2")

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": ua})
        if resp.status_code >= 400:
            return {"ok": False, "url": url, "error": f"HTTP {resp.status_code}", "has_body": False}
        ctype = (resp.headers.get("content-type") or "").lower()
        if "html" not in ctype and "text" not in ctype and "json" not in ctype:
            return {
                "ok": False,
                "url": url,
                "error": f"unsupported content-type: {ctype}",
                "has_body": False,
            }
        body = _extract_body(resp.text)
        if count_tokens(body) > max_tokens:
            # rough trim by chars
            body = body[: max_tokens * 4]
            truncated = True
        else:
            truncated = False
        ok_body = len(body.strip()) >= 200
        return {
            "ok": ok_body,
            "source_type": "web_body",
            "url": url,
            "text": body,
            "has_body": ok_body,
            "truncated": truncated,
            "chars": len(body),
            "error": None if ok_body else "body_too_short_or_empty",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": url, "error": str(exc), "has_body": False}


def fetch_pages(urls: list[str]) -> dict:
    cfg = load_tools_cfg().get("fetch_page", {})
    limit = int(cfg.get("max_urls_per_turn", 2))
    urls = urls[:limit]
    pages = [fetch_page(u) for u in urls]
    return {"ok": any(p.get("ok") for p in pages), "pages": pages}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    args = parser.parse_args(argv)
    print(json.dumps(fetch_pages(args.urls), ensure_ascii=False, indent=2))


def get_tool_spec():
    from src.tools.protocol import ToolSpec

    def handler(args: dict) -> dict:
        if args.get("urls"):
            return fetch_pages([str(u) for u in args["urls"]])
        return fetch_page(str(args.get("url") or ""))

    return ToolSpec(
        name="fetch_page",
        description="Fetch URL(s) and extract main body text for evidence.",
        args_schema={"url": "str?", "urls": "list[str]?"},
        handler=handler,
        timeout_sec=60.0,
        permissions=["web_fetch"],
        evidence_kind="web_body",
        requires_body=True,
    )


if __name__ == "__main__":
    main()
