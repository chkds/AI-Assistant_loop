"""Tavily web search — discovery only (title/url/snippet). Body via fetch_page."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_tools_cfg, load_yaml, read_api_key


def tavily_search(query: str, max_results: int | None = None) -> dict:
    load_yaml.cache_clear()
    cfg = load_tools_cfg().get("tavily_search", {})
    key = read_api_key(cfg.get("api_key_file", "setting/API-key/Tavily-API-key.txt"))
    n = int(max_results or cfg.get("max_results", 3))
    depth = cfg.get("search_depth", "basic")

    payload = {
        "api_key": key,
        "query": query,
        "max_results": n,
        "search_depth": depth,
        "include_raw_content": bool(cfg.get("include_raw_content", False)),
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post("https://api.tavily.com/search", json=payload)
    if resp.status_code >= 400:
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:400]}"}
    data = resp.json()
    results = []
    for r in data.get("results") or []:
        results.append(
            {
                "source_type": "web_snippet",
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "score": r.get("score"),
                "has_body": False,
                "note": "snippet_only; call fetch_page for body",
            }
        )
    return {
        "ok": True,
        "query": query,
        "results": results,
        "answer": data.get("answer"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--max-results", type=int, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(tavily_search(args.query, args.max_results), ensure_ascii=False, indent=2))


def get_tool_spec():
    from src.tools.protocol import ToolSpec

    def handler(args: dict) -> dict:
        return tavily_search(str(args.get("query") or ""), args.get("max_results"))

    return ToolSpec(
        name="tavily_search",
        description="Web search discovery (title/url/snippet only). Must follow with fetch_page for body.",
        args_schema={"query": "str", "max_results": "int?"},
        handler=handler,
        timeout_sec=60.0,
        permissions=["web_search"],
        evidence_kind="web_snippet",
        requires_body=False,
    )


if __name__ == "__main__":
    main()
