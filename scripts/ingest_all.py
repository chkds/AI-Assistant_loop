"""Batch ingest MinerU papers under raw/pdf2md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ingest_one import ingest_one
from src.ingest.mineru_loader import list_doc_dirs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch ingest MinerU papers")
    parser.add_argument("--limit", type=int, default=0, help="Max docs (0 = all)")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--recreate-collection", action="store_true", help="Recreate collection before first embed")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args(argv)

    dirs = list_doc_dirs()
    dirs = dirs[args.offset :]
    if args.limit and args.limit > 0:
        dirs = dirs[: args.limit]

    results = []
    for i, d in enumerate(dirs):
        recreate = args.recreate_collection and i == 0 and args.embed
        print(f"[{i + 1}/{len(dirs)}] {d.name}", flush=True)
        try:
            stats = ingest_one(d.name, embed=args.embed, recreate_collection=recreate)
            results.append({"ok": True, **stats})
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}", flush=True)
            results.append({"ok": False, "doc_id": d.name, "error": str(exc)})

    summary = {
        "total": len(results),
        "ok": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
        "chunks": sum(r.get("chunks", 0) for r in results if r.get("ok")),
        "embedded": sum(r.get("embedded", 0) for r in results if r.get("ok")),
    }
    print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
