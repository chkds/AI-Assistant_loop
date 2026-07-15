"""Load MinerU output directories into a normalized document model."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import load_paths, resolve_path

IMG_MD_RE = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")


@dataclass
class MinerUBlock:
    type: str
    text: str = ""
    page_idx: int | None = None
    bbox: list[float] | None = None
    text_level: int | None = None
    img_path: str | None = None
    image_caption: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class MinerUDocument:
    doc_id: str
    title: str
    source_dir: Path
    md_path: Path | None
    image_dir: Path | None
    content_list_path: Path | None
    blocks: list[MinerUBlock] = field(default_factory=list)
    asset_map: dict[str, Path] = field(default_factory=dict)

    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for b in self.blocks:
            counts[b.type] = counts.get(b.type, 0) + 1
        return counts


def list_doc_dirs(root: Path | None = None) -> list[Path]:
    paths = load_paths()
    base = resolve_path(root or paths["raw_pdf2md"])
    if not base.exists():
        raise FileNotFoundError(f"raw_pdf2md not found: {base}")
    return sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def _pick_content_list(doc_dir: Path) -> Path | None:
    """Prefer non-v2 content_list.json."""
    candidates = sorted(doc_dir.glob("*content_list*.json"))
    non_v2 = [p for p in candidates if "content_list_v2" not in p.name]
    if non_v2:
        return non_v2[0]
    return candidates[0] if candidates else None


def _normalize_block(raw: dict[str, Any]) -> MinerUBlock:
    btype = str(raw.get("type", "text"))
    text = ""
    if btype == "text":
        text = str(raw.get("text") or "")
    elif btype == "table":
        text = str(raw.get("table_body") or raw.get("text") or raw.get("html") or "")
        if not text and isinstance(raw.get("table_caption"), list):
            text = "\n".join(str(x) for x in raw["table_caption"])
    elif btype == "equation":
        text = str(raw.get("text") or raw.get("latex") or "")
    elif btype in {"image", "chart"}:
        caps = raw.get("image_caption") or []
        if isinstance(caps, list):
            text = "\n".join(str(c) for c in caps)
        else:
            text = str(caps or raw.get("content") or "")
    else:
        text = str(raw.get("text") or raw.get("content") or "")

    caption = raw.get("image_caption") or []
    if not isinstance(caption, list):
        caption = [str(caption)] if caption else []

    # Keep chart as distinct in raw stats; chunker maps chart→image
    return MinerUBlock(
        type=btype,
        text=text,
        page_idx=raw.get("page_idx"),
        bbox=raw.get("bbox"),
        text_level=raw.get("text_level"),
        img_path=raw.get("img_path"),
        image_caption=[str(c) for c in caption],
        raw=raw,
    )


def _build_asset_map(doc_dir: Path, blocks: list[MinerUBlock], md_path: Path | None) -> dict[str, Path]:
    asset_map: dict[str, Path] = {}
    image_dir = doc_dir / "images"
    for b in blocks:
        if b.img_path:
            rel = b.img_path.replace("\\", "/")
            full = doc_dir / rel
            if full.exists():
                asset_map[rel] = full
            elif image_dir.exists():
                name = Path(rel).name
                candidate = image_dir / name
                if candidate.exists():
                    asset_map[f"images/{name}"] = candidate

    if md_path and md_path.exists():
        md = md_path.read_text(encoding="utf-8", errors="ignore")
        for match in IMG_MD_RE.finditer(md):
            rel = match.group(1).replace("\\", "/")
            full = doc_dir / rel
            if full.exists():
                asset_map[rel] = full

    if image_dir.exists():
        for img in image_dir.iterdir():
            if img.is_file():
                rel = f"images/{img.name}"
                asset_map.setdefault(rel, img)
    return asset_map


def load_document(doc_dir: str | Path) -> MinerUDocument:
    doc_dir = Path(doc_dir)
    if not doc_dir.is_absolute():
        # allow relative to raw_pdf2md or project
        paths = load_paths()
        candidate = resolve_path(paths["raw_pdf2md"]) / doc_dir
        if candidate.exists():
            doc_dir = candidate
        else:
            doc_dir = resolve_path(doc_dir)

    if not doc_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {doc_dir}")

    md_path = doc_dir / "full.md"
    if not md_path.exists():
        md_path = None

    content_list_path = _pick_content_list(doc_dir)
    blocks: list[MinerUBlock] = []
    if content_list_path:
        data = json.loads(content_list_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    blocks.append(_normalize_block(item))

    image_dir = doc_dir / "images"
    asset_map = _build_asset_map(doc_dir, blocks, md_path)

    title = doc_dir.name
    for b in blocks:
        if b.type == "text" and b.text_level == 1 and b.text.strip():
            title = b.text.strip()
            break

    return MinerUDocument(
        doc_id=doc_dir.name,
        title=title,
        source_dir=doc_dir,
        md_path=md_path,
        image_dir=image_dir if image_dir.exists() else None,
        content_list_path=content_list_path,
        blocks=blocks,
        asset_map=asset_map,
    )


def find_document_by_substr(substr: str) -> MinerUDocument:
    needle = substr.lower()
    for d in list_doc_dirs():
        if needle in d.name.lower():
            return load_document(d)
    raise FileNotFoundError(f"No document matching: {substr}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect MinerU document stats")
    parser.add_argument("--doc", required=True, help="Folder name or substring")
    args = parser.parse_args()
    doc = find_document_by_substr(args.doc) if not Path(args.doc).exists() else load_document(args.doc)
    print(f"doc_id: {doc.doc_id}")
    print(f"title: {doc.title}")
    print(f"blocks: {len(doc.blocks)}")
    print(f"type_counts: {doc.type_counts()}")
    print(f"assets: {len(doc.asset_map)}")
    print(f"content_list: {doc.content_list_path}")
