"""Hierarchical parent-child multimodal chunker for MinerU documents."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import tiktoken

from src import load_chunking
from src.ingest.chunker.models import Chunk
from src.ingest.mineru_loader import MinerUBlock, MinerUDocument

_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        try:
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            _ENC = None
    return _ENC


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    # Fallback: ~4 chars per token
    return max(1, len(text) // 4)


def _stable_id(*parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return h


NOISE_TYPES = {
    "page_number",
    "footer",
    "header",
    "aside_text",
    "page_footnote",
    "ref_text",
}

FIGURE_TYPES = {"image", "chart"}


def _is_heading(block: MinerUBlock) -> bool:
    if block.type != "text":
        return False
    if block.text_level is not None and block.text_level <= 2:
        return True
    text = block.text.strip()
    if not text or len(text) > 120:
        return False
    # Common paper section patterns: "I. INTRODUCTION", "II. RELATED WORK", "A. Foo"
    if re.match(r"^[IVXLC]+\.\s+\S+", text):
        return True
    if re.match(r"^[A-Z]\.\s+[A-Z]", text):
        return True
    if text.isupper() and 3 <= len(text.split()) <= 12:
        return True
    return False


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _pack_paragraphs(paragraphs: list[str], min_tokens: int, max_tokens: int, overlap_ratio: float) -> list[str]:
    if not paragraphs:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current))
            # overlap: keep trailing paragraphs
            if overlap_ratio > 0 and chunks:
                keep_tokens = int(max_tokens * overlap_ratio)
                kept: list[str] = []
                tok = 0
                for p in reversed(current):
                    t = count_tokens(p)
                    if tok + t > keep_tokens and kept:
                        break
                    kept.insert(0, p)
                    tok += t
                current = kept
                current_tokens = sum(count_tokens(p) for p in current)
            else:
                current = []
                current_tokens = 0

    for para in paragraphs:
        t = count_tokens(para)
        if t > max_tokens:
            # hard-split long paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf: list[str] = []
            buf_tok = 0
            for s in sentences:
                st = count_tokens(s)
                if buf and buf_tok + st > max_tokens:
                    chunks.append(" ".join(buf))
                    # small overlap of last sentence
                    buf = [buf[-1]] if buf and overlap_ratio > 0 else []
                    buf_tok = count_tokens(buf[0]) if buf else 0
                buf.append(s)
                buf_tok += st
            if buf:
                if current and current_tokens + buf_tok <= max_tokens:
                    current.extend(buf)
                    current_tokens += buf_tok
                else:
                    flush()
                    chunks.append(" ".join(buf))
            continue

        if current and current_tokens + t > max_tokens and current_tokens >= min_tokens:
            flush()
        current.append(para)
        current_tokens += t

    if current:
        # final flush without forcing overlap cycle
        chunks.append("\n\n".join(current))
    return chunks


class MultiModalChunker:
    def __init__(self, config: dict | None = None):
        self.cfg = config or load_chunking()
        self.domain = self.cfg.get("domain", "research")
        child = self.cfg.get("child", {})
        self.min_tokens = int(child.get("min_tokens", 400))
        self.max_tokens = int(child.get("max_tokens", 800))
        self.overlap_ratio = float(child.get("overlap_ratio", 0.12))
        fig = self.cfg.get("figure", {})
        self.nearby_blocks = int(fig.get("nearby_blocks", 2))

    def process(self, doc: MinerUDocument) -> list[Chunk]:
        sections = self._group_sections(doc.blocks)
        chunks: list[Chunk] = []
        # map img rel path -> chunk ids that mention it (for reverse links later)
        figure_index: dict[str, list[str]] = defaultdict(list)

        for sec_idx, section in enumerate(sections):
            header = section["header"]
            headers_path = header or f"Section {sec_idx + 1}"
            parent_id = f"p_{_stable_id(doc.doc_id, headers_path, str(sec_idx))}"
            parent_parts: list[str] = []
            if header:
                parent_parts.append(header)

            page_idxs = [b.page_idx for b in section["blocks"] if b.page_idx is not None]
            parent_meta = {
                "doc_id": doc.doc_id,
                "source_dir": str(doc.source_dir),
                "headers_path": headers_path,
                "page_start": min(page_idxs) if page_idxs else None,
                "page_end": max(page_idxs) if page_idxs else None,
                "role": "parent",
            }

            text_buffer: list[str] = []
            block_window: list[MinerUBlock] = section["blocks"]

            for i, block in enumerate(block_window):
                if block.type == "text":
                    if _is_heading(block) and block.text.strip() == (header or ""):
                        continue
                    text_buffer.append(block.text.strip())
                    parent_parts.append(block.text.strip())
                elif block.type in {"table", "equation", "image"}:
                    # flush text first
                    chunks.extend(
                        self._flush_text(
                            doc,
                            text_buffer,
                            parent_id=parent_id,
                            headers_path=headers_path,
                            sec_idx=sec_idx,
                        )
                    )
                    text_buffer = []

                    if block.type == "table":
                        caption = "\n".join(block.raw.get("table_caption") or [])
                        body = block.text
                        text = f"{caption}\n{body}".strip() if caption else body
                        assets = []
                        if block.img_path:
                            assets.append(block.img_path.replace("\\", "/"))
                        cid = f"c_{_stable_id(doc.doc_id, parent_id, 'table', text[:80])}"
                        chunks.append(
                            Chunk(
                                id=cid,
                                text=text,
                                content_type="table",
                                domain=self.domain,
                                parent_id=parent_id,
                                related_assets=assets,
                                metadata={
                                    **parent_meta,
                                    "role": "child",
                                    "page_idx": block.page_idx,
                                    "bbox": block.bbox,
                                },
                            )
                        )
                        parent_parts.append(text)
                    elif block.type == "equation":
                        cid = f"c_{_stable_id(doc.doc_id, parent_id, 'eq', block.text[:80])}"
                        chunks.append(
                            Chunk(
                                id=cid,
                                text=block.text,
                                content_type="formula",
                                domain=self.domain,
                                parent_id=parent_id,
                                metadata={
                                    **parent_meta,
                                    "role": "child",
                                    "page_idx": block.page_idx,
                                    "bbox": block.bbox,
                                },
                            )
                        )
                        parent_parts.append(block.text)
                    else:  # image
                        nearby = self._nearby_text(block_window, i)
                        caption = "\n".join(block.image_caption) or block.text
                        rel = (block.img_path or "").replace("\\", "/")
                        parts = [p for p in [caption, nearby] if p]
                        if rel:
                            parts.append(f"[image: {rel}]")
                        text = "\n\n".join(parts).strip()
                        cid = f"c_{_stable_id(doc.doc_id, parent_id, 'fig', rel or text[:80])}"
                        assets = [rel] if rel else []
                        chunks.append(
                            Chunk(
                                id=cid,
                                text=text,
                                content_type="figure",
                                domain=self.domain,
                                parent_id=parent_id,
                                related_assets=assets,
                                metadata={
                                    **parent_meta,
                                    "role": "child",
                                    "page_idx": block.page_idx,
                                    "bbox": block.bbox,
                                    "img_path": rel,
                                },
                            )
                        )
                        if rel:
                            figure_index[rel].append(cid)
                        parent_parts.append(caption or rel)

            chunks.extend(
                self._flush_text(
                    doc,
                    text_buffer,
                    parent_id=parent_id,
                    headers_path=headers_path,
                    sec_idx=sec_idx,
                )
            )

            parent_text = "\n\n".join(p for p in parent_parts if p).strip()
            if parent_text:
                # truncate parent store text lightly for JSONL size; retrieval expands from children
                if count_tokens(parent_text) > int(self.cfg.get("parent", {}).get("max_tokens", 4000)):
                    # keep head
                    paras = _split_paragraphs(parent_text)
                    packed = _pack_paragraphs(paras, self.min_tokens, int(self.cfg.get("parent", {}).get("max_tokens", 4000)), 0)
                    parent_text = packed[0] if packed else parent_text[:12000]
                chunks.append(
                    Chunk(
                        id=parent_id,
                        text=parent_text,
                        content_type="section",
                        domain=self.domain,
                        parent_id=None,
                        metadata={**parent_meta, "role": "parent"},
                    )
                )

        # reverse-link figures onto nearby text children in same parent
        by_parent: dict[str | None, list[Chunk]] = defaultdict(list)
        for c in chunks:
            by_parent[c.parent_id].append(c)
        for c in chunks:
            if c.content_type != "text" or not c.parent_id:
                continue
            siblings = by_parent.get(c.parent_id, [])
            for s in siblings:
                if s.content_type == "figure" and s.related_assets:
                    for asset in s.related_assets:
                        if asset not in c.related_assets:
                            c.related_assets.append(asset)

        return chunks

    def _group_sections(self, blocks: list[MinerUBlock]) -> list[dict]:
        sections: list[dict] = []
        current: dict = {"header": None, "blocks": []}
        for b in blocks:
            if b.type in NOISE_TYPES:
                continue
            # Normalize chart → treat as image downstream
            if b.type == "chart":
                b.type = "image"
            if _is_heading(b):
                if current["blocks"] or current["header"]:
                    sections.append(current)
                current = {"header": b.text.strip(), "blocks": [b]}
            else:
                current["blocks"].append(b)
        if current["blocks"] or current["header"]:
            sections.append(current)
        if not sections:
            sections = [{"header": None, "blocks": blocks}]
        return sections

    def _nearby_text(self, blocks: list[MinerUBlock], index: int) -> str:
        parts: list[str] = []
        for j in range(max(0, index - self.nearby_blocks), index):
            if blocks[j].type == "text" and blocks[j].text.strip():
                parts.append(blocks[j].text.strip())
        for j in range(index + 1, min(len(blocks), index + 1 + self.nearby_blocks)):
            if blocks[j].type == "text" and blocks[j].text.strip():
                parts.append(blocks[j].text.strip())
                break  # prefer first following paragraph
        return "\n".join(parts)

    def _flush_text(
        self,
        doc: MinerUDocument,
        paragraphs_src: list[str],
        *,
        parent_id: str,
        headers_path: str,
        sec_idx: int,
    ) -> list[Chunk]:
        text = "\n\n".join(p for p in paragraphs_src if p).strip()
        if not text:
            return []
        paras = _split_paragraphs(text)
        pieces = _pack_paragraphs(paras, self.min_tokens, self.max_tokens, self.overlap_ratio)
        out: list[Chunk] = []
        for i, piece in enumerate(pieces):
            cid = f"c_{_stable_id(doc.doc_id, parent_id, 'text', str(i), piece[:64])}"
            out.append(
                Chunk(
                    id=cid,
                    text=piece,
                    content_type="text",
                    domain=self.domain,
                    parent_id=parent_id,
                    continuation_id=f"{parent_id}:text" if len(pieces) > 1 else None,
                    metadata={
                        "doc_id": doc.doc_id,
                        "source_dir": str(doc.source_dir),
                        "headers_path": headers_path,
                        "role": "child",
                        "piece_index": i,
                        "sec_idx": sec_idx,
                    },
                )
            )
        return out
