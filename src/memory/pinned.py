"""Load pinned paper text for broker injection (skip vector search)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.ingest.chunker.multimodal import count_tokens
from src.ingest.mineru_loader import find_document_by_substr, list_doc_dirs, load_document


def extract_pinned_refs(query: str) -> list[str]:
    """Parse pinned doc refs from user text: 钉住 X / pinned:X / @DocName."""
    q = query or ""
    refs: list[str] = []
    for m in re.finditer(r"(?:钉住|指定论文|pinned\s*[:=])\s*[「\"']?([^\s「\"'，,。]+)", q, re.I):
        refs.append(m.group(1).strip())
    for m in re.finditer(r"@([A-Za-z0-9_\-\.]{2,80})", q):
        refs.append(m.group(1).strip())
    seen: set[str] = set()
    out: list[str] = []
    for r in refs:
        key = r.lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def resolve_doc_dir(ref: str) -> Path | None:
    ref = (ref or "").strip()
    if not ref:
        return None
    p = Path(ref)
    if p.is_dir() and ((p / "full.md").exists() or list(p.glob("*content_list*.json"))):
        return p
    try:
        for d in list_doc_dirs():
            if d.name == ref or ref.lower() in d.name.lower():
                return d
    except FileNotFoundError:
        return None
    try:
        doc = find_document_by_substr(ref)
        return Path(doc.source_dir)
    except Exception:  # noqa: BLE001
        return None


def load_pinned_documents(
    refs: list[str],
    *,
    max_tokens: int = 8000,
) -> list[dict[str, Any]]:
    """Load MinerU full.md (truncated) for each ref. Returns evidence-shaped dicts."""
    if not refs:
        return []
    per = max(400, int(max_tokens / max(1, len(refs))))
    loaded: list[dict[str, Any]] = []
    used = 0
    for ref in refs:
        d = resolve_doc_dir(ref)
        if d is None:
            loaded.append(
                {
                    "source_type": "kb_body",
                    "doc_id": ref,
                    "text": f"[pinned load failed: {ref}]",
                    "has_body": False,
                    "pinned": True,
                    "error": "not_found",
                }
            )
            continue
        doc = load_document(d)
        md_path = doc.md_path or (doc.source_dir / "full.md")
        if md_path and Path(md_path).exists():
            text = Path(md_path).read_text(encoding="utf-8", errors="replace")
        else:
            text = "\n\n".join(b.text for b in doc.blocks if b.text)
        if count_tokens(text) > per:
            approx = max(500, per * 3)
            text = text[:approx]
            while count_tokens(text) > per and len(text) > 400:
                text = text[: int(len(text) * 0.85)]
            text = text.rstrip() + "\n\n[… pinned truncated …]"
        tcount = count_tokens(text)
        if loaded and used + tcount > max_tokens:
            break
        loaded.append(
            {
                "source_type": "kb_body",
                "doc_id": doc.doc_id,
                "text": text,
                "has_body": len(text.strip()) >= 50,
                "pinned": True,
                "headers_path": ["pinned"],
                "content_type": "text",
            }
        )
        used += tcount
    return loaded


def inject_pinned_evidence(
    state: dict[str, Any],
    *,
    max_tokens: int = 8000,
    add_evidence,
) -> dict[str, Any]:
    """Ensure pinned docs appear once in state.evidence."""
    refs = list(state.get("pinned_docs") or [])
    if not refs:
        return state
    if state.get("pinned_loaded"):
        return state
    payloads = load_pinned_documents(refs, max_tokens=max_tokens)
    ids: list[str] = []
    for p in payloads:
        before = len(state.get("evidence") or [])
        add_evidence(state, p)
        after = state.get("evidence") or []
        if len(after) > before and after[-1].get("id"):
            ids.append(str(after[-1]["id"]))
    state["broker_pinned_ids"] = ids
    state["pinned_loaded"] = True
    return state
