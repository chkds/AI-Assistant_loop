"""Workspace snapshot, diff, and confirm gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT, load_session_cfg, resolve_path
from src.agent.checkpoint import latest_plan_checkpoint, list_checkpoints, rollback_to
from src.agent.session import SessionStore


def _hash_file(path: Path, max_bytes: int) -> str | None:
    if not path.is_file():
        return None
    if path.stat().st_size > max_bytes:
        return f"size:{path.stat().st_size}"
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def watch_paths(store: SessionStore, state: dict[str, Any]) -> list[Path]:
    cfg = load_session_cfg()
    paths: list[Path] = []
    if cfg.get("file_watch", {}).get("include_session_artifacts", True):
        paths.append(store.root / "artifacts")
    for rel in cfg.get("file_watch", {}).get("include_paths") or []:
        paths.append(resolve_path(rel))
    for p in state.get("file_watch_paths") or []:
        paths.append(resolve_path(p) if not Path(p).is_absolute() else Path(p))
    # expand directories to files
    files: list[Path] = []
    exts = set(cfg.get("file_watch", {}).get("pin_extensions") or [".md", ".py", ".yaml", ".yml", ".txt", ".json"])
    for p in paths:
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in exts:
                    files.append(f)
    # unique
    uniq = []
    seen = set()
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(f.resolve())
    return uniq


def take_snapshot(store: SessionStore, state: dict[str, Any]) -> dict[str, Any]:
    cfg = load_session_cfg()
    max_bytes = int(cfg.get("snapshot", {}).get("max_file_bytes", 5_000_000))
    snap: dict[str, Any] = {}
    for f in watch_paths(store, state):
        try:
            st = f.stat()
            snap[str(f)] = {
                "mtime": st.st_mtime,
                "size": st.st_size,
                "sha256": _hash_file(f, max_bytes),
            }
        except OSError:
            continue
    store.snapshot_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    return snap


def load_snapshot(store: SessionStore) -> dict[str, Any]:
    if not store.snapshot_path.exists():
        return {}
    return json.loads(store.snapshot_path.read_text(encoding="utf-8"))


def diff_workspace(store: SessionStore, state: dict[str, Any]) -> list[dict[str, Any]]:
    old = load_snapshot(store)
    cfg = load_session_cfg()
    max_bytes = int(cfg.get("snapshot", {}).get("max_file_bytes", 5_000_000))
    current: dict[str, Any] = {}
    for f in watch_paths(store, state):
        try:
            st = f.stat()
            current[str(f)] = {
                "mtime": st.st_mtime,
                "size": st.st_size,
                "sha256": _hash_file(f, max_bytes),
            }
        except OSError:
            continue
    changes: list[dict[str, Any]] = []
    for path, meta in current.items():
        if path not in old:
            changes.append({"path": path, "change": "added", **meta})
        elif old[path].get("sha256") != meta.get("sha256"):
            changes.append({"path": path, "change": "modified", **meta})
    for path in old:
        if path not in current:
            changes.append({"path": path, "change": "deleted"})
    return changes


def require_confirm_if_changed(store: SessionStore, state: dict[str, Any]) -> dict[str, Any]:
    changes = diff_workspace(store, state)
    if not changes:
        return {"needs_confirm": False, "changes": []}
    pending = {
        "reason": "file_diff",
        "changes": changes,
        "suggested_resume": _suggest_resume(changes),
    }
    store.save_pending(pending)
    state["status"] = "awaiting_confirm"
    state["pending_file_confirm"] = pending
    store.append_event({"type": "file_diff", "count": len(changes)})
    store.save_state(state)
    return {"needs_confirm": True, "changes": changes, "pending": pending}


def _suggest_resume(changes: list[dict[str, Any]]) -> str:
    cfg = load_session_cfg().get("confirm", {})
    for c in changes:
        p = c.get("path", "").replace("\\", "/")
        if "/config/" in p or p.endswith(".yaml") or p.endswith(".yml"):
            return cfg.get("config_change_resume_node", "classify")
        if "full.md" in p or "/raw/pdf2md/" in p:
            return cfg.get("source_md_change_resume_node", "gate")
        if "/artifacts/" in p:
            return cfg.get("artifact_change_resume_node", "plan")
    return "plan"


def confirm_files(
    store: SessionStore,
    accepted: list[str] | None = None,
    rejected: list[str] | None = None,
) -> dict[str, Any]:
    state = store.load_state()
    pending = store.load_pending() or state.get("pending_file_confirm") or {}
    changes = pending.get("changes") or []
    accepted = accepted or [c["path"] for c in changes]
    rejected = set(rejected or [])
    accepted_set = set(accepted) - rejected

    resume_node = pending.get("suggested_resume") or _suggest_resume(
        [c for c in changes if c.get("path") in accepted_set]
    )
    # apply impact
    if resume_node in {"classify", "gate", "plan"}:
        target = None
        for cid in reversed(list_checkpoints(store)):
            if cid.endswith(f"_{resume_node}"):
                target = cid
                break
        if target is None and resume_node == "plan":
            target = latest_plan_checkpoint(store)
        if target:
            state = rollback_to(store, state, target)
        else:
            state["resume_from"] = resume_node
    else:
        state["resume_from"] = resume_node

    # refresh snapshot after confirm
    take_snapshot(store, state)
    store.clear_pending()
    state["pending_file_confirm"] = None
    state["status"] = "interrupted"  # ready for explicit resume
    state["resume_hint"] = resume_node
    store.append_event(
        {
            "type": "confirm_files",
            "accepted": list(accepted_set),
            "rejected": list(rejected),
            "resume_node": resume_node,
        }
    )
    store.save_state(state)
    return {"ok": True, "resume_node": resume_node, "state": state}
