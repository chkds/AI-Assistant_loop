"""
MinerU PDF vectorization via v1 Agent API (all curl commands).

Three-step flow for local files:
1. POST  /api/v1/agent/parse/file  → get task_id + presigned OSS upload URL
2. PUT   file to presigned URL     → upload file directly to OSS
3. Poll  /api/v1/agent/parse/{id}  → wait for markdown result

Usage:
    from Tools.Scripts.mineru_client import parse_file, parse_url

    md = parse_file("E:/datasets/Zotero/storage/XXX/paper.pdf")
    md = parse_url("https://example.com/paper.pdf")
"""
from __future__ import annotations

import json
import subprocess
import time
import re
from pathlib import Path
from typing import Any, Optional

_BASE = "https://mineru.net/api/v1/agent"
_BASE_V4 = "https://mineru.net/api/v4"
_V4_TOKEN_PATH = Path("E:/project/copilot-assistant/only_read_for_agent/MinerU-API-Token.txt")
_OUTPUT_DIR = Path("E:/project/copilot-assistant/files/download/embedding/pdf")
_POLL_INTERVAL = 3
_MAX_WAIT = 600

_STATE_LABELS = {
    "pending": "queued",
    "running": "parsing",
    "waiting-file": "waiting upload",
}


def _run_curl(args: list, timeout: int = 30, raw: bool = False) -> Any:
    """Execute curl, return parsed JSON or raw text."""
    cmd = ["curl", "-s", "--location"] + args
    proc = subprocess.run(
        cmd, capture_output=True, timeout=timeout, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    stdout = (proc.stdout or "").strip()
    if proc.returncode != 0 or not stdout:
        raise RuntimeError(f"curl failed (rc={proc.returncode}): {proc.stderr[:200]}")
    if raw:
        return stdout
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        m = re.search(r"\{.+\}", stdout, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise RuntimeError(f"curl returned non-JSON: {stdout[:300]}")


def _save_markdown(content: str, source_name: str, output_dir: Optional[str] = None) -> Path:
    """Save parsed markdown text to output dir (for v1 API — markdown only)."""
    dest = Path(output_dir) if output_dir else _OUTPUT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name).stem
    out_path = dest / f"{stem}.md"
    out_path.write_text(content, encoding="utf-8")
    print(f"[mineru] saved markdown → {out_path}", flush=True)
    return out_path


def _download_and_extract_zip(zip_url: str, source_name: str,
                               output_dir: Optional[str] = None) -> Optional[str]:
    """Download v4 result ZIP, extract contents to output_dir, return markdown.

    v4 ZIP contains:
      - full.md (or {doc}.md) — the main markdown file
      - images/ — extracted images referenced by the markdown
      - content_list.json, layout.json, etc.

    All files are extracted to: output_dir / {stem} /
    Returns the markdown text, or None on failure.
    """
    import tempfile, zipfile, shutil
    dest = Path(output_dir) if output_dir else _OUTPUT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name).stem
    extract_dir = dest / stem  # e.g. E:/.../pdf/my_paper/

    # Download ZIP to temp file
    tmp_dir = tempfile.mkdtemp(prefix="mineru_v4_")
    zip_path = Path(tmp_dir) / "result.zip"
    try:
        cmd = [
            "curl", "-s", "--location",
            "--request", "GET", zip_url,
            "--output", str(zip_path),
            "--max-time", "120",
            "--header", "Accept: */*",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, timeout=130, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"zip download failed: {proc.stderr[:200]}")

        # Extract to target dir (overwrite if exists)
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(extract_dir))

        # List extracted files
        all_files = list(extract_dir.rglob("*"))
        md_files = [f for f in all_files if f.suffix == ".md" and f.is_file()]
        img_files = [f for f in all_files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}]
        json_files = [f for f in all_files if f.suffix == ".json" and f.is_file()]

        print(f"[mineru-v4] extracted {len(all_files)} files → {extract_dir}",
              flush=True)
        if md_files:
            print(f"[mineru-v4]   markdown: {len(md_files)} files", flush=True)
        if img_files:
            print(f"[mineru-v4]   images:   {len(img_files)} files", flush=True)
        if json_files:
            print(f"[mineru-v4]   json:     {len(json_files)} files", flush=True)

        # Read the main markdown file
        if md_files:
            main = [f for f in md_files if f.stem in ("full", stem)]
            chosen = (main or md_files)[0]
            return chosen.read_text(encoding="utf-8", errors="replace")
        return None
    except Exception:
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _http_put(url: str, file_path: str, timeout: int = 120) -> None:
    """PUT raw file content to a presigned URL via curl -T."""
    abs_path = str(Path(file_path).resolve())
    cmd = [
        "curl", "-s", "--location",
        "--request", "PUT",
        "--upload-file", abs_path,
        "--max-time", str(timeout),
        url,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout + 10,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"PUT upload failed (rc={proc.returncode}): {proc.stderr[:200]}")


def parse_file(file_path: str, language: str = "ch",
               enable_table: bool = True, is_ocr: bool = False,
               enable_formula: bool = True, timeout: int = _MAX_WAIT,
               output_dir: Optional[str] = None) -> Optional[str]:
    """Parse a local PDF (or docx/pptx/xlsx/image/html) file via MinerU.

    Steps: request presigned URL → curl PUT file → poll → return markdown text.
    If successful, also saves {stem}.md to output_dir (default: files/download/embedding/pdf).
    Returns the parsed markdown string, or None if failed.
    """
    abs_path = str(Path(file_path).resolve())
    if not Path(abs_path).exists():
        raise FileNotFoundError(f"file not found: {abs_path}")
    file_name = Path(abs_path).name

    # 1. Request presigned upload URL
    print(f"[mineru] requesting upload URL for '{file_name}'...", flush=True)
    payload = json.dumps({
        "file_name": file_name,
        "language": language,
        "enable_table": enable_table,
        "is_ocr": is_ocr,
        "enable_formula": enable_formula,
    })
    args = [
        "--request", "POST", f"{_BASE}/parse/file",
        "--header", "Content-Type: application/json",
        "--header", "Accept: */*",
        "--data-raw", payload,
    ]
    resp = _run_curl(args)
    if resp.get("code") != 0:
        raise RuntimeError(f"request upload URL failed: {resp.get('msg', resp)}")
    data = resp["data"]
    task_id = data["task_id"]
    file_url = data["file_url"]
    print(f"[mineru] task_id={task_id}", flush=True)

    # 2. PUT file to OSS presigned URL
    print("[mineru] uploading via PUT...", flush=True)
    _http_put(file_url, abs_path)
    print("[mineru] upload done, waiting for parse...", flush=True)

    # 3. Poll for result
    md = _poll_result(task_id, timeout=timeout)
    if md:
        _save_markdown(md, file_name, output_dir)
    return md


def parse_url(url: str, language: str = "ch",
              enable_table: bool = True, is_ocr: bool = False,
              enable_formula: bool = True, timeout: int = _MAX_WAIT,
              output_dir: Optional[str] = None) -> Optional[str]:
    """Parse a remote document (by URL) via MinerU. Returns markdown text.

    If successful, also saves {url_stem}.md to output_dir.
    """
    file_name = url.split("/")[-1].split("?")[0] or "document.pdf"
    payload = json.dumps({
        "file_name": file_name,
        "url": url,
        "language": language,
        "enable_table": enable_table,
        "is_ocr": is_ocr,
        "enable_formula": enable_formula,
    })
    args = [
        "--request", "POST", f"{_BASE}/parse/url",
        "--header", "Content-Type: application/json",
        "--header", "Accept: */*",
        "--data-raw", payload,
    ]
    resp = _run_curl(args)
    if resp.get("code") != 0:
        raise RuntimeError(f"parse_url failed: {resp.get('msg', resp)}")
    task_id = resp["data"]["task_id"]
    print(f"[mineru] url submitted → task_id={task_id}", flush=True)
    md = _poll_result(task_id, timeout=timeout)
    if md:
        _save_markdown(md, file_name, output_dir)
    return md


def _poll_result(task_id: str, timeout: int = _MAX_WAIT,
                 interval: int = _POLL_INTERVAL) -> Optional[str]:
    """Poll GET /api/v1/agent/parse/{task_id} until done. Returns markdown text."""
    args = [
        "--request", "GET", f"{_BASE}/parse/{task_id}",
        "--header", "Accept: */*",
    ]
    start = time.time()
    while time.time() - start < timeout:
        resp = _run_curl(args)
        if resp.get("code") != 0:
            raise RuntimeError(f"poll failed: {resp.get('msg', resp)}")
        data = resp.get("data", {})
        state = data.get("state", "unknown")
        elapsed = int(time.time() - start)

        if state == "done":
            md_url = data.get("markdown_url", "")
            print(f"[mineru] [{elapsed}s] done, downloading markdown...", flush=True)
            if md_url:
                raw = _run_curl([
                    "--request", "GET", md_url,
                    "--header", "Accept: */*",
                ], raw=True)
                return raw
            return data.get("markdown", "")

        if state == "failed":
            err = data.get("err_msg", "unknown error")
            print(f"[mineru] [{elapsed}s] FAILED: {err}", flush=True)
            return None

        label = _STATE_LABELS.get(state, state)
        print(f"[mineru] [{elapsed}s] {label}...", flush=True)
        time.sleep(interval)

    print(f"[mineru] poll timeout ({timeout}s), task_id={task_id}", flush=True)
    return None


def extract_pdf(path_or_url: str, **kwargs) -> Optional[str]:
    """Auto-detect: local file → parse_file (v1), URL → parse_url (v1)."""
    if path_or_url.startswith(("http://", "https://")):
        return parse_url(path_or_url, **kwargs)
    return parse_file(path_or_url, **kwargs)


# ═══════════════════════════════════════════════════════════════
# V4 API — single-file precise extraction (batch upload, JWT auth)
# ═══════════════════════════════════════════════════════════════

def _get_v4_token() -> str:
    return _V4_TOKEN_PATH.read_text(encoding="utf-8").strip()


def _v4_auth() -> list:
    return ["--header", f"Authorization: Bearer {_get_v4_token()}"]


def parse_file_v4(file_path: str, model_version: str = "vlm",
                  is_ocr: bool = False, enable_formula: bool = True,
                  enable_table: bool = True, timeout: int = _MAX_WAIT,
                  output_dir: Optional[str] = None) -> Optional[str]:
    """Parse a local PDF via MinerU v4 batch API (precise, JWT auth).

    1. POST /api/v4/file-urls/batch  → batch_id + presigned URLs
    2. PUT file to presigned URL
    3. Poll /api/v4/extract-results/batch/{id} → download zip → extract markdown
    If successful, saves {stem}.md to output_dir.
    """
    abs_path = str(Path(file_path).resolve())
    if not Path(abs_path).exists():
        raise FileNotFoundError(f"file not found: {abs_path}")
    file_name = Path(abs_path).name

    # 1. Request batch upload URLs
    print(f"[mineru-v4] requesting batch upload for '{file_name}'...", flush=True)
    payload = json.dumps({
        "files": [{"name": file_name, "data_id": "0"}],
        "model_version": model_version,
    })
    args = [
        "--request", "POST", f"{_BASE_V4}/file-urls/batch",
        "--header", "Content-Type: application/json",
        "--header", "Accept: */*",
        *_v4_auth(),
        "--data-raw", payload,
    ]
    resp = _run_curl(args)
    if resp.get("code") != 0:
        raise RuntimeError(f"v4 batch request failed: {resp.get('msg', resp)}")
    data = resp["data"]
    batch_id = data["batch_id"]
    urls = data["file_urls"]
    print(f"[mineru-v4] batch_id={batch_id}, urls={len(urls)}", flush=True)

    # 2. PUT file to presigned URL
    print("[mineru-v4] uploading via PUT...", flush=True)
    _http_put(urls[0], abs_path)
    print("[mineru-v4] upload done, polling batch...", flush=True)

    # 3. Poll batch extract results — downloads ZIP + extracts to output_dir
    return _poll_v4_batch(batch_id, file_name, output_dir, timeout=timeout)


def _poll_v4_batch(batch_id: str, source_name: str,
                   output_dir: Optional[str] = None,
                   timeout: int = _MAX_WAIT,
                   interval: int = _POLL_INTERVAL) -> Optional[str]:
    """Poll GET /api/v4/extract-results/batch/{batch_id} until done.

    Downloads the result ZIP and extracts all contents (markdown, images/,
    tables, JSON) to output_dir/{stem}/.  Returns markdown text.
    """
    args = [
        "--request", "GET", f"{_BASE_V4}/extract-results/batch/{batch_id}",
        "--header", "Accept: */*",
        *_v4_auth(),
    ]
    start = time.time()
    while time.time() - start < timeout:
        resp = _run_curl(args)
        if resp.get("code") != 0:
            raise RuntimeError(f"v4 poll batch failed: {resp.get('msg', resp)}")
        data = resp.get("data", {})
        results = data.get("extract_result", [])
        elapsed = int(time.time() - start)

        for fi, r in enumerate(results):
            state = r.get("state", "unknown")
            fname = r.get("file_name", f"file_{fi}")
            if state == "done":
                zip_url = r.get("full_zip_url", "")
                print(f"[mineru-v4] [{elapsed}s] {fname} done, downloading...", flush=True)
                if zip_url:
                    md = _download_and_extract_zip(zip_url, source_name, output_dir)
                    if md:
                        return md
                return str(r)
            if state == "failed":
                err = r.get("err_msg", "")
                print(f"[mineru-v4] [{elapsed}s] {fname} FAILED: {err}", flush=True)
                return None

            progress = r.get("extract_progress", {})
            if progress:
                pct = progress.get("extracted_pages", 0) / max(progress.get("total_pages", 1), 1) * 100
                print(f"[mineru-v4] [{elapsed}s] {fname}: {state} ({pct:.0f}%)...", flush=True)
            else:
                print(f"[mineru-v4] [{elapsed}s] {fname}: {state}...", flush=True)

        if not results:
            print(f"[mineru-v4] [{elapsed}s] waiting for results...", flush=True)

        time.sleep(interval)

    print(f"[mineru-v4] poll timeout ({timeout}s)", flush=True)
    return None


