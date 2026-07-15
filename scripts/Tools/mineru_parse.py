from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from mineru_kie_sdk import MineruKIEClient

DEFAULT_TIMEOUT = 120
DEFAULT_POLL_INTERVAL = 5
WORKSPACE_KEY = "COPILOT_ASSISTANT_WORKSPACE_ROOT"


def _workspace_root() -> Path:
    import os
    raw = str(os.getenv(WORKSPACE_KEY, "")).strip()
    if raw:
        return Path(raw).resolve()
    return Path(__file__).resolve().parent.parent.parent


def _read_secret(filename: str) -> str:
    path = _workspace_root() / "only_read_for_agent" / filename
    text = path.read_text(encoding="utf-8").strip()
    for line in text.splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith("#"):
            return candidate
    return text


def _read_pipeline_id() -> str:
    key = _read_secret("MinerU-API-key.txt")
    if not key:
        raise ValueError("MinerU API key (pipeline_id) not found in only read for agent/MinerU-API-key.txt")
    return key


def _read_token() -> str:
    token = _read_secret("MinuerU-API-Token.txt")
    if not token:
        raise ValueError("MinerU API token not found in only read for agent/MinuerU-API-Token.txt")
    return token


def _resolve_path(file_path: str) -> Path:
    path = Path(file_path)
    if not path.is_absolute():
        path = _workspace_root() / path
    path = path.resolve()
    if not path.exists():
        raise ValueError(f"file not found: {file_path}")
    return path


def _validate_file_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    supported = {".pdf", ".jpg", ".jpeg", ".png"}
    if suffix not in supported:
        raise ValueError(f"unsupported file type: {suffix}, must be one of {sorted(supported)}")
    return suffix


def parse_document(
    file_path: str,
    timeout: int = DEFAULT_TIMEOUT,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    base_url: str = "https://mineru.net/api/kie",
) -> Dict[str, Any]:
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path is required")

    resolved = _resolve_path(file_path.strip())
    _validate_file_type(resolved)

    pipeline_id = _read_pipeline_id()
    token = _read_token()

    to = max(10, int(timeout or DEFAULT_TIMEOUT))
    pi = max(2, int(poll_interval or DEFAULT_POLL_INTERVAL))

    try:
        client = MineruKIEClient(
            pipeline_id=pipeline_id,
            base_url=str(base_url or "https://mineru.net/api/kie"),
            timeout=60,
        )
        client.session.headers["Authorization"] = f"Bearer {token}"
        client.headers["Authorization"] = f"Bearer {token}"
    except Exception as exc:
        return {
            "status": "error",
            "code": "PARSE_CLIENT_INIT_FAILED",
            "error": f"failed to initialize MinerU client: {type(exc).__name__}: {exc}",
            "file_path": str(resolved.name),
        }

    try:
        file_ids = client.upload_file(str(resolved))
    except Exception as exc:
        return {
            "status": "error",
            "code": "PARSE_UPLOAD_FAILED",
            "error": f"upload failed: {type(exc).__name__}: {exc}",
            "file_path": str(resolved.name),
        }

    try:
        results = client.get_result(
            file_ids=file_ids,
            timeout=to,
            poll_interval=pi,
        )
    except TimeoutError as exc:
        return {
            "status": "error",
            "code": "PARSE_TIMEOUT",
            "error": f"parse timed out after {to}s: {exc}",
            "file_path": str(resolved.name),
            "file_ids": file_ids,
        }
    except Exception as exc:
        return {
            "status": "error",
            "code": "PARSE_RESULT_FAILED",
            "error": f"result query failed: {type(exc).__name__}: {exc}",
            "file_path": str(resolved.name),
            "file_ids": file_ids,
        }

    output: Dict[str, Any] = {
        "status": "ok",
        "code": "PARSE_OK",
        "file_path": str(resolved.name),
        "file_ids": file_ids,
    }

    parse_result = results.get("parse")
    if isinstance(parse_result, dict):
        output["parse"] = _truncate_dict(parse_result)

    split_result = results.get("split")
    if isinstance(split_result, dict):
        output["split"] = _truncate_dict(split_result)

    extract_result = results.get("extract")
    if isinstance(extract_result, dict):
        output["extract"] = _truncate_dict(extract_result)

    if not output.get("parse") and not output.get("split") and not output.get("extract"):
        output["raw_response"] = str(results)[:4000]

    return output


def _truncate_dict(data: Dict[str, Any], max_value_len: int = 16000) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_value_len:
            output[key] = value[:max_value_len] + f"...[truncated {len(value)} chars]"
        elif isinstance(value, dict):
            output[key] = _truncate_dict(value, max_value_len)
        elif isinstance(value, list):
            output[key] = [
                _truncate_dict(item, max_value_len) if isinstance(item, dict)
                else (str(item)[:max_value_len] if isinstance(item, str) and len(str(item)) > max_value_len else item)
                for item in value[:20]
            ]
        else:
            output[key] = value
    return output


if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read())
    output = parse_document(
        file_path=str(input_data.get("file_path") or ""),
        timeout=int(input_data.get("timeout") or DEFAULT_TIMEOUT),
        poll_interval=int(input_data.get("poll_interval") or DEFAULT_POLL_INTERVAL),
        base_url=str(input_data.get("base_url") or "https://mineru.net/api/kie"),
    )
    print(json.dumps(output, ensure_ascii=False))
