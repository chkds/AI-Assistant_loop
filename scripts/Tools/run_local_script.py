"""Run a whitelisted local script under scripts/Tools (sandboxed)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_tools_cfg, load_yaml, resolve_path


def run_local_script(script: str, args: list[str] | None = None) -> dict:
    load_yaml.cache_clear()
    cfg = load_tools_cfg().get("run_local_script", {})
    whitelist = [resolve_path(d) for d in cfg.get("whitelist_dirs") or ["scripts/Tools"]]
    timeout = int(cfg.get("timeout_sec", 60))
    max_out = int(cfg.get("max_stdout_chars", 8000))

    script_path = Path(script)
    if not script_path.is_absolute():
        # prefer scripts/Tools
        candidate = resolve_path("scripts/Tools") / script_path.name
        if candidate.exists():
            script_path = candidate
        else:
            script_path = resolve_path(script)
    script_path = script_path.resolve()

    allowed = False
    for w in whitelist:
        try:
            script_path.relative_to(w.resolve())
            allowed = True
            break
        except ValueError:
            continue
    if not allowed or not script_path.exists():
        return {
            "ok": False,
            "error": f"script not allowed or missing: {script_path}",
            "stdout": "",
            "stderr": "",
        }

    cmd = [sys.executable, str(script_path), *(args or [])]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(ROOT),
        )
        stdout = (proc.stdout or "")[:max_out]
        stderr = (proc.stderr or "")[:max_out]
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "script": str(script_path),
            "stdout": stdout,
            "stderr": stderr,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "script": str(script_path)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "script": str(script_path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="Script name or path under whitelist")
    parser.add_argument("script_args", nargs="*", default=[])
    args = parser.parse_args(argv)
    print(json.dumps(run_local_script(args.script, args.script_args), ensure_ascii=False, indent=2))


def get_tool_spec():
    from src.tools.protocol import ToolSpec

    def handler(args: dict) -> dict:
        return run_local_script(str(args.get("script") or ""), list(args.get("args") or []))

    return ToolSpec(
        name="run_local_script",
        description="Run a whitelisted script under scripts/Tools.",
        args_schema={"script": "str", "args": "list[str]?"},
        handler=handler,
        timeout_sec=90.0,
        permissions=["script_exec"],
        evidence_kind="none",
        requires_body=False,
    )


if __name__ == "__main__":
    main()
