"""Run the Phase-0 LangGraph echo ring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.graph import run_echo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="hello from phase0")
    args = parser.parse_args()
    print(json.dumps(run_echo(args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
