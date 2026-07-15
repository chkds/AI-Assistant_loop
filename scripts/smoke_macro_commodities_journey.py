"""Deep research journey: geopolitics / commodity trade / US macro ↔ Au·Cu futures.

Script I/O only (no mental scoring). Evaluates:
  1) work_item chain reasonableness (facets → correlate; no matlab)
  2) live AnySearch retrieval accuracy proxies (keyword coverage)
  3) timeliness proxies (year/date signals in returned text)

Usage:
  E:\\application\\miniforge3\\envs\\copilot-agent\\python.exe scripts\\smoke_macro_commodities_journey.py
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_yaml  # noqa: E402
from src.agent.graph import AgentRuntime, _add_evidence  # noqa: E402
from src.agent.orchestrator import decompose, orchestrate_step  # noqa: E402
from src.agent.session import SessionStore  # noqa: E402
from src.agent.tool_io import apply_tool_io  # noqa: E402
from src.tools.mcp_adapter import close_all_mcp_clients  # noqa: E402
from src.tools.registry import get_registry  # noqa: E402
from src.util.console_io import configure_stdout_utf8, dump_json, safe_print  # noqa: E402

configure_stdout_utf8()

DEEP_QUERY = (
    "对照近一年地域冲突、大宗商品贸易和美国的重要经济指标发布情况，"
    "核查黄金、铜的期货价格变动情况与以上事件的关联。"
)

# Focused retrieval queries aligned to each facet work_item
FACET_SEARCHES: list[dict[str, Any]] = [
    {
        "facet": "Geopolitics past year",
        "query": "regional geopolitical conflicts 2025 2026 timeline impact markets",
        "must_any": ["conflict", "war", "geopolit", "冲突", "地缘", "中东", "乌克兰", "red sea"],
        "fresh_any": ["2025", "2026", "2024"],
    },
    {
        "facet": "Commodity trade flows",
        "query": "commodity trade sanctions copper export restrictions 2025 2026",
        "must_any": ["commodity", "trade", "copper", "sanction", "贸易", "铜", "制裁", "出口"],
        "fresh_any": ["2025", "2026", "2024"],
    },
    {
        "facet": "US macro releases",
        "query": "US CPI nonfarm payrolls FOMC interest rate decisions 2025 2026",
        "must_any": ["cpi", "payroll", "fomc", "fed", "inflation", "非农", "美联储", "利率"],
        "fresh_any": ["2025", "2026", "2024"],
    },
    {
        "facet": "Au/Cu futures moves",
        "query": "gold copper futures price moves 2025 2026 COMEX",
        "must_any": ["gold", "copper", "futures", "price", "黄金", "铜", "期货", "xau", "comex"],
        "fresh_any": ["2025", "2026", "2024"],
    },
]

ALLOWED = ["web_research", "matlab_assist", "research_qa", "general_qa", "finance_qa"]


def _dump(label: str, obj: Any) -> None:
    dump_json(label, obj)


def _item_snap(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": i.get("id"),
            "title": i.get("title"),
            "task_type": i.get("task_type"),
            "status": i.get("status"),
            "depends_on": i.get("depends_on") or [],
            "goal": (i.get("goal") or "")[:160],
            "expect": i.get("expect"),
        }
        for i in items
    ]


def _text_blob(result: dict[str, Any]) -> str:
    parts: list[str] = []
    if result.get("text"):
        parts.append(str(result["text"]))
    raw = result.get("raw")
    if isinstance(raw, dict):
        parts.append(json.dumps(raw, ensure_ascii=False)[:8000])
    elif raw is not None:
        parts.append(str(raw)[:8000])
    for k in ("content", "results", "data"):
        if k in result and result[k] is not None:
            parts.append(json.dumps(result[k], ensure_ascii=False)[:8000])
    return "\n".join(parts)


def _score_retrieval(blob: str, *, must_any: list[str], fresh_any: list[str]) -> dict[str, Any]:
    low = blob.lower()
    must_hits = [k for k in must_any if k.lower() in low]
    fresh_hits = [k for k in fresh_any if k.lower() in low]
    # ISO-ish dates
    date_hits = re.findall(r"20(2[4-6])[-/](0?[1-9]|1[0-2])", blob)
    accuracy = len(must_hits) / max(1, min(3, len(must_any)))  # saturate at 3 hits
    accuracy = min(1.0, accuracy)
    freshness = 1.0 if fresh_hits or date_hits else 0.0
    if fresh_hits and ("2025" in fresh_hits or "2026" in fresh_hits):
        freshness = 1.0
    elif fresh_hits:
        freshness = 0.6
    return {
        "chars": len(blob),
        "must_hits": must_hits[:12],
        "fresh_hits": fresh_hits,
        "date_pattern_count": len(date_hits),
        "accuracy_score": round(accuracy, 3),
        "freshness_score": round(freshness, 3),
        "preview": blob[:400].replace("\n", " "),
    }


def check_chain() -> dict[str, Any]:
    safe_print("\n=== CHECK A: work_item chain reasonableness ===")
    items = decompose(DEEP_QUERY, available_task_types=ALLOWED, use_llm=False)
    snap = _item_snap(items)
    _dump("decompose_deep_query", {"query": DEEP_QUERY, "items": snap})

    types = [i["task_type"] for i in items]
    titles = [i["title"] for i in items]
    no_matlab = "matlab_assist" not in types
    # Expect 4 evidence facets + 1 correlate
    ok_len = len(items) >= 5
    correlate = items[-1]
    ok_correlate = (
        "Correlate" in (correlate.get("title") or "")
        or "关联" in (correlate.get("goal") or "")
    )
    ok_deps = len(correlate.get("depends_on") or []) >= 3
    facet_ready = all(
        (i.get("status") in {"ready", "pending", "blocked"}) and i.get("task_type") == "web_research"
        for i in items[:-1]
    )
    # facets should NOT depend on correlate; correlate depends on facets
    facet_ids = {i["id"] for i in items[:-1]}
    ok_edge = set(correlate.get("depends_on") or []).issubset(facet_ids) and ok_deps
    # No matlab; all research
    ok = no_matlab and ok_len and ok_correlate and ok_edge and facet_ready and all(
        t == "web_research" for t in types
    )

    # Also bind via orchestrate_step + SessionStore
    store = SessionStore()
    state = store.new_state(DEEP_QUERY)
    state = orchestrate_step(state, available_task_types=ALLOWED)
    store.save_state(state)
    _dump(
        "orchestrate_bound",
        {
            "session_id": store.session_id,
            "current": state.get("current_work_item_id"),
            "items": _item_snap(state.get("work_items") or []),
        },
    )

    safe_print("RESULT checkA:", "PASS" if ok else "FAIL")
    return {
        "ok": ok,
        "session_id": store.session_id,
        "titles": titles,
        "no_matlab": no_matlab,
        "ok_len": ok_len,
        "ok_correlate": ok_correlate,
        "ok_edge": ok_edge,
        "items": snap,
    }


def check_live_retrieval() -> dict[str, Any]:
    safe_print("\n=== CHECK B: live AnySearch accuracy + freshness ===")
    load_yaml.cache_clear()
    reg = get_registry(reload=True)
    names = set(reg.names())
    search_tool = "mcp_anysearch_search"
    if search_tool not in names:
        alts = sorted(n for n in names if n.startswith("mcp_anysearch_"))
        _dump("registry_anysearch", {"alts": alts})
        if not alts:
            safe_print("RESULT checkB: FAIL (anysearch not registered)")
            return {"ok": False, "reason": "no_anysearch"}
        search_tool = alts[0]

    store = SessionStore()
    state = store.new_state(DEEP_QUERY + " [retrieval probe]")
    state["task_type"] = "web_research"
    _ = AgentRuntime(store, state, registry=reg)  # bind session tooling path

    facet_reports: list[dict[str, Any]] = []
    for facet in FACET_SEARCHES:
        args = {"query": facet["query"], "max_results": 3}
        safe_print(f"\n>> search facet={facet['facet']!r} q={facet['query']!r}")
        result = reg.call(search_tool, args)
        ok_call = bool(result.get("ok"))
        blob = _text_blob(result)
        scores = _score_retrieval(
            blob,
            must_any=list(facet["must_any"]),
            fresh_any=list(facet["fresh_any"]),
        )
        obs = apply_tool_io(
            state,
            tool=search_tool,
            arguments=args,
            result=result,
            evidence_kind=reg.get(search_tool).evidence_kind if reg.get(search_tool) else "mcp",
            add_evidence=_add_evidence,
        )
        report = {
            "facet": facet["facet"],
            "query": facet["query"],
            "call_ok": ok_call,
            "obs_ok": obs.get("ok"),
            "evidence_ids": obs.get("evidence_ids"),
            "source_type": result.get("source_type"),
            "has_body": result.get("has_body"),
            **scores,
        }
        facet_reports.append(report)
        _dump(f"facet_{facet['facet']}", report)

    store.save_state(state)
    mcp_ev = [e for e in state.get("evidence") or [] if e.get("source_type") == "mcp"]
    avg_acc = sum(r["accuracy_score"] for r in facet_reports) / max(1, len(facet_reports))
    avg_fresh = sum(r["freshness_score"] for r in facet_reports) / max(1, len(facet_reports))
    calls_ok = all(r["call_ok"] for r in facet_reports)
    # Pass bar: all calls ok, mean accuracy>=0.34 (~1/3 keywords), freshness>=0.5
    ok = calls_ok and avg_acc >= 0.34 and avg_fresh >= 0.5 and len(mcp_ev) >= 1

    summary = {
        "ok": ok,
        "session_id": store.session_id,
        "search_tool": search_tool,
        "utc_now": datetime.now(timezone.utc).isoformat(),
        "calls_ok": calls_ok,
        "avg_accuracy_score": round(avg_acc, 3),
        "avg_freshness_score": round(avg_fresh, 3),
        "evidence_mcp_count": len(mcp_ev),
        "facets": facet_reports,
    }
    _dump("retrieval_summary", summary)
    # Persist report next to session
    report_path = store.root / "macro_commodities_retrieval_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    safe_print("report_path:", report_path)
    safe_print("RESULT checkB:", "PASS" if ok else "FAIL")
    return summary


def main() -> int:
    safe_print("=== smoke_macro_commodities_journey ===")
    safe_print("query:", DEEP_QUERY)
    results: dict[str, Any] = {}
    try:
        results["chain"] = check_chain()
        results["retrieval"] = check_live_retrieval()
    except Exception as exc:  # noqa: BLE001
        safe_print("EXCEPTION:", exc)
        traceback.print_exc()
        results["exception"] = str(exc)
        close_all_mcp_clients()
        _dump("RESULTS_PARTIAL", {k: (v.get("ok") if isinstance(v, dict) else v) for k, v in results.items()})
        safe_print("RESULT: FAIL")
        return 2
    finally:
        close_all_mcp_clients()

    summary = {
        "chain_ok": bool(results.get("chain", {}).get("ok")),
        "retrieval_ok": bool(results.get("retrieval", {}).get("ok")),
        "avg_accuracy": results.get("retrieval", {}).get("avg_accuracy_score"),
        "avg_freshness": results.get("retrieval", {}).get("avg_freshness_score"),
        "titles": results.get("chain", {}).get("titles"),
    }
    _dump("SUMMARY", summary)
    all_ok = summary["chain_ok"] and summary["retrieval_ok"]
    # Also write under Development/log
    log_path = ROOT / "Development" / "log" / "2026-07-15-macro-commodities-smoke.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"query": DEEP_QUERY, "summary": summary, "results": results}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    safe_print("log_path:", log_path)
    safe_print("\nRESULT:", "SUCCESS" if all_ok else "FAIL")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
