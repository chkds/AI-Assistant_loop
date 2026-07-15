"""Artifact contract for matlab_assist work items (collaboration F7).

MATLAB MCP is a **coding worker side-line**, not the main orchestrator loop.
The orchestrator schedules a `matlab_assist` work_item only when the user goal
needs MATLAB code/edit/run — not on every session.

## Input (Envelope projection)
- `goal`: narrow coding ask (problem statement)
- optional artifact refs: existing `.m` path under session artifacts / workspace
- `budget.allowed_tools`: typically `mcp:matlab`, `run_local_script`

## Output (Result.expect = artifact)
- modified or new `.m` path (under `data/sessions/{id}/artifacts/` when possible)
- run / eval log summary in evidence (`source_type=mcp`, `has_body=true`)
- self-check pass/fail if `check_matlab_code` / tests were used

## Failure → Feedback
- execution_error → reopen **same** work_item (`needs_revise`), do not invent output
- missing MATLAB MCP → tell user to enable `matlab` in `config/mcp_servers.yaml`
  (lazy connect; process starts on first tool call only)

Do not dump full MATLAB session state into global broker text; use projection.
"""

from __future__ import annotations

from src.tasks.protocol import TaskSpec


def get_task_spec() -> TaskSpec:
    return TaskSpec(
        task_type="matlab_assist",
        domain="matlab",
        description=(
            "Coding worker: edit/run/debug MATLAB via local matlab-mcp-server. "
            "Scheduled by orchestrator only when a coding work_item needs it."
        ),
        default_knowledge_mode="none",
        # mcp:matlab → all tools named mcp_matlab_*
        allowed_tools=["mcp:matlab", "run_local_script", "kb_retrieve"],
        planner_hints=(
            "Prefer mcp_matlab_evaluate_matlab_code / run_matlab_file for execution. "
            "Use check_matlab_code before destructive runs when reviewing scripts. "
            "Do not invent MATLAB output — only report tool observations. "
            "Emit artifact path + log summary as the work_item result."
        ),
        reflect_rules=[
            "Answers about MATLAB results must cite MCP tool observations",
            "If MATLAB MCP server is disabled, tell the user to enable it in mcp_servers.yaml",
            "On tool failure, revise this work_item (execution_error), do not invent numbers",
        ],
        require_body_evidence=False,
    )
