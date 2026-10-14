"""Technical specialist with simulated log search."""

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.graph.state import SupportState
from app.services.tool_runner import run_tool_loop
from app.services.workflow_compat import evidence_for, tools_for
from app.tools.technical import search_technical_logs as lookup_logs


def technical_agent(state: SupportState, config: RunnableConfig | None = None) -> dict:
    device_id = state.get("entities", {}).get("device_id")
    if not device_id:
        return {
            "specialist": "technical",
            "reply": (
                "Please provide your device ID so I can check logs for the correct device. "
                "I haven't checked any device logs yet."
            ),
            "evidence": [],
            "response_kind": "clarification",
            "schema_version": 2,
            "awaiting_device_id": True,
            "tool_calls": tools_for(state),
            "steps": ["technical specialist: requested device ID before log lookup"],
        }

    evidence = [item for item in evidence_for(state) if item.get("scope", {}).get("device_id") == device_id]

    @tool
    def search_technical_logs(search_term: str) -> str:
        """Search simulated logs for the customer's supplied device; use issue keywords or an empty query."""
        # The model controls keywords, but cannot choose or replace the device ID.
        return lookup_logs.invoke({"search_term": search_term, "device_id": device_id})

    result = run_tool_loop(
        {**state, "evidence": evidence}, config, specialist="technical", tools=[search_technical_logs],
        instructions=(
            "Technical logs are historical demo records, not current service status. "
            "The log tool is restricted to the device ID supplied by the customer. "
            "Describe results as records for that device ID, not proof of customer ownership. "
            "Do not claim the current issue is diagnosed or resolved from historical logs. "
            "If there are no matching records, say so; do not infer another device or diagnosis. "
        ),
    )
    return {
        **result,
        "awaiting_device_id": False,
    }
