"""Technical specialist with simulated log search."""

import json

from langchain_core.tools import tool

from app.agents.tool_agent import run_tool_agent
from app.graph.state import SupportState
from app.tools.technical import search_technical_logs as lookup_logs


def technical_agent(state: SupportState) -> dict:
    device_id = state.get("entities", {}).get("device_id")
    if not device_id:
        return {
            "specialist": "technical",
            "findings": (
                "Please provide your device ID so I can check logs for the correct device. "
                "I haven't checked any device logs yet."
            ),
            "technical_context": {"device_id": None, "lookups": []},
            "current_tool_calls": [],
            "steps": ["technical specialist: requested device ID before log lookup"],
        }

    previous = state.get("technical_context", {})
    lookups = list(previous.get("lookups", [])) if previous.get("device_id") == device_id else []

    @tool
    def search_technical_logs(search_term: str) -> str:
        """Search simulated logs for the customer's supplied device; use issue keywords or an empty query."""
        # The model controls keywords, but cannot choose or replace the device ID.
        result = lookup_logs.invoke({"search_term": search_term, "device_id": device_id})
        lookups.append(json.loads(result))
        return result

    result = run_tool_agent(state, specialist="technical", tools=[search_technical_logs])
    return {**result, "technical_context": {"device_id": device_id, "lookups": lookups}}
