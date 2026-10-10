"""Billing specialist with read-only PostgreSQL tools."""

from app.agents.tool_agent import run_tool_agent
from app.graph.state import SupportState
from app.tools.billing import BILLING_TOOLS


def billing_agent(state: SupportState) -> dict:
    return run_tool_agent(state, specialist="billing", tools=BILLING_TOOLS)
