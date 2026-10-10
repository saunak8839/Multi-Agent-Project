"""Technical specialist with simulated log search."""

from app.agents.tool_agent import run_tool_agent
from app.graph.state import SupportState
from app.tools.technical import TECHNICAL_TOOLS


def technical_agent(state: SupportState) -> dict:
    return run_tool_agent(state, specialist="technical", tools=TECHNICAL_TOOLS)
