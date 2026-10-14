"""Billing specialist responsible for refund proposal orchestration."""

from langchain_core.runnables import RunnableConfig

from app.graph.state import SupportState
from app.services.tool_runner import run_tool_loop
from app.tools.billing import BILLING_TOOLS


BILLING_INSTRUCTIONS = (
    "Only if the customer requests a refund and tool results verify an excess successful "
    "payment, use propose_refund for the verified excess in cents. This records a proposal "
    "for human approval and does not issue money. State clearly that it is awaiting approval. "
)


def billing_agent(state: SupportState, config: RunnableConfig | None = None) -> dict:
    result = run_tool_loop(
        state, config, specialist="billing", tools=BILLING_TOOLS,
        instructions=BILLING_INSTRUCTIONS,
    )
    proposed_action = state.get("proposed_action")
    for item in result["evidence"]:
        if item["kind"] == "tool" and item["tool"] == "propose_refund" and item["outcome"] == "ok":
            proposal = item["data"]
            if isinstance(proposal, dict) and proposal.get("status") == "proposed":
                proposed_action = proposal
    return {**result, "proposed_action": proposed_action}
