"""Execute approved demo actions or prepare a rejection handoff."""

from app.graph.state import SupportState
from app.tools.billing import execute_approved_refund


def execute_approved_action(state: SupportState) -> dict:
    action = state.get("proposed_action", {})
    if action.get("action") != "refund":
        raise ValueError("Only a proposed refund can be executed in this iteration.")
    if state.get("approval_decision", {}).get("approved") is not True:
        raise PermissionError("Refund execution requires explicit human approval.")

    result = execute_approved_refund(action)
    amount = int(result["amount_cents"]) / 100
    currency = result["currency"].strip()
    response = (
        f"Your approved refund of {currency} {amount:,.2f} for order {result['order_id']} "
        "has been recorded in the demo refund ledger. This demo does not contact a payment processor."
    )
    return {
        "action_result": result,
        "draft_reply": response,
        "steps": ["action executor: recorded approved refund in demo ledger"],
    }


def escalate_rejected_action(state: SupportState) -> dict:
    action = state.get("proposed_action", {})
    order_id = action.get("order_id", "the order")
    return {
        "escalation_context": (
            f"Human approval was declined for the refund proposed on {order_id}. "
            "No refund was executed."
        ),
        "draft_reply": (
            f"The refund request for {order_id} was not approved, so no refund was executed. "
            "The request needs follow-up from support."
        ),
        "steps": ["escalation: recorded declined refund for support follow-up"],
    }
