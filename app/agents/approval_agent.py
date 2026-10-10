"""Pause for a human decision before any refund is recorded."""

from langgraph.types import interrupt

from app.graph.state import SupportState


def approval_gate(state: SupportState) -> dict:
    action = state.get("proposed_action")
    if not action or not state.get("requires_approval"):
        raise ValueError("Approval gate reached without a pending approval-required action.")

    decision = interrupt({
        "type": "refund_approval",
        "action": action,
        "qa_result": state.get("qa_result"),
        "draft_reply": state.get("draft_reply"),
    })
    if isinstance(decision, dict):
        approved = decision.get("approved") is True
        note = str(decision.get("note", ""))
    else:
        approved = decision is True
        note = ""
    return {
        "approval_decision": {"approved": approved, "note": note},
        "steps": [f"approval gate: {'approved' if approved else 'rejected'} by human"],
    }
