"""Deterministic approval, action, handoff, fallback, and history operations."""

from datetime import datetime, timezone

from langgraph.types import interrupt

from app.graph.routing import requires_approval
from app.graph.state import SupportState
from app.services.workflow_compat import reply_for
from app.tools.billing import execute_approved_refund


def approval_gate(state: SupportState) -> dict:
    action = state.get("proposed_action")
    if not requires_approval(state):
        raise ValueError("Approval gate reached without a pending approval-required action.")

    decision = interrupt({
        "type": "refund_approval",
        "action": action,
        "qa_result": state.get("qa_result"),
        "draft_reply": reply_for(state),
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


def execute_approved_action(state: SupportState) -> dict:
    action = state.get("proposed_action") or {}
    if action.get("action") != "refund":
        raise ValueError("Only a proposed refund can be executed in this iteration.")
    if (state.get("approval_decision") or {}).get("approved") is not True:
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
        "reply": response,
        "steps": ["action executor: recorded approved refund in demo ledger"],
    }


def escalate_rejected_action(state: SupportState) -> dict:
    action = state.get("proposed_action") or {}
    order_id = action.get("order_id", "the order")
    return {
        "escalation_context": (
            f"Human approval was declined for the refund proposed on {order_id}. "
            "No refund was executed."
        ),
        "reply": (
            f"The refund request for {order_id} was not approved, so no refund was executed. "
            "The request needs follow-up from support."
        ),
        "steps": ["escalation: recorded declined refund for support follow-up"],
    }


def finalize_turn(state: SupportState) -> dict:
    reply = reply_for(state)
    return {
        "conversation_history": [{
            "role": "assistant",
            "content": reply,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }],
        "steps": ["conversation: saved assistant response to thread history"],
    }


def urgent_handoff(state: SupportState) -> dict:
    security_report = state.get("intent") == "escalation" and any(
        term in state.get("customer_message", "").lower()
        for term in ("security", "breach", "hacked", "compromised", "unauthorized", "unauthorised", "phishing")
    )
    message = (
        "This security report needs immediate human review. If you believe your account is at risk, "
        "use the official account recovery channel and change your password from a trusted device."
        if security_report else
        "Your request has been marked urgent and needs prompt review by a support specialist."
    )
    return {
        "specialist": "escalation",
        "reply": message,
        "response_kind": "handoff",
        "evidence": [],
        "escalation_context": "High urgency or security risk requires human review.",
        "steps": ["urgent escalation: marked for prompt human support review"],
    }


def safe_fallback(state: SupportState) -> dict:
    """Return an accurate holding response after review cannot verify the answer."""
    return {
        "reply": (
            "I couldn't verify an answer that meets our quality checks. "
            "Your request needs review by a support specialist."
        ),
        "response_kind": "handoff",
        "proposed_action": None,
        "steps": ["qa fallback: replaced unverified answer with a holding response"],
    }
