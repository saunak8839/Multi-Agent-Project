"""Conditional routing based on the intent produced by intake."""

from app.graph.state import SupportState
from app.services.workflow_compat import review_for


MAX_QA_ATTEMPTS = 3


def route_to_specialist(state: SupportState) -> str:
    intent = state.get("intent", "faq")
    if intent == "escalation" or state.get("urgency") == "high":
        return "escalation"
    if intent in {"billing", "technical"}:
        return intent
    return "faq"


def route_after_qa(state: SupportState) -> str:
    review = review_for(state)
    if review.get("passed"):
        return "approval" if requires_approval(state) else "pass"
    if review["attempt"] >= MAX_QA_ATTEMPTS:
        return "fallback"
    if state.get("response_kind", "generated") != "generated":
        return "fallback"
    specialist = state.get("specialist", "faq")
    return specialist if specialist in {"faq", "billing", "technical"} else "fallback"


def route_after_approval(state: SupportState) -> str:
    if (state.get("approval_decision") or {}).get("approved") is True:
        return "execute"
    return "escalate"


def requires_approval(state: SupportState) -> bool:
    """Only complete refund proposals with no decision or result need approval."""
    action = state.get("proposed_action")
    if not isinstance(action, dict) or state.get("approval_decision") or state.get("action_result"):
        return False
    return (
        action.get("action") == "refund"
        and all(isinstance(action.get(key), str) and action[key].strip() for key in ("proposal_id", "order_id", "currency"))
        and len(action["currency"].strip()) == 3
        and type(action.get("amount_cents")) is int
        and action["amount_cents"] > 0
        and action.get("status", "proposed") == "proposed"
        and action.get("requires_human_approval", True) is True
    )
