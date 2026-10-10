"""Conditional routing based on the intent produced by intake."""

from app.graph.state import SupportState
from app.agents.qa_agent import MAX_QA_ATTEMPTS


def route_to_specialist(state: SupportState) -> str:
    intent = state.get("intent", "faq")
    if intent in {"billing", "technical"}:
        return intent
    return "faq"


def route_after_qa(state: SupportState) -> str:
    review = state.get("qa_result", {})
    if review.get("passed"):
        return "approval" if state.get("requires_approval") else "pass"
    if state.get("qa_attempts", 0) >= MAX_QA_ATTEMPTS:
        return "fallback"
    specialist = state.get("specialist", "faq")
    return specialist if specialist in {"faq", "billing", "technical"} else "fallback"


def route_after_approval(state: SupportState) -> str:
    if state.get("approval_decision", {}).get("approved") is True:
        return "execute"
    return "escalate"
