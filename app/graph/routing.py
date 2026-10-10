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
        return "pass"
    if state.get("qa_attempts", 0) >= MAX_QA_ATTEMPTS:
        return "fallback"
    specialist = state.get("specialist", "faq")
    return specialist if specialist in {"faq", "billing", "technical"} else "fallback"
