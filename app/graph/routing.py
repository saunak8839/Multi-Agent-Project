"""Conditional routing based on the intent produced by intake."""

from app.graph.state import SupportState


def route_to_specialist(state: SupportState) -> str:
    intent = state.get("intent", "faq")
    if intent in {"billing", "technical"}:
        return intent
    return "faq"
