"""Create a safe handoff for urgent or high-risk customer reports."""

from app.graph.state import SupportState


def urgent_escalation_agent(state: SupportState) -> dict:
    security_report = state.get("intent") == "escalation" and any(
        term in state.get("customer_message", "").lower()
        for term in ("security", "breach", "hacked", "compromised", "unauthorized", "unauthorised", "phishing")
    )
    message = (
        "I’m escalating this security report for immediate human review. If you believe your account is at risk, "
        "use the official account recovery channel and change your password from a trusted device."
        if security_report else
        "Your request has been marked urgent and needs prompt review by a support specialist."
    )
    return {
        "specialist": "escalation",
        "findings": message,
        "escalation_context": "High urgency or security risk requires human review.",
        "steps": ["urgent escalation: marked for prompt human support review"],
    }
