"""A deterministic first-pass classifier for Iteration 1."""

import re

from app.graph.state import SupportState


def intake_agent(state: SupportState) -> dict:
    message = state["customer_message"]
    text = message.lower()

    billing_terms = (
        "bill", "billing", "charged", "charge", "payment", "invoice",
        "duplicate charge", "double charged", "refund me", "refund for",
        "request a refund", "refund status", "where is my refund",
    )
    faq_terms = (
        "policy", "how do i", "how can i", "what is", "what are",
        "where can i", "when can i", "shipping", "warranty", "return policy",
    )
    technical_terms = ("error", "broken", "doesn't work", "does not work", "not working", "can't", "cannot", "internet", "login", "crash")
    if any(term in text for term in billing_terms):
        intent = "billing"
    elif any(term in text for term in technical_terms):
        intent = "technical"
    elif any(term in text for term in faq_terms):
        intent = "faq"
    else:
        intent = "faq"

    urgency = "high" if any(term in text for term in ("urgent", "asap", "right now", "emergency")) else "normal"
    order_match = re.search(r"\b(?:order\s*)?#?(\d{4,})\b", message, re.IGNORECASE)
    entities = {"order_id": order_match.group(1)} if order_match else {}

    return {
        "intent": intent,
        "entities": entities,
        "urgency": urgency,
        "steps": [f"intake: intent={intent}, urgency={urgency}, entities={entities}"],
    }
