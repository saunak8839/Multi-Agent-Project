"""A deterministic first-pass classifier for Iteration 1."""

import re
from datetime import datetime, timezone

from app.graph.state import SupportState


def intake_agent(state: SupportState) -> dict:
    message = state["customer_message"]
    text = message.lower()
    previous_intent = state.get("intent")
    previous_entities = state.get("entities", {})

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

    follow_up_markers = ("it", "that", "those", "there", "yes", "no", "same", "still", "also", "what about")
    words = text.split()
    is_short_follow_up = len(words) <= 5 or any(marker in text for marker in follow_up_markers)
    if intent == "faq" and previous_intent in {"billing", "technical", "faq"} and is_short_follow_up:
        intent = previous_intent

    urgency = "high" if any(term in text for term in ("urgent", "asap", "right now", "emergency")) else "normal"
    order_match = re.search(r"\b(?:order\s*)?#?(\d{4,})\b", message, re.IGNORECASE)
    entities = {"order_id": order_match.group(1)} if order_match else {}
    if is_short_follow_up and not entities:
        entities = previous_entities

    history_entry = {
        "role": "user",
        "content": message,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "customer_message": message,
        "intent": intent,
        "entities": entities,
        "urgency": urgency,
        "findings": "",
        "retrieved_context": [],
        "draft_reply": "",
        "current_tool_calls": [],
        # Never trust action or approval fields supplied with the initial
        # customer input; only downstream graph nodes may populate them.
        "proposed_action": {},
        "requires_approval": False,
        "approval_decision": {},
        "qa_result": {},
        "qa_feedback": "",
        "qa_attempts": 0,
        "action_result": {},
        "escalation_context": "",
        "conversation_history": [history_entry],
        "steps": [f"intake: intent={intent}, urgency={urgency}, entities={entities}"],
    }
