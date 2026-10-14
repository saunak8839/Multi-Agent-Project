"""A deterministic first-pass classifier for Iteration 1."""

import re
from datetime import datetime, timezone

from app.graph.state import SupportState
from app.tools.technical import extract_device_ids


def intake(state: SupportState) -> dict:
    message = state["customer_message"]
    text = message.lower()
    previous_intent = state.get("intent")
    previous_entities = state.get("entities", {})

    security_terms = (
        "security breach", "data breach", "hacked", "account compromised",
        "compromised account", "unauthorized access", "unauthorised access",
        "someone accessed", "got into my account", "stolen password", "stolen credentials", "phishing",
        "suspicious login", "account takeover",
    )
    urgent_terms = ("urgent", "asap", "right now", "immediately", "emergency", "critical")
    security_risk = any(term in text for term in security_terms)
    urgency = "high" if security_risk or any(term in text for term in urgent_terms) else "normal"

    billing_terms = (
        "bill", "billing", "charged", "charge", "payment", "invoice", "refund",
        "duplicate charge", "double charged", "order payment",
    )
    faq_terms = (
        "policy", "how do i", "how can i", "what is", "what are",
        "where can i", "where do i", "when can i", "shipping", "warranty",
        "return policy", "tracking", "parcel", "delivery", "address",
    )
    technical_terms = (
        "error", "broken", "doesn't work", "does not work", "not working", "internet",
        "wifi", "wi-fi", "router", "login", "log in", "sign in", "login issue",
        "can't connect", "cannot connect",
        "crash", "disconnect", "keeps dropping", "won't load", "not loading",
    )
    if security_risk or urgency == "high":
        intent = "escalation"
    elif any(term in text for term in billing_terms):
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

    device_ids = extract_device_ids(message)
    history = state.get("conversation_history", [])
    # Recognize clarification prompts saved before awaiting_device_id was introduced.
    legacy_device_prompt = (
        bool(history) and history[-1].get("role") == "assistant"
        and "provide your device id" in history[-1].get("content", "").lower()
    )
    awaiting_device = state.get("awaiting_device_id", False) or legacy_device_prompt
    bare_identifier = message.strip().lower()
    if (
        not device_ids and awaiting_device and previous_intent == "technical" and intent == "technical"
        and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", bare_identifier)
        and any(character.isdigit() for character in bare_identifier)
    ):
        device_ids = {bare_identifier}

    order_pattern = r"\border\s*#?(\d{4,})\b" if device_ids else r"\b(?:order\s*)?#?(\d{4,})\b"
    order_match = re.search(order_pattern, message, re.IGNORECASE)
    entities = {"order_id": order_match.group(1)} if order_match else {}
    if is_short_follow_up and not entities:
        entities = previous_entities
    if device_ids:
        entities = {key: value for key, value in entities.items() if key != "device_id"}
        if len(device_ids) == 1:
            entities["device_id"] = next(iter(device_ids))
            if not order_match and entities.get("order_id") == entities["device_id"]:
                entities.pop("order_id")

    history_entry = {
        "role": "user",
        "content": message,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        **initialize_turn(),
        "customer_message": message,
        "intent": intent,
        "entities": entities,
        "urgency": urgency,
        "conversation_history": [history_entry],
        "steps": [f"intake: intent={intent}, urgency={urgency}, entities={entities}"],
    }


def initialize_turn() -> dict:
    """Reset turn-local state after intake reads the previous continuation context."""
    return {
        "schema_version": 2,
        "specialist": "",
        "reply": "",
        "evidence": [],
        "response_kind": "generated",
        "tool_calls": [],
        "awaiting_device_id": False,
        "proposed_action": None,
        "approval_decision": None,
        "qa_result": None,
        "action_result": None,
        "escalation_context": "",
        "steps": [],
        "routing_path": [],
    }
