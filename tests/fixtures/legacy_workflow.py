"""Frozen pre-refactor checkpoint contract and deterministic nodes, for migration tests only."""


# Snapshot: app/graph/state.py
"""Shared state passed between nodes in the support workflow."""

import operator
from typing import Annotated, Any, TypedDict


class SupportState(TypedDict, total=False):
    run_id: str
    thread_id: str
    customer_message: str
    intent: str
    entities: dict[str, Any]
    urgency: str
    specialist: str
    findings: str
    retrieved_context: list[dict[str, Any]]
    technical_context: dict[str, Any]
    awaiting_device_id: bool
    draft_reply: str
    qa_result: dict[str, Any]
    qa_feedback: str
    qa_attempts: int
    proposed_action: dict[str, Any]
    requires_approval: bool
    approval_decision: dict[str, Any]
    action_result: dict[str, Any]
    escalation_context: str
    current_tool_calls: list[str]
    # Minimal UI progress for the latest turn; reset at intake, retained across retries.
    routing_path: list[dict[str, Any]]
    conversation_history: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[str], operator.add]
    # LangGraph combines each node's step entry with the existing list.
    steps: Annotated[list[str], operator.add]


# Snapshot: app/agents/intake_agent.py
"""A deterministic first-pass classifier for Iteration 1."""

import re
from datetime import datetime, timezone

from app.tools.technical import extract_device_ids


def intake_agent(state: SupportState) -> dict:
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
        "customer_message": message,
        "intent": intent,
        "entities": entities,
        "urgency": urgency,
        "findings": "",
        "retrieved_context": [],
        "draft_reply": "",
        "current_tool_calls": [],
        "technical_context": {},
        "awaiting_device_id": False,
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


# Snapshot: app/agents/draft_agent.py
"""Turn specialist findings into a simple customer-facing reply."""



def draft_agent(state: SupportState) -> dict:
    reply = f"Thanks for contacting support. {state['findings']}"
    return {
        "draft_reply": reply,
        "steps": ["draft: created a reply from the specialist finding"],
    }


# Snapshot: app/agents/approval_agent.py
"""Pause for a human decision before any refund is recorded."""

from langgraph.types import interrupt



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


# Snapshot: app/agents/conversation_agent.py
"""Persist the completed assistant turn in the checkpointed thread state."""

from datetime import datetime, timezone



def finalize_turn(state: SupportState) -> dict:
    reply = state.get("draft_reply", "")
    return {
        "conversation_history": [{
            "role": "assistant",
            "content": reply,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }],
        "steps": ["conversation: saved assistant response to thread history"],
    }


# Snapshot: app/agents/action_agent.py
"""Execute approved demo actions or prepare a rejection handoff."""

from app.tools.billing import execute_approved_refund


def execute_approved_action(state: SupportState) -> dict:
    action = state.get("proposed_action", {})
    if action.get("action") != "refund":
        raise ValueError("Only a proposed refund can be executed in this iteration.")
    if state.get("approval_decision", {}).get("approved") is not True:
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
        "draft_reply": response,
        "steps": ["action executor: recorded approved refund in demo ledger"],
    }


def escalate_rejected_action(state: SupportState) -> dict:
    action = state.get("proposed_action", {})
    order_id = action.get("order_id", "the order")
    return {
        "escalation_context": (
            f"Human approval was declined for the refund proposed on {order_id}. "
            "No refund was executed."
        ),
        "draft_reply": (
            f"The refund request for {order_id} was not approved, so no refund was executed. "
            "The request needs follow-up from support."
        ),
        "steps": ["escalation: recorded declined refund for support follow-up"],
    }


# Snapshot: app/agents/escalation_agent.py
"""Create a safe handoff for urgent or high-risk customer reports."""



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



def build_legacy_graph(checkpointer, stop_before=None):
    """Produce old-format checkpoints, including real approval interrupts."""
    from langgraph.graph import END, START, StateGraph

    def specialist(state):
        result = {"specialist": state["intent"], "findings": "Verified answer; refund awaiting approval."}
        if state["intent"] == "billing":
            result.update(proposed_action={
                "action": "refund", "proposal_id": "legacy-proposal", "order_id": "12345",
                "amount_cents": 499900, "currency": "INR",
            }, requires_approval=True)
        return result

    def review(state):
        return {"qa_result": {"passed": True, "issues": [], "feedback": "", "attempt": 1}, "qa_attempts": 1}

    graph = StateGraph(SupportState)
    for name, node in {
        "intake": intake_agent, "faq": specialist, "billing": specialist, "technical": specialist,
        "escalation": urgent_escalation_agent, "draft": draft_agent, "qa": review,
        "approval_gate": approval_gate, "execute_action": execute_approved_action,
        "escalate": escalate_rejected_action, "finalize_turn": finalize_turn,
    }.items():
        graph.add_node(name, node)
    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", lambda state: state["intent"], {name: name for name in ("faq", "billing", "technical", "escalation")})
    for name in ("faq", "billing", "technical", "escalation"):
        graph.add_edge(name, "draft")
    graph.add_edge("draft", "qa")
    graph.add_conditional_edges("qa", lambda state: "approval_gate" if state.get("requires_approval") else "finalize_turn", {"approval_gate": "approval_gate", "finalize_turn": "finalize_turn"})
    graph.add_conditional_edges("approval_gate", lambda state: "execute_action" if state["approval_decision"]["approved"] else "escalate", {"execute_action": "execute_action", "escalate": "escalate"})
    graph.add_edge("execute_action", "finalize_turn")
    graph.add_edge("escalate", "finalize_turn")
    graph.add_edge("finalize_turn", END)
    return graph.compile(checkpointer=checkpointer, interrupt_before=[stop_before] if stop_before else None)
