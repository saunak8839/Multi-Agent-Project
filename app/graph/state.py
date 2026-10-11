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
    conversation_history: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[str], operator.add]
    # LangGraph combines each node's step entry with the existing list.
    steps: Annotated[list[str], operator.add]
