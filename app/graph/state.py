"""Shared state passed between nodes in the support workflow."""

import operator
from typing import Annotated, Any, TypedDict


class SupportState(TypedDict, total=False):
    customer_message: str
    intent: str
    entities: dict[str, Any]
    urgency: str
    specialist: str
    findings: str
    retrieved_context: list[dict[str, Any]]
    draft_reply: str
    qa_result: dict[str, Any]
    qa_feedback: str
    qa_attempts: int
    tool_calls: Annotated[list[str], operator.add]
    # LangGraph combines each node's step entry with the existing list.
    steps: Annotated[list[str], operator.add]
