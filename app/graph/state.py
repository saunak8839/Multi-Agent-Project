"""Flat checkpoint contract: conversation context plus explicitly reset turn data."""

import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from app.services.workflow_compat import LegacyCheckpointFields


class HistoryEntry(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class KnowledgeChunk(TypedDict):
    source: str
    chunk_index: int
    content: str
    distance: float


class KnowledgeEvidence(TypedDict):
    kind: Literal["knowledge"]
    data: KnowledgeChunk


class ToolEvidence(TypedDict):
    kind: Literal["tool"]
    tool: str
    scope: dict[str, Any]
    outcome: Literal["ok", "error"]
    data: Any


Evidence = KnowledgeEvidence | ToolEvidence


class RefundProposal(TypedDict):
    action: Literal["refund"]
    proposal_id: str
    order_id: str
    amount_cents: int
    currency: str
    status: NotRequired[Literal["proposed"]]
    requires_human_approval: NotRequired[bool]
    message: NotRequired[str]


class ReviewResult(TypedDict):
    passed: bool
    issues: list[str]
    feedback: str
    attempt: int


class ApprovalDecision(TypedDict):
    approved: bool
    note: str


class RefundResult(TypedDict):
    refund_id: str
    proposal_id: str
    order_id: str
    amount_cents: int
    currency: str
    status: str
    created_at: NotRequired[Any]


class RoutingStep(TypedDict):
    node: str
    status: Literal["completed", "passed", "failed", "approved", "rejected"]


class SupportState(LegacyCheckpointFields, total=False):
    # Continuing context; only history uses an append reducer.
    schema_version: int
    run_id: str
    thread_id: str
    conversation_history: Annotated[list[HistoryEntry], operator.add]
    intent: str
    entities: dict[str, str]
    urgency: str
    awaiting_device_id: bool

    # Current turn. Intake resets these; list writers return complete values.
    customer_message: str
    specialist: str
    reply: str
    evidence: list[Evidence]
    response_kind: Literal["generated", "clarification", "abstention", "handoff"]
    qa_result: ReviewResult | None
    proposed_action: RefundProposal | None
    approval_decision: ApprovalDecision | None
    action_result: RefundResult | None
    escalation_context: str
    tool_calls: list[str]
    steps: list[str]
    routing_path: list[RoutingStep]
