"""Review drafted replies for grounding, correctness, tone, and citations."""

import re

from langchain_core.messages import HumanMessage, SystemMessage
from app.llm.factory import chat_model_name, get_chat_model
from pydantic import BaseModel, Field

from app.graph.state import SupportState
from app.services.observability import model_usage


MAX_QA_ATTEMPTS = 3


class QAReview(BaseModel):
    passed: bool = Field(description="Whether the draft is safe, grounded, and ready to send")
    issues: list[str] = Field(description="Specific problems that must be fixed; empty when passed")
    feedback: str = Field(description="Concise, actionable directions for improving a failed draft")


def qa_reviewer_agent(state: SupportState) -> dict:
    attempt = state.get("qa_attempts", 0) + 1
    model = get_chat_model().with_structured_output(QAReview, include_raw=True)
    evidence = {
        "specialist": state.get("specialist"),
        "specialist_findings": state.get("findings", ""),
        "retrieved_knowledge_chunks": state.get("retrieved_context", []),
        "tool_calls": state.get("current_tool_calls", []),
        "proposed_action": state.get("proposed_action"),
        "technical_log_evidence": state.get("technical_context", {}),
    }
    review_response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strict customer-support quality reviewer. Check that every factual claim in "
                    "the draft is supported by specialist findings or retrieved knowledge, that the reply "
                    "answers the customer's request, follows the evidence and policy, and has an appropriate "
                    "professional tone. For FAQ replies, require a citation that exactly matches a retrieved "
                    "source chunk. Fail drafts with unsupported promises, invented actions, missing required "
                    "citations, or material factual errors. Pass only when there are no material issues. "
                    "If a refund is pending approval, require the draft to say it is only proposed or awaiting "
                    "approval and must not claim it has been issued. "
                    "When failing, give concise, actionable feedback for the specialist to address."
                    " For technical replies, check the raw technical_log_evidence rather than trusting "
                    "specialist summaries alone. Without a customer-provided device ID and matching log "
                    "records, reject claims of a device-specific diagnosis or resolution. Historical demo "
                    "logs do not prove customer ownership or the current connection's status."
                )
            ),
            HumanMessage(
                content=(
                    f"Customer message:\n{state['customer_message']}\n\n"
                    f"Evidence:\n{evidence}\n\nDraft reply:\n{state.get('draft_reply', '')}"
                )
            ),
        ]
    )

    review = review_response["parsed"]
    issues = list(review.issues)
    if state.get("specialist") == "faq" and state.get("retrieved_context"):
        valid_citations = {
            f"[{chunk['source']}#chunk-{chunk['chunk_index']:03d}]"
            for chunk in state["retrieved_context"]
        }
        draft_citations = set(re.findall(r"\[[^\]\n]+#chunk-\d{3}\]", state.get("draft_reply", "")))
        if not draft_citations or not draft_citations.intersection(valid_citations):
            issues.append("Add a citation that exactly matches a retrieved knowledge source chunk.")
        elif not draft_citations.issubset(valid_citations):
            issues.append("Remove citations that do not match a retrieved knowledge source chunk.")

    if state.get("requires_approval"):
        draft = state.get("draft_reply", "").lower()
        approval_language = (
            "awaiting approval", "awaiting human approval", "pending approval",
            "pending human approval", "proposed for approval", "requires your approval",
            "requires human approval", "awaiting your approval", "pending your approval",
            "subject to approval",
        )
        premature_claims = (
            "refund has been issued", "refund was issued", "refund has been processed",
            "refund was processed", "refund is complete", "refund has been sent",
            "refund was sent", "refund initiated", "initiated your refund",
            "i've refunded", "i have refunded", "i issued your refund",
        )
        if not any(phrase in draft for phrase in approval_language):
            issues.append("State clearly that the refund is proposed and awaiting human approval.")
        if any(phrase in draft for phrase in premature_claims):
            issues.append("Remove any claim that the refund was executed before human approval.")

    passed = review.passed and not issues
    if not passed and not issues:
        issues.append("Reviewer requested a revision; see the feedback below.")
    feedback = review.feedback.strip()
    if issues and not feedback:
        feedback = "Fix the QA issues: " + "; ".join(issues)

    result = {
        "passed": passed,
        "issues": issues,
        "feedback": feedback,
        "attempt": attempt,
    }
    return {
        "qa_result": result,
        "qa_feedback": "" if passed else feedback,
        "qa_attempts": attempt,
        "model_usage": [model_usage(review_response["raw"], chat_model_name())],
        "steps": [
            f"qa reviewer: {'passed' if passed else 'failed'} on attempt {attempt}"
            + (f" ({len(issues)} issue(s))" if issues else "")
        ],
    }


def qa_fallback(state: SupportState) -> dict:
    """Replace a repeatedly rejected draft with a safe holding response."""
    return {
        "draft_reply": (
            "Thanks for contacting support. I couldn't verify an answer that meets our quality checks, "
            "so I’m passing your request to a support specialist for review."
        ),
        "steps": ["qa fallback: retry limit reached; replaced draft with safe handoff response"],
    }
