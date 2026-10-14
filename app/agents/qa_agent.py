"""Review drafted replies for grounding, correctness, tone, and citations."""

import re

from langchain_core.messages import HumanMessage, SystemMessage
from app.llm.factory import chat_model_name, get_chat_model
from pydantic import BaseModel, Field

from app.graph.state import SupportState
from app.graph.routing import requires_approval
from app.services.workflow_compat import evidence_for, reply_for, review_for
from app.services.observability import model_usage


class QAReview(BaseModel):
    passed: bool = Field(description="Whether the draft is safe, grounded, and ready to send")
    issues: list[str] = Field(description="Specific problems that must be fixed; empty when passed")
    feedback: str = Field(description="Concise, actionable directions for improving a failed draft")


def qa_reviewer_agent(state: SupportState) -> dict:
    attempt = review_for(state)["attempt"] + 1
    records = evidence_for(state)
    reply = reply_for(state)
    # Old billing checkpoints retained summaries, not raw lookup results.
    if state.get("specialist") == "billing" and "evidence" not in state:
        return {
            "qa_result": {
                "passed": False, "attempt": 1,
                "issues": ["Legacy billing evidence is missing."],
                "feedback": "Look up the order and payments again; supply actual tool results for review.",
            },
            "response_kind": "generated",
            "steps": ["qa reviewer: requested fresh billing evidence for legacy checkpoint"],
        }
    model = get_chat_model().with_structured_output(QAReview, include_raw=True)
    evidence = {
        "specialist": state.get("specialist"),
        "records": records,
        "proposed_action": state.get("proposed_action"),
        "entities": state.get("entities", {}),
        "response_kind": state.get("response_kind", "generated"),
    }
    review_response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strict customer-support quality reviewer. Check that every factual claim in "
                    "the reply is supported by actual tool results or retrieved knowledge, that the reply "
                    "answers the customer's request, follows the evidence and policy, and has an appropriate "
                    "professional tone. For FAQ replies, require a citation that exactly matches a retrieved "
                    "source chunk. Fail drafts with unsupported promises, invented actions, missing required "
                    "citations, or material factual errors. Pass only when there are no material issues. "
                    "If a refund is pending approval, require the draft to say it is only proposed or awaiting "
                    "approval and must not claim it has been issued. "
                    "When failing, give concise, actionable feedback for the specialist to address."
                    " For billing and technical replies, inspect the raw records rather than trusting "
                    "specialist summaries alone. For technical replies, without a customer-provided device ID and matching log "
                    "records, reject claims of a device-specific diagnosis or resolution. Historical demo "
                    "logs do not prove customer ownership or the current connection's status."
                )
            ),
            HumanMessage(
                content=(
                    f"Customer message:\n{state['customer_message']}\n\n"
                    f"Recent conversation:\n{state.get('conversation_history', [])[-9:-1]}\n\n"
                    f"Evidence:\n{evidence}\n\nDraft reply:\n{reply}"
                )
            ),
        ]
    )

    review = review_response.get("parsed")
    parsed = isinstance(review, QAReview) and not review_response.get("parsing_error")
    issues = list(review.issues) if parsed else ["QA output could not be parsed; submit the answer for review again."]
    chunks = [item["data"] for item in records if item["kind"] == "knowledge"]
    if state.get("specialist") == "faq" and chunks:
        valid_citations = {
            f"[{chunk['source']}#chunk-{chunk['chunk_index']:03d}]"
            for chunk in chunks
        }
        draft_citations = set(re.findall(r"\[[^\]\n]+#chunk-\d{3}\]", reply))
        if not draft_citations or not draft_citations.intersection(valid_citations):
            issues.append("Add a citation that exactly matches a retrieved knowledge source chunk.")
        elif not draft_citations.issubset(valid_citations):
            issues.append("Remove citations that do not match a retrieved knowledge source chunk.")

    if requires_approval(state):
        draft = reply.lower()
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

    passed = parsed and review.passed and not issues
    if not passed and not issues:
        issues.append("Reviewer requested a revision; see the feedback below.")
    feedback = review.feedback.strip() if parsed else ""
    if issues:
        feedback = " ".join(filter(None, [feedback, "Fix the QA issues: " + "; ".join(issues)]))

    result = {
        "passed": passed,
        "issues": issues,
        "feedback": feedback,
        "attempt": attempt,
    }
    return {
        "qa_result": result,
        "model_usage": [model_usage(review_response.get("raw"), chat_model_name())],
        "steps": [
            f"qa reviewer: {'passed' if passed else 'failed'} on attempt {attempt}"
            + (f" ({len(issues)} issue(s))" if issues else "")
        ],
    }
