"""Temporary reads for pre-v2 checkpoints; retire after legacy pending turns drain.

Keep the draft adapter registered until no saved task targets ``draft``. New turns
write only the v2 contract; legacy channels remain declared to deserialize old state.
"""

from typing import Any, TypedDict


class LegacyCheckpointFields(TypedDict, total=False):
    findings: str
    draft_reply: str
    retrieved_context: list[dict[str, Any]]
    technical_context: dict[str, Any]
    current_tool_calls: list[str]
    requires_approval: bool
    qa_feedback: str
    qa_attempts: int


def reply_for(state: dict) -> str:
    if "reply" in state:
        return state["reply"]
    return state.get("draft_reply") or state.get("findings", "")


def review_for(state: dict) -> dict:
    review = dict(state.get("qa_result") or {})
    legacy = state.get("schema_version", 1) < 2
    review.setdefault("attempt", state.get("qa_attempts", 0) if legacy else 0)
    review.setdefault("feedback", state.get("qa_feedback", "") if legacy else "")
    return review


def tools_for(state: dict) -> list[str]:
    if state.get("schema_version", 1) < 2:
        return list(state.get("current_tool_calls", []))
    return list(state.get("tool_calls", []))


def evidence_for(state: dict) -> list[dict]:
    if "evidence" in state:
        return list(state["evidence"])
    evidence = [{"kind": "knowledge", "data": chunk} for chunk in state.get("retrieved_context", [])]
    for lookup in state.get("technical_context", {}).get("lookups", []):
        evidence.append({
            "kind": "tool", "tool": "search_technical_logs",
            "scope": {"device_id": lookup.get("device_id")},
            "outcome": "error" if lookup.get("error") else "ok", "data": lookup,
        })
    return evidence


def legacy_draft(state: dict) -> dict:
    """Finish an old pending draft task; no new graph edge selects this node."""
    # A specialist may have revised findings after an older draft failed review.
    reply = state.get("reply") or state.get("findings") or state.get("draft_reply", "")
    return {"reply": reply, "response_kind": "generated", "steps": ["compatibility: resumed legacy draft"]}
