"""Persist the completed assistant turn in the checkpointed thread state."""

from datetime import datetime, timezone

from app.graph.state import SupportState


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
