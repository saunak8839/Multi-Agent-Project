"""Turn specialist findings into a simple customer-facing reply."""

from app.graph.state import SupportState


def draft_agent(state: SupportState) -> dict:
    reply = f"Thanks for contacting support. {state['findings']}"
    return {
        "draft_reply": reply,
        "steps": ["draft: created a reply from the specialist finding"],
    }
