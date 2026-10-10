"""FAQ specialist placeholder; knowledge retrieval is a later iteration."""

from app.graph.state import SupportState


def faq_agent(state: SupportState) -> dict:
    return {
        "specialist": "faq",
        "findings": "No knowledge base is connected yet. The question has been classified as a general FAQ.",
        "steps": ["faq specialist: prepared a knowledge-base placeholder finding"],
    }
