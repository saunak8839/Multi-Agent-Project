"""Technical specialist placeholder; diagnostics are a later iteration."""

from app.graph.state import SupportState


def technical_agent(state: SupportState) -> dict:
    return {
        "specialist": "technical",
        "findings": "Technical issue received. No device, service-status, or log lookup tools are connected yet; ask the customer what they have tried and for any error message.",
        "steps": ["technical specialist: prepared a diagnostics placeholder finding"],
    }
