"""Billing specialist placeholder; billing tools are a later iteration."""

from app.graph.state import SupportState


def billing_agent(state: SupportState) -> dict:
    order_id = state.get("entities", {}).get("order_id")
    if order_id:
        findings = f"Billing request received for order {order_id}. No billing system is connected yet, so no account or payment was looked up."
    else:
        findings = "Billing request received. Ask the customer for an order or invoice number so it can be investigated. No billing system is connected yet."
    return {
        "specialist": "billing",
        "findings": findings,
        "steps": ["billing specialist: prepared a safe placeholder finding"],
    }
