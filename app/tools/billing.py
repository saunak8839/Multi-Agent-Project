"""Read-only billing tools exposed to the billing agent."""

import json

from langchain_core.tools import tool

from app.models.database import query_all, query_one


@tool
def get_order(order_id: str) -> str:
    """Look up a seeded order by its order ID. Amount is in currency units, not cents. Use this when a customer asks about an order or its billing."""
    order = query_one(
        "SELECT order_id, customer_name, status, total_cents, "
        "ROUND(total_cents / 100.0, 2) AS total_amount, currency, created_at "
        "FROM orders WHERE order_id = %s",
        (order_id,),
    )
    return json.dumps(order or {"error": f"Order {order_id} was not found."}, default=str)


@tool
def get_payments(order_id: str) -> str:
    """List payment attempts recorded for an order, including status and amount in currency units (not cents). Use to check for duplicate charges."""
    payments = query_all(
        "SELECT payment_id, order_id, amount_cents, "
        "ROUND(amount_cents / 100.0, 2) AS amount, currency, status, created_at "
        "FROM payments WHERE order_id = %s ORDER BY created_at",
        (order_id,),
    )
    return json.dumps({"order_id": order_id, "payments": payments}, default=str)


BILLING_TOOLS = [get_order, get_payments]
