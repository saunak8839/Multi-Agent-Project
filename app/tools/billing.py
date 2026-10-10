"""Billing lookup and refund-proposal tools; execution remains separately gated."""

import json
import uuid

from langchain_core.tools import tool

from app.models.database import connect_db, query_all, query_one


CREATE_REFUNDS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS refunds (
    refund_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL UNIQUE,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    currency CHAR(3) NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def _refund_capacity(order_id: str) -> dict:
    order = query_one(
        "SELECT order_id, total_cents, currency FROM orders WHERE order_id = %s",
        (order_id,),
    )
    if not order:
        return {"error": f"Order {order_id} was not found."}
    payments = query_all(
        "SELECT COALESCE(SUM(amount_cents), 0) AS paid_cents FROM payments "
        "WHERE order_id = %s AND status = 'succeeded'",
        (order_id,),
    )
    with connect_db() as connection:
        connection.execute(CREATE_REFUNDS_TABLE_SQL)
        refunds = connection.execute(
            "SELECT COALESCE(SUM(amount_cents), 0) AS refunded_cents FROM refunds "
            "WHERE order_id = %s AND status = 'issued'",
            (order_id,),
        ).fetchone()
    paid_cents = int(payments[0]["paid_cents"])
    refunded_cents = int(refunds["refunded_cents"])
    refundable_cents = max(0, paid_cents - int(order["total_cents"]) - refunded_cents)
    return {
        "order_id": order_id,
        "currency": order["currency"],
        "refundable_cents": refundable_cents,
        "paid_cents": paid_cents,
        "refunded_cents": refunded_cents,
    }


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


@tool
def propose_refund(order_id: str, amount_cents: int) -> str:
    """Propose, but never issue, a refund for a verified duplicate charge. Use the excess payment amount in cents; a human must approve before execution."""
    capacity = _refund_capacity(order_id)
    if "error" in capacity:
        return json.dumps(capacity)
    if amount_cents <= 0 or amount_cents > capacity["refundable_cents"]:
        return json.dumps({
            "error": "Requested refund exceeds the verified refundable duplicate-charge amount.",
            **capacity,
        })
    return json.dumps({
        "status": "proposed",
        "proposal_id": uuid.uuid4().hex,
        "action": "refund",
        "order_id": order_id,
        "amount_cents": amount_cents,
        "currency": capacity["currency"],
        "requires_human_approval": True,
        "message": "Proposal recorded in workflow state only; no refund has been issued.",
    })


BILLING_TOOLS = [get_order, get_payments, propose_refund]


def execute_approved_refund(action: dict) -> dict:
    """Write an approved refund to the demo ledger, with capacity rechecked."""
    order_id = action["order_id"]
    amount_cents = int(action["amount_cents"])
    proposal_id = action["proposal_id"]
    with connect_db() as connection:
        connection.execute(CREATE_REFUNDS_TABLE_SQL)
        # Serialize approvals for an order so concurrent approvals cannot
        # spend the same duplicate-charge capacity twice.
        order = connection.execute(
            "SELECT order_id, total_cents, currency FROM orders WHERE order_id = %s FOR UPDATE",
            (order_id,),
        ).fetchone()
        if not order:
            raise ValueError(f"Order {order_id} was not found while executing the approved refund.")
        existing = connection.execute(
            "SELECT refund_id, proposal_id, order_id, amount_cents, currency, status "
            "FROM refunds WHERE proposal_id = %s",
            (proposal_id,),
        ).fetchone()
        if existing:
            return dict(existing)
        payment = connection.execute(
            "SELECT COALESCE(SUM(amount_cents), 0) AS paid_cents FROM payments "
            "WHERE order_id = %s AND status = 'succeeded'",
            (order_id,),
        ).fetchone()
        refunded = connection.execute(
            "SELECT COALESCE(SUM(amount_cents), 0) AS refunded_cents FROM refunds "
            "WHERE order_id = %s AND status = 'issued'",
            (order_id,),
        ).fetchone()
        capacity = max(
            0,
            int(payment["paid_cents"])
            - int(order["total_cents"])
            - int(refunded["refunded_cents"]),
        )
        if amount_cents <= 0 or amount_cents > capacity:
            raise ValueError("The approved refund no longer fits the verified duplicate-charge amount.")
        refund_id = uuid.uuid4().hex
        row = connection.execute(
            "INSERT INTO refunds (refund_id, proposal_id, order_id, amount_cents, currency, status) "
            "VALUES (%s, %s, %s, %s, %s, 'issued') "
            "RETURNING refund_id, proposal_id, order_id, amount_cents, currency, status, created_at",
            (refund_id, proposal_id, order_id, amount_cents, order["currency"]),
        ).fetchone()
        return dict(row)
