"""Opt-in checks using real PostgreSQL transactions and fresh saver connections."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import pytest
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from app.graph import steps, workflow
from app.tools import billing


PROPOSAL = {
    "action": "refund", "proposal_id": "persisted-proposal", "order_id": "12345",
    "amount_cents": 499900, "currency": "INR", "status": "proposed",
}


@pytest.fixture
def business_database(postgres_url, monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(tmp_path / "trace.jsonl"))
    with psycopg.connect(postgres_url, autocommit=True) as connection:
        connection.execute((Path(__file__).resolve().parents[1] / "data/seed/001_seed.sql").read_text())
        connection.execute(billing.CREATE_REFUNDS_TABLE_SQL)
    return postgres_url


@pytest.mark.parametrize("stage", ["awaiting_approval", "after_ledger_commit", "before_finalization"])
def test_refund_resume_with_fresh_connection_is_idempotent(business_database, monkeypatch, stage):
    monkeypatch.setattr(workflow, "billing_agent", lambda state: {
        "specialist": "billing", "reply": "Refund awaiting approval.", "evidence": [], "proposed_action": PROPOSAL,
    })
    monkeypatch.setattr(workflow, "qa_reviewer_agent", lambda state: {"qa_result": {"passed": True, "attempt": 1}})

    def fail_after_commit(state):
        steps.execute_approved_action(state)
        raise RuntimeError("Simulated failure after ledger commit, before graph checkpoint.")

    def fail_finalization(state):
        raise RuntimeError("Simulated failure before history finalization.")

    if stage == "after_ledger_commit":
        monkeypatch.setattr(workflow, "execute_approved_action", fail_after_commit)
    if stage == "before_finalization":
        monkeypatch.setattr(workflow, "finalize_turn", fail_finalization)

    config = {"configurable": {"thread_id": "persistent-refund", "run_id": "first"}}
    with PostgresSaver.from_conn_string(business_database) as saver:
        saver.setup()
        graph = workflow.build_graph(saver)
        graph.invoke({"customer_message": "Refund order 12345"}, config)
        assert graph.get_state(config).next == ("approval_gate",)
        if stage != "awaiting_approval":
            with pytest.raises(RuntimeError):
                graph.invoke(Command(resume={"approved": True}), config)

    # Rebuild using an independently opened connection, as after process restart.
    monkeypatch.setattr(workflow, "execute_approved_action", steps.execute_approved_action)
    monkeypatch.setattr(workflow, "finalize_turn", steps.finalize_turn)
    resumed = {"configurable": {"thread_id": "persistent-refund", "run_id": "resumed"}}
    with PostgresSaver.from_conn_string(business_database) as saver:
        graph = workflow.build_graph(saver)
        command = Command(resume={"approved": True}) if stage == "awaiting_approval" else None
        result = graph.invoke(command, resumed)
        assert graph.get_state(resumed).next == ()
        assert result["action_result"]["status"] == "issued"
        assert len(result["conversation_history"]) == 2
        graph.invoke(None, resumed)
        assert len(graph.get_state(resumed).values["conversation_history"]) == 2
    with psycopg.connect(business_database) as connection:
        count, total = connection.execute("SELECT COUNT(*), SUM(amount_cents) FROM refunds").fetchone()
    assert count == 1 and total == 499900
    assert billing._refund_capacity("12345")["refundable_cents"] == 0


def test_concurrent_approvals_cannot_spend_same_capacity_twice(business_database):
    def execute(index):
        try:
            return billing.execute_approved_refund({**PROPOSAL, "proposal_id": f"concurrent-{index}"})
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, [1, 2]))
    issued = [result for result in results if result is not None]
    assert len(issued) == 1
    replay = billing.execute_approved_refund({**PROPOSAL, "proposal_id": issued[0]["proposal_id"]})
    assert replay["refund_id"] == issued[0]["refund_id"]
