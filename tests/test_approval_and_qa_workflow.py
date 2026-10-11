import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agents import action_agent
from app.agents.qa_agent import MAX_QA_ATTEMPTS
from app.graph import workflow


def _proposed_refund(_state):
    return {
        "specialist": "billing",
        "findings": "Verified duplicate payment; refund proposal awaits approval.",
        "proposed_action": {
            "action": "refund", "proposal_id": "proposal-1", "order_id": "12345",
            "amount_cents": 500, "currency": "USD",
        },
        "requires_approval": True,
    }


def _passing_qa(_state):
    return {
        "qa_result": {"passed": True, "issues": [], "feedback": "", "attempt": 1},
        "qa_attempts": 1,
    }


def _refund_graph(monkeypatch, trace_path, execute):
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(trace_path))
    monkeypatch.setattr(workflow, "billing_agent", _proposed_refund)
    monkeypatch.setattr(workflow, "qa_reviewer_agent", _passing_qa)
    monkeypatch.setattr(workflow, "execute_approved_action", execute)
    return workflow.build_graph(InMemorySaver())


@pytest.mark.parametrize("approved", [False, True])
def test_refund_pauses_until_decision_and_follows_approval_branch(
    monkeypatch, tmp_path, approved
):
    executions = []

    def execute(state):
        assert state["approval_decision"]["approved"] is True
        executions.append(state["proposed_action"]["proposal_id"])
        return {"action_result": {"status": "issued"}, "draft_reply": "Approved refund recorded."}

    graph = _refund_graph(monkeypatch, tmp_path / "traces.jsonl", execute)
    config = {"configurable": {"thread_id": f"approval-{approved}", "run_id": "approval-run"}}
    paused = graph.invoke({"customer_message": "Refund the duplicate charge"}, config)

    assert paused.get("__interrupt__")
    assert executions == []
    assert "approval_gate" in graph.get_state(config).next

    result = graph.invoke(Command(resume={"approved": approved, "note": "reviewed"}), config)
    if approved:
        assert result["action_result"]["status"] == "issued"
        assert executions == ["proposal-1"]
        try:
            graph.invoke(Command(resume={"approved": True, "note": "replayed"}), config)
        except (ValueError, RuntimeError):
            # A completed thread may reject a stale resume command; it must not
            # cause the already committed action node to run again.
            pass
        assert executions == ["proposal-1"]
    else:
        assert executions == []
        assert "not approved" in result["draft_reply"]
        assert result["escalation_context"]


def test_qa_failure_retries_only_to_limit_then_uses_safe_fallback(monkeypatch, tmp_path):
    specialist_calls = []
    qa_calls = []
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(tmp_path / "traces.jsonl"))

    def specialist(_state):
        specialist_calls.append(1)
        return {"specialist": "faq", "findings": "Unverified draft evidence."}

    def failing_qa(state):
        attempt = state.get("qa_attempts", 0) + 1
        qa_calls.append(attempt)
        return {
            "qa_result": {"passed": False, "issues": ["unsupported"], "feedback": "Use verified evidence."},
            "qa_feedback": "Use verified evidence.",
            "qa_attempts": attempt,
        }

    monkeypatch.setattr(workflow, "faq_agent", specialist)
    monkeypatch.setattr(workflow, "qa_reviewer_agent", failing_qa)
    graph = workflow.build_graph(InMemorySaver())
    result = graph.invoke(
        {"customer_message": "What is the policy?"},
        {"configurable": {"thread_id": "qa-limit", "run_id": "qa-limit-run"}},
    )

    assert qa_calls == list(range(1, MAX_QA_ATTEMPTS + 1))
    assert len(specialist_calls) == MAX_QA_ATTEMPTS
    assert "qa fallback" in result["steps"][-2]
    assert "couldn't verify an answer" in result["draft_reply"]
    events = [json.loads(line) for line in (tmp_path / "traces.jsonl").read_text().splitlines()]
    assert sum(event.get("node") == "qa" for event in events) == MAX_QA_ATTEMPTS


def test_action_executor_rejects_unapproved_refund(monkeypatch):
    def should_not_execute(_action):
        pytest.fail("refund tool ran without approval")

    monkeypatch.setattr(action_agent, "execute_approved_refund", should_not_execute)
    with pytest.raises(PermissionError):
        action_agent.execute_approved_action({
            "proposed_action": {"action": "refund"},
            "approval_decision": {"approved": False},
        })


def test_refund_execution_failure_is_traced_without_sensitive_error_text(monkeypatch, tmp_path):
    trace_path = tmp_path / "refund-failure.jsonl"
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(trace_path))

    def fail(_action):
        raise RuntimeError("database failure for private proposal proposal-7")

    monkeypatch.setattr(action_agent, "execute_approved_refund", fail)
    from app.services.observability import observe_node
    observed_executor = observe_node("execute_action", action_agent.execute_approved_action)
    with pytest.raises(RuntimeError):
        observed_executor({
            "proposed_action": {"action": "refund", "proposal_id": "proposal-7"},
            "approval_decision": {"approved": True},
        }, {"configurable": {"run_id": "refund-error", "thread_id": "thread-error"}})
    event = json.loads(trace_path.read_text())
    assert event["node"] == "execute_action"
    assert event["status"] == "error"
    assert "proposal-7" not in trace_path.read_text()


def test_refund_ledger_execution_is_idempotent_by_proposal_id(monkeypatch):
    from app.tools import billing

    class Cursor:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    class FakeConnection:
        def __init__(self):
            self.issued = {}
            self.insert_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params=()):
            if "SELECT order_id, total_cents" in sql:
                return Cursor({"order_id": "12345", "total_cents": 2000, "currency": "USD"})
            if "WHERE proposal_id = %s" in sql:
                return Cursor(self.issued.get(params[0]))
            if "AS paid_cents" in sql:
                return Cursor({"paid_cents": 3000})
            if "AS refunded_cents" in sql:
                return Cursor({"refunded_cents": 0})
            if "INSERT INTO refunds" in sql:
                self.insert_count += 1
                row = {
                    "refund_id": "refund-1", "proposal_id": params[1], "order_id": params[2],
                    "amount_cents": params[3], "currency": params[4], "status": "issued",
                }
                self.issued[params[1]] = row
                return Cursor(row)
            return Cursor()

    connection = FakeConnection()
    monkeypatch.setattr(billing, "connect_db", lambda: connection)
    action = {
        "proposal_id": "proposal-once", "order_id": "12345", "amount_cents": 1000,
        "action": "refund",
    }
    first = billing.execute_approved_refund(action)
    second = billing.execute_approved_refund(action)
    assert first == second
    assert connection.insert_count == 1
