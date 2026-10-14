"""Resume actual pre-refactor channel layouts and tasks with the new graph."""

from types import SimpleNamespace

import pytest
from langgraph.types import Command

from app import api
from app.agents import qa_agent
from app.graph import steps, workflow
from app.services.workflow_compat import legacy_draft, reply_for, review_for
from fixtures import legacy_workflow


@pytest.fixture(autouse=True)
def fake_dependencies(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(tmp_path / "traces.jsonl"))
    executions = []

    def record_refund(action):
        executions.append(action["proposal_id"])
        return {**action, "refund_id": "r1", "status": "issued"}

    monkeypatch.setattr(steps, "execute_approved_refund", record_refund)
    monkeypatch.setattr(legacy_workflow, "execute_approved_refund", record_refund)

    class Reviewer:
        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            return {"parsed": qa_agent.QAReview(passed=True, issues=[], feedback=""), "raw": SimpleNamespace(usage_metadata={})}

    monkeypatch.setattr(qa_agent, "get_chat_model", lambda: Reviewer())
    return executions


@pytest.mark.parametrize("stage", ["completed", "billing", "draft", "qa", "approval_gate", "execute_action", "escalate", "finalize_turn"])
def test_saved_legacy_turn_resumes_with_new_contract(checkpoint_backend, monkeypatch, fake_dependencies, stage):
    config = {"configurable": {"thread_id": "legacy-" + stage, "run_id": "old-run"}}
    stop = None if stage in {"completed", "approval_gate"} else stage
    old = legacy_workflow.build_legacy_graph(checkpoint_backend, stop_before=stop)
    message = "What is the policy?" if stage == "completed" else "Refund order 12345"
    old.invoke({"customer_message": message, **config["configurable"]}, config)
    if stage in {"execute_action", "escalate", "finalize_turn"}:
        old.invoke(Command(resume={"approved": stage != "escalate", "note": "old reviewer"}), config)
    before = old.get_state(config)
    assert before.next == (() if stage == "completed" else (stage,))

    investigations = []

    def billing(state):
        investigations.append(1)
        return {
            "schema_version": 2, "specialist": "billing", "response_kind": "generated",
            "reply": "Verified duplicate payment; refund awaiting approval.",
            "evidence": [{"kind": "tool", "tool": "get_payments", "scope": {"order_id": "12345"}, "outcome": "ok", "data": {"payments": ["fresh raw record"]}}],
            "proposed_action": state.get("proposed_action") or {
                "action": "refund", "proposal_id": "fresh-proposal", "order_id": "12345", "amount_cents": 499900, "currency": "INR",
            },
        }

    monkeypatch.setattr(workflow, "billing_agent", billing)
    graph = workflow.build_graph(checkpoint_backend)
    resumed = {"configurable": {**config["configurable"], "run_id": "new-run"}}
    if stage == "completed":
        assert api.view_for(graph, resumed).history[-1].content == before.values["draft_reply"]
        monkeypatch.setattr(workflow, "faq_agent", lambda state: {"specialist": "faq", "reply": "New answer", "evidence": [], "response_kind": "generated"})
        graph = workflow.build_graph(checkpoint_backend)
        result = graph.invoke({"customer_message": "What is the policy?"}, resumed)
        assert len(result["conversation_history"]) == 4
        assert result["conversation_history"][:2] == before.values["conversation_history"]
        assert result["schema_version"] == 2
        assert result["qa_result"]["attempt"] == 1
        assert result["tool_calls"] == []
        assert "draft" not in [step["node"] for step in result["routing_path"]]
        return

    command = Command(resume={"approved": True, "note": "new reviewer"}) if stage == "approval_gate" else None
    if stage == "approval_gate":
        view = api.view_for(graph, resumed)
        assert view.pending_approval.draft_reply == before.values["draft_reply"]
        assert len(view.history) == 1
    result = graph.invoke(command, resumed)
    if graph.get_state(resumed).next == ("approval_gate",):
        result = graph.invoke(Command(resume={"approved": True}), resumed)
    assert graph.get_state(resumed).next == ()
    assert len(result["conversation_history"]) == 2
    assert result["conversation_history"][-1]["content"] == reply_for(result)
    if stage == "escalate":
        assert fake_dependencies == []
        assert "not approved" in reply_for(result)
    else:
        assert len(fake_dependencies) == 1
        assert "demo refund ledger" in reply_for(result)
    if stage in {"billing", "draft", "qa"}:
        assert investigations == [1]
    else:
        assert investigations == []
    if stage == "finalize_turn":
        assert result["conversation_history"][-1]["content"] == before.values["draft_reply"]
    # Reopening an already-completed turn must not append or execute again.
    graph.invoke(None, resumed)
    assert len(graph.get_state(resumed).values["conversation_history"]) == 2


def test_legacy_qa_never_accepts_generated_findings_as_payment_evidence(monkeypatch):
    monkeypatch.setattr(qa_agent, "get_chat_model", lambda: pytest.fail("Missing raw legacy evidence must trigger investigation before model review."))
    result = qa_agent.qa_reviewer_agent({
        "customer_message": "Refund order 12345", "specialist": "billing",
        "findings": "I verified the payment.", "draft_reply": "Refund awaiting approval.",
        "qa_attempts": 3,
    })
    assert result["qa_result"]["passed"] is False
    assert result["qa_result"]["attempt"] == 1
    assert "actual tool results" in result["qa_result"]["feedback"]


def test_legacy_pending_draft_uses_newly_revised_findings():
    result = legacy_draft({"findings": "Revised answer", "draft_reply": "Old rejected answer"})
    assert result["reply"] == "Revised answer"


def test_new_turn_ignores_stale_legacy_review_channels():
    from app.graph.intake import intake

    old = {"customer_message": "What is the policy?", "qa_attempts": 3, "qa_feedback": "Old feedback", "qa_result": {"passed": False, "attempt": 3}}
    state = {**old, **intake(old)}
    assert review_for(state)["attempt"] == 0
    assert review_for(state)["feedback"] == ""
