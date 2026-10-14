import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.agents import billing_agent, faq_agent, qa_agent, technical_agent
from app.graph import workflow
from app.graph.intake import intake
from app.graph.routing import requires_approval
from app.services import tool_runner


PROPOSAL = {
    "action": "refund", "proposal_id": "p1", "order_id": "12345",
    "amount_cents": 100, "currency": "INR", "status": "proposed",
}
PAYMENT_EVIDENCE = {
    "kind": "tool", "tool": "get_payments", "scope": {"order_id": "12345"}, "outcome": "ok",
    "data": {"order_id": "12345", "payments": [{"payment_id": "p-private", "amount_cents": 100, "status": "succeeded"}]},
}


class ReviewModel:
    def __init__(self, review=None, inspect=None):
        self.review = review
        self.inspect = inspect
        self.calls = 0

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.inspect:
            self.inspect(messages)
        return {
            "parsed": self.review,
            "parsing_error": ValueError("private provider details") if self.review is None else None,
            "raw": SimpleNamespace(usage_metadata={"input_tokens": 2, "output_tokens": 1}),
        }


@pytest.fixture(autouse=True)
def local_trace(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(tmp_path / "trace.jsonl"))


def test_faq_reply_and_raw_passage_reach_qa_without_draft(monkeypatch):
    chunk = {"source": "shipping.md", "chunk_index": 1, "content": "Use the tracking email.", "distance": 0.1}
    monkeypatch.setattr(faq_agent, "retrieve_faq_context", lambda query: [chunk])
    monkeypatch.setattr(faq_agent, "get_chat_model", lambda: SimpleNamespace(invoke=lambda messages: AIMessage(content="Use your tracking email [shipping.md#chunk-001].")))

    def inspect(messages):
        assert chunk["content"] in messages[-1].content
        assert "Use your tracking email" in messages[-1].content

    reviewer = ReviewModel(qa_agent.QAReview(passed=True, issues=[], feedback=""), inspect)
    monkeypatch.setattr(qa_agent, "get_chat_model", lambda: reviewer)
    graph = workflow.build_graph(InMemorySaver())
    result = graph.invoke({"customer_message": "Where is my parcel?"}, {"configurable": {"thread_id": "faq"}})
    assert result["qa_result"]["passed"] is True
    assert result["evidence"] == [{"kind": "knowledge", "data": chunk}]
    assert [step["node"] for step in result["routing_path"]] == ["intake", "faq", "qa", "finalize_turn"]
    assert result["conversation_history"][-1]["content"] == result["reply"]
    assert "findings" not in result and "draft_reply" not in result


def test_faq_revision_reuses_passages_and_complete_feedback(monkeypatch):
    chunk = {"source": "shipping.md", "chunk_index": 1, "content": "Use the tracking email.", "distance": 0.1}
    monkeypatch.setattr(faq_agent, "retrieve_faq_context", lambda query: pytest.fail("A revision should reuse observed evidence."))

    def answer(messages):
        assert "Correct the unsupported date" in messages[0].content
        assert chunk["content"] in messages[-1].content
        return AIMessage(content="Use the tracking email [shipping.md#chunk-001].")

    monkeypatch.setattr(faq_agent, "get_chat_model", lambda: SimpleNamespace(invoke=answer))
    evidence = [{"kind": "knowledge", "data": chunk}]
    result = faq_agent.faq_agent({
        "customer_message": "When will it arrive?", "schema_version": 2, "evidence": evidence,
        "qa_result": {"passed": False, "attempt": 1, "feedback": "Correct the unsupported date"},
    })
    assert result["evidence"] == evidence


def test_billing_qa_receives_raw_records_and_rejects_unsupported_claim(monkeypatch):
    def inspect(messages):
        assert "p-private" in messages[-1].content
        assert "amount_cents" in messages[-1].content
        assert "Previous customer question" in messages[-1].content

    reviewer = ReviewModel(qa_agent.QAReview(passed=False, issues=["Claim exceeds recorded payment."], feedback="Use recorded amounts."), inspect)
    monkeypatch.setattr(qa_agent, "get_chat_model", lambda: reviewer)
    result = qa_agent.qa_reviewer_agent({
        "customer_message": "Check it", "specialist": "billing", "reply": "You paid 9999.",
        "evidence": [PAYMENT_EVIDENCE], "conversation_history": [
            {"role": "user", "content": "Previous customer question"}, {"role": "user", "content": "Check it"},
        ],
    })
    assert result["qa_result"]["passed"] is False
    assert "Claim exceeds recorded payment" in result["qa_result"]["feedback"]


def test_deterministic_qa_issues_are_always_in_revision_feedback(monkeypatch):
    reviewer = ReviewModel(qa_agent.QAReview(passed=True, issues=[], feedback="Make the tone warmer."))
    monkeypatch.setattr(qa_agent, "get_chat_model", lambda: reviewer)
    result = qa_agent.qa_reviewer_agent({
        "customer_message": "Refund order 12345", "specialist": "billing", "evidence": [PAYMENT_EVIDENCE],
        "reply": "Your refund has been issued.", "proposed_action": PROPOSAL,
    })
    review = result["qa_result"]
    assert review["passed"] is False
    assert "Make the tone warmer" in review["feedback"]
    assert "awaiting human approval" in review["feedback"]
    assert "before human approval" in review["feedback"]


def test_malformed_qa_output_consumes_budget_and_falls_back(monkeypatch):
    reviewer = ReviewModel()
    monkeypatch.setattr(qa_agent, "get_chat_model", lambda: reviewer)
    monkeypatch.setattr(workflow, "faq_agent", lambda state: {"specialist": "faq", "reply": "Answer", "response_kind": "generated", "evidence": []})
    graph = workflow.build_graph(InMemorySaver())
    result = graph.invoke({"customer_message": "Policy?"}, {"configurable": {"thread_id": "malformed"}})
    assert reviewer.calls == 3
    assert result["qa_result"]["attempt"] == 3
    assert "could not be parsed" in result["qa_result"]["feedback"]
    assert result["response_kind"] == "handoff"
    assert "private provider details" not in str(result)


@pytest.mark.parametrize("message", ["My router is broken", "Urgent: security breach", "What is the policy?"])
def test_failed_fixed_response_uses_fallback_without_identical_retries(monkeypatch, message):
    monkeypatch.setattr(faq_agent, "retrieve_faq_context", lambda question: [])
    reviewer = ReviewModel(qa_agent.QAReview(passed=False, issues=["Could not verify."], feedback="Ask support."))
    monkeypatch.setattr(qa_agent, "get_chat_model", lambda: reviewer)
    graph = workflow.build_graph(InMemorySaver())
    result = graph.invoke({"customer_message": message}, {"configurable": {"thread_id": message}})
    assert reviewer.calls == 1
    assert result["qa_result"]["attempt"] == 1
    assert result["routing_path"][-2]["node"] == "qa_fallback"
    assert "passing your request" not in result["reply"]


def test_tool_evidence_reused_without_duplication_and_business_errors_preserved(monkeypatch):
    @tool
    def get_payments(order_id: str) -> str:
        """Read payments."""
        return json.dumps(PAYMENT_EVIDENCE["data"])

    @tool
    def propose_refund(order_id: str) -> str:
        """Propose a refund."""
        return json.dumps({"error": "No excess", "order_id": order_id})

    responses = iter([
        AIMessage(content="", tool_calls=[{"name": "get_payments", "args": {"order_id": "12345"}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[{"name": "propose_refund", "args": {"order_id": "12345"}, "id": "c2"}]),
        AIMessage(content="No excess available."),
    ])

    class Model:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            assert "p-private" in messages[1].content
            return next(responses)

    monkeypatch.setattr(tool_runner, "get_chat_model", lambda: Model())
    state = {"customer_message": "Please refund", "schema_version": 2, "evidence": [PAYMENT_EVIDENCE], "tool_calls": ["get_payments"]}
    result = tool_runner.run_tool_loop(state, specialist="billing", tools=[get_payments, propose_refund])
    assert len(result["evidence"]) == 2
    assert result["evidence"][-1]["outcome"] == "error"
    assert result["evidence"][-1]["scope"] == {"order_id": "12345"}
    assert result["tool_calls"] == ["get_payments", "get_payments", "propose_refund"]
    reset = intake({**state, **result, "intent": "billing", "proposed_action": PROPOSAL, "customer_message": "Check order 12345"})
    assert reset["tool_calls"] == reset["evidence"] == []
    assert reset["proposed_action"] is None and reset["qa_result"] is None


def test_actual_technical_tool_evidence_uses_bound_device(monkeypatch):
    responses = iter([
        AIMessage(content="", tool_calls=[{"name": "search_technical_logs", "args": {"search_term": "", "device_id": "web-client"}, "id": "c1"}]),
        AIMessage(content="Historical router records."),
    ])
    model = SimpleNamespace(bind_tools=lambda tools: SimpleNamespace(invoke=lambda messages: next(responses)))
    monkeypatch.setattr(tool_runner, "get_chat_model", lambda: model)
    result = technical_agent.technical_agent(intake({"customer_message": "My router-01 is broken"}))
    evidence = result["evidence"][0]
    assert evidence["scope"] == {"device_id": "router-01"}
    assert all(record["device_id"] == "router-01" for record in evidence["data"]["matches"])


@pytest.mark.parametrize("action", [{}, {"action": "refund"}, {**PROPOSAL, "amount_cents": True}, {**PROPOSAL, "amount_cents": -1}, {**PROPOSAL, "currency": ""}, {**PROPOSAL, "status": "issued"}])
def test_incomplete_or_invalid_actions_do_not_enter_approval(action):
    assert requires_approval({"proposed_action": action}) is False


def test_telemetry_disabled_preserves_routing_and_turn_progress(monkeypatch):
    monkeypatch.setattr(workflow, "observe_node", lambda name, node: node)
    monkeypatch.setattr(workflow, "observe_decision", lambda name, node: node)
    monkeypatch.setattr(workflow, "faq_agent", lambda state: {"specialist": "faq", "reply": "Answer", "evidence": [], "model_usage": [{"input_tokens": 5}], "steps": ["answer created"]})
    monkeypatch.setattr(workflow, "qa_reviewer_agent", lambda state: {"qa_result": {"passed": True, "attempt": 1}})
    graph = workflow.build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "no-telemetry"}}
    first = graph.invoke({"customer_message": "Policy?"}, config)
    second = graph.invoke({"customer_message": "Explain that"}, config)
    assert first["routing_path"] == second["routing_path"]
    assert first["steps"][1:] == second["steps"][1:]
    assert len(second["conversation_history"]) == 4
    assert "model_usage" not in second


def test_resumed_billing_node_and_tool_use_current_invocation_ids(monkeypatch, tmp_path):
    trace = tmp_path / "resume.jsonl"
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(trace))

    @tool
    def get_payments(order_id: str) -> str:
        """Read payment records."""
        return json.dumps(PAYMENT_EVIDENCE["data"])

    responses = iter([
        AIMessage(content="", tool_calls=[{"name": "get_payments", "args": {"order_id": "12345"}, "id": "c1"}]),
        AIMessage(content="Recorded payment found."),
    ])
    monkeypatch.setattr(tool_runner, "get_chat_model", lambda: SimpleNamespace(bind_tools=lambda tools: SimpleNamespace(invoke=lambda messages: next(responses))))
    monkeypatch.setattr(billing_agent, "BILLING_TOOLS", [get_payments])
    attempts = []

    def billing(state, config):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("temporary interruption")
        return billing_agent.billing_agent(state, config)

    monkeypatch.setattr(workflow, "billing_agent", billing)
    monkeypatch.setattr(workflow, "qa_reviewer_agent", lambda state: {"qa_result": {"passed": True, "attempt": 1}})
    graph = workflow.build_graph(InMemorySaver())
    first = {"configurable": {"thread_id": "trace", "run_id": "old-run"}}
    with pytest.raises(RuntimeError):
        graph.invoke({"customer_message": "Check payments for order 12345", **first["configurable"]}, first)
    graph.invoke(None, {"configurable": {"thread_id": "trace", "run_id": "resume-run"}})
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    tool_event = next(event for event in events if event.get("tool") == "get_payments")
    billing_event = next(event for event in events if event.get("node") == "billing" and event["status"] == "ok")
    assert tool_event["run_id"] == billing_event["run_id"] == "resume-run"
    assert billing_event["model_usage"]
    assert "p-private" not in trace.read_text()
