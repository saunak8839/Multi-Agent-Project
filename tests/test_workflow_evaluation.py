import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import workflow


@pytest.mark.parametrize(
    ("message", "expected_specialist"),
    [
        ("What is your return policy?", "faq"),
        ("I was charged twice for order 12345", "billing"),
        ("My router internet connection is down", "technical"),
    ],
)
def test_graph_routes_and_traces_end_to_end_without_external_services(
    monkeypatch, tmp_path, message, expected_specialist
):
    trace_file = tmp_path / "traces.jsonl"
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(trace_file))

    def specialist(state):
        return {"specialist": expected_specialist, "findings": f"Verified {expected_specialist} result."}

    def pass_qa(_state):
        return {"qa_result": {"passed": True, "issues": [], "feedback": ""}, "qa_attempts": 1}

    for name in ("faq_agent", "billing_agent", "technical_agent"):
        monkeypatch.setattr(workflow, name, specialist)
    monkeypatch.setattr(workflow, "qa_reviewer_agent", pass_qa)
    graph = workflow.build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "thread-eval", "run_id": "run-eval"}}
    result = graph.invoke({
        "customer_message": message,
        "run_id": "run-eval",
        "thread_id": "thread-eval",
    }, config)

    assert result["specialist"] == expected_specialist
    assert result["qa_result"]["passed"] is True
    assert result["conversation_history"][-1]["role"] == "assistant"
    assert result["draft_reply"].startswith("Thanks for contacting support.")
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    completed_nodes = {event["node"] for event in events if event["event"] == "node"}
    assert {"intake", expected_specialist, "draft", "qa", "finalize_turn"} <= completed_nodes
    assert any(
        event.get("event") == "decision" and event.get("decision") == expected_specialist
        for event in events
    )
