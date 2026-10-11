import json

import pytest

from app.services.observability import (
    model_usage,
    observe_node,
    record_tool_event,
    summarize_run,
)


def test_node_trace_records_latency_and_safe_state_summary(monkeypatch, tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(trace_file))
    node = observe_node("example", lambda state: {"specialist": "faq", "findings": "private reply"})
    updates = node(
        {"customer_message": "private customer message", "intent": "faq"},
        {"configurable": {"run_id": "run-1", "thread_id": "thread-1"}},
    )
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    assert updates["node_traces"][0]["node"] == "example"
    assert events[0]["status"] == "ok"
    assert events[0]["latency_ms"] >= 0
    assert "private customer message" not in trace_file.read_text()
    assert "private reply" not in trace_file.read_text()


def test_node_error_trace_identifies_failing_node(monkeypatch, tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(trace_file))

    def fail(_state):
        raise RuntimeError("database unavailable")

    node = observe_node("billing", fail)
    with pytest.raises(RuntimeError):
        node({}, {"configurable": {"run_id": "run-2", "thread_id": "thread-2"}})
    event = json.loads(trace_file.read_text())
    assert (event["node"], event["status"], event["error"]) == (
        "billing", "error", "database unavailable"
    )


def test_tool_failure_and_token_cost_are_counted(monkeypatch, tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(trace_file))
    monkeypatch.setenv("LLM_INPUT_COST_PER_1M", "1")
    monkeypatch.setenv("LLM_OUTPUT_COST_PER_1M", "2")
    usage = model_usage(
        type("Response", (), {"usage_metadata": {"input_tokens": 100, "output_tokens": 50}})(),
        "sample-model",
    )
    record_tool_event(
        tool="lookup", run_id="run-3", thread_id="thread-3", started=0,
        status="error", error=ValueError("missing record"),
    )
    report = summarize_run([{
        "event": "node", "run_id": "run-3", "latency_ms": 12, "model_usage": [usage]
    }, json.loads(trace_file.read_text())], "run-3")
    assert usage["cost_usd"] == 0.0002
    assert report["input_tokens"] == 100
    assert report["output_tokens"] == 50
    assert report["tool_failures"] == 1
    assert report["latency_ms"] >= 12
