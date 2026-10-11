import json

import pytest
from langchain_core.tools import tool

from app.agents import tool_agent
from app.services.observability import observe_node


class FakeResponse:
    content = "verified response"
    usage_metadata = {"input_tokens": 3, "output_tokens": 2}

    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return next(self.responses)


def test_tool_agent_dispatches_only_selected_allowlisted_tool(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(trace_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    calls = []

    @tool
    def lookup(record_id: str) -> str:
        """Find a record by id."""
        calls.append(record_id)
        return "record verified"

    model = FakeModel([
        FakeResponse([{"name": "lookup", "args": {"record_id": "r-1"}, "id": "call-1", "type": "tool_call"}]),
        FakeResponse(),
    ])
    monkeypatch.setattr(tool_agent, "get_chat_model", lambda: model)
    result = tool_agent.run_tool_agent(
        {"customer_message": "look up r-1", "run_id": "tool-run", "thread_id": "tool-thread"},
        specialist="billing",
        tools=[lookup],
    )
    assert calls == ["r-1"]
    assert result["current_tool_calls"] == ["lookup"]
    assert result["findings"] == "verified response"
    assert len(result["model_usage"]) == 2


def test_tool_failure_trace_is_sanitized_and_preserves_prior_model_usage(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SUPPORT_TRACE_FILE", str(trace_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    fake_tool = type("Tool", (), {
        "name": "lookup",
        "invoke": lambda _self, _args: (_ for _ in ()).throw(ValueError("sensitive customer record r-99")),
    })()
    model = FakeModel([
        FakeResponse([{"name": "lookup", "args": {}, "id": "call-1", "type": "tool_call"}]),
    ])
    monkeypatch.setattr(tool_agent, "get_chat_model", lambda: model)
    node = observe_node(
        "billing",
        lambda state: tool_agent.run_tool_agent(state, specialist="billing", tools=[fake_tool]),
    )
    with pytest.raises(ValueError):
        node(
            {"customer_message": "sensitive customer text", "run_id": "tool-fail", "thread_id": "t-1"},
            {"configurable": {"run_id": "tool-fail", "thread_id": "t-1"}},
        )
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert any(event["event"] == "model_usage" and event["model_usage"][0]["total_tokens"] == 5 for event in events)
    assert any(event["event"] == "tool" and event["status"] == "error" for event in events)
    assert any(event["event"] == "node" and event["status"] == "error" for event in events)
    assert "r-99" not in trace_path.read_text()
    assert "sensitive customer text" not in trace_path.read_text()
