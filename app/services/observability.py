"""Lightweight local tracing and run-metric helpers."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import inspect
import time
from typing import Any, Callable
from uuid import uuid4


def new_run_id() -> str:
    return uuid4().hex


def record_event(event: dict[str, Any]) -> None:
    """Append one JSONL event; failures here must not fail customer workflows."""
    trace_path = Path(os.getenv("SUPPORT_TRACE_FILE", "data/observability/traces.jsonl"))
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
    except OSError:
        # The current workflow result still carries successful node traces.
        # A missing file-system trace sink should not take down support.
        return


def model_usage(response: Any, model_name: str) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    result: dict[str, Any] = {
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    try:
        input_rate = float(os.environ["LLM_INPUT_COST_PER_1M"])
        output_rate = float(os.environ["LLM_OUTPUT_COST_PER_1M"])
    except (KeyError, ValueError):
        return result
    result["cost_usd"] = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return result


def _state_summary(state: dict[str, Any], updates: dict[str, Any] | None = None) -> dict[str, Any]:
    current = {**state, **(updates or {})}
    context = current.get("retrieved_context", [])
    qa = current.get("qa_result", {})
    approval = current.get("approval_decision", {})
    return {
        "intent": current.get("intent"),
        "specialist": current.get("specialist"),
        "tool_calls": current.get("current_tool_calls", []),
        "retrieval": {
            "count": len(context),
            "sources": [f"{item.get('source')}#chunk-{item.get('chunk_index', 0):03d}" for item in context],
            "distances": [item.get("distance") for item in context],
        },
        "qa": {"passed": qa.get("passed"), "attempt": qa.get("attempt")},
        "approval": {"approved": approval.get("approved")},
        "action_status": (current.get("action_result") or {}).get("status"),
    }


def observe_node(name: str, node: Callable) -> Callable:
    """Wrap a graph node with timing, outcome, and state-summary telemetry."""
    def wrapped(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        configurable = config.get("configurable", {})
        base = {
            "event": "node",
            "run_id": configurable.get("run_id", "unknown"),
            "thread_id": configurable.get("thread_id", "unknown"),
            "node": name,
            "started_at": started_at,
        }
        try:
            updates = node(state, config) if len(inspect.signature(node).parameters) > 1 else node(state)
        except Exception as exc:
            interrupted = exc.__class__.__name__ == "GraphInterrupt"
            event = {
                **base,
                "status": "interrupted" if interrupted else "error",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error_type": exc.__class__.__name__,
                "error": str(exc)[:500],
                "state": _state_summary(state),
            }
            record_event(event)
            raise

        usages = updates.get("model_usage", [])
        event = {
            **base,
            "status": "ok",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "state": _state_summary(state, updates),
            "model_usage": usages,
        }
        record_event(event)
        return {**updates, "node_traces": [event]}

    wrapped.__name__ = getattr(node, "__name__", name)
    return wrapped


def observe_decision(name: str, decision: Callable) -> Callable:
    def wrapped(state: dict[str, Any], config: dict[str, Any] | None = None) -> str:
        started = time.perf_counter()
        result = decision(state)
        configurable = (config or {}).get("configurable", {})
        record_event({
            "event": "decision",
            "run_id": configurable.get("run_id", state.get("run_id", "unknown")),
            "thread_id": configurable.get("thread_id", state.get("thread_id", "unknown")),
            "node": name,
            "decision": result,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return result
    return wrapped


def record_tool_event(*, tool: str, run_id: str, thread_id: str, started: float,
                      status: str, error: Exception | None = None) -> None:
    event = {
        "event": "tool",
        "run_id": run_id,
        "thread_id": thread_id,
        "tool": tool,
        "status": status,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    if error:
        event.update(error_type=error.__class__.__name__, error=str(error)[:500])
    record_event(event)


def summarize_run(events: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    selected = [event for event in events if event.get("run_id") == run_id]
    tokens_in = tokens_out = 0
    cost = 0.0
    has_cost = False
    for event in selected:
        for usage in event.get("model_usage", []):
            tokens_in += int(usage.get("input_tokens", 0))
            tokens_out += int(usage.get("output_tokens", 0))
            if "cost_usd" in usage:
                cost += float(usage["cost_usd"])
                has_cost = True
    return {
        "events": len(selected),
        "node_failures": sum(e.get("event") == "node" and e.get("status") == "error" for e in selected),
        "tool_failures": sum(e.get("event") == "tool" and e.get("status") == "error" for e in selected),
        "latency_ms": round(sum(float(e.get("latency_ms", 0)) for e in selected), 2),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cost_usd": round(cost, 8) if has_cost else None,
    }


def load_run_events(run_id: str) -> list[dict[str, Any]]:
    trace_path = Path(os.getenv("SUPPORT_TRACE_FILE", "data/observability/traces.jsonl"))
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("run_id") == run_id:
            events.append(event)
    return events


def routing_accuracy(actual: list[str], expected: list[str]) -> float:
    if not expected or len(actual) != len(expected):
        return 0.0
    return sum(a == e for a, e in zip(actual, expected)) / len(expected)


def retrieval_recall_at_k(retrieved_sources: list[str], expected_sources: set[str]) -> float:
    if not expected_sources:
        return 1.0
    return len(set(retrieved_sources) & expected_sources) / len(expected_sources)
