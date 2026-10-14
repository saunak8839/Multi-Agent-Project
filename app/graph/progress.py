"""Application progress and invocation context, independent of telemetry."""

import inspect
from collections.abc import Callable

from langchain_core.runnables import RunnableConfig


def workflow_node(name: str, node: Callable) -> Callable:
    accepts_config = len(inspect.signature(node).parameters) > 1

    def wrapped(state: dict, config: RunnableConfig) -> dict:
        configurable = config.get("configurable", {})
        invocation = {**state, **{key: configurable[key] for key in ("run_id", "thread_id") if key in configurable}}
        updates = node(invocation, config) if accepts_config else node(invocation)
        path = [] if name == "intake" else list(state.get("routing_path", []))
        status = "completed"
        if name == "qa":
            status = "passed" if (updates.get("qa_result") or {}).get("passed") else "failed"
        elif name == "approval_gate":
            status = "approved" if updates["approval_decision"]["approved"] else "rejected"
        steps = [] if name == "intake" else list(state.get("steps", []))
        if state.get("schema_version", 1) < 2 and name != "intake":
            # Old checkpoints kept lifetime CLI steps. Retain the latest turn only.
            last_intake = next((i for i in range(len(steps) - 1, -1, -1) if steps[i].startswith("intake:")), 0)
            steps = steps[last_intake:]
        return {
            **{key: value for key, value in updates.items() if key != "model_usage"},
            "routing_path": [*path, {"node": name, "status": status}],
            "steps": [*steps, *updates.get("steps", [])],
        }

    return wrapped
