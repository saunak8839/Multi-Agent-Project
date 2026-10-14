"""Shared LLM tool-calling loop used by specialist nodes."""

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.graph.state import SupportState
from app.llm.factory import chat_model_name, get_chat_model, response_text
from app.services.observability import model_usage, record_model_usage_event, record_tool_event
from app.services.workflow_compat import evidence_for, review_for, tools_for


MAX_TOOL_ROUNDS = 4


def run_tool_loop(
    state: SupportState, config: RunnableConfig | None = None, *, specialist: str, tools: list,
    instructions: str = "", context: str = "",
) -> dict:
    configurable = (config or {}).get("configurable", {})
    invocation = {**state, **{key: configurable[key] for key in ("run_id", "thread_id") if key in configurable}}
    evidence = evidence_for(state)
    model = get_chat_model().bind_tools(tools)
    messages = [
        SystemMessage(
            content=(
                f"You are the {specialist} specialist in a customer support desk. "
                "Use only the tools provided to you. Treat lookup tools as read-only. Only claim a lookup "
                "succeeded when a supplied or newly returned tool result verifies it. Do not invent records, promise "
                "unverified actions, or execute side effects. Summarize verified results and missing information. "
                + instructions
            )
        ),
        *[
            HumanMessage(content=f"Previous {entry['role']} message: {entry['content']}")
            for entry in state.get("conversation_history", [])[-9:-1]
        ],
        HumanMessage(
            content=(
                f"Customer message: {state['customer_message']}\n"
                f"Intake entities: {state.get('entities', {})}. Do not guess missing identifiers.\n"
                + context
                + f"Previous verified evidence for this turn: {evidence}\n"
                + (
                    f"QA reviewer feedback to address on this retry: {review_for(state)['feedback']}"
                    if review_for(state)["feedback"] else ""
                )
            )
        ),
    ]
    tool_by_name = {tool.name: tool for tool in tools}
    called_tools = tools_for(state)
    usage_events: list[dict] = []

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = model.invoke(messages)
            usage_events.append(model_usage(response, chat_model_name()))
            messages.append(response)
            if not response.tool_calls:
                return {
                    "specialist": specialist,
                    "reply": response_text(response),
                    "response_kind": "generated",
                    "schema_version": 2,
                    "evidence": evidence,
                    "tool_calls": called_tools,
                    "model_usage": usage_events,
                    "steps": [f"{specialist} specialist: completed after {len(called_tools)} tool call(s)"],
                }

            for tool_call in response.tool_calls:
                tool_started = time.perf_counter()
                selected_tool = tool_by_name.get(tool_call["name"])
                if selected_tool is None:
                    result = f"Tool {tool_call['name']} is not available to this specialist."
                    record_tool_event(
                        tool=tool_call["name"], run_id=invocation.get("run_id", "unknown"),
                        thread_id=invocation.get("thread_id", "unknown"), started=tool_started, status="error",
                        error=ValueError(result),
                    )
                else:
                    try:
                        result = selected_tool.invoke(tool_call["args"])
                    except Exception as exc:
                        record_tool_event(
                            tool=tool_call["name"], run_id=invocation.get("run_id", "unknown"),
                            thread_id=invocation.get("thread_id", "unknown"), started=tool_started,
                            status="error", error=exc,
                        )
                        raise
                    record_tool_event(
                        tool=tool_call["name"], run_id=invocation.get("run_id", "unknown"),
                        thread_id=invocation.get("thread_id", "unknown"), started=tool_started, status="ok",
                    )
                    called_tools.append(tool_call["name"])
                    try:
                        data = json.loads(str(result))
                    except (TypeError, ValueError):
                        data = str(result)
                    scope = {
                        key: data.get(key, tool_call["args"].get(key)) if isinstance(data, dict) else tool_call["args"].get(key)
                        for key in ("order_id", "device_id")
                        if key in tool_call["args"] or isinstance(data, dict) and key in data
                    }
                    item = {
                        "kind": "tool", "tool": tool_call["name"], "scope": scope,
                        "outcome": "error" if isinstance(data, dict) and "error" in data else "ok",
                        "data": data,
                    }
                    if item not in evidence:
                        evidence.append(item)
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        raise RuntimeError(f"{specialist} specialist exceeded the {MAX_TOOL_ROUNDS}-round tool-call limit.")
    except Exception:
        for usage in usage_events:
            record_model_usage_event(
                model=usage["model"], run_id=invocation.get("run_id", "unknown"),
                thread_id=invocation.get("thread_id", "unknown"), specialist=specialist, usage=usage,
            )
        raise
