"""Shared LLM tool-calling loop used by specialist nodes."""

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from app.llm.factory import chat_model_name, get_chat_model, response_text

from app.graph.state import SupportState
from app.services.observability import model_usage, record_model_usage_event, record_tool_event


MAX_TOOL_ROUNDS = 4


def run_tool_agent(state: SupportState, *, specialist: str, tools: list) -> dict:
    model = get_chat_model().bind_tools(tools)
    messages = [
        SystemMessage(
            content=(
                f"You are the {specialist} specialist in a customer support desk. "
                "Use only the tools provided to you. Treat lookup tools as read-only. Never claim a lookup "
                "succeeded unless you called the tool and saw its result. Do not invent records, promise "
                "refunds, or execute side effects. Summarize verified results and missing information. "
                + (
                    "Only if the customer requests a refund and tool results verify an excess successful "
                    "payment, use propose_refund for the verified excess in cents. This records a proposal "
                    "for human approval and does not issue money. State clearly that it is awaiting approval. "
                    if specialist == "billing" else ""
                )
                + (
                    "Technical logs are historical demo records, not current service status. "
                    "The log tool is restricted to the device ID supplied by the customer. "
                    "Describe results as records for that device ID, not proof of customer ownership. "
                    "Do not claim the current issue is diagnosed or resolved from historical logs. "
                    "If there are no matching records, say so; do not infer another device or diagnosis. "
                    if specialist == "technical" else ""
                )
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
                + (
                    f"Previous lookups for this turn: {state.get('technical_context', {})}\n"
                    if specialist == "technical" else ""
                )
                + (
                    f"QA reviewer feedback to address on this retry: {state['qa_feedback']}"
                    if state.get("qa_feedback") else ""
                )
            )
        ),
    ]
    tool_by_name = {tool.name: tool for tool in tools}
    called_tools: list[str] = []
    usage_events: list[dict] = []
    proposed_action = state.get("proposed_action")

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = model.invoke(messages)
            usage_events.append(model_usage(response, chat_model_name()))
            messages.append(response)
            if not response.tool_calls:
                return {
                    "specialist": specialist,
                    "findings": response_text(response),
                    "tool_calls": called_tools,
                    "current_tool_calls": called_tools,
                    "model_usage": usage_events,
                    "proposed_action": proposed_action,
                    "requires_approval": bool(proposed_action),
                    "steps": [f"{specialist} specialist: completed after {len(called_tools)} tool call(s)"],
                }

            for tool_call in response.tool_calls:
                tool_started = time.perf_counter()
                selected_tool = tool_by_name.get(tool_call["name"])
                if selected_tool is None:
                    result = f"Tool {tool_call['name']} is not available to this specialist."
                    record_tool_event(
                        tool=tool_call["name"], run_id=state.get("run_id", "unknown"),
                        thread_id=state.get("thread_id", "unknown"), started=tool_started, status="error",
                        error=ValueError(result),
                    )
                else:
                    try:
                        result = selected_tool.invoke(tool_call["args"])
                    except Exception as exc:
                        record_tool_event(
                            tool=tool_call["name"], run_id=state.get("run_id", "unknown"),
                            thread_id=state.get("thread_id", "unknown"), started=tool_started,
                            status="error", error=exc,
                        )
                        raise
                    record_tool_event(
                        tool=tool_call["name"], run_id=state.get("run_id", "unknown"),
                        thread_id=state.get("thread_id", "unknown"), started=tool_started, status="ok",
                    )
                    called_tools.append(tool_call["name"])
                    if tool_call["name"] == "propose_refund":
                        proposal = json.loads(str(result))
                        if proposal.get("status") == "proposed":
                            proposed_action = proposal
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        raise RuntimeError(f"{specialist} specialist exceeded the {MAX_TOOL_ROUNDS}-round tool-call limit.")
    except Exception:
        for usage in usage_events:
            record_model_usage_event(
                model=usage["model"], run_id=state.get("run_id", "unknown"),
                thread_id=state.get("thread_id", "unknown"), specialist=specialist, usage=usage,
            )
        raise
