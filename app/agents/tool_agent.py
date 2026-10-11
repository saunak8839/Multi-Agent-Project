"""Shared LLM tool-calling loop used by specialist nodes."""

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.graph.state import SupportState


MAX_TOOL_ROUNDS = 4


def run_tool_agent(state: SupportState, *, specialist: str, tools: list) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env to use billing or technical tool calling.")

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    ).bind_tools(tools)
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
                    f"QA reviewer feedback to address on this retry: {state['qa_feedback']}"
                    if state.get("qa_feedback") else ""
                )
            )
        ),
    ]
    tool_by_name = {tool.name: tool for tool in tools}
    called_tools: list[str] = []
    proposed_action = state.get("proposed_action")

    for _ in range(MAX_TOOL_ROUNDS):
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {
                "specialist": specialist,
                "findings": str(response.content),
                "tool_calls": called_tools,
                "current_tool_calls": called_tools,
                "proposed_action": proposed_action,
                "requires_approval": bool(proposed_action),
                "steps": [f"{specialist} specialist: completed after {len(called_tools)} tool call(s)"],
            }

        for tool_call in response.tool_calls:
            selected_tool = tool_by_name.get(tool_call["name"])
            if selected_tool is None:
                result = f"Tool {tool_call['name']} is not available to this specialist."
            else:
                result = selected_tool.invoke(tool_call["args"])
                called_tools.append(tool_call["name"])
                if tool_call["name"] == "propose_refund":
                    proposal = json.loads(str(result))
                    if proposal.get("status") == "proposed":
                        proposed_action = proposal
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    raise RuntimeError(f"{specialist} specialist exceeded the {MAX_TOOL_ROUNDS}-round tool-call limit.")
