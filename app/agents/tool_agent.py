"""Shared LLM tool-calling loop used by specialist nodes."""

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
                "Use the provided read-only tools when they can verify facts. Never claim a lookup succeeded "
                "unless you called the tool and saw its result. Do not invent records, promise refunds, "
                "or perform actions. Summarize the verified results and any missing information concisely."
            )
        ),
        HumanMessage(
            content=(
                f"Customer message: {state['customer_message']}\n"
                f"Intake entities: {state.get('entities', {})}. Do not guess missing identifiers."
            )
        ),
    ]
    tool_by_name = {tool.name: tool for tool in tools}
    called_tools: list[str] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {
                "specialist": specialist,
                "findings": str(response.content),
                "tool_calls": called_tools,
                "steps": [f"{specialist} specialist: completed after {len(called_tools)} tool call(s)"],
            }

        for tool_call in response.tool_calls:
            selected_tool = tool_by_name.get(tool_call["name"])
            if selected_tool is None:
                result = f"Tool {tool_call['name']} is not available to this specialist."
            else:
                result = selected_tool.invoke(tool_call["args"])
                called_tools.append(tool_call["name"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    raise RuntimeError(f"{specialist} specialist exceeded the {MAX_TOOL_ROUNDS}-round tool-call limit.")
