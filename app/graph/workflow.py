"""Build and compile the Iteration 1 support graph."""

from langgraph.graph import END, START, StateGraph

from app.agents.billing_agent import billing_agent
from app.agents.draft_agent import draft_agent
from app.agents.faq_agent import faq_agent
from app.agents.intake_agent import intake_agent
from app.agents.technical_agent import technical_agent
from app.graph.routing import route_to_specialist
from app.graph.state import SupportState


def build_graph():
    graph = StateGraph(SupportState)
    graph.add_node("intake", intake_agent)
    graph.add_node("faq", faq_agent)
    graph.add_node("billing", billing_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("draft", draft_agent)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        route_to_specialist,
        {"faq": "faq", "billing": "billing", "technical": "technical"},
    )
    for specialist in ("faq", "billing", "technical"):
        graph.add_edge(specialist, "draft")
    graph.add_edge("draft", END)
    return graph.compile()


support_graph = build_graph()
