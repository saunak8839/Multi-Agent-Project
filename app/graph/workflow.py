"""Build and compile the support desk graph."""

from langgraph.graph import END, START, StateGraph
from app.agents.action_agent import escalate_rejected_action, execute_approved_action
from app.agents.approval_agent import approval_gate
from app.agents.billing_agent import billing_agent
from app.agents.draft_agent import draft_agent
from app.agents.escalation_agent import urgent_escalation_agent
from app.agents.faq_agent import faq_agent
from app.agents.intake_agent import intake_agent
from app.agents.qa_agent import qa_fallback, qa_reviewer_agent
from app.agents.technical_agent import technical_agent
from app.agents.conversation_agent import finalize_turn
from app.graph.routing import route_after_approval, route_after_qa, route_to_specialist
from app.graph.state import SupportState
from app.services.observability import observe_decision, observe_node


def build_graph(checkpointer):
    graph = StateGraph(SupportState)
    def add_node(name, node):
        graph.add_node(name, observe_node(name, node, track_routing=True))

    add_node("intake", intake_agent)
    add_node("faq", faq_agent)
    add_node("billing", billing_agent)
    add_node("technical", technical_agent)
    add_node("escalation", urgent_escalation_agent)
    add_node("draft", draft_agent)
    add_node("qa", qa_reviewer_agent)
    add_node("qa_fallback", qa_fallback)
    add_node("approval_gate", approval_gate)
    add_node("execute_action", execute_approved_action)
    add_node("escalate", escalate_rejected_action)
    add_node("finalize_turn", finalize_turn)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        observe_decision("route_to_specialist", route_to_specialist),
        {"faq": "faq", "billing": "billing", "technical": "technical", "escalation": "escalation"},
    )
    for specialist in ("faq", "billing", "technical", "escalation"):
        graph.add_edge(specialist, "draft")
    graph.add_edge("draft", "qa")
    graph.add_conditional_edges(
        "qa",
        observe_decision("route_after_qa", route_after_qa),
        {
            "pass": "finalize_turn",
            "approval": "approval_gate",
            "faq": "faq",
            "billing": "billing",
            "technical": "technical",
            "escalation": "escalation",
            "fallback": "qa_fallback",
        },
    )
    graph.add_edge("qa_fallback", "finalize_turn")
    graph.add_conditional_edges(
        "approval_gate",
        observe_decision("route_after_approval", route_after_approval),
        {"execute": "execute_action", "escalate": "escalate"},
    )
    graph.add_edge("execute_action", "finalize_turn")
    graph.add_edge("escalate", "finalize_turn")
    graph.add_edge("finalize_turn", END)
    return graph.compile(checkpointer=checkpointer)
