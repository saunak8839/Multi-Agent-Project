import json
from pathlib import Path

from app.agents.intake_agent import intake_agent
from app.graph.routing import route_to_specialist
from app.services.observability import retrieval_recall_at_k, routing_accuracy


ROOT = Path(__file__).resolve().parents[1]


def test_labeled_routing_cases_meet_expected_accuracy():
    cases = json.loads((ROOT / "data/evaluation/routing_cases.json").read_text())
    actual = [
        route_to_specialist(intake_agent({"customer_message": case["message"]}))
        for case in cases
    ]
    expected = [case["expected"] for case in cases]
    assert routing_accuracy(actual, expected) == 1.0


def test_retrieval_recall_at_k_handles_expected_source_set():
    assert retrieval_recall_at_k(
        ["billing_and_refunds.md", "orders_and_shipping.md"],
        {"billing_and_refunds.md"},
    ) == 1.0
    assert retrieval_recall_at_k(["orders_and_shipping.md"], {"billing_and_refunds.md"}) == 0.0


def test_empty_retrieval_gold_set_is_not_penalized():
    assert retrieval_recall_at_k([], set()) == 1.0
