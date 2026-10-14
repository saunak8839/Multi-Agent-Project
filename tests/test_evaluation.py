import json
from pathlib import Path

from app.graph.intake import intake as intake_agent
from app.graph.routing import route_to_specialist
from app.rag.chunking import chunk_text
from app.services.observability import retrieval_mrr_at_k, retrieval_recall_at_k, routing_accuracy


ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_labels_reference_existing_chunks():
    evidence = {
        f"{path.name}#chunk-{index:03d}"
        for path in (ROOT / "data/knowledge_base").glob("*.md")
        for index, _ in enumerate(chunk_text(path.read_text()), start=1)
    }
    cases = json.loads((ROOT / "data/evaluation/retrieval_cases.json").read_text())
    for case in cases:
        assert set(case["expected_evidence"]) <= evidence, case["query"]


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


def test_chunk_evidence_metric_rewards_early_rank_and_distinguishes_chunks():
    expected = {"billing_and_refunds.md#chunk-001"}
    assert retrieval_recall_at_k(["billing_and_refunds.md#chunk-001"], expected) == 1.0
    assert retrieval_recall_at_k(["billing_and_refunds.md#chunk-000"], expected) == 0.0
    assert retrieval_mrr_at_k(["other.md#chunk-000", "billing_and_refunds.md#chunk-001"], expected) == 0.5


def test_ambiguous_and_high_risk_messages_route_by_topic_or_escalate():
    examples = {
        "I can't see my refund yet": "billing",
        "My wifi keeps dropping": "technical",
        "URGENT: security breach on my account": "escalation",
        "I can't sign in after resetting my password": "technical",
        "My order has not shipped and I see a duplicate charge": "billing",
        "What is the return policy?": "faq",
    }
    for message, expected in examples.items():
        state = intake_agent({"customer_message": message})
        assert route_to_specialist(state) == expected
