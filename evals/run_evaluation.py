"""Run labeled routing and live FAQ retrieval evaluations."""

import json
from pathlib import Path

from app.agents.intake_agent import intake_agent
from app.graph.routing import route_to_specialist
from app.rag.retriever import retrieve_faq_context
from app.services.observability import retrieval_recall_at_k, routing_accuracy


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    routing_cases = json.loads((ROOT / "data/evaluation/routing_cases.json").read_text())
    predicted = [
        route_to_specialist(intake_agent({"customer_message": case["message"]}))
        for case in routing_cases
    ]
    expected = [case["expected"] for case in routing_cases]
    route_score = routing_accuracy(predicted, expected)

    retrieval_cases = json.loads((ROOT / "data/evaluation/retrieval_cases.json").read_text())
    retrieval_scores = []
    retrieval_results = []
    for case in retrieval_cases:
        retrieved = retrieve_faq_context(case["query"])
        sources = [item["source"] for item in retrieved]
        score = retrieval_recall_at_k(sources, set(case["expected_sources"]))
        retrieval_scores.append(score)
        retrieval_results.append({"query": case["query"], "recall": score, "sources": sources})

    mean_recall = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 1.0
    print(json.dumps({
        "routing_accuracy": route_score,
        "routing_cases": len(routing_cases),
        "retrieval_recall_at_4": mean_recall,
        "retrieval_cases": retrieval_results,
    }, indent=2))
    if route_score < 1.0 or mean_recall < 1.0:
        raise SystemExit("Evaluation thresholds were not met.")


if __name__ == "__main__":
    main()
