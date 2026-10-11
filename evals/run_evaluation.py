"""Run labeled routing and live FAQ retrieval evaluations."""

import json
import argparse
from dotenv import load_dotenv
from pathlib import Path

from app.agents.intake_agent import intake_agent
from app.graph.routing import route_to_specialist
from app.rag.retriever import retrieve_faq_context
from app.services.observability import retrieval_mrr_at_k, retrieval_recall_at_k, routing_accuracy


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--routing-only", action="store_true",
        help="Skip retrieval, which requires the database and embedding credentials",
    )
    args = parser.parse_args()
    routing_cases = json.loads((ROOT / "data/evaluation/routing_cases.json").read_text())
    predicted = [
        route_to_specialist(intake_agent({"customer_message": case["message"]}))
        for case in routing_cases
    ]
    expected = [case["expected"] for case in routing_cases]
    route_score = routing_accuracy(predicted, expected)
    routing_results = [
        {"message": case["message"], "expected": target, "predicted": actual, "correct": target == actual}
        for case, target, actual in zip(routing_cases, expected, predicted)
    ]

    if args.routing_only:
        print(json.dumps({
            "routing_accuracy": route_score,
            "routing_cases": len(routing_cases),
            "routing_results": routing_results,
        }, indent=2))
        return

    retrieval_cases = json.loads((ROOT / "data/evaluation/retrieval_cases.json").read_text())
    retrieval_scores = []
    retrieval_mrr_scores = []
    retrieval_results = []
    for case in retrieval_cases:
        retrieved = retrieve_faq_context(case["query"])
        evidence = [f"{item['source']}#chunk-{item['chunk_index']:03d}" for item in retrieved]
        expected = set(case["expected_evidence"])
        score = retrieval_recall_at_k(evidence, expected)
        mrr = retrieval_mrr_at_k(evidence, expected)
        retrieval_scores.append(score)
        retrieval_mrr_scores.append(mrr)
        retrieval_results.append({"query": case["query"], "evidence_recall": score, "mrr": mrr, "evidence": evidence})

    mean_recall = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 1.0
    mean_mrr = sum(retrieval_mrr_scores) / len(retrieval_mrr_scores) if retrieval_mrr_scores else 1.0
    print(json.dumps({
        "routing_accuracy": route_score,
        "routing_cases": len(routing_cases),
        "routing_results": routing_results,
        "retrieval_evidence_recall_at_4": mean_recall,
        "retrieval_mrr_at_4": mean_mrr,
        "retrieval_cases": retrieval_results,
    }, indent=2))


if __name__ == "__main__":
    main()
