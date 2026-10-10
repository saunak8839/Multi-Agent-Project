"""Run the support workflow from the command line."""

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from app.graph.workflow import support_graph


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the support desk graph")
    parser.add_argument("message", help="Customer message to triage")
    args = parser.parse_args()

    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    result = support_graph.invoke({"customer_message": args.message}, config)
    while result.get("__interrupt__"):
        pending = result["__interrupt__"][0].value
        action = pending["action"]
        amount = int(action["amount_cents"]) / 100
        currency = action["currency"]
        print("\nRefund approval required")
        print(f"Order: {action['order_id']}")
        print(f"Amount: {currency} {amount:,.2f}")
        print(f"QA-approved draft: {pending['draft_reply']}")
        print("Approval records this in the local demo ledger; it does not contact a payment processor.")
        approved = input("Approve this refund? [y/N]: ").strip().lower() in {"y", "yes"}
        note = input("Decision note (optional): ").strip()
        result = support_graph.invoke(
            Command(resume={"approved": approved, "note": note}),
            config,
        )
    print(f"Intent: {result['intent']}")
    print(f"Route:  {result['specialist']}")
    print("Path:")
    for step in result["steps"]:
        print(f"  - {step}")
    if result.get("tool_calls"):
        print(f"Tools called: {', '.join(result['tool_calls'])}")
    if result.get("qa_result"):
        status = "passed" if result["qa_result"].get("passed") else "failed"
        print(f"QA: {status} after {result.get('qa_attempts', 0)} attempt(s)")
        if result["qa_result"].get("issues"):
            print(f"QA issues: {'; '.join(result['qa_result']['issues'])}")
    if result.get("approval_decision"):
        print(f"Human approval: {'approved' if result['approval_decision']['approved'] else 'rejected'}")
    print(f"\nFinal reply: {result['draft_reply']}")


if __name__ == "__main__":
    main()
