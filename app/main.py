"""Run or resume support conversations from the command line."""

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from app.services.conversation import open_support_graph


def _print_history(graph, config: dict) -> None:
    snapshot = graph.get_state(config)
    history = snapshot.values.get("conversation_history", []) if snapshot.values else []
    if not history:
        print("No conversation history found for that ID.")
        return
    for entry in history:
        print(f"[{entry.get('created_at', 'unknown time')}] {entry['role']}: {entry['content']}")


def _show_result(result: dict, thread_id: str) -> None:
    print(f"Conversation ID: {thread_id}")
    print(f"Intent: {result.get('intent', 'unknown')}")
    print(f"Route:  {result.get('specialist', 'unknown')}")
    print("Path:")
    for step in result.get("steps", []):
        print(f"  - {step}")
    if result.get("tool_calls"):
        print(f"Tools called: {', '.join(result['tool_calls'])}")
    if result.get("qa_result"):
        status = "passed" if result["qa_result"].get("passed") else "failed"
        print(f"QA: {status} after {result.get('qa_attempts', 0)} attempt(s)")
        if result["qa_result"].get("issues"):
            print(f"QA issues: {'; '.join(result['qa_result']['issues'])}")
    if result.get("approval_decision"):
        decision = "approved" if result["approval_decision"].get("approved") else "rejected"
        print(f"Human approval: {decision}")
    print(f"\nFinal reply: {result.get('draft_reply', '')}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run and resume support conversations")
    parser.add_argument("message", nargs="?", help="Customer message to triage")
    parser.add_argument("--thread-id", help="Resume or continue this conversation ID")
    parser.add_argument("--history", action="store_true", help="Print saved conversation history")
    parser.add_argument("--decision", choices=("approve", "reject"), help="Resolve a pending refund approval")
    parser.add_argument("--note", default="", help="Optional human approval/rejection note")
    args = parser.parse_args()

    if args.history and not args.thread_id:
        parser.error("--history requires --thread-id")
    if args.decision and not args.thread_id:
        parser.error("--decision requires --thread-id")
    if args.decision and args.message:
        parser.error("Do not pass a new message when resolving a pending approval")
    if not args.history and not args.decision and not args.message:
        parser.error("provide a customer message, --history, or --decision")

    thread_id = args.thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    with open_support_graph() as support_graph:
        if args.history:
            _print_history(support_graph, config)
            return

        if args.decision:
            snapshot = support_graph.get_state(config)
            if "approval_gate" not in snapshot.next:
                parser.error("that conversation has no pending approval to resolve")
            result = support_graph.invoke(
                Command(resume={"approved": args.decision == "approve", "note": args.note}), config
            )
        elif args.thread_id:
            snapshot = support_graph.get_state(config)
            if snapshot.next:
                parser.error(
                    "this conversation has a pending graph step; resolve its approval with --decision first"
                )
            result = support_graph.invoke({"customer_message": args.message}, config)
        else:
            result = support_graph.invoke({"customer_message": args.message}, config)

        while result.get("__interrupt__"):
            pending = result["__interrupt__"][0].value
            action = pending["action"]
            amount = int(action["amount_cents"]) / 100
            print(f"\nConversation ID: {thread_id}")
            print("Refund approval required")
            print(f"Order: {action['order_id']}")
            print(f"Amount: {action['currency']} {amount:,.2f}")
            print(f"QA-approved draft: {pending['draft_reply']}")
            print("Approval records this in the local demo ledger; it does not contact a payment processor.")
            approved = input("Approve this refund? [y/N]: ").strip().lower() in {"y", "yes"}
            note = input("Decision note (optional): ").strip()
            result = support_graph.invoke(
                Command(resume={"approved": approved, "note": note}), config
            )
        _show_result(result, thread_id)


if __name__ == "__main__":
    main()
