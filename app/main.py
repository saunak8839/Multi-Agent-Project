"""Run or resume support conversations from the command line."""

import argparse
import os
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from app.services.conversation import open_support_graph
from app.services.observability import load_run_events, new_run_id, summarize_run


def _print_history(graph, config: dict) -> None:
    snapshot = graph.get_state(config)
    history = snapshot.values.get("conversation_history", []) if snapshot.values else []
    if not history:
        print("No conversation history found for that ID.")
        return
    for entry in history:
        print(f"[{entry.get('created_at', 'unknown time')}] {entry['role']}: {entry['content']}")


def _show_result(result: dict, thread_id: str, run_id: str) -> None:
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
    report = summarize_run(load_run_events(run_id), run_id)
    print(
        f"Trace: {report['events']} events, {report['latency_ms']:.1f} ms summed node/tool/decision time, "
        f"{report['input_tokens']} input + {report['output_tokens']} output tokens"
    )
    if report["cost_usd"] is not None:
        print(f"Estimated model cost: ${report['cost_usd']:.6f}")
    if report["node_failures"] or report["tool_failures"]:
        print(f"Failures: {report['node_failures']} node, {report['tool_failures']} tool")
    print(f"Trace file: {os.getenv('SUPPORT_TRACE_FILE', 'data/observability/traces.jsonl')}")


def _invoke(graph, command, config: dict, run_id: str):
    try:
        return graph.invoke(command, config)
    except Exception:
        failures = [
            event for event in load_run_events(run_id)
            if event.get("status") == "error"
        ]
        for event in failures:
            location = event.get("node") or event.get("tool") or "unknown component"
            print(f"Trace failure at {location}: {event.get('error_type')}: {event.get('error')}")
        print(f"Inspect full trace at {os.getenv('SUPPORT_TRACE_FILE', 'data/observability/traces.jsonl')}")
        raise


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
    run_id = new_run_id()
    config = {"configurable": {"thread_id": thread_id, "run_id": run_id}}
    with open_support_graph() as support_graph:
        if args.history:
            _print_history(support_graph, config)
            return

        if args.decision:
            snapshot = support_graph.get_state(config)
            if "approval_gate" not in snapshot.next:
                parser.error("that conversation has no pending approval to resolve")
            result = _invoke(
                support_graph,
                Command(resume={"approved": args.decision == "approve", "note": args.note}),
                config,
                run_id,
            )
        elif args.thread_id:
            snapshot = support_graph.get_state(config)
            if snapshot.next:
                parser.error(
                    "this conversation has a pending graph step; resolve its approval with --decision first"
                )
            result = _invoke(
                support_graph,
                {"customer_message": args.message, "run_id": run_id, "thread_id": thread_id},
                config,
                run_id,
            )
        else:
            result = _invoke(
                support_graph,
                {"customer_message": args.message, "run_id": run_id, "thread_id": thread_id},
                config,
                run_id,
            )

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
            result = _invoke(
                support_graph, Command(resume={"approved": approved, "note": note}), config, run_id
            )
        _show_result(result, thread_id, run_id)


if __name__ == "__main__":
    main()
