"""Run the support workflow from the command line."""

import argparse

from dotenv import load_dotenv

from app.graph.workflow import support_graph


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the basic support desk graph")
    parser.add_argument("message", help="Customer message to triage")
    args = parser.parse_args()

    result = support_graph.invoke({"customer_message": args.message})
    print(f"Intent: {result['intent']}")
    print(f"Route:  {result['specialist']}")
    print("Path:")
    for step in result["steps"]:
        print(f"  - {step}")
    if result.get("tool_calls"):
        print(f"Tools called: {', '.join(result['tool_calls'])}")
    print(f"\nDraft reply: {result['draft_reply']}")


if __name__ == "__main__":
    main()
