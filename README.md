# Multi-Agent Support Desk

A graph-based customer support system that triages incoming messages, sends them to a focused specialist, and drafts a response. The project is being built in iterations to make the workflow architecture easy to understand and extend.

## Architecture

```text
Customer message
      ↓
Intake agent — classify intent, detect urgency, extract entities
      ↓
Conditional router
   ┌──┼───────────┐
   ↓  ↓           ↓
 FAQ Billing   Technical
   └──┼───────────┘
      ↓
Draft reply
      ↓
     END
```

The workflow is implemented with LangGraph. Each node reads the shared workflow state and returns updates. A conditional edge uses the intent identified by intake to select the specialist. Specialist output is passed to the draft node, which produces a customer-facing reply.

The broader project design can grow to include an FAQ knowledge base (RAG), controlled billing and diagnostic tools, escalation to a human, quality review with a feedback loop, and approval before risky actions such as refunds. Those capabilities are planned; they are not part of the current basic graph.

## Current implementation

Iteration 1 provides deterministic keyword-based triage and FAQ, billing, and technical specialist placeholders. It runs without model credentials or external services. Billing and technical nodes do not access real customer systems, and the FAQ node does not yet retrieve documents.

The shared state currently carries:

- The original customer message
- Classified intent, urgency, and extracted entities
- Selected specialist and its findings
- Draft reply and a list of workflow steps

## Project structure

```text
app/
├── agents/       # Intake, specialist, and draft nodes
├── graph/        # State schema, routing, graph construction
└── main.py       # Command-line entry point
data/             # Knowledge base and seed data (future use)
scripts/          # Utility scripts (future use)
tests/            # Test suite (future use)
requirements.txt  # Python dependencies
```

## Run the current workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main "I was charged twice for order 12345"
```

Try a general question or a technical issue to see the conditional router choose a different specialist.
