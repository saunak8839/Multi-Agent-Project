# Multi-Agent Support Desk

A graph-based customer support system that triages incoming messages, routes them to focused specialists, and drafts replies. The project is built in iterations to make agent workflows and their boundaries clear.

## Architecture

```text
Customer message
      ↓
Intake — classify intent, detect urgency, extract entities
      ↓
Conditional router
   ┌──┼───────────┐
   ↓  ↓           ↓
 FAQ Billing   Technical
       ↓           ↓
       └─── Draft ─┘
             ↓
            END
```

LangGraph passes a shared state through nodes. Intake adds the intent, urgency, and extracted entities. A conditional edge selects FAQ, Billing, or Technical. The selected specialist adds findings, and Draft turns those findings into a response.

## Iteration 2: Tool calling

Billing and Technical are now tool-using LLM specialists. The model receives descriptions and argument schemas for only its permitted tools. It can request a call; application code dispatches that request to a Python function, adds the result to the model conversation, and lets the specialist continue. A bounded loop limits tool-call rounds.

```text
Billing model ──requests──> get_order / get_payments
                                  ↓
                         Python tool function
                                  ↓
                              PostgreSQL
                                  ↓
Billing model <──tool result─────┘

Technical model ──requests──> search_technical_logs
                                      ↓
                              Python tool function
                                      ↓
                             Simulated log data
```

The Billing tools are read-only and can look up seeded orders and payment attempts. The Technical tool searches simulated log records in `data/seed/technical_logs.json`. No refund, account change, or other side-effect tool is exposed.

## Set up and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add an OpenAI API key to `.env`, then start PostgreSQL. The first database startup loads the schema and seed rows from `data/seed/001_seed.sql`.

```bash
docker compose up -d postgres
python -m app.main "I was charged twice for order 12345"
python -m app.main "My router internet connection is down"
```

Order `12345` has two successful seeded payment records to inspect. Order `12346` has one. The technical log seed includes router connectivity, login, and checkout examples. FAQ messages still use a deterministic placeholder and do not require an LLM call.

`DATABASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` are configured in `.env.example`. Billing and Technical require the model API key; Billing also requires the local PostgreSQL service.

## State and code map

The shared state carries the original message, intake classification and entities, selected specialist, findings, tool-call names, draft reply, and a step trace. Nodes return partial updates; LangGraph merges them into the state.

```text
app/
├── agents/       # Intake, specialist tool-call loop, and draft nodes
├── graph/        # State schema, routing, graph construction
├── models/       # PostgreSQL access layer
├── tools/        # Read-only billing and simulated log tools
└── main.py       # Command-line entry point
data/seed/        # PostgreSQL seed SQL and simulated technical logs
docker-compose.yml
requirements.txt
```

## Next iterations

Planned work includes FAQ retrieval (RAG), escalation to a human, quality review with a feedback loop, and approval before risky actions such as refunds. These are not implemented yet.
