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
   ↓     ↓          ↓
Retrieve  PostgreSQL  Simulated logs
   └──────┼──────────┘
          ↓
       Draft reply
          ↓
         END
```

LangGraph passes a shared state through nodes. Intake adds the intent, urgency, and extracted entities. A conditional edge selects FAQ, Billing, or Technical. Specialists add findings, and Draft turns those findings into a response.

## Iteration 2: Tool calling

Billing and Technical are LLM specialists with access to allowlisted Python tools. The model requests a tool call using its description and argument schema. Application code dispatches the request to the matching function, appends the result to the model conversation, and lets the specialist continue. A bounded loop limits tool-call rounds.

Billing tools read seeded orders and payment attempts from PostgreSQL. The Technical tool searches simulated records in `data/seed/technical_logs.json`. These tools are read-only; no refund or account-change action is available.

## Iteration 3: FAQ RAG

Markdown documents in `data/knowledge_base/` are split into overlapping chunks, embedded with OpenAI's `text-embedding-3-small`, and stored in PostgreSQL with pgvector. For an FAQ question, the agent embeds the query, retrieves the closest chunks by cosine distance, and gives those excerpts to the model. The model is instructed to answer only from the retrieved text. If no chunk is sufficiently relevant, the agent abstains. Answers include references in the form `[source-file.md#chunk-001]`.

The ingester replaces chunks for a document when it is rerun. Run it again after adding or editing knowledge-base Markdown files.

## Set up and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add an OpenAI API key to `.env`, then start PostgreSQL with pgvector. The first database startup creates the vector extension and seeds order and payment rows.

```bash
docker compose up -d postgres
python -m app.rag.ingestion
python -m app.main "What is the return policy?"
python -m app.main "I was charged twice for order 12345"
python -m app.main "My router internet connection is down"
```

Order `12345` has two successful seeded payment records; order `12346` has one. The knowledge base contains sample billing/refund, shipping, and technical guidance. These are demo policies for this project, not real company policies.

`DATABASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` are configured in `.env.example`. Billing and Technical require the model API key; Billing also requires PostgreSQL. FAQ requires PostgreSQL, ingested documents, and the OpenAI API key for embeddings and answer generation.

## State and code map

The shared state carries the original message, intake classification and entities, selected specialist, findings, retrieved context, tool-call names, draft reply, and a step trace. Nodes return partial updates; LangGraph merges them into the state.

```text
app/
├── agents/       # Intake, specialist, tool-call, and draft nodes
├── graph/        # State schema, routing, graph construction
├── models/       # PostgreSQL access layer
├── rag/          # Chunking, embeddings, ingestion, retrieval
├── tools/        # Read-only billing and simulated log tools
└── main.py       # Command-line entry point
data/
├── knowledge_base/ # Source Markdown used by FAQ retrieval
└── seed/          # PostgreSQL SQL and simulated technical logs
```

## Next iterations

Planned work includes quality review with a feedback loop, escalation to a human, and approval before risky actions such as refunds. These are not implemented yet.
