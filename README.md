# Multi-Agent Support Desk

A graph-based customer support system that triages incoming messages, routes them to focused specialists, and drafts replies. The project is built in iterations to make agent workflows and their boundaries clear.

## Architecture

```text
Customer message
      ↓
Intake — classify intent, detect urgency, extract entities
      ↓
Conditional router
   ┌──┼───────────┬──────────────┐
   ↓  ↓           ↓              ↓
 FAQ Billing   Technical   Urgent/security
   ↓     ↓          ↓        escalation
Retrieve  PostgreSQL  Simulated logs
   └──────┼──────────┴──────────────┘
          ↓
       Draft reply
          ↓
       QA reviewer
       ↙       ↘
     FAIL      PASS
      ↓          ↓
  Specialist   Risky action?
      (retry)    ↙       ↘
              yes         no
               ↓           ↓
          Approval gate   END
           ↙       ↘
       Approve    Reject
          ↓          ↓
        Execute   Escalate
          ↓          ↓
          END        END
```

LangGraph passes a shared state through nodes. Intake adds the intent, urgency, and extracted entities. A conditional edge selects FAQ, Billing, or Technical. Specialists add findings, Draft turns those findings into a response, and QA decides whether to accept it or return feedback to the same specialist.

## Iteration 2: Tool calling

Billing and Technical are LLM specialists with access to allowlisted Python tools. The model requests a tool call using its description and argument schema. Application code dispatches the request to the matching function, appends the result to the model conversation, and lets the specialist continue. A bounded loop limits tool-call rounds.

Billing tools read seeded orders and payment attempts from PostgreSQL. A verified duplicate charge can produce a refund proposal, but proposing does not issue money. The Technical tool searches simulated records in `data/seed/technical_logs.json`.

## Iteration 3: FAQ RAG

Markdown documents in `data/knowledge_base/` are split into overlapping chunks, embedded with OpenAI's `text-embedding-3-small`, and stored in PostgreSQL with pgvector. For an FAQ question, the agent embeds the query, retrieves the closest chunks by cosine distance, and gives those excerpts to the model. The model is instructed to answer only from the retrieved text. If no chunk is sufficiently relevant, the agent abstains. Answers include references in the form `[source-file.md#chunk-001]`.

The ingester replaces chunks for a document when it is rerun. Run it again after adding or editing knowledge-base Markdown files.

## Iteration 4: QA and feedback loops

After each draft, a structured QA reviewer checks grounding, factual consistency with specialist evidence, completeness, tone, and FAQ citations. A passing response goes to the end of the graph. A failed response sends actionable feedback to the same specialist, which runs again before a fresh draft and review.

The workflow allows three QA attempts (the first review plus up to two retries). If the draft still fails, the graph replaces it with a safe handoff response rather than returning a repeatedly rejected draft. The command-line output shows the review status, issues, and attempt count.

## Iteration 5: Approval gate

If Billing proposes a refund, QA must pass before the graph interrupts and shows the order, amount, and reviewed draft. The CLI asks a human to approve or reject. Approval rechecks the duplicate-payment capacity and records the refund in the local demo refund ledger. Rejection records an escalation summary and executes no refund.

## Iteration 6: Persistence and conversation history

The graph uses LangGraph's PostgreSQL checkpointer. Each conversation has a `thread_id`; graph state, pending interrupts, and the user/assistant transcript are saved in PostgreSQL. Reusing the ID continues the same conversation, while `--history` retrieves its transcript. A pending refund approval can be resumed from a new process with `--decision approve` or `--decision reject`. Recent turns are supplied to specialists as context, while FAQ answers remain grounded in retrieved knowledge-base chunks. The refund ledger is simulated and does not contact a payment processor.

## Iteration 7: Evaluation and observability

Each graph node, routing decision, and tool call appends a JSON event to `data/observability/traces.jsonl` (or `SUPPORT_TRACE_FILE`). Events include run/thread IDs, status, elapsed milliseconds, redacted failure details, selected route, QA outcome, retrieval chunks/distances, tool names, and provider-reported token usage. Customer message and generated reply text are excluded from the trace log. Traces stay outside checkpointed conversation state so they do not grow with each follow-up. The CLI reports accumulated component time separately from the current process wall time: component durations overlap when tools run inside nodes. Set `LLM_INPUT_COST_PER_1M` and `LLM_OUTPUT_COST_PER_1M` to the provider's current USD rates per million tokens to enable cost estimates.

The offline tests cover labeled routing, specialist-to-QA paths, approval interruption and resumption, QA retry bounds, tool dispatch, refund idempotency, and trace/error/token accounting without OpenAI or PostgreSQL. The deterministic data evaluation measures route accuracy on a 22-case labeled set. The live retrieval evaluation measures chunk-evidence recall and reciprocal rank (MRR) at four against the ingested knowledge base. Chunk-level labels distinguish retrieval quality even though the corpus has only three source documents:

```bash
pytest
python -m evals.run_evaluation --routing-only
python -m evals.run_evaluation
```

The routing score only measures this rule-based classifier against its labeled examples; it does not establish performance on arbitrary customer messages. Retrieval recall measures whether expected evidence chunks occur in the first four results, while MRR rewards a relevant chunk appearing earlier. Evaluation examples are in `data/evaluation/`; add representative and ambiguous cases as routing or corpus behavior changes. The retrieval evaluation requires PostgreSQL, ingested FAQ chunks, and OpenAI embedding credentials. The workflow tests mock model judgments and data access, so they verify application control flow and do not measure real model QA or tool-choice quality.

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
python -m app.main "I was charged twice for order 12345. Please refund one payment."
python -m app.main "My router internet connection is down"
```

The CLI prints each conversation ID. Save it to continue that thread or inspect its transcript:

```bash
python -m app.main --thread-id <conversation-id> "Can you explain that?"
python -m app.main --thread-id <conversation-id> --history
python -m app.main --thread-id <conversation-id> --decision approve --note "Verified by support"
```

If a refund approval is interrupted, the ID printed before the prompt can be used to resolve it after restarting. The same `DATABASE_URL` must be available, and PostgreSQL must remain running. Checkpointing uses LangGraph's Postgres saver tables, initialized automatically on startup.

Order `12345` has two successful seeded payment records; order `12346` has one. The knowledge base contains sample billing/refund, shipping, and technical guidance. These are demo policies for this project, not real company policies.

`DATABASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `LANGGRAPH_STRICT_MSGPACK`, and optional model cost rates are configured in `.env.example`. All routes use the OpenAI API for QA review; Billing and Technical also use it for specialist work. Billing and durable checkpoints require PostgreSQL, and FAQ requires PostgreSQL plus ingested documents for retrieval and embeddings.

## State and code map

The shared state carries the current message, intake classification and entities, selected specialist, findings, retrieved context, tool-call names, proposed action, QA results, approval decision, draft reply, conversation transcript, and a step trace. LangGraph checkpoints that state under the conversation thread ID. Nodes return partial updates; LangGraph merges them into the state.

```text
app/
├── agents/       # Intake, specialists, draft, QA, approval, action, and transcript nodes
├── graph/        # State schema, routing, graph construction
├── models/       # PostgreSQL access layer
├── rag/          # Chunking, embeddings, ingestion, retrieval
├── tools/        # Billing lookups/proposals and simulated log tools
├── services/     # Durable graph/checkpointer lifecycle
└── main.py       # Command-line entry point
data/
├── knowledge_base/ # Source Markdown used by FAQ retrieval
└── seed/          # PostgreSQL SQL and simulated technical logs
```

## Next iteration

Further work can connect escalation to a human support system and replace the demo refund ledger with an authorized payment provider integration.

## Provider configuration

Agents obtain LangChain models from `app/llm/factory.py`; provider configuration is centralized there. Existing `OPENAI_API_KEY` and `OPENAI_MODEL` settings still work. Chat and embeddings can use separate providers and keys. Tests can inject fake models by replacing an agent module's `get_chat_model` factory.

To use an OpenAI-compatible chat endpoint, set `CHAT_PROVIDER=openai_compatible`, `CHAT_MODEL`, `CHAT_API_KEY`, and `CHAT_BASE_URL` in `.env`. The endpoint and model must support tool calling and structured output, including LangChain's `include_raw=True` response. Compatibility must be verified with the chosen provider. Native providers with different APIs need an adapter in the factory; agents do not need provider configuration changes.

Embeddings use `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, and `EMBEDDING_BASE_URL` independently. The compatible adapter sends text inputs rather than OpenAI token IDs. Set `EMBEDDING_DIMENSIONS` to the model's actual output size; this setting validates output and defines the database column, rather than requesting reduced dimensions. Re-ingest documents after switching embedding providers or models. When dimensions change, migrate or recreate the `faq_chunks` table first; ingestion does not automatically alter existing columns. Compatible provider rows use a provider/endpoint/model identity so retrieval does not mix them with existing OpenAI vectors.

The evaluation command loads `.env` automatically, like the CLI and ingestion command.
