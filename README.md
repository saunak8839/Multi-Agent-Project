# Multi-Agent Support Desk

A graph-based customer support system that triages incoming messages, routes them to focused specialists, and drafts replies. The project is built in iterations to make agent workflows and their boundaries clear.

## Architecture

```text
Customer message -> Intake -> FAQ / Billing / Technical -> QA -> Save reply
                       |                                  |
                       +-> Urgent guidance ---------------+
                                                          +-> Revision (up to 3 reviews)
                                                          +-> Safe fallback -> Save reply
                                                          +-> Human approval
                                                               | Approve -> Execute -> Save reply
                                                               | Reject  -> Rejection reply -> Save reply
```

The graph contains both model-based agents and deterministic workflow operations. A graph node is a step, not necessarily an agent. Specialists produce a customer-ready `reply`, structured `evidence`, and an optional `proposed_action`. QA checks the reply against observed evidence; finalization appends the completed reply to conversation history. There is no separate draft stage or automatic greeting prefix.

| Component | Responsibility |
| --- | --- |
| Intake | Deterministic classification, entity extraction, and current-turn initialization |
| FAQ agent | Retrieve knowledge passages and answer from those passages |
| Billing agent | Select billing tools, interpret results, and handle refund proposals |
| Technical agent | Investigate historical logs scoped to the supplied device ID |
| QA reviewer | Review the reply and raw evidence; provide complete revision feedback |
| Approval gate | Pause durably for an explicit human decision |
| Action executor | Recheck and record an approved demo refund |
| Handoff/fallback | Provide accurate guidance when human review is needed |
| Finalization | Save one completed assistant response |

Deterministic operations live in `app/graph/intake.py` and `app/graph/steps.py`. The shared model/tool loop lives in `app/services/tool_runner.py`; specialist instructions and proposal interpretation stay in the specialist modules. Database operations remain in `app/tools/`.

## Evidence, tools, and knowledge

Billing tools read seeded orders/payment attempts from PostgreSQL. They can propose a verified duplicate-charge refund for human approval; proposing does not issue money. Tool results are stored as tagged evidence with the tool name, effective scope, outcome, and returned data. QA receives actual payment/order results rather than only the billing model's summary. Business-error results remain distinguishable from successful lookups.

Technical lookups search `data/seed/technical_logs.json`. Without a device ID, the technical agent requests clarification before any model or log lookup. The application binds the tool to the supplied ID; model arguments cannot override it. Historical records do not establish ownership or current service status. A production service must bind lookup scope to the authenticated customer's authorized devices and orders.

Only FAQ currently retrieves knowledge-base articles. Markdown in `data/knowledge_base/` is chunked, embedded, and stored with pgvector. FAQ retrieves passages, drops those beyond the configured-in-code distance threshold, and abstains when none are sufficiently relevant. Answers currently include internal `[source-file.md#chunk-001]` citations. Re-run ingestion after changing articles. QA revisions reuse the observed passages for the turn.

## Review, approval, and persistence

Generated replies can be revised up to three total QA attempts. Reviewer feedback includes both model issues and application checks. Malformed structured QA output consumes a review attempt; provider failures retain the interrupted-execution recovery path. Fixed clarification, abstention, and urgent-guidance responses fall back immediately if rejected rather than repeating identical work. Fallback indicates that human review is needed; it does not claim an external support ticket was created.

A valid refund proposal goes to `approval_gate` only after QA passes. Approval rechecks duplicate-payment capacity under an order lock and records the refund in the local demo ledger. Proposal IDs make execution idempotent. Rejection records a response and executes no refund. No payment processor is contacted. Business schema migrations and independent refund-history features remain separate follow-up work; refund table creation still occurs in the billing tools.

PostgreSQL checkpoints retain conversation history, current-turn data, and pending steps under a `thread_id`. Intake reads continuation context before resetting the reply, evidence, proposal, review, results, tools, and progress for the new turn. Evidence survives revisions within a turn. Public API names remain compatible, including mapping internal `reply` to `pending_approval.draft_reply`.

Pre-refactor checkpoints are supported through temporary legacy-field reads and a compatibility-only `draft` node. No new turn routes through it. A legacy billing QA task missing raw evidence requests a fresh investigation. Retire this compatibility layer only after old pending turns are handled; do not clear the database to upgrade. Implementation and migration details are in [docs/implementation-plan.md](docs/implementation-plan.md).

## Evaluation and observability

Node, decision, and tool telemetry is written as JSONL to `data/observability/traces.jsonl` or `SUPPORT_TRACE_FILE`. Customer messages, replies, and raw lookup records are excluded. Routing progress is maintained independently of telemetry. Resumed tool calls use the current invocation's run/thread IDs. Model usage stays outside durable graph state. CLI reports overlapping accumulated component time separately from process wall time; optional `LLM_INPUT_COST_PER_1M` and `LLM_OUTPUT_COST_PER_1M` enable cost estimates.

```bash
python -m pytest -q
python -m evals.run_evaluation --routing-only
python -m evals.run_evaluation
```

Offline tests verify control flow, evidence delivery, review bounds, approval/resumption, compatibility, and tracing with fake providers. Routing evaluation measures the 22 labeled examples; it does not establish general classifier quality. Live retrieval evaluation requires PostgreSQL and embedding credentials and reports evidence recall/MRR at four before FAQ distance filtering.

To also run PostgreSQL migration, restart/replay, and concurrent-refund checks, point `SUPPORT_TEST_DATABASE_URL` at a disposable test database. Each test creates and removes a uniquely named schema; the configured test user needs schema-creation permission. Never use this variable as a substitute for an intentional test database.

```bash
SUPPORT_TEST_DATABASE_URL=postgresql://test_user@localhost/test_db python -m pytest -q
```

The same frozen pre-refactor fixtures run against both memory and PostgreSQL savers. PostgreSQL checks are skipped when the test URL is absent. Fake review judgments do not establish live model QA quality.

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

`DATABASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `LANGGRAPH_STRICT_MSGPACK`, and optional model cost rates are configured in `.env.example`. All routes use the configured chat provider for QA review; Billing and Technical also use it for specialist work. Billing and durable checkpoints require PostgreSQL, and FAQ requires PostgreSQL plus ingested documents for retrieval and embeddings.

## State and code map

The flat state contract in `app/graph/state.py` uses typed records for evidence, proposals, review results, decisions, and outcomes. Conversation history appends; turn-local tool and readable-step lists are explicitly replaced/reset. Each producer returns local progress entries; the graph progress wrapper combines them for the current turn.

```text
app/
├── agents/       # FAQ, billing, technical, and QA model-based components
├── graph/        # Intake, deterministic steps, typed state, routing, and assembly
├── rag/          # Chunking, embeddings, ingestion, retrieval
├── tools/        # Billing database operations and scoped simulated log lookup
├── models/       # PostgreSQL access layer
├── services/     # Shared tool runner, persistence, telemetry, legacy compatibility
├── web/          # Buildless browser interface
├── api.py        # HTTP boundary and compatible public response models
└── main.py       # CLI entry point
data/
├── knowledge_base/
└── seed/
```

## Next iteration

Further work can connect escalation to a human support system and replace the demo refund ledger with an authorized payment provider integration.

## Provider configuration

Agents obtain LangChain models from `app/llm/factory.py`; provider configuration is centralized there. Existing `OPENAI_API_KEY` and `OPENAI_MODEL` settings still work. Chat and embeddings can use separate providers and keys. Tests can inject fake models by replacing an agent module's `get_chat_model` factory.

To use an OpenAI-compatible chat endpoint, set `CHAT_PROVIDER=openai_compatible`, `CHAT_MODEL`, `CHAT_API_KEY`, and `CHAT_BASE_URL` in `.env`. The endpoint and model must support tool calling and structured output, including LangChain's `include_raw=True` response. Compatibility must be verified with the chosen provider. Native providers with different APIs need an adapter in the factory; agents do not need provider configuration changes.

Embeddings use `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, and `EMBEDDING_BASE_URL` independently. The compatible adapter sends text inputs rather than OpenAI token IDs. Set `EMBEDDING_DIMENSIONS` to the model's actual output size; this setting validates output and defines the database column, rather than requesting reduced dimensions. Re-ingest documents after switching embedding providers or models. When dimensions change, migrate or recreate the `faq_chunks` table first; ingestion does not automatically alter existing columns. Compatible provider rows use a provider/endpoint/model identity so retrieval does not mix them with existing OpenAI vectors.

The evaluation command loads `.env` automatically, like the CLI and ingestion command.

For embedding endpoints that support the OpenAI `dimensions` parameter, set `EMBEDDING_OUTPUT_DIMENSIONS` to request a specific output size (for example, `1536`). Keep `EMBEDDING_DIMENSIONS` equal to that size. Leave the output-size request blank for endpoints that do not support it; the schema setting alone does not change API output.

For Gemini chat, prefer `CHAT_PROVIDER=google_genai` with `CHAT_MODEL` and `CHAT_API_KEY` set to your Google model ID and API key. This uses the native `langchain-google-genai` adapter and preserves thought signatures across tool calls, which Gemini 3 models require. `CHAT_BASE_URL` is unused by this native adapter. Embeddings can continue to use the OpenAI-compatible Google endpoint independently. Model availability depends on the Google project; use an available model rather than assuming older model IDs are still enabled.

## Local API and browser interface

The FastAPI server serves both the HTTP API and a buildless HTML/CSS/JavaScript frontend. No Node build is needed. After installing `requirements.txt`, starting PostgreSQL, and ingesting the knowledge base:

```bash
source .venv/bin/activate
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** for chat, **http://127.0.0.1:8000/docs** for interactive API documentation, or `/openapi.json` for the schema. The health endpoint only checks the server process; database and model errors surface when a conversation is used.

The browser supports new conversations, follow-ups, reopening saved threads, a separate support review panel, and resuming interrupted turns. Conversation IDs and short titles are remembered in local browser storage; transcripts remain in PostgreSQL. You can paste an existing CLI conversation UUID using **Open with conversation ID**, or bookmark a browser conversation's URL. Replies are displayed as plain text so model content cannot execute as HTML.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/conversations` | Start a conversation with `{"message": "..."}` |
| `GET /api/conversations/{id}` | Get transcript, status, QA summary, and pending proposal |
| `POST /api/conversations/{id}/messages` | Send a follow-up with `{"message": "..."}` |
| `POST /api/conversations/{id}/retry` | Resume the checkpointed unfinished turn without adding another user message |
| `POST /api/conversations/{id}/approval` | Resolve a proposal with `{"approved": true, "note": "..."}` |
| `GET /api/health` | Check process health |

Approval and rejection require a separate `SUPPORT_APPROVAL_KEY` in `.env` and an `Authorization: Bearer <key>` header. Use the **Support view** button to review the proposal and enter that key. The UI never stores the key in local storage. If the environment setting is blank, decisions are disabled. Generate a separate random secret; do not reuse a model-provider key. Approving records the refund in the demo database ledger, not a payment processor.

HTTP `409` means a conversation is busy or has a pending step that must be resolved first. A provider rate limit returns `429` with `Retry-After: 60`. Dependency failures return a redacted `503`; failed execution responses include a conversation ID so the UI can load its checkpoint and offer **Resume conversation**. If failure occurred before any checkpoint existed, resend the initial message after correcting the dependency.

This version is for **local development**: bind to loopback and run one worker. The server restricts hostnames and cross-origin browser writes, but conversation UUIDs are not customer authentication. Approval access uses a shared reviewer key, and mutation locks are process-local. Before network deployment, add customer/session authentication, per-conversation and per-device authorization, individual support roles, and cross-process concurrency control. No hosted deployment is included.

API tests use an in-memory checkpointer and fake specialists, so `python -m pytest -q` needs no running database or provider calls.

The separate **Agent routing** panel shows the latest turn's actual node sequence, selected specialist, QA revision loops, fallback, and approval outcomes. It opens by default and can be toggled independently from the support approval view. Minimal node/outcome records are checkpointed per turn, without entity values, prompts, or raw log data. The panel updates when a request finishes or pauses; it does not stream live node execution. Older checkpoints without routing records show their route after the next message.
