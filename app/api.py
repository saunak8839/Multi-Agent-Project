"""Local HTTP interface to the durable support workflow."""

from contextlib import contextmanager
from hmac import compare_digest
import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from langgraph.types import Command
from pydantic import BaseModel, Field, StrictBool, field_validator

from app.services.conversation import open_support_graph
from app.services.workflow_compat import reply_for, review_for, tools_for

load_dotenv()
app = FastAPI(title="Support Desk API", version="0.1.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
WEB_DIR = Path(__file__).resolve().parent / "web"
# Bounded process-local locks. Run one worker; production requires distributed locking.
_THREAD_LOCKS = [Lock() for _ in range(64)]


class MessageInput(BaseModel):
    message: str = Field(min_length=1, max_length=8000)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must not be blank.")
        return value


class ApprovalInput(BaseModel):
    approved: StrictBool
    note: str = Field(default="", max_length=2000)


class HistoryEntry(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class PendingApproval(BaseModel):
    order_id: str
    amount_cents: int
    currency: str
    draft_reply: str


class RoutingStep(BaseModel):
    node: str
    status: Literal["completed", "passed", "failed", "approved", "rejected", "awaiting_approval", "interrupted"]


class ConversationView(BaseModel):
    thread_id: str
    status: Literal["ready", "awaiting_approval", "interrupted"]
    history: list[HistoryEntry]
    specialist: str | None = None
    qa_passed: bool | None = None
    qa_attempts: int = 0
    tools: list[str] = Field(default_factory=list)
    pending_approval: PendingApproval | None = None
    routing_path: list[RoutingStep] = Field(default_factory=list)


def require_support(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("SUPPORT_APPROVAL_KEY", "")
    if not expected:
        raise HTTPException(503, "Support approval access is not configured on this server.")
    supplied = (authorization or "").removeprefix("Bearer ")
    if not authorization or not authorization.startswith("Bearer ") or not compare_digest(
        supplied.encode(), expected.encode()
    ):
        raise HTTPException(401, "A valid support key is required.", headers={"WWW-Authenticate": "Bearer"})


@contextmanager
def conversation_lock(thread_id: str):
    lock = _THREAD_LOCKS[hash(thread_id) % len(_THREAD_LOCKS)]
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "This conversation is busy. Please try again shortly.")
    try:
        yield
    finally:
        lock.release()


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "run_id": uuid4().hex}}


def view_for(graph, config: dict) -> ConversationView:
    snapshot = graph.get_state(config)
    state = snapshot.values or {}
    if not state:
        raise HTTPException(404, "Conversation not found.")
    status = "awaiting_approval" if "approval_gate" in snapshot.next else "interrupted" if snapshot.next else "ready"
    pending = None
    routing_path = list(state.get("routing_path", []))
    for node in snapshot.next:
        routing_path.append({"node": node, "status": "awaiting_approval" if node == "approval_gate" else "interrupted"})
    if status == "awaiting_approval":
        action = state.get("proposed_action", {})
        pending = PendingApproval(
            order_id=action["order_id"], amount_cents=action["amount_cents"],
            currency=action["currency"].strip(), draft_reply=reply_for(state),
        )
    return ConversationView(
        thread_id=config["configurable"]["thread_id"], status=status,
        history=state.get("conversation_history", []), specialist=state.get("specialist"),
        qa_passed=review_for(state).get("passed"), qa_attempts=review_for(state)["attempt"],
        tools=tools_for(state), pending_approval=pending,
        routing_path=routing_path,
    )


@contextmanager
def api_graph(thread_id: str):
    """Translate infrastructure failures without returning credentials or provider payloads."""
    try:
        with open_support_graph() as graph:
            yield graph
    except HTTPException:
        raise
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        rate_limited = status == 429 or "ratelimit" in type(exc).__name__.lower()
        raise HTTPException(
            429 if rate_limited else 503,
            detail={
                "message": "Provider rate limit reached. Wait before resuming this conversation."
                if rate_limited else "The support service could not complete the request. Try resuming this conversation.",
                "thread_id": thread_id,
            },
            headers={"Retry-After": "60"} if rate_limited else None,
        ) from exc


@app.get("/api/health")
def health():
    """Process health only; live dependency failures surface on conversation requests."""
    return {"status": "ok"}


@app.post("/api/conversations", response_model=ConversationView, status_code=201)
def create_conversation(body: MessageInput):
    thread_id = uuid4().hex
    config = config_for(thread_id)
    with conversation_lock(thread_id), api_graph(thread_id) as graph:
        graph.invoke({"customer_message": body.message, **config["configurable"]}, config)
        return view_for(graph, config)


@app.get("/api/conversations/{conversation_id}", response_model=ConversationView)
def get_conversation(conversation_id: UUID):
    config = config_for(conversation_id.hex)
    with api_graph(conversation_id.hex) as graph:
        return view_for(graph, config)


@app.post("/api/conversations/{conversation_id}/messages", response_model=ConversationView)
def send_message(conversation_id: UUID, body: MessageInput):
    thread_id = conversation_id.hex
    config = config_for(thread_id)
    with conversation_lock(thread_id), api_graph(thread_id) as graph:
        current = view_for(graph, config)
        if current.status != "ready":
            raise HTTPException(409, "Resolve the pending approval or resume the interrupted turn first.")
        graph.invoke({"customer_message": body.message, **config["configurable"]}, config)
        return view_for(graph, config)


@app.post("/api/conversations/{conversation_id}/retry", response_model=ConversationView)
def retry_turn(conversation_id: UUID):
    thread_id = conversation_id.hex
    config = config_for(thread_id)
    with conversation_lock(thread_id), api_graph(thread_id) as graph:
        current = view_for(graph, config)
        if current.status != "interrupted":
            raise HTTPException(409, "Only an interrupted turn can be resumed here.")
        graph.invoke(None, config)
        return view_for(graph, config)


@app.post("/api/conversations/{conversation_id}/approval", response_model=ConversationView,
          dependencies=[Depends(require_support)])
def decide_approval(conversation_id: UUID, body: ApprovalInput):
    thread_id = conversation_id.hex
    config = config_for(thread_id)
    with conversation_lock(thread_id), api_graph(thread_id) as graph:
        current = view_for(graph, config)
        if current.status != "awaiting_approval":
            raise HTTPException(409, "This conversation has no pending approval.")
        graph.invoke(Command(resume=body.model_dump()), config)
        return view_for(graph, config)


@app.middleware("http")
async def local_request_headers(request, call_next):
    # Same-origin browser UI only. This is not a substitute for production authentication.
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse(status_code=403, content={"detail": "Cross-origin writes are not allowed."})
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    if request.url.path in {"/docs", "/redoc"}:
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; font-src 'self' https://fonts.gstatic.com; connect-src 'self'; frame-ancestors 'none'"
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")
