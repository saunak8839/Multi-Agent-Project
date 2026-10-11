"""Open the support graph with a durable PostgreSQL checkpointer."""

from contextlib import contextmanager
import os
from typing import Iterator

from langgraph.checkpoint.postgres import PostgresSaver

from app.graph.workflow import build_graph


@contextmanager
def open_support_graph() -> Iterator:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for durable conversation checkpoints.")

    # Checkpoints can contain serialized state; strict msgpack avoids unsafe
    # fallback deserialization of arbitrary Python objects.
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.setup()
        yield build_graph(checkpointer)
