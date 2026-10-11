"""Semantic retrieval over ingested FAQ chunks."""

from pgvector import Vector
from pgvector.psycopg import register_vector

from app.models.database import connect_db
from app.rag.embeddings import embedding_identity, get_embeddings


def retrieve_faq_context(question: str, *, limit: int = 4) -> list[dict]:
    query_vector = Vector(get_embeddings().embed_query(question))
    with connect_db() as connection:
        register_vector(connection)
        rows = connection.execute(
            "SELECT source, chunk_index, content, embedding <=> %s AS distance "
            "FROM faq_chunks WHERE embedding_model = %s "
            "ORDER BY embedding <=> %s LIMIT %s",
            (query_vector, embedding_identity(), query_vector, limit),
        ).fetchall()

    return [
        {
            "source": row["source"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "distance": float(row["distance"]),
        }
        for row in rows
    ]
