"""Ingest Markdown knowledge files into the PostgreSQL pgvector store."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.models.database import connect_db
from app.rag.chunking import chunk_text
from app.rag.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, get_embeddings

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS faq_chunks (
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding VECTOR({EMBEDDING_DIMENSIONS}) NOT NULL,
    PRIMARY KEY (source, chunk_index)
)
"""


def ingest_documents(directory: Path = KNOWLEDGE_BASE_DIR) -> int:
    files = sorted(directory.glob("*.md"))
    if not files:
        raise RuntimeError(f"No Markdown knowledge files found in {directory}.")

    chunks_by_source = {
        path.name: chunk_text(path.read_text(encoding="utf-8"))
        for path in files
    }
    all_chunks = [
        (source, index, content)
        for source, chunks in chunks_by_source.items()
        for index, content in enumerate(chunks, start=1)
    ]
    if not all_chunks:
        raise RuntimeError(f"Knowledge files in {directory} are empty.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before ingesting documents.")

    embeddings = get_embeddings()
    vectors = embeddings.embed_documents([content for _, _, content in all_chunks])
    rows_by_source: dict[str, list[tuple]] = {}
    for (source, index, content), vector in zip(all_chunks, vectors):
        rows_by_source.setdefault(source, []).append(
            (source, index, content, EMBEDDING_MODEL, Vector(vector))
        )

    with connect_db() as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)
        connection.execute(CREATE_TABLE_SQL)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS faq_chunks_embedding_idx "
            "ON faq_chunks USING hnsw (embedding vector_cosine_ops)"
        )

        # Replace each source's chunk set as one transaction so re-ingestion
        # removes stale chunks when a Markdown file gets shorter.
        for source, chunks in chunks_by_source.items():
            connection.execute("DELETE FROM faq_chunks WHERE source = %s", (source,))
            if not chunks:
                continue
            connection.executemany(
                "INSERT INTO faq_chunks "
                "(source, chunk_index, content, embedding_model, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                rows_by_source[source],
            )

    return len(all_chunks)


def main() -> None:
    load_dotenv()
    count = ingest_documents()
    print(f"Ingested {count} FAQ chunks from {KNOWLEDGE_BASE_DIR} using {EMBEDDING_MODEL}.")


if __name__ == "__main__":
    main()
