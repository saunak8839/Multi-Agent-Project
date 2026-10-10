"""Shared embedding model configuration for FAQ ingestion and retrieval."""

from langchain_openai import OpenAIEmbeddings


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
