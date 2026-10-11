"""Provider adapters shared by agents and retrieval code.

Factories return LangChain interfaces; provider-specific configuration lives here.
"""

import os
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def response_text(response: Any) -> str:
    """Extract visible text without exposing provider metadata or reasoning."""
    content = response.content
    if isinstance(content, str):
        return content
    return "\n".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
    )


def chat_model_name() -> str:
    return os.getenv("CHAT_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"


def embedding_dimensions() -> int:
    value = int(os.getenv("EMBEDDING_DIMENSIONS") or "1536")
    if value <= 0:
        raise ValueError("EMBEDDING_DIMENSIONS must be positive.")
    return value


def embedding_identity() -> str:
    # Separate stores logically when a provider or endpoint changes.
    provider = os.getenv("EMBEDDING_PROVIDER") or "openai"
    model = embedding_model_name()
    if provider == "openai":
        return model  # Preserve existing OpenAI rows.
    return f"{provider}:{os.getenv('EMBEDDING_BASE_URL', '')}:{model}"


def _connection_options(prefix: str) -> dict:
    provider = os.getenv(f"{prefix}_PROVIDER") or "openai"
    if provider not in {"openai", "openai_compatible"}:
        raise ValueError(f"Unsupported {prefix}_PROVIDER: {provider}. Add a provider adapter first.")
    key = os.getenv(f"{prefix}_API_KEY") or (
        os.getenv("OPENAI_API_KEY") if provider == "openai" else None
    )
    if not key:
        raise ValueError(f"Set {prefix}_API_KEY" + (" or OPENAI_API_KEY." if provider == "openai" else "."))
    options = {"api_key": key}
    if provider == "openai_compatible":
        url = os.getenv(f"{prefix}_BASE_URL")
        if not url:
            raise ValueError(f"Set {prefix}_BASE_URL for openai_compatible.")
        if not os.getenv(f"{prefix}_MODEL"):
            raise ValueError(f"Set {prefix}_MODEL for openai_compatible.")
        options["base_url"] = url
    return options


def get_chat_model() -> BaseChatModel:
    if os.getenv("CHAT_PROVIDER") == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = os.getenv("CHAT_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key or not os.getenv("CHAT_MODEL"):
            raise ValueError("Set CHAT_API_KEY (or GOOGLE_API_KEY) and CHAT_MODEL for google_genai.")
        return ChatGoogleGenerativeAI(
            model=chat_model_name(), google_api_key=key, vertexai=False,
        )
    return ChatOpenAI(model=chat_model_name(), temperature=0, **_connection_options("CHAT"))


def get_embeddings() -> Embeddings:
    options = _connection_options("EMBEDDING")
    requested_dimensions = os.getenv("EMBEDDING_OUTPUT_DIMENSIONS")
    if requested_dimensions:
        dimensions = int(requested_dimensions)
        if dimensions <= 0:
            raise ValueError("EMBEDDING_OUTPUT_DIMENSIONS must be positive.")
        options["dimensions"] = dimensions
    return OpenAIEmbeddings(
        model=embedding_model_name(),
        # Compatible APIs may accept text but not OpenAI token IDs.
        check_embedding_ctx_length=(os.getenv("EMBEDDING_PROVIDER", "openai") != "openai_compatible"),
        **options,
    )
