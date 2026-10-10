"""Small, deterministic character chunker for Markdown knowledge files."""


def chunk_text(text: str, *, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    text = text.strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer ending on a paragraph or sentence boundary when it is
            # reasonably close to the target chunk size.
            lower_bound = start + chunk_size // 2
            boundary = max(
                text.rfind("\n\n", lower_bound, end),
                text.rfind(". ", lower_bound, end),
            )
            if boundary >= lower_bound:
                end = boundary + (2 if text[boundary : boundary + 2] == ". " else 0)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
