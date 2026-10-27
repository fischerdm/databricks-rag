"""Text chunking. Standard library only, so the Databricks ingest job can import it without Chroma."""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows, preferring whitespace boundaries."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Back off to the last whitespace so words are not cut in half.
            boundary = text.rfind(" ", start + overlap + 1, end)
            if boundary != -1:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]
