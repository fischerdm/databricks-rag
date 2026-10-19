"""The retrieval half of a query: filter extraction, filtered search and the empty-result fallback.

Shared by the /query endpoint and the evaluation scripts, so both run the same pipeline.
"""

import asyncio

from app.schemas import RetrievedChunk, SearchFilters
from app.services import filters, vector


async def retrieve(
    question: str, top_k: int, explicit: SearchFilters, auto_filters: bool = True
) -> tuple[SearchFilters, list[RetrievedChunk]]:
    """Return the filters actually applied and the retrieved chunks, closest first."""
    applied = explicit
    if auto_filters:
        applied = filters.merge_filters(explicit, await filters.extract_filters(question))

    # Chroma and the embedding call are synchronous; keep them off the event loop.
    chunks = await asyncio.to_thread(vector.search, question, top_k, applied)
    if not chunks and applied != explicit:
        # Extracted filters can be too narrow (wrong topic, date outside the corpus); fall back.
        applied = explicit
        chunks = await asyncio.to_thread(vector.search, question, top_k, applied)
    return applied, chunks
