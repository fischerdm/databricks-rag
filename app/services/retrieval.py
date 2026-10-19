"""The retrieval half of a query: filter extraction, filtered search and the empty-result fallback.

Shared by the /query endpoint and the evaluation scripts, so both run the same pipeline.
"""

import asyncio

from app.config import TopicMode, get_settings
from app.schemas import RetrievedChunk, SearchFilters
from app.services import filters, vector

# With a topic boost, rerank this many times top_k candidates, so that articles just below the
# cut can move up.
BOOST_POOL = 5


def boost_topic(chunks: list[RetrievedChunk], topic: str, amount: float, top_k: int) -> list[RetrievedChunk]:
    """The top_k chunks after subtracting `amount` from the distance of chunks labelled `topic`.

    The returned chunks keep their real distances; only the order changes.
    """
    def score(chunk: RetrievedChunk) -> float:
        return chunk.distance - (amount if topic in chunk.metadata.get("topics", []) else 0.0)

    return sorted(chunks, key=score)[:top_k]


async def retrieve(
    question: str,
    top_k: int,
    explicit: SearchFilters,
    auto_filters: bool = True,
    topic_mode: TopicMode | None = None,
    topic_boost: float | None = None,
) -> tuple[SearchFilters, str | None, list[RetrievedChunk]]:
    """Return the filters actually applied, the boosted topic (if any) and the chunks, best first.

    topic_mode and topic_boost default to the settings; the evaluation overrides them to compare modes.
    """
    settings = get_settings()
    topic_mode = topic_mode or settings.topic_mode
    topic_boost = settings.topic_boost if topic_boost is None else topic_boost

    applied, boosted = explicit, None
    if auto_filters:
        extracted = await filters.extract_filters(question)
        if extracted and extracted.topic and topic_mode != "filter":
            # A topic the client set still filters (explicit wins); an extracted one is boosted or dropped.
            if topic_mode == "boost" and not explicit.topic:
                boosted = extracted.topic
            extracted = extracted.model_copy(update={"topic": None})
        applied = filters.merge_filters(explicit, extracted)

    pool = top_k * BOOST_POOL if boosted else top_k
    # Chroma and the embedding call are synchronous; keep them off the event loop.
    chunks = await asyncio.to_thread(vector.search, question, pool, applied)
    if not chunks and applied != explicit:
        # Extracted filters can be too narrow (wrong topic, date outside the corpus); fall back.
        applied = explicit
        chunks = await asyncio.to_thread(vector.search, question, pool, applied)
    if boosted:
        chunks = boost_topic(chunks, boosted, topic_boost, top_k)
    return applied, boosted, chunks
