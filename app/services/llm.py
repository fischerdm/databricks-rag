"""Prompt construction and streaming answer generation."""

import json
from collections.abc import AsyncIterator
from functools import lru_cache

from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas import RetrievedChunk, SearchFilters

SYSTEM_PROMPT = (
    "You are a financial news assistant. Answer the question using only the provided context. "
    "If the context does not contain the answer, say so. Cite sources as [1], [2], ... "
    "matching the numbered context passages."
)


@lru_cache
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks, start=1))


def _sse(event: str, data: object) -> str:
    # JSON-encode the payload so newlines in tokens cannot break SSE framing.
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_answer(
    question: str, chunks: list[RetrievedChunk], filters: SearchFilters
) -> AsyncIterator[str]:
    """Yield Server-Sent Events: 'filters' and 'sources' once, many 'token' events, then 'done'."""
    yield _sse("filters", filters.model_dump(mode="json"))
    yield _sse("sources", [chunk.model_dump() for chunk in chunks])

    stream = await _client().chat.completions.create(
        model=get_settings().chat_model,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"},
        ],
    )
    async for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield _sse("token", event.choices[0].delta.content)

    yield _sse("done", None)
