"""Chunking, embedding and storage in the Chroma vector database."""

from functools import lru_cache

import chromadb
from openai import OpenAI

from app.config import get_settings
from app.schemas import MetadataValue, RetrievedChunk


@lru_cache
def _openai_client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


@lru_cache
def _collection() -> chromadb.Collection:
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


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


def embed(texts: list[str]) -> list[list[float]]:
    response = _openai_client().embeddings.create(
        model=get_settings().embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def add_document(document_id: str, content: str, metadata: dict[str, MetadataValue]) -> int:
    """Chunk, embed and store a document. Returns the number of chunks written."""
    settings = get_settings()
    chunks = chunk_text(content, settings.chunk_size, settings.chunk_overlap)
    _collection().upsert(
        ids=[f"{document_id}:{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
        metadatas=[{**metadata, "document_id": document_id, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


def search(query: str, top_k: int) -> list[RetrievedChunk]:
    result = _collection().query(query_embeddings=embed([query]), n_results=top_k)
    return [
        RetrievedChunk(text=text, metadata=metadata or {}, distance=distance)
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
