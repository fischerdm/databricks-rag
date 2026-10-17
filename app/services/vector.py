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


def add_documents(documents: list[tuple[str, str, dict[str, MetadataValue]]]) -> int:
    """Chunk, embed and store (document_id, content, metadata) tuples in one embedding call.

    Returns the number of chunks written. Upserts by chunk id, so re-adding a document
    with the same id overwrites it instead of duplicating it.
    """
    settings = get_settings()
    ids, chunks, metadatas = [], [], []
    for document_id, content, metadata in documents:
        for i, chunk in enumerate(chunk_text(content, settings.chunk_size, settings.chunk_overlap)):
            ids.append(f"{document_id}:{i}")
            chunks.append(chunk)
            metadatas.append({**metadata, "document_id": document_id, "chunk": i})
    if chunks:
        _collection().upsert(ids=ids, documents=chunks, embeddings=embed(chunks), metadatas=metadatas)
    return len(chunks)


def add_document(document_id: str, content: str, metadata: dict[str, MetadataValue]) -> int:
    """Chunk, embed and store a document. Returns the number of chunks written."""
    return add_documents([(document_id, content, metadata)])


def search(query: str, top_k: int) -> list[RetrievedChunk]:
    result = _collection().query(query_embeddings=embed([query]), n_results=top_k)
    return [
        RetrievedChunk(text=text, metadata=metadata or {}, distance=distance)
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
