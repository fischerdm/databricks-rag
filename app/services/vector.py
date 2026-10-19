"""Chunking, embedding and storage in the Chroma vector database."""

from functools import lru_cache

import chromadb
from openai import OpenAI

from app.config import get_settings
from app.schemas import MetadataValue, RetrievedChunk, SearchFilters


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


def update_metadata(documents: list[tuple[str, str, dict[str, MetadataValue | None]]]) -> int:
    """Rewrite chunk metadata in place, keeping the stored embeddings.

    Takes the same tuples as add_documents. Chroma merges metadata on update, so a key
    set to None is deleted. Returns the number of chunks updated.
    """
    settings = get_settings()
    ids, metadatas = [], []
    for document_id, content, metadata in documents:
        for i, _ in enumerate(chunk_text(content, settings.chunk_size, settings.chunk_overlap)):
            ids.append(f"{document_id}:{i}")
            metadatas.append({**metadata, "document_id": document_id, "chunk": i})
    if ids:
        _collection().update(ids=ids, metadatas=metadatas)
    return len(ids)


def document_metadata(document_ids: list[str]) -> dict[str, dict[str, MetadataValue]]:
    """Return each indexed document's metadata, read from its first chunk. Unknown ids are left out."""
    if not document_ids:
        return {}
    result = _collection().get(ids=[f"{d}:0" for d in document_ids], include=["metadatas"])
    return {m["document_id"]: m for m in result["metadatas"]}


def build_filter(filters: SearchFilters) -> dict | None:
    """Translate search filters into a Chroma `where` clause."""
    conditions = []
    if filters.date_from:
        conditions.append({"date_int": {"$gte": int(filters.date_from.strftime("%Y%m%d"))}})
    if filters.date_to:
        conditions.append({"date_int": {"$lte": int(filters.date_to.strftime("%Y%m%d"))}})
    if filters.topic:
        conditions.append({"topics": {"$contains": filters.topic.lower()}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(query: str, top_k: int, filters: SearchFilters | None = None) -> list[RetrievedChunk]:
    where = build_filter(filters) if filters else None
    result = _collection().query(query_embeddings=embed([query]), n_results=top_k, where=where)
    return [
        RetrievedChunk(text=text, metadata=metadata or {}, distance=distance)
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
