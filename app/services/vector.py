"""Chunking, embedding and storage in the Chroma vector database.

With VECTOR_BACKEND=databricks, search and lookups go to Databricks Vector Search instead
(vector_databricks.py); writing then happens in the ingest job, not here.
"""

from functools import lru_cache
from types import ModuleType

import chromadb
from openai import OpenAI

from app.config import get_settings
from app.schemas import MetadataValue, RetrievedChunk, SearchFilters
from app.services.chunking import chunk_text


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


def _databricks() -> ModuleType | None:
    """The Databricks backend module, or None when Chroma is used. Imported lazily, so Chroma needs no SDK."""
    if get_settings().vector_backend != "databricks":
        return None
    from app.services import vector_databricks

    return vector_databricks


def _require_chroma(action: str) -> None:
    if _databricks():
        raise RuntimeError(f"{action} is not supported with VECTOR_BACKEND=databricks; run the ingest job instead")


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
    _require_chroma("Indexing")
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
    _require_chroma("Updating metadata")
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
    if databricks := _databricks():
        return databricks.document_metadata(document_ids)
    if not document_ids:
        return {}
    result = _collection().get(ids=[f"{d}:0" for d in document_ids], include=["metadatas"])
    return {m["document_id"]: m for m in result["metadatas"]}


def document_distances(query: str, document_ids: list[str]) -> dict[str, float]:
    """Cosine distance from the query to the closest chunk of each document, as search ranks them.

    Unlike search, this ignores filters and top_k, so it also works for documents search did not return.
    """
    if not document_ids:
        return {}
    query_vec = embed([query])[0]
    if databricks := _databricks():
        return databricks.document_distances(query_vec, document_ids)
    query_norm = sum(x * x for x in query_vec) ** 0.5
    result = _collection().get(where={"document_id": {"$in": document_ids}}, include=["embeddings", "metadatas"])
    distances: dict[str, float] = {}
    for vec, metadata in zip(result["embeddings"], result["metadatas"], strict=True):
        dot = sum(a * b for a, b in zip(query_vec, vec, strict=True))
        similarity = dot / (query_norm * sum(x * x for x in vec) ** 0.5)
        doc_id = metadata["document_id"]
        distances[doc_id] = min(distances.get(doc_id, 2.0), float(1 - similarity))
    return distances


def similar_documents(document_id: str, max_distance: float, limit: int = 10) -> dict[str, float]:
    """Other documents whose first chunk lies within max_distance of this document's first chunk.

    Comparing first chunks (title, date and opening paragraph) finds re-sent and rewritten
    versions of one news story. Returns {document_id: distance}, closest first.
    """
    if databricks := _databricks():
        return databricks.similar_documents(document_id, max_distance, limit)
    own = _collection().get(ids=[f"{document_id}:0"], include=["embeddings"])
    if not len(own["ids"]):
        return {}
    result = _collection().query(query_embeddings=[own["embeddings"][0]], n_results=limit + 1, where={"chunk": 0})
    return {
        metadata["document_id"]: distance
        for metadata, distance in zip(result["metadatas"][0], result["distances"][0], strict=True)
        if metadata["document_id"] != document_id and distance <= max_distance
    }


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
    if databricks := _databricks():
        return databricks.search(embed([query])[0], top_k, filters)
    where = build_filter(filters) if filters else None
    result = _collection().query(query_embeddings=embed([query]), n_results=top_k, where=where)
    return [
        RetrievedChunk(text=text, metadata=metadata or {}, distance=distance)
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        )
    ]
