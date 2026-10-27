"""Search in a Databricks Vector Search index, the counterpart of the Chroma functions in vector.py.

Used when VECTOR_BACKEND=databricks. The index is a Delta Sync index on the chunk table that
jobs/ingest_chunks.py writes, so writing happens there, not here. Query embeddings still come
from vector.embed, so they match the embeddings the ingest job stored.

Authentication is the Databricks SDK's default: the CLI profile locally (databricks auth login),
the app's service principal inside a Databricks App.
"""

import json
from functools import lru_cache

from databricks.sdk import WorkspaceClient

from app.config import get_settings
from app.schemas import MetadataValue, RetrievedChunk, SearchFilters

COLUMNS = [
    "text",
    "chunk_id",
    "document_id",
    "chunk",
    "title",
    "date",
    "date_int",
    "topics",
    "places",
    "reuters_id",
    "source",
]
# Vector Search returns every number as a float.
INT_COLUMNS = {"chunk", "date_int", "reuters_id"}
# The ingest job's embedding size (text-embedding-3-small); a zero vector of it serves lookups by id.
DIMENSION = 1536


@lru_cache
def _client() -> WorkspaceClient:
    return WorkspaceClient()


def _distance(score: float) -> float:
    """Cosine distance, as Chroma reports it, from a Vector Search score.

    The index ranks by L2 distance and reports score = 1 / (1 + L2²). For unit-length embeddings
    (OpenAI's are), L2² = 2 · (1 − cosine similarity), so the cosine distance is L2² / 2.
    """
    return (1 / score - 1) / 2


def _metadata(row: dict) -> dict[str, MetadataValue]:
    """A result row as Chroma-style chunk metadata: ints restored, empty values left out."""
    metadata = {}
    for key, value in row.items():
        if key in ("text", "score", "embedding", "chunk_id") or value is None or value == []:
            continue
        metadata[key] = int(value) if key in INT_COLUMNS else value
    return metadata


def _query(vector: list[float], k: int, filters: dict | None = None, columns: list[str] = COLUMNS) -> list[dict]:
    response = _client().vector_search_indexes.query_index(
        index_name=get_settings().databricks_index,
        columns=columns,
        query_vector=vector,
        num_results=k,
        filters_json=json.dumps(filters) if filters else None,
    )
    names = [column.name for column in response.manifest.columns]
    return [dict(zip(names, row, strict=True)) for row in response.result.data_array or []]


def build_filter(filters: SearchFilters) -> dict | None:
    """Translate search filters into a Vector Search `filters_json` dict (standard endpoint syntax)."""
    conditions = {}
    if filters.date_from:
        conditions["date_int >="] = int(filters.date_from.strftime("%Y%m%d"))
    if filters.date_to:
        conditions["date_int <="] = int(filters.date_to.strftime("%Y%m%d"))
    if filters.topic:
        # On an array column, a value matches rows whose array contains it.
        conditions["topics"] = filters.topic.lower()
    return conditions or None


def search(query_vector: list[float], top_k: int, filters: SearchFilters | None = None) -> list[RetrievedChunk]:
    rows = _query(query_vector, top_k, build_filter(filters) if filters else None)
    return [RetrievedChunk(text=row["text"], metadata=_metadata(row), distance=_distance(row["score"])) for row in rows]


def document_metadata(document_ids: list[str]) -> dict[str, dict[str, MetadataValue]]:
    """Each document's metadata, read from its first chunk. Unknown ids are left out."""
    if not document_ids:
        return {}
    rows = _query([0.0] * DIMENSION, len(document_ids), {"chunk_id": [f"{d}:0" for d in document_ids]})
    return {row["document_id"]: _metadata(row) for row in rows}


def document_distances(query_vector: list[float], document_ids: list[str]) -> dict[str, float]:
    """Cosine distance from the query to the closest chunk of each document, ignoring filters and top_k."""
    if not document_ids:
        return {}
    # Results come closest first, so the first row seen per document is its closest chunk.
    rows = _query(query_vector, 100 * len(document_ids), {"document_id": document_ids}, ["document_id"])
    distances: dict[str, float] = {}
    for row in rows:
        distances.setdefault(row["document_id"], _distance(row["score"]))
    return distances


def similar_documents(document_id: str, max_distance: float, limit: int = 10) -> dict[str, float]:
    """Other documents whose first chunk lies within max_distance of this document's first chunk."""
    own = _query([0.0] * DIMENSION, 1, {"chunk_id": f"{document_id}:0"}, ["chunk_id", "embedding"])
    if not own:
        return {}
    rows = _query(own[0]["embedding"], limit + 1, {"chunk": 0}, ["document_id"])
    return {
        row["document_id"]: distance
        for row in rows
        if row["document_id"] != document_id and (distance := _distance(row["score"])) <= max_distance
    }
