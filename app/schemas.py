from typing import Union

from pydantic import BaseModel, Field

# Chroma only accepts flat, primitive metadata values.
MetadataValue = Union[str, int, float, bool]


class DocumentInput(BaseModel):
    content: str = Field(min_length=1)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class IndexResponse(BaseModel):
    status: str
    document_id: str
    message: str


class QueryInput(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrievedChunk(BaseModel):
    text: str
    metadata: dict[str, MetadataValue]
    distance: float
