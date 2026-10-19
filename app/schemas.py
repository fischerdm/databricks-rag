from datetime import date
from typing import Union

from pydantic import BaseModel, Field, model_validator

# Chroma only accepts flat metadata: primitives, or non-empty lists of them.
MetadataValue = Union[str, int, float, bool, list[str]]


class DocumentInput(BaseModel):
    content: str = Field(min_length=1)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class IndexResponse(BaseModel):
    status: str
    document_id: str
    message: str


class SearchFilters(BaseModel):
    date_from: date | None = Field(default=None, description="Earliest article date (inclusive)")
    date_to: date | None = Field(default=None, description="Latest article date (inclusive)")
    topic: str | None = Field(default=None, description="Reuters topic code, e.g. 'crude', 'acq', 'earn'")

    @model_validator(mode="after")
    def check_range(self) -> "SearchFilters":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self


class QueryInput(SearchFilters):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    auto_filters: bool = Field(
        default=True,
        description=(
            "Extract filters from the question: the date range, and the topic only if the server's "
            "TOPIC_MODE uses it (default: dates only). Explicit filters take precedence"
        ),
    )


class RetrievedChunk(BaseModel):
    text: str
    metadata: dict[str, MetadataValue]
    distance: float
