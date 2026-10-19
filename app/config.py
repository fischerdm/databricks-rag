from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

TopicMode = Literal["filter", "boost", "off"]


class Settings(BaseSettings):
    """Application settings, loaded from environment variables or the .env file."""

    # Ignore unrelated keys in .env (e.g. API keys used by other tools).
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    # Extracts date/topic filters from questions. gpt-4o-mini mapped "OPEC" to "oilseed"
    # in testing; the gpt-5.4 family got it right and nano was the fastest.
    filter_model: str = "gpt-5.4-nano"
    # How an extracted topic is used: "filter" keeps only articles with that label, "boost" ranks
    # them higher without excluding the rest, "off" ignores it. A topic set by the client always filters.
    # "off" since the evaluation showed the extracted topic is too often wrong to filter by.
    topic_mode: TopicMode = "off"
    # With "boost": subtracted from the cosine distance of chunks labelled with the extracted topic,
    # so the topic becomes a second ranking signal instead of a hard filter. It reorders, never adds:
    # only worth it with a reliable topic. See docs/evaluation.md, 'topic as filter, boost, or not at all'.
    topic_boost: float = 0.05

    chroma_path: str = "data/chroma"
    collection_name: str = "documents"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4

    # Where the Chainlit UI (app/ui.py) reaches the API; it calls /query over HTTP.
    api_url: str = "http://127.0.0.1:8000"

    # Only for answer scoring (evals/score_answers.py). The judge comes from another provider than
    # chat_model, so the answers are not graded by a model of their own family.
    anthropic_api_key: str | None = None
    judge_model: str = "claude-sonnet-5-5"


@lru_cache
def get_settings() -> Settings:
    return Settings()
