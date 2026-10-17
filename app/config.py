from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    chroma_path: str = "data/chroma"
    collection_name: str = "documents"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4

    # Where the Chainlit UI (app/ui.py) reaches the API; it calls /query over HTTP.
    api_url: str = "http://127.0.0.1:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
