from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables or the .env file."""

    # Ignore unrelated keys in .env (e.g. API keys used by other tools).
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    chroma_path: str = "data/chroma"
    collection_name: str = "documents"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
