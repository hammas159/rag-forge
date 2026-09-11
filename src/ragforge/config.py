"""Single source of truth for configuration. Everything reads from here."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Storage
    postgres_dsn: str = "postgresql://ragforge:ragforge@localhost:5433/ragforge"
    redis_url: str = "redis://localhost:6380/0"

    # LLM backend - one interface, two implementations. See llm.py.
    llm_backend: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    # Models
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    embed_dim: int = 1024

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60

    # Chunking
    chunk_tokens: int = 512
    chunk_overlap: int = 64

    # Answer only when retrieved support clears this bar; otherwise refuse.
    grounding_threshold: float = 0.55

    cache_ttl_seconds: int = 3600


@lru_cache
def get_settings() -> Settings:
    return Settings()
