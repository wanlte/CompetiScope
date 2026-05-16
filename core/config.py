"""Unified Pydantic BaseSettings configuration.

Reads from environment variables (and .env file) with sensible defaults.
Replaces the scattered class-attribute config in config/settings.py.
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent


class AppConfig(BaseSettings):
    """Unified application configuration — all settings in one place."""

    model_config = {"env_file": str(BASE_DIR / ".env"), "env_file_encoding": "utf-8", "extra": "ignore"}

    # ---- LLM ----
    llm_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    llm_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    llm_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    llm_timeout: int = Field(default=60, alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    llm_input_price_per_1k: float = Field(default=0.00014, alias="LLM_INPUT_PRICE")
    llm_output_price_per_1k: float = Field(default=0.00028, alias="LLM_OUTPUT_PRICE")

    # ---- Cache ----
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_dir: str = Field(default=str(BASE_DIR / ".cache" / "llm"), alias="CACHE_DIR")
    cache_ttl_seconds: int = Field(default=86400, alias="CACHE_TTL")

    # ---- Agent Loop ----
    agent_max_steps: int = Field(default=12, alias="AGENT_MAX_STEPS")
    agent_reflection_enabled: bool = Field(default=True, alias="AGENT_REFLECTION_ENABLED")
    agent_reflection_rounds: int = Field(default=2, alias="AGENT_REFLECTION_ROUNDS")
    agent_reflection_threshold: int = Field(default=7, alias="AGENT_REFLECTION_THRESHOLD")

    # ---- Embedding ----
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    local_embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2", alias="LOCAL_EMBEDDING_MODEL"
    )
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    openai_embedding_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # ---- Knowledge Base ----
    kb_enabled: bool = Field(default=True, alias="ENABLE_KNOWLEDGE_BASE")
    kb_persist_dir: str = Field(default=str(BASE_DIR / ".chromadb"), alias="KB_PERSIST_DIR")
    kb_rag_n_results: int = Field(default=5, alias="RAG_N_RESULTS")
    kb_vector_weight: float = Field(default=0.6, alias="RAG_VECTOR_WEIGHT")
    kb_keyword_weight: float = Field(default=0.4, alias="RAG_KEYWORD_WEIGHT")
    kb_chunk_size: int = Field(default=500, alias="RAG_CHUNK_SIZE")
    kb_chunk_overlap: int = Field(default=50, alias="RAG_CHUNK_OVERLAP")

    # ---- Search ----
    search_engine: str = Field(default="duckduckgo", alias="SEARCH_ENGINE")
    search_default_results: int = Field(default=10, alias="DEFAULT_SEARCH_RESULTS")
    search_timeout: int = Field(default=30, alias="SEARCH_TIMEOUT")

    # ---- Output ----
    output_dir: str = Field(default=str(BASE_DIR / "output"), alias="OUTPUT_DIR")
    output_format: str = Field(default="markdown", alias="OUTPUT_FORMAT")

    # ---- Server ----
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_cors_origins: str = Field(default="*", alias="API_CORS_ORIGINS")

    # ---- Logging ----
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get or create the singleton AppConfig instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
