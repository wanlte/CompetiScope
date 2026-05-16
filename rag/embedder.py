"""Embedding service abstraction.

Supports two backends:
- OpenAIEmbedder: OpenAI-compatible API (text-embedding-3-small / text-embedding-ada-002)
- LocalEmbedder: SentenceTransformer local model (free, no API calls)

DeepSeek does not currently offer a dedicated embedding API, so we default to
local SentenceTransformer for cost-free embeddings.
"""

from abc import ABC, abstractmethod
from typing import Optional
from loguru import logger


class BaseEmbedder(ABC):
    """Abstract embedding service."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns list of embedding vectors."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible embedder
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    """Embedding via OpenAI-compatible API.

    Works with:
    - OpenAI: text-embedding-3-small (1536d), text-embedding-3-large (3072d), text-embedding-ada-002 (1536d)
    - Any OpenAI-compatible embedding service
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        timeout: int = 30,
        batch_size: int = 20,
    ):
        import asyncio
        from openai import AsyncOpenAI

        self.model = model
        self.batch_size = batch_size
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        # Known dimensions per model
        _dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = _dims.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = await self._client.embeddings.create(model=self.model, input=batch)
            results.extend([d.embedding for d in resp.data])
        return results

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]


# ---------------------------------------------------------------------------
# Local SentenceTransformer embedder (free, no API key needed)
# ---------------------------------------------------------------------------

class LocalEmbedder(BaseEmbedder):
    """Local embedding via SentenceTransformer.

    Default model: all-MiniLM-L6-v2 (384d, fast, good for Chinese via multilingual).
    For better Chinese support: paraphrase-multilingual-MiniLM-L12-v2 (384d).
    """

    _DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        model_name = model_name or self._DEFAULT_MODEL
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"  embedding dimension = {self._dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self._model.encode, texts, None, "cpu")
        return [e.tolist() for e in embeddings]

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_embedder(
    provider: str = "local",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> BaseEmbedder:
    """Factory: create an embedder by provider name.

    Args:
        provider: "local" (default, SentenceTransformer) or "openai"
        api_key: required for "openai"
        base_url: optional, for OpenAI-compatible services
        model: optional model name override
    """
    if provider == "openai":
        if not api_key:
            raise ValueError("api_key is required for OpenAI embedder")
        return OpenAIEmbedder(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            model=model or "text-embedding-3-small",
        )
    # default: local
    return LocalEmbedder(model_name=model or None)
