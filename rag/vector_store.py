"""ChromaDB vector store wrapper.

Manages persistent collections for:
- competitor_data: ingested competitor profiles and collected data
- analysis_reports: historical analysis reports (SWOT, insights, landscape)
- search_cache: cached search results (reuse across sessions)

Each collection stores document chunks with embeddings and metadata.
"""

from pathlib import Path
from typing import Optional
from loguru import logger


class VectorStore:
    """ChromaDB-backed vector store for RAG."""

    # Standard collections used by CompetiScope
    COLLECTION_COMPETITOR_DATA = "competitor_data"
    COLLECTION_ANALYSIS_REPORTS = "analysis_reports"
    COLLECTION_SEARCH_CACHE = "search_cache"

    def __init__(self, persist_dir: str, embedder=None, embedding_function=None):
        """
        Args:
            persist_dir: directory for ChromaDB persistence
            embedder: embedding function (optional, use ChromaDB's default if None)
            embedding_function: ChromaDB-compatible embedding function (optional).
                If not provided, uses a safe fallback to avoid ONNX dependency on Windows.
        """
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._persist_dir = str(persist_dir)
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedder = embedder

        # Set up embedding function: prefer explicit, then embedder adapter, then safe default
        if embedding_function is not None:
            self._ef = embedding_function
        elif embedder is not None and hasattr(embedder, 'dimension'):
            self._ef = _EmbedderAdapter(embedder)
        else:
            self._ef = _safe_default_ef()

        logger.info(f"VectorStore initialized at {self._persist_dir}")

    # ---- collection management ----

    def get_or_create(self, name: str) -> "chromadb.Collection":
        """Get or create a named collection."""
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]

    def delete_collection(self, name: str):
        try:
            self._client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
        except Exception as exc:
            logger.warning(f"Failed to delete collection '{name}': {exc}")

    # ---- document operations ----

    def add(
        self,
        collection_name: str,
        chunks: list[dict],
        embeddings: Optional[list[list[float]]] = None,
    ):
        """Add chunks to a collection.

        Args:
            collection_name: target collection name
            chunks: list of {content, metadata, ...} dicts
            embeddings: pre-computed embeddings (optional, computed via embedder if None)
        """
        if not chunks:
            return

        col = self.get_or_create(collection_name)

        ids = [f"{collection_name}_{c.get('chunk_index', i)}" for i, c in enumerate(chunks)]
        documents = [c["content"] for c in chunks]
        metadatas = [{k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                       for k, v in c.get("metadata", {}).items()} for c in chunks]

        col.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas if metadatas[0] else None,
            embeddings=embeddings,
        )
        logger.debug(f"Added {len(chunks)} chunks to '{collection_name}'")

    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Search a collection by text query (ChromaDB auto-embeds).

        Returns list of {id, content, metadata, distance}
        """
        try:
            col = self.get_or_create(collection_name)
        except Exception:
            logger.warning(f"Collection '{collection_name}' not found")
            return []

        try:
            results = col.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error(f"Search failed in '{collection_name}': {exc}")
            return []

        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })
        return hits

    def search_by_embedding(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Search using a pre-computed embedding vector."""
        try:
            col = self.get_or_create(collection_name)
        except Exception:
            return []

        try:
            results = col.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error(f"Embedding search failed in '{collection_name}': {exc}")
            return []

        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })
        return hits

    def count(self, collection_name: str) -> int:
        """Count documents in a collection."""
        try:
            col = self.get_or_create(collection_name)
            return col.count()
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Embedding function adapters (for ChromaDB integration)
# ---------------------------------------------------------------------------

class _EmbedderAdapter:
    """Wrap a rag.embedder.BaseEmbedder as a ChromaDB EmbeddingFunction."""

    def __init__(self, embedder):
        self._embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._embedder.embed(input))


def _safe_default_ef():
    """Return a safe embedding function.

    On Windows systems where torch/onnxruntime have DLL conflicts, defaults
    to a hash-based dummy embedding to avoid process crashes. Set
    CHROMADB_EF=st or CHROMADB_EF=onnx to force a specific backend.
    """
    import os
    mode = os.getenv("CHROMADB_EF", "dummy")  # dummy is safe default for all platforms

    if mode == "st":
        return _make_st_ef()
    elif mode == "onnx":
        import chromadb.utils.embedding_functions as ef
        return ef.DefaultEmbeddingFunction()
    elif mode == "dummy":
        return _DummyEmbeddingFunction(dim=384)
    else:
        # auto: try ST first, then ONNX, then dummy
        try:
            return _make_st_ef()
        except Exception:
            try:
                import chromadb.utils.embedding_functions as ef
                fn = ef.DefaultEmbeddingFunction()
                _ = fn(["test"])
                return fn
            except Exception:
                logger.warning("No embedding backend available — using dummy EF")
                return _DummyEmbeddingFunction(dim=384)


def _make_st_ef():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    class STEmbeddingFunction:
        def __call__(self, input: list[str]) -> list[list[float]]:
            embeddings = model.encode(input, show_progress_bar=False)
            return [e.tolist() for e in embeddings]
    logger.info("Using SentenceTransformer embedding function")
    return STEmbeddingFunction()


class _DummyEmbeddingFunction:
    """Deterministic hash-based embedding for environments where ONNX/torch crash.

    Uses Python's built-in hash — NOT semantically meaningful, but allows
    ChromaDB to function without crashing for development/testing.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    def name(self) -> str:
        return "dummy-hash-ef"

    def __call__(self, input: list[str]) -> list[list[float]]:
        import hashlib
        results = []
        for text in input:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = []
            for i in range(self._dim):
                byte_val = h[i % len(h)] / 255.0
                vec.append((byte_val - 0.5) * 2.0)  # normalize to [-1, 1]
            results.append(vec)
        return results
