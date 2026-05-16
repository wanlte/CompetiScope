"""KnowledgeBase orchestrator.

Ties together embedder, chunker, and vector store into a coherent RAG pipeline.
Designed to be injected into Collector, Analyst, and Writer agents.

Lifecycle:
1. After each analysis: ingest_report() / ingest_collected_data()
2. Before each search: search_similar() to check if fresh data exists
3. During analysis: get_competitor_history() to enrich with past findings
"""

from typing import Optional
from loguru import logger

from rag.embedder import BaseEmbedder, LocalEmbedder
from rag.chunker import TextChunker
from rag.vector_store import VectorStore


class KnowledgeBase:
    """Orchestrates ingestion and retrieval of competitive intelligence knowledge."""

    def __init__(
        self,
        persist_dir: str = "./knowledge_base",
        embedder: Optional[BaseEmbedder] = None,
        chunker: Optional[TextChunker] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._embedder = embedder
        self._chunker = chunker or TextChunker(chunk_size=500, chunk_overlap=50)
        self._vector_store: Optional[VectorStore] = None

        if enabled:
            self._vector_store = VectorStore(
                persist_dir=persist_dir,
                embedder=embedder,
            )

        logger.info(f"KnowledgeBase initialized (enabled={enabled}, dir={persist_dir})")

    # ---- ingestion ----

    async def ingest_collected_data(
        self, competitor: str, collected_data: dict
    ):
        """Ingest collected competitor data into the vector store."""
        if not self.enabled or not self._vector_store:
            return

        text = _dict_to_text(collected_data, f"Collected data for {competitor}")
        chunks = self._chunker.chunk(text, metadata={
            "competitor": competitor,
            "type": "collected_data",
            "source": "collector",
        })

        embeddings = None
        if self._embedder:
            contents = [c["content"] for c in chunks]
            embeddings = await self._embedder.embed(contents)

        self._vector_store.add(
            VectorStore.COLLECTION_COMPETITOR_DATA,
            chunks,
            embeddings=embeddings,
        )
        logger.info(f"Ingested {len(chunks)} chunks for '{competitor}' → competitor_data")

    async def ingest_report(
        self, competitors: list[str], report: str, report_type: str = "full"
    ):
        """Ingest a generated analysis report."""
        if not self.enabled or not self._vector_store:
            return

        competitor_str = ", ".join(competitors)
        chunks = self._chunker.chunk(report, metadata={
            "competitors": competitor_str,
            "type": "analysis_report",
            "report_type": report_type,
        })

        embeddings = None
        if self._embedder:
            contents = [c["content"] for c in chunks]
            embeddings = await self._embedder.embed(contents)

        self._vector_store.add(
            VectorStore.COLLECTION_ANALYSIS_REPORTS,
            chunks,
            embeddings=embeddings,
        )
        logger.info(f"Ingested {len(chunks)} report chunks → analysis_reports")

    async def ingest_search_cache(
        self, competitor: str, query: str, results: str
    ):
        """Cache search results for reuse."""
        if not self.enabled or not self._vector_store:
            return

        chunks = self._chunker.chunk(results, metadata={
            "competitor": competitor,
            "query": query,
            "type": "search_cache",
        })

        embeddings = None
        if self._embedder:
            contents = [c["content"] for c in chunks]
            embeddings = await self._embedder.embed(contents)

        self._vector_store.add(
            VectorStore.COLLECTION_SEARCH_CACHE,
            chunks,
            embeddings=embeddings,
        )

    # ---- retrieval ----

    def search_similar(self, query: str, collection: str, n_results: int = 5) -> list[dict]:
        """Semantic search across a named collection."""
        if not self.enabled or not self._vector_store:
            return []
        return self._vector_store.search(collection, query, n_results=n_results)

    def get_competitor_history(self, competitor: str, n_results: int = 5) -> dict:
        """Retrieve historical data about a competitor.

        Returns dict with keys: collected_data, reports, search_cache
        """
        if not self.enabled or not self._vector_store:
            return {"collected_data": [], "reports": [], "search_cache": []}

        return {
            "collected_data": self._vector_store.search(
                VectorStore.COLLECTION_COMPETITOR_DATA,
                competitor, n_results=n_results,
                where={"competitor": competitor},
            ),
            "reports": self._vector_store.search(
                VectorStore.COLLECTION_ANALYSIS_REPORTS,
                competitor, n_results=n_results,
            ),
            "search_cache": self._vector_store.search(
                VectorStore.COLLECTION_SEARCH_CACHE,
                competitor, n_results=n_results,
                where={"competitor": competitor},
            ),
        }

    def search_all(self, query: str, n_results: int = 3) -> dict:
        """Search across all collections."""
        if not self.enabled or not self._vector_store:
            return {"competitor_data": [], "reports": [], "search_cache": []}

        return {
            "competitor_data": self._vector_store.search(
                VectorStore.COLLECTION_COMPETITOR_DATA, query, n_results=n_results,
            ),
            "reports": self._vector_store.search(
                VectorStore.COLLECTION_ANALYSIS_REPORTS, query, n_results=n_results,
            ),
            "search_cache": self._vector_store.search(
                VectorStore.COLLECTION_SEARCH_CACHE, query, n_results=n_results,
            ),
        }

    # ---- management ----

    def clear_competitor(self, competitor: str):
        """Remove all data for a specific competitor."""
        if not self.enabled or not self._vector_store:
            return
        for col in [
            VectorStore.COLLECTION_COMPETITOR_DATA,
            VectorStore.COLLECTION_ANALYSIS_REPORTS,
            VectorStore.COLLECTION_SEARCH_CACHE,
        ]:
            self._vector_store.delete_collection(col)
            logger.info(f"Cleared collection '{col}' for '{competitor}'")

    def get_stats(self) -> dict:
        """Return collection statistics."""
        if not self.enabled or not self._vector_store:
            return {"enabled": False}

        return {
            "enabled": True,
            "collections": {
                col: self._vector_store.count(col)
                for col in self._vector_store.list_collections()
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_text(data: dict, title: str = "") -> str:
    """Flatten a nested dict into a readable text block."""
    import json

    parts = [f"# {title}\n"] if title else []
    for key, value in data.items():
        parts.append(f"## {key}")
        if isinstance(value, (dict, list)):
            parts.append(json.dumps(value, ensure_ascii=False, indent=2))
        else:
            parts.append(str(value))
        parts.append("")
    return "\n".join(parts)
