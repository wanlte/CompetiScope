"""Hybrid search combining vector similarity with BM25 keyword matching.

Why hybrid:
- Vector search excels at semantic similarity ("cheap pricing" ≈ "affordable plans")
- BM25 excels at exact keyword matching ("Notion" must match "Notion", not "motion")
- Combining both gives higher recall/precision than either alone

Scoring: weighted reciprocal rank fusion (RRF) — the standard approach.
"""

import re
from typing import Optional
from math import log
from loguru import logger

from rag.embedder import BaseEmbedder
from rag.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Minimal BM25 implementation
# ---------------------------------------------------------------------------

class BM25:
    """Simple BM25 scorer for in-memory document sets."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._avgdl: float = 0.0
        self._df: dict[str, int] = {}       # document frequency
        self._idf: dict[str, float] = {}    # pre-computed IDF

    def fit(self, documents: list[str]):
        """Tokenize and index documents."""
        self._docs = [_tokenize(d) for d in documents]
        total_docs = len(self._docs)
        if total_docs == 0:
            return

        # Average document length
        lengths = [len(d) for d in self._docs]
        self._avgdl = sum(lengths) / max(total_docs, 1)

        # Document frequency
        self._df.clear()
        for doc in self._docs:
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1

        # IDF
        self._idf.clear()
        for term, df in self._df.items():
            self._idf[term] = log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str, doc_idx: int) -> float:
        """Score a single document against a query."""
        if doc_idx >= len(self._docs):
            return 0.0

        doc = self._docs[doc_idx]
        query_tokens = _tokenize(query)
        doc_len = len(doc)

        total = 0.0
        for token in query_tokens:
            if token not in self._idf:
                continue
            tf = doc.count(token)
            numerator = self._idf[token] * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1))
            total += numerator / max(denominator, 0.001)

        return total

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return top-k (doc_idx, score) tuples sorted by score descending."""
        scores = [(i, self.score(query, i)) for i in range(len(self._docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def _tokenize(text: str) -> list[str]:
    """Minimal Chinese + English tokenizer.

    For Chinese: splits on each character (bigram would be better but this is simple).
    For English: splits on whitespace and punctuation.
    """
    # Extract Chinese characters and English words
    tokens: list[str] = []
    # English words (2+ chars)
    eng_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    tokens.extend(eng_words)
    # Chinese characters
    ch_chars = re.findall(r'[一-鿿]', text)
    tokens.extend(ch_chars)
    # Numbers
    numbers = re.findall(r'\d+', text)
    tokens.extend(numbers)
    return tokens or text.lower().split()


# ---------------------------------------------------------------------------
# Hybrid searcher
# ---------------------------------------------------------------------------

class HybridSearcher:
    """Combines vector search (ChromaDB) with BM25 keyword search.

    Usage:
        searcher = HybridSearcher(vector_store, embedder)
        results = searcher.search("Notion 定价", "competitor_data", n_results=5)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Optional[BaseEmbedder] = None,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ):
        self._store = vector_store
        self._embedder = embedder
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight

    def search(
        self,
        query: str,
        collection_name: str,
        n_results: int = 5,
        fetch_k: int = 20,
    ) -> list[dict]:
        """Hybrid search: vector + BM25.

        Args:
            query: search query
            collection_name: ChromaDB collection name
            n_results: number of final results
            fetch_k: larger pool for vector search (before fusion)

        Returns list of {id, content, metadata, hybrid_score}
        """
        # 1. Vector search (fetch more candidates)
        vector_hits = self._store.search(
            collection_name, query, n_results=fetch_k,
        )

        if not vector_hits:
            return []

        # 2. BM25 keyword search on the same candidates
        documents = [h["content"] for h in vector_hits]
        bm25 = BM25()
        bm25.fit(documents)
        bm25_hits = bm25.search(query, k=len(documents))

        # Map BM25 index → score
        bm25_map = {idx: score for idx, score in bm25_hits}

        # 3. Reciprocal rank fusion (RRF)
        rrf_scores: dict[str, float] = {}
        rrf_k = 60  # standard RRF constant

        # Vector rank
        for rank, hit in enumerate(vector_hits):
            doc_id = hit["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self._vector_weight / (rrf_k + rank + 1)

        # BM25 rank
        for rank, (idx, _) in enumerate(bm25_hits):
            doc_id = vector_hits[idx]["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self._keyword_weight / (rrf_k + rank + 1)

        # 4. Re-rank by fused score
        scored_hits = []
        for hit in vector_hits:
            hit["hybrid_score"] = rrf_scores.get(hit["id"], 0)
            scored_hits.append(hit)

        scored_hits.sort(key=lambda h: h["hybrid_score"], reverse=True)
        return scored_hits[:n_results]
