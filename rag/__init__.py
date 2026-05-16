"""RAG (Retrieval-Augmented Generation) module.

Components:
- embedder: Text embedding abstraction (OpenAI / local SentenceTransformer)
- chunker: Recursive text splitting
- vector_store: ChromaDB wrapper
- knowledge_base: Orchestrator for ingest + retrieval
- hybrid_search: Vector similarity + BM25 keyword matching
"""

from rag.embedder import BaseEmbedder, OpenAIEmbedder, LocalEmbedder
from rag.chunker import TextChunker
from rag.vector_store import VectorStore
from rag.knowledge_base import KnowledgeBase
from rag.hybrid_search import HybridSearcher

__all__ = [
    "BaseEmbedder",
    "OpenAIEmbedder",
    "LocalEmbedder",
    "TextChunker",
    "VectorStore",
    "KnowledgeBase",
    "HybridSearcher",
]
