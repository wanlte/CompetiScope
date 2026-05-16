"""Knowledge API routes — POST /knowledge/search, GET /knowledge/competitors/{name}."""

from fastapi import APIRouter
from loguru import logger

from api.schemas import (
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeSearchResult,
)
from rag.knowledge_base import KnowledgeBase
from rag.hybrid_search import HybridSearcher
from config.settings import KnowledgeBaseConfig

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


def _make_kb() -> KnowledgeBase:
    return KnowledgeBase(
        persist_dir=KnowledgeBaseConfig.PERSIST_DIR,
        enabled=KnowledgeBaseConfig.ENABLED,
    )


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(req: KnowledgeSearchRequest):
    """Search across knowledge base collections.

    If `collection` is specified, searches only that collection.
    Otherwise searches all three (competitor_data, analysis_reports, search_cache).
    """
    kb = _make_kb()
    if not kb.enabled:
        return KnowledgeSearchResponse(query=req.query, results=[], total=0)

    results = []
    try:
        if req.collection:
            hits = await kb.search_similar(req.query, req.collection, n_results=req.n_results)
            results = [
                KnowledgeSearchResult(
                    id=h.get("id", ""),
                    content=h.get("content", "")[:500],
                    distance=h.get("distance", 0.0),
                    collection=req.collection,
                )
                for h in hits
            ]
        else:
            all_hits = await kb.search_all(req.query, n_results=req.n_results)
            for col_name, hits in all_hits.items():
                for h in hits:
                    results.append(KnowledgeSearchResult(
                        id=h.get("id", ""),
                        content=h.get("content", "")[:500],
                        distance=h.get("distance", 0.0),
                        collection=col_name,
                    ))
            results.sort(key=lambda r: r.distance)
            results = results[:req.n_results]
    except Exception as exc:
        logger.error(f"Knowledge search failed: {exc}")
        return KnowledgeSearchResponse(query=req.query, results=[], total=0)

    return KnowledgeSearchResponse(query=req.query, results=results, total=len(results))


@router.get("/competitors/{name}")
async def get_competitor_history(name: str):
    """Get knowledge base history for a specific competitor."""
    kb = _make_kb()
    if not kb.enabled:
        return {"competitor": name, "data": {}, "message": "Knowledge base is disabled"}

    try:
        history = kb.get_competitor_history(name, n_results=10)
        # Truncate content for API response
        truncated = {}
        for col, items in history.items():
            truncated[col] = [
                {"id": i.get("id", ""), "content": i.get("content", "")[:400], "distance": i.get("distance", 0.0)}
                for i in items
            ]
        return {"competitor": name, "data": truncated, "total_items": sum(len(v) for v in truncated.values())}
    except Exception as exc:
        logger.error(f"Competitor history lookup failed: {exc}")
        return {"competitor": name, "data": {}, "error": str(exc)}
