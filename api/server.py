"""FastAPI application factory for CompetiScope.

Usage:
    uvicorn api.server:app --reload
    python -m api.server
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is on sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes.analysis import router as analysis_router
from api.routes.knowledge import router as knowledge_router
from api.schemas import HealthResponse, CostSummary, MetricsSummary
from observability.cost_tracker import cost_tracker
from observability.metrics import metrics
from core.config import get_config


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    cfg = get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("CompetiScope API v2.0 starting up")
        logger.info(f"  LLM configured: {bool(cfg.llm_api_key)}")
        logger.info(f"  KB enabled: {cfg.kb_enabled}")
        logger.info(f"  Docs: http://{cfg.api_host}:{cfg.api_port}/docs")
        yield
        logger.info("CompetiScope API shutting down")

    app = FastAPI(
        title="CompetiScope API",
        description="Enterprise Competitor Analysis Agent — ReAct Agent + RAG + Async",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    origins = cfg.api_cors_origins.split(",") if cfg.api_cors_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(analysis_router)
    app.include_router(knowledge_router)

    # Health + observability endpoints
    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            llm_configured=bool(cfg.llm_api_key),
            kb_enabled=cfg.kb_enabled,
        )

    @app.get("/api/v1/costs", response_model=CostSummary)
    async def get_costs():
        """Get current session cost summary."""
        s = cost_tracker.summary()
        return CostSummary(
            total_input_tokens=s["total_input_tokens"],
            total_output_tokens=s["total_output_tokens"],
            total_tokens=s["total_tokens"],
            total_cost_usd=s["total_cost_usd"],
            call_count=s["call_count"],
        )

    @app.get("/api/v1/metrics", response_model=MetricsSummary)
    async def get_metrics():
        """Get operational metrics summary."""
        s = metrics.summary()
        return MetricsSummary(**s)

    return app


# Module-level app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run("api.server:app", host=cfg.api_host, port=cfg.api_port, reload=True)
