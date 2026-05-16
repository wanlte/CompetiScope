"""API request/response Pydantic models."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ---- Request models ----

class AnalysisRequest(BaseModel):
    competitors: list[str] = Field(..., min_length=1, description="竞品名称列表")
    report_type: str = Field(default="full", pattern="^(full|summary|snapshot)$")
    dimensions: Optional[list[str]] = Field(default=None, description="分析维度")
    our_product: Optional[str] = Field(default=None, description="我方产品名")


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: Optional[str] = Field(default=None, description="可选: competitor_data, analysis_reports, search_cache")
    n_results: int = Field(default=5, ge=1, le=20)


# ---- Response models ----

class TaskResponse(BaseModel):
    task_id: str
    status: str  # pending, running, completed, failed
    competitors: list[str] = []
    report_type: str = "full"
    created_at: str = ""
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    total: int


class CostSummary(BaseModel):
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0


class MetricsSummary(BaseModel):
    total_analyses: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 1.0
    avg_duration_seconds: float = 0.0
    avg_agent_steps: float = 0.0
    running_count: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_configured: bool = False
    kb_enabled: bool = False


class KnowledgeSearchResult(BaseModel):
    id: str
    content: str
    distance: float = 0.0
    collection: str = ""


class KnowledgeSearchResponse(BaseModel):
    query: str
    results: list[KnowledgeSearchResult]
    total: int
