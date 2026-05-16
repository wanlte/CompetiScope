"""Analysis API routes — POST /analyze, GET /task/{id}, GET /tasks, POST /analyze/stream."""

import asyncio
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from api.schemas import (
    AnalysisRequest,
    TaskResponse,
    TaskListResponse,
)
from core.exceptions import TaskNotFoundError
from agents.manager_agent import ManagerAgent
from observability.metrics import metrics
from observability.cost_tracker import cost_tracker

router = APIRouter(prefix="/api/v1", tags=["analysis"])

# In-memory task store (replace with DB for production)
_tasks: dict[str, dict] = {}


def _make_agent():
    """Create a fresh ManagerAgent. Centralized so we can swap impl later."""
    return ManagerAgent()


@router.post("/analyze", response_model=TaskResponse)
async def start_analysis(req: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start an async competitor analysis. Returns immediately with task_id."""
    agent = _make_agent()
    dimensions = req.dimensions or ["产品功能", "市场表现", "用户评价", "战略动态", "商业模式"]

    task_id = agent._generate_task_id()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_doc = {
        "task_id": task_id,
        "status": "pending",
        "competitors": req.competitors,
        "report_type": req.report_type,
        "created_at": now,
        "completed_at": None,
        "error": None,
        "result": None,
    }
    _tasks[task_id] = task_doc

    # Track metrics
    analysis_id = metrics.start_analysis(competitor_count=len(req.competitors))

    async def _run():
        task_doc["status"] = "running"
        cost_tracker.reset()
        try:
            result = await agent.analyze_async(
                competitors=req.competitors,
                analysis_dimensions=dimensions,
                report_type=req.report_type,
                our_product=req.our_product,
                show_progress=False,
            )
            task_doc["result"] = {
                "report": result.get("report", ""),
                "agent_steps": result.get("agent_steps", 0),
                "reflection_rounds": result.get("reflection_rounds", 0),
                "reflection_scores": result.get("reflection_scores", []),
                "cost": result.get("cost", {}),
            }
            task_doc["status"] = "completed" if result.get("success") else "failed"
            task_doc["error"] = result.get("error")

            metrics.finish_analysis(
                analysis_id=analysis_id,
                success=result.get("success", False),
                agent_steps=result.get("agent_steps", 0),
                reflection_rounds=result.get("reflection_rounds", 0),
                error=result.get("error"),
                competitor_count=len(req.competitors),
            )
        except Exception as exc:
            logger.error(f"Analysis task {task_id} failed: {exc}")
            task_doc["status"] = "failed"
            task_doc["error"] = str(exc)
            metrics.finish_analysis(analysis_id=analysis_id, success=False, error=str(exc))

        task_doc["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    background_tasks.add_task(_run)
    return TaskResponse(**task_doc)


@router.get("/task/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get task status and results by ID."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return TaskResponse(**_tasks[task_id])


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(limit: int = 20, status: Optional[str] = None):
    """List recent tasks, optionally filtered by status."""
    filtered = list(_tasks.values())
    if status:
        filtered = [t for t in filtered if t["status"] == status]
    filtered.sort(key=lambda t: t["created_at"], reverse=True)
    return TaskListResponse(
        tasks=[TaskResponse(**t) for t in filtered[:limit]],
        total=len(filtered),
    )


@router.post("/analyze/stream")
async def stream_analysis(req: AnalysisRequest):
    """Run analysis with SSE streaming progress updates.

    Events:
      - progress: {phase, message}
      - step: {step_num, thought, action}
      - reflection: {round, score}
      - cost: {total_cost_usd, total_tokens}
      - result: {report (chunked)}
      - done: {success, task_id, summary}
    """
    agent = _make_agent()
    dimensions = req.dimensions or ["产品功能", "市场表现", "用户评价", "战略动态", "商业模式"]

    async def event_stream():
        task_id = agent._generate_task_id()
        analysis_id = metrics.start_analysis(competitor_count=len(req.competitors))
        cost_tracker.reset()

        try:
            yield {"event": "progress", "data": json.dumps({"phase": "start", "message": f"开始分析: {', '.join(req.competitors)}", "task_id": task_id}, ensure_ascii=False)}

            result = await agent.analyze_async(
                competitors=req.competitors,
                analysis_dimensions=dimensions,
                report_type=req.report_type,
                our_product=req.our_product,
                show_progress=False,
            )

            if result.get("agent_step_details"):
                for step_detail in result["agent_step_details"]:
                    yield {"event": "step", "data": json.dumps(step_detail, ensure_ascii=False)}

            if result.get("reflection_scores"):
                for i, score in enumerate(result["reflection_scores"]):
                    yield {"event": "reflection", "data": json.dumps({"round": i + 1, "score": score}, ensure_ascii=False)}

            if result.get("cost"):
                yield {"event": "cost", "data": json.dumps(result["cost"], ensure_ascii=False)}

            if result.get("report"):
                # Stream report in chunks
                report = result["report"]
                chunk_size = 500
                for i in range(0, len(report), chunk_size):
                    chunk = report[i:i + chunk_size]
                    yield {"event": "result", "data": json.dumps({"chunk": chunk, "index": i // chunk_size}, ensure_ascii=False)}
                    await asyncio.sleep(0.05)  # small delay for readable streaming

            success = result.get("success", False)
            metrics.finish_analysis(
                analysis_id=analysis_id,
                success=success,
                agent_steps=result.get("agent_steps", 0),
                reflection_rounds=result.get("reflection_rounds", 0),
                error=result.get("error"),
                competitor_count=len(req.competitors),
            )

            yield {"event": "done", "data": json.dumps({
                "success": success,
                "task_id": task_id,
                "agent_steps": result.get("agent_steps", 0),
                "cost": result.get("cost", {}),
            }, ensure_ascii=False)}

        except Exception as exc:
            logger.error(f"Stream analysis failed: {exc}")
            metrics.finish_analysis(analysis_id=analysis_id, success=False, error=str(exc))
            yield {"event": "error", "data": json.dumps({"error": str(exc)}, ensure_ascii=False)}

    return EventSourceResponse(event_stream())
