"""Tests for FastAPI application and endpoints.

Tests that need real API access are skipped when DEEPSEEK_API_KEY is not set.
"""

import os
import pytest
from fastapi.testclient import TestClient


# Check if real API key is available
HAS_API_KEY = bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def client(monkeypatch):
    """Fresh TestClient per test. Sets a mock API key for schema/routing tests
    so that FastAPI validation + task creation don't crash on missing credentials.
    The mock key won't work for real API calls, but that's fine for routing tests."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "mock-test-key-for-routing")
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key-for-routing")
    from api.server import create_app
    app = create_app()
    return TestClient(app)


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"

    def test_health_has_llm_status(self, client):
        r = client.get("/health")
        assert "llm_configured" in r.json()
        assert "kb_enabled" in r.json()


class TestCosts:
    def test_costs_ok(self, client):
        r = client.get("/api/v1/costs")
        assert r.status_code == 200
        data = r.json()
        assert "total_cost_usd" in data
        assert "total_tokens" in data


class TestMetrics:
    def test_metrics_ok(self, client):
        r = client.get("/api/v1/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "total_analyses" in data
        assert "success_rate" in data


class TestAnalysisRouteValidation:
    def test_empty_competitors_rejected(self, client):
        r = client.post("/api/v1/analyze", json={"competitors": []})
        assert r.status_code == 422

    def test_missing_competitors_rejected(self, client):
        r = client.post("/api/v1/analyze", json={})
        assert r.status_code == 422

    def test_invalid_report_type_rejected(self, client):
        r = client.post("/api/v1/analyze", json={
            "competitors": ["Foo"],
            "report_type": "invalid_type",
        })
        assert r.status_code == 422

    def test_task_not_found(self, client):
        r = client.get("/api/v1/task/nonexistent-id")
        assert r.status_code == 404


class TestAnalysisRouteWithMockKey:
    """Test routes that need agent initialization but not real API calls."""

    def test_create_task_returns_200(self, client):
        r = client.post("/api/v1/analyze", json={
            "competitors": ["TestCoA"],
            "report_type": "snapshot",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["task_id"]
        assert data["competitors"] == ["TestCoA"]
        assert data["report_type"] == "snapshot"
        # Task should be pending or running (background task starts immediately)
        assert data["status"] in ("pending", "running")

    def test_get_created_task(self, client):
        r = client.post("/api/v1/analyze", json={"competitors": ["Foo"]})
        task_id = r.json()["task_id"]

        r2 = client.get(f"/api/v1/task/{task_id}")
        assert r2.status_code == 200
        assert r2.json()["task_id"] == task_id
        assert "error" in r2.json()  # May have error from failed API call, but task exists

    def test_list_tasks(self, client):
        client.post("/api/v1/analyze", json={"competitors": ["A"]})
        client.post("/api/v1/analyze", json={"competitors": ["B"]})
        r = client.get("/api/v1/tasks")
        assert r.status_code == 200
        assert r.json()["total"] >= 2

    def test_list_tasks_filtered(self, client):
        client.post("/api/v1/analyze", json={"competitors": ["A"]})
        r = client.get("/api/v1/tasks?status=pending")
        assert r.status_code == 200
        for t in r.json()["tasks"]:
            assert t["status"] == "pending"

    def test_stream_endpoint_returns_sse(self, client):
        """Streaming endpoint should return text/event-stream even if analysis fails."""
        r = client.post("/api/v1/analyze/stream", json={
            "competitors": ["TestCo"],
            "report_type": "snapshot",
        })
        # Even with mock key, the endpoint starts streaming (may fail later)
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")


class TestKnowledgeRoutes:
    def test_search_knowledge(self, client):
        r = client.post("/api/v1/knowledge/search", json={
            "query": "Notion pricing",
            "n_results": 3,
        })
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "query" in data

    def test_get_competitor_history(self, client):
        r = client.get("/api/v1/knowledge/competitors/Notion")
        assert r.status_code == 200
        data = r.json()
        assert data["competitor"] == "Notion"
        assert "data" in data

    def test_search_empty_query_rejected(self, client):
        r = client.post("/api/v1/knowledge/search", json={"query": ""})
        assert r.status_code == 422
