"""Tests for SSE streaming and related functionality."""

import pytest
import asyncio
import json

from agents.writer_agent import WriterAgent


@pytest.fixture
def writer(mock_provider):
    """Writer agent with mock provider."""
    return WriterAgent(provider=mock_provider)


class TestWriterStreaming:
    def test_astream_full_report_yields_sections(self, writer):
        """astream_full_report should yield all sections with proper JSON."""
        from agents.analyst_agent import AnalysisReport, SWOTAnalysis, FeatureComparison

        report = AnalysisReport(
            competitors=["TestCo"],
            swot_analysis=[
                SWOTAnalysis(
                    competitor="TestCo",
                    strengths=["Good product"],
                    weaknesses=["Slow growth"],
                    opportunities=["New market"],
                    threats=["Competition"],
                )
            ],
            feature_comparison=[
                FeatureComparison(
                    feature_name="Pricing",
                    competitor_scores={"TestCo": 4},
                )
            ],
            key_insights=["Insight 1"],
            risks=["Risk 1"],
            opportunities=["Opportunity 1"],
            competitive_landscape={
                "leader": "TestCo",
                "market_gaps": ["Gap 1"],
            },
        )
        collected_data = []

        async def _collect():
            sections = []
            async for chunk in writer.astream_full_report(report, collected_data):
                sections.append(chunk)
            return sections

        sections = asyncio.run(_collect())

        assert len(sections) >= 8  # cover + 10 report sections
        for s in sections:
            parsed = json.loads(s)
            assert "section" in parsed
            assert "content" in parsed

    def test_stream_sections_have_expected_names(self, writer):
        """All expected section names should be present in streaming output."""
        from agents.analyst_agent import AnalysisReport, SWOTAnalysis, FeatureComparison

        report = AnalysisReport(
            competitors=["TestCo"],
            swot_analysis=[
                SWOTAnalysis(
                    competitor="TestCo",
                    strengths=["A"],
                    weaknesses=["B"],
                    opportunities=["C"],
                    threats=["D"],
                )
            ],
            feature_comparison=[],
            key_insights=[],
            risks=[],
            opportunities=[],
            competitive_landscape={},
        )

        async def _run():
            names = []
            async for chunk in writer.astream_full_report(report, []):
                names.append(json.loads(chunk)["section"])
            return names

        names = asyncio.run(_run())
        assert "cover" in names
        assert "executive_summary" in names
        assert "swot" in names
        assert "strategy" in names
        assert "appendix" in names

    def test_stream_handles_empty_data(self, writer):
        """Streaming should not crash with minimal/empty report data."""
        from agents.analyst_agent import AnalysisReport

        report = AnalysisReport(
            competitors=["EmptyCo"],
            swot_analysis=[],
            feature_comparison=[],
            key_insights=[],
            risks=[],
            opportunities=[],
            competitive_landscape={},
        )

        async def _run():
            count = 0
            async for _ in writer.astream_full_report(report, []):
                count += 1
            return count

        count = asyncio.run(_run())
        assert count > 0  # Still yields sections, just empty-ish


class TestObservability:
    def test_cost_tracker_record(self):
        from observability.cost_tracker import CostTracker
        tracker = CostTracker()
        tracker.record(input_tokens=100, output_tokens=50, model="test", call_type="analyze")
        assert tracker.total_tokens == 150

    def test_cost_tracker_summary(self):
        from observability.cost_tracker import CostTracker
        tracker = CostTracker()
        tracker.record(input_tokens=10, output_tokens=5, call_type="collect")
        tracker.record(input_tokens=20, output_tokens=10, call_type="analyze")
        s = tracker.summary()
        assert s["call_count"] == 2
        assert "collect" in s["by_type"]
        assert "analyze" in s["by_type"]

    def test_cost_tracker_reset(self):
        from observability.cost_tracker import CostTracker
        tracker = CostTracker()
        tracker.record(input_tokens=100, output_tokens=50)
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.call_count == 0

    def test_metrics_collector_start_finish(self):
        from observability.metrics import MetricsCollector
        mc = MetricsCollector()
        aid = mc.start_analysis(competitor_count=3)
        mc.finish_analysis(aid, success=True, agent_steps=8, reflection_rounds=2)
        assert mc.total_analyses == 1
        assert mc.success_count == 1
        assert mc.success_rate == 1.0

    def test_metrics_collector_failure(self):
        from observability.metrics import MetricsCollector
        mc = MetricsCollector()
        aid = mc.start_analysis()
        mc.finish_analysis(aid, success=False, error="test error")
        assert mc.total_analyses == 1
        assert mc.failure_count == 1
        assert mc.success_rate == 0.0

    def test_metrics_collector_summary(self):
        from observability.metrics import MetricsCollector
        mc = MetricsCollector()
        for _ in range(3):
            aid = mc.start_analysis()
            mc.finish_analysis(aid, success=True)
        s = mc.summary()
        assert s["total_analyses"] == 3
        assert s["success_rate"] == 1.0
        assert "recent" in s


class TestCoreConfig:
    def test_app_config_defaults(self):
        from core.config import AppConfig
        cfg = AppConfig()
        assert cfg.llm_model == "deepseek-chat"
        assert cfg.llm_base_url == "https://api.deepseek.com"
        assert cfg.api_host == "0.0.0.0"
        assert cfg.api_port == 8000

    def test_app_config_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_MODEL", "custom-model")
        monkeypatch.setenv("API_PORT", "9000")
        from core.config import AppConfig
        cfg = AppConfig()
        assert cfg.llm_model == "custom-model"
        assert cfg.api_port == 9000

    def test_get_config_singleton(self):
        from core.config import get_config, AppConfig
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2
        assert isinstance(cfg1, AppConfig)


class TestCoreExceptions:
    def test_exception_hierarchy(self):
        from core.exceptions import (
            CompetiScopeError, ConfigError, AgentError, ToolError, APIError, TaskNotFoundError
        )
        assert issubclass(ConfigError, CompetiScopeError)
        assert issubclass(AgentError, CompetiScopeError)
        assert issubclass(ToolError, AgentError)
        assert issubclass(APIError, CompetiScopeError)
        assert issubclass(TaskNotFoundError, APIError)
