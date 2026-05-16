"""Tests for AnalystAgent (v2)."""

import pytest
from unittest.mock import patch

from agents.analyst_agent import AnalystAgent, AnalysisReport, SWOTAnalysis, FeatureComparison
from agents.collector_agent import CollectedData


def make_sample_data(competitor: str) -> CollectedData:
    """Helper to create sample collected data."""
    return CollectedData(
        competitor=competitor,
        basic_info={"company_name": competitor, "founded": "2020"},
        product_features={"features": [], "pricing": {}},
        market_performance={"metrics": {}, "growth_trends": []},
        user_reviews={"positive": [], "negative": []},
        strategic_news={"latest_news": [], "strategic_moves": []},
    )


class TestAnalystAgent:

    def test_init_with_provider(self, mock_provider):
        agent = AnalystAgent(provider=mock_provider)
        assert agent.provider == mock_provider

    def test_analyze_feature_comparison(self, mock_provider):
        """Should generate feature comparison from collected data."""
        mock_provider.responses = ["""[
          {"feature_name": "UI Design", "description": "User interface quality", "competitor_scores": {"A": 4, "B": 3}},
          {"feature_name": "Performance", "description": "Speed and stability", "competitor_scores": {"A": 3, "B": 5}}
        ]"""]
        agent = AnalystAgent(provider=mock_provider)
        data = [make_sample_data("A"), make_sample_data("B")]
        result = agent.analyze_feature_comparison(data)
        assert len(result) == 2
        assert result[0].feature_name == "UI Design"
        assert result[0].competitor_scores == {"A": 4, "B": 3}

    def test_analyze_swot(self, mock_provider_swot):
        """Should generate SWOT analysis per competitor."""
        agent = AnalystAgent(provider=mock_provider_swot)
        data = [make_sample_data("TestCo")]
        result = agent.analyze_swot(data)
        assert len(result) == 1
        assert result[0].competitor == "TestCo"
        assert len(result[0].strengths) == 3
        assert "Strong brand" in result[0].strengths

    def test_extract_key_insights(self, mock_provider_insights):
        """Should extract key insights from analysis."""
        agent = AnalystAgent(provider=mock_provider_insights)
        data = [make_sample_data("TestCo")]
        swot = [SWOTAnalysis(competitor="TestCo", strengths=["S1"], weaknesses=["W1"], opportunities=["O1"], threats=["T1"])]
        landscape = {"leader": "TestCo"}
        insights, risks, opportunities = agent.extract_key_insights(data, swot, landscape)
        assert len(insights) == 2
        assert "Market is growing rapidly" in insights

    def test_analyze_all_integration(self, mock_provider_swot, mock_provider_insights):
        """Should run complete analysis pipeline."""
        agent = AnalystAgent(provider=mock_provider_swot)
        data = [make_sample_data("TestCo")]

        comparisons = [FeatureComparison(feature_name="Speed", competitor_scores={"TestCo": 4})]
        swot = [SWOTAnalysis(competitor="TestCo", strengths=["Fast"], weaknesses=["Expensive"], opportunities=["AI"], threats=["Competition"])]
        landscape = {"leader": "TestCo"}

        with patch.object(agent, "analyze_feature_comparison", return_value=comparisons), \
             patch.object(agent, "analyze_swot", return_value=swot), \
             patch.object(agent, "analyze_competitive_landscape", return_value=landscape), \
             patch.object(agent, "extract_key_insights", return_value=(["Insight 1"], ["Risk 1"], ["Opportunity 1"])):
            result = agent.analyze_all(data)
            assert isinstance(result, AnalysisReport)
            assert "TestCo" in result.competitors
            assert len(result.key_insights) == 1
