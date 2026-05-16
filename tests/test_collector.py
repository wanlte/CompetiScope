"""Tests for CollectorAgent (async v2)."""

import pytest
from unittest.mock import patch, MagicMock

from agents.collector_agent import CollectorAgent, CollectedData


class TestCollectorAgent:
    """CollectorAgent tests."""

    def test_init_with_provider(self, mock_provider):
        with patch("tools.search_tool.DUCKDUCKGO_AVAILABLE", True):
            agent = CollectorAgent(provider=mock_provider)
            assert agent.provider == mock_provider

    @pytest.mark.asyncio
    async def test_collect_batch_concurrent(self, mock_provider):
        with patch("tools.search_tool.DUCKDUCKGO_AVAILABLE", True):
            agent = CollectorAgent(provider=mock_provider)

            async def fake_collect(competitor):
                return CollectedData(competitor=competitor)

            with patch.object(agent, "_collect_all_async", side_effect=fake_collect):
                result = await agent.collect_batch(["A", "B", "C"])
                assert len(result) == 3
                assert result[0].competitor == "A"
                assert result[1].competitor == "B"
                assert result[2].competitor == "C"

    @pytest.mark.asyncio
    async def test_collect_all_dimensions_concurrent(self, mock_provider):
        with patch("tools.search_tool.DUCKDUCKGO_AVAILABLE", True):
            agent = CollectorAgent(provider=mock_provider)

            basic = {"company_name": "Test", "founders": ["Alice"]}
            product = {"competitor": "Test", "features": []}
            market = {"competitor": "Test", "metrics": {}}
            reviews = {"competitor": "Test", "positive": []}
            news = {"competitor": "Test", "latest_news": []}

            with patch.object(agent, "_collect_basic_info_async", return_value=basic), \
                 patch.object(agent, "_collect_product_features_async", return_value=product), \
                 patch.object(agent, "_collect_market_performance_async", return_value=market), \
                 patch.object(agent, "_collect_user_reviews_async", return_value=reviews), \
                 patch.object(agent, "_collect_strategic_news_async", return_value=news):
                result = await agent._collect_all_async("Test")
                assert result.competitor == "Test"
                assert result.basic_info == basic
                assert result.product_features == product
                assert result.market_performance == market
                assert result.user_reviews == reviews
                assert result.strategic_news == news

    @pytest.mark.asyncio
    async def test_extract_basic_info_with_llm(self, mock_provider):
        mock_provider.responses = ['{"company_name": "TestCo", "founded": "2020", "ceo": "Alice"}']
        with patch("tools.search_tool.DUCKDUCKGO_AVAILABLE", True):
            agent = CollectorAgent(provider=mock_provider)
            result = await agent._extract_basic_info_with_llm_async("TestCo", [])
            assert result["company_name"] == "TestCo"
            assert result["founded"] == "2020"

    @pytest.mark.asyncio
    async def test_error_isolation_in_batch(self, mock_provider):
        with patch("tools.search_tool.DUCKDUCKGO_AVAILABLE", True):
            agent = CollectorAgent(provider=mock_provider)

            async def raise_on_c(c):
                if c == "C":
                    raise RuntimeError("Search failed for C")
                return CollectedData(competitor=c)

            with patch.object(agent, "_collect_all_async", side_effect=raise_on_c):
                result = await agent.collect_batch(["A", "B", "C"])
                assert len(result) == 2  # C is excluded
                assert result[0].competitor == "A"
                assert result[1].competitor == "B"
