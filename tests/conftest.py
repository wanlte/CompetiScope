"""Test fixtures for CompetiScope."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncIterator

from llm.provider import BaseLLMProvider
from llm.types import LLMResponse, TokenUsage, ProviderConfig


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing — no API calls needed."""

    def __init__(self, responses: list[str] | None = None):
        config = ProviderConfig(
            api_key="mock-key",
            base_url="https://mock.api",
            model="mock-model",
        )
        super().__init__(config)
        self.responses = responses or ["mock response"]
        self.call_count = 0
        self.call_history: list[list[dict]] = []

    async def ainvoke(self, messages: list[dict], **kwargs) -> LLMResponse:
        self.call_count += 1
        self.call_history.append(messages)
        idx = min(self.call_count - 1, len(self.responses) - 1)
        usage = TokenUsage(
            input_tokens=10,
            output_tokens=len(self.responses[idx]),
            total_tokens=10 + len(self.responses[idx]),
            model="mock-model",
        )
        self._accumulate_usage(usage)
        return LLMResponse(content=self.responses[idx], model="mock-model", usage=usage)

    async def astream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        self.call_count += 1
        idx = min(self.call_count - 1, len(self.responses) - 1)
        for word in self.responses[idx].split():
            yield word + " "


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider with default responses."""
    return MockLLMProvider(responses=['{"test": "ok"}'])


@pytest.fixture
def mock_provider_swot():
    """Create a mock provider that returns valid SWOT JSON."""
    swot_json = """{
  "competitor": "TestCo",
  "strengths": ["Strong brand", "Innovative tech", "Great team"],
  "weaknesses": ["High price", "Limited market", "Slow support"],
  "opportunities": ["Market expansion", "AI integration", "New segments"],
  "threats": ["New entrants", "Price wars", "Regulation"]
}"""
    return MockLLMProvider(responses=[swot_json])


@pytest.fixture
def mock_provider_insights():
    """Create a mock provider that returns insights JSON."""
    insights_json = """{
  "key_insights": ["Market is growing rapidly", "Competition intensifying"],
  "risks": ["Supply chain disruption", "Talent shortage"],
  "opportunities": ["Untapped SMB market", "International expansion"]
}"""
    return MockLLMProvider(responses=[insights_json])


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {"title": "Test Result 1", "url": "https://example.com/1", "snippet": "Test snippet 1", "source": "mock"},
        {"title": "Test Result 2", "url": "https://example.com/2", "snippet": "Test snippet 2", "source": "mock"},
    ]
