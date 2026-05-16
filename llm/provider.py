from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from loguru import logger

from .types import LLMResponse, TokenUsage, ProviderConfig


class BaseLLMProvider(ABC):
    """Abstract base for LLM providers. Supports OpenAI-compatible APIs."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._total_usage = TokenUsage(model=config.model)

    @abstractmethod
    async def ainvoke(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Async single invocation."""

    @abstractmethod
    async def astream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """Async streaming invocation. Yields content chunks."""

    @property
    def total_usage(self) -> TokenUsage:
        return self._total_usage

    def _accumulate_usage(self, usage: TokenUsage):
        self._total_usage.input_tokens += usage.input_tokens
        self._total_usage.output_tokens += usage.output_tokens
        self._total_usage.total_tokens += usage.total_tokens
        self._total_usage.cost_usd += usage.cost_usd

    def get_cost_summary(self) -> dict:
        return {
            "model": self._total_usage.model,
            "total_input_tokens": self._total_usage.input_tokens,
            "total_output_tokens": self._total_usage.output_tokens,
            "total_tokens": self._total_usage.total_tokens,
            "total_cost_usd": round(self._total_usage.cost_usd, 6),
        }

    @staticmethod
    def _convert_messages(messages: list) -> list[dict]:
        """Convert LangChain message objects to plain dicts."""
        converted = []
        for msg in messages:
            if hasattr(msg, "type"):
                role = msg.type
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = "user"
                content = str(msg)
            converted.append({"role": role, "content": content})
        return converted
