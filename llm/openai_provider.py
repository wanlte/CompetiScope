import asyncio
from typing import AsyncIterator, Optional
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from .provider import BaseLLMProvider
from .types import LLMResponse, TokenUsage, ProviderConfig


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible async LLM provider (works with DeepSeek, OpenAI, etc.)."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,  # We handle retries ourselves
        )
        self._semaphore = asyncio.Semaphore(5)  # Rate limiting: max 5 concurrent
        logger.info(f"OpenAIProvider initialized: model={config.model}, base_url={config.base_url}")

    async def ainvoke(self, messages: list[dict], **kwargs) -> LLMResponse:
        async with self._semaphore:
            response = await self._call_with_retry(messages, stream=False, **kwargs)
            return self._parse_response(response)

    async def astream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        async with self._semaphore:
            stream = await self._call_with_retry(messages, stream=True, **kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def _call_with_retry(self, messages: list[dict], stream: bool = False, **kwargs):
        converted = self._convert_messages(messages)
        model = kwargs.pop("model", self.config.model)
        temperature = kwargs.pop("temperature", self.config.temperature)
        max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)

        return await self._client.chat.completions.create(
            model=model,
            messages=converted,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

    def _parse_response(self, response: ChatCompletion) -> LLMResponse:
        usage_dict = response.usage.model_dump() if response.usage else {}
        usage = TokenUsage.from_response(
            model=response.model,
            usage=usage_dict,
            pricing=self.config.pricing_dict,
        )
        self._accumulate_usage(usage)
        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage=usage,
            finish_reason=response.choices[0].finish_reason,
        )
