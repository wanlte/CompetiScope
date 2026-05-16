from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any
from datetime import datetime


class TokenUsage(BaseModel):
    """Token usage tracking for cost monitoring."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_response(cls, model: str, usage: dict, pricing: dict | None = None) -> "TokenUsage":
        input_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else getattr(usage, "prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else getattr(usage, "completion_tokens", 0)
        total = input_tokens + output_tokens
        cost = 0.0
        if pricing:
            cost = (input_tokens * pricing.get("input_per_1k", 0) + output_tokens * pricing.get("output_per_1k", 0)) / 1000
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            model=model,
            cost_usd=cost,
        )


class LLMResponse(BaseModel):
    """Standardized LLM response wrapper."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: Optional[str] = None
    raw_response: Any = None


class ProviderConfig(BaseModel):
    """Provider-level configuration."""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 2.0
    request_interval: float = 1.0

    # Cost tracking (USD per 1K tokens)
    input_price_per_1k: float = 0.00014   # DeepSeek pricing
    output_price_per_1k: float = 0.00028

    @property
    def pricing_dict(self) -> dict:
        return {
            "input_per_1k": self.input_price_per_1k,
            "output_per_1k": self.output_price_per_1k,
        }
