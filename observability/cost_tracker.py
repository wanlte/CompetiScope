"""LLM cost tracking — intercepts provider calls and records token usage.

Provides a global singleton `cost_tracker` for per-session tracking,
plus per-call record keeping for detailed cost breakdown.
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class CostRecord:
    """Single LLM call cost record."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    call_type: str = ""  # e.g. "collect", "analyze", "reflect", "write"


class CostTracker:
    """Track LLM costs across a session.

    Thread-safe accumulation of token usage and cost. Attach as an observer
    to BaseLLMProvider or call `record()` explicitly after each invocation.

    Usage:
        tracker = CostTracker()
        tracker.record(usage, call_type="analyze")
        print(tracker.summary())
    """

    def __init__(self):
        self._records: list[CostRecord] = []
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0

    def record(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        cost_usd: float = 0.0,
        call_type: str = "",
    ):
        total = input_tokens + output_tokens
        rec = CostRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            cost_usd=cost_usd,
            call_type=call_type,
        )
        self._records.append(rec)
        self._total_input += input_tokens
        self._total_output += output_tokens
        self._total_cost += cost_usd

    def record_from_usage(self, usage, call_type: str = ""):
        """Record from a TokenUsage or usage dict."""
        if hasattr(usage, "input_tokens"):
            self.record(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                model=getattr(usage, "model", ""),
                cost_usd=getattr(usage, "cost_usd", 0.0),
                call_type=call_type,
            )
        elif isinstance(usage, dict):
            self.record(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                model=usage.get("model", ""),
                call_type=call_type,
            )

    @property
    def total_input_tokens(self) -> int:
        return self._total_input

    @property
    def total_output_tokens(self) -> int:
        return self._total_output

    @property
    def total_tokens(self) -> int:
        return self._total_input + self._total_output

    @property
    def total_cost_usd(self) -> float:
        return round(self._total_cost, 6)

    @property
    def call_count(self) -> int:
        return len(self._records)

    def summary(self) -> dict:
        """Return a summary dict for API responses / CLI display."""
        by_type: dict[str, dict] = {}
        for r in self._records:
            if r.call_type not in by_type:
                by_type[r.call_type] = {"calls": 0, "tokens": 0, "cost": 0.0}
            by_type[r.call_type]["calls"] += 1
            by_type[r.call_type]["tokens"] += r.total_tokens
            by_type[r.call_type]["cost"] += r.cost_usd

        return {
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "call_count": self.call_count,
            "by_type": by_type,
            "records": [
                {
                    "timestamp": r.timestamp,
                    "model": r.model,
                    "tokens": r.total_tokens,
                    "cost": r.cost_usd,
                    "type": r.call_type,
                }
                for r in self._records
            ],
        }

    def reset(self):
        """Reset all counters for a new session."""
        self._records.clear()
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0


# Global singleton
cost_tracker = CostTracker()
