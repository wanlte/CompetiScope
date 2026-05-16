"""Analysis metrics collector — tracks success rates, latency, and throughput.

Global singleton `metrics` for aggregating operational metrics across analyses.
"""

import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class AnalysisRecord:
    """Record of a single analysis run."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    competitor_count: int = 0
    success: bool = False
    duration_seconds: float = 0.0
    agent_steps: int = 0
    reflection_rounds: int = 0
    error: Optional[str] = None


class MetricsCollector:
    """Collect and aggregate operational metrics for analyses.

    Usage:
        metrics = MetricsCollector()
        metrics.start_analysis(competitor_count=3)
        # ... run analysis ...
        metrics.finish_analysis(success=True, agent_steps=8, reflection_rounds=2)
        print(metrics.summary())
    """

    def __init__(self, max_history: int = 100):
        self._records: list[AnalysisRecord] = []
        self._max_history = max_history
        self._running_count = 0
        self._start_times: dict[int, float] = {}
        self._analysis_id = 0

    def start_analysis(self, competitor_count: int = 0) -> int:
        """Begin tracking an analysis. Returns analysis_id."""
        self._analysis_id += 1
        aid = self._analysis_id
        self._start_times[aid] = time.time()
        self._running_count += 1
        return aid

    def finish_analysis(
        self,
        analysis_id: int,
        success: bool = True,
        agent_steps: int = 0,
        reflection_rounds: int = 0,
        error: Optional[str] = None,
        competitor_count: int = 0,
    ):
        """Record completion of an analysis."""
        start = self._start_times.pop(analysis_id, time.time())
        self._running_count -= 1

        rec = AnalysisRecord(
            competitor_count=competitor_count,
            success=success,
            duration_seconds=round(time.time() - start, 2),
            agent_steps=agent_steps,
            reflection_rounds=reflection_rounds,
            error=error,
        )
        self._records.append(rec)
        if len(self._records) > self._max_history:
            self._records.pop(0)

    @property
    def total_analyses(self) -> int:
        return len(self._records)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self._records if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self._records if not r.success)

    @property
    def success_rate(self) -> float:
        if not self._records:
            return 1.0
        return round(self.success_count / len(self._records), 4)

    @property
    def avg_duration_seconds(self) -> float:
        if not self._records:
            return 0.0
        return round(sum(r.duration_seconds for r in self._records) / len(self._records), 2)

    @property
    def avg_agent_steps(self) -> float:
        if not self._records:
            return 0.0
        return round(sum(r.agent_steps for r in self._records) / len(self._records), 1)

    @property
    def running_count(self) -> int:
        return self._running_count

    def summary(self) -> dict:
        """Return a summary dict for API / dashboard."""
        return {
            "total_analyses": self.total_analyses,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "avg_duration_seconds": self.avg_duration_seconds,
            "avg_agent_steps": self.avg_agent_steps,
            "running_count": self.running_count,
            "recent": [
                {
                    "timestamp": r.timestamp,
                    "competitor_count": r.competitor_count,
                    "success": r.success,
                    "duration": r.duration_seconds,
                    "agent_steps": r.agent_steps,
                }
                for r in self._records[-5:]
            ],
        }

    def reset(self):
        self._records.clear()
        self._running_count = 0
        self._start_times.clear()
        self._analysis_id = 0


# Global singleton
metrics = MetricsCollector()
