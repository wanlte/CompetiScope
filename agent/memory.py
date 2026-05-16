"""Agent memory system — conversation history with sliding window and working storage."""

from dataclasses import dataclass, field
from typing import Any
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)


@dataclass
class ConversationMemory:
    """Sliding-window conversation memory.

    Stores the full agent conversation and automatically summarizes
    older messages when the window exceeds the configured size.
    """

    max_messages: int = 20
    summary_trigger: int = 15  # start summarizing when exceeding this

    _messages: list[BaseMessage] = field(default_factory=list)
    _summary: str = ""

    # ---- write ----

    def add_system(self, content: str) -> None:
        self._messages.append(SystemMessage(content=content))

    def add_user(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))

    def add_assistant(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))

    def add_tool_result(self, tool_name: str, result: str) -> None:
        self._messages.append(ToolMessage(content=result, tool_call_id=tool_name))

    def set_summary(self, summary: str) -> None:
        self._summary = summary

    # ---- read ----

    def get_messages(self) -> list[BaseMessage]:
        """Return messages with sliding window applied."""
        msgs = list(self._messages)
        if len(msgs) <= self.max_messages:
            return msgs

        system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
        non_system = [m for m in msgs if not isinstance(m, SystemMessage)]

        keep = self.max_messages - len(system_msgs) - (1 if self._summary else 0)
        recent = non_system[-keep:]

        result = list(system_msgs)
        if self._summary:
            result.append(SystemMessage(
                content=f"[Summarized earlier conversation]\n{self._summary}"
            ))
        result.extend(recent)
        return result

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""


@dataclass
class WorkingMemory:
    """Structured storage for intermediate agent results.

    Holds search results, collected competitor data, and analysis
    outputs that the agent produces during its run.
    """

    search_cache: dict[str, list[dict]] = field(default_factory=dict)
    collected_data: dict[str, dict] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    task_state: dict[str, Any] = field(default_factory=dict)

    # ---- search ----

    def store_search(self, query: str, results: list[dict]) -> None:
        self.search_cache[query] = results

    def get_search(self, query: str) -> list[dict] | None:
        return self.search_cache.get(query)

    # ---- collected ----

    def store_collected(self, competitor: str, data: dict) -> None:
        self.collected_data[competitor] = data

    def get_collected(self, competitor: str) -> dict | None:
        return self.collected_data.get(competitor)

    @property
    def all_competitors(self) -> list[str]:
        return list(self.collected_data.keys())

    # ---- analysis ----

    def store_analysis(self, competitor: str, result: Any) -> None:
        self.analysis[competitor] = result

    def get_analysis(self, competitor: str) -> Any | None:
        return self.analysis.get(competitor)

    # ---- overview ----

    def snapshot(self) -> dict:
        """Return a summary of current working memory state."""
        return {
            "competitors_collected": self.all_competitors,
            "search_queries": list(self.search_cache.keys()),
            "analyzed_competitors": list(self.analysis.keys()),
        }

    def clear(self) -> None:
        self.search_cache.clear()
        self.collected_data.clear()
        self.analysis.clear()
        self.task_state.clear()
