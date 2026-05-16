"""Tool registry — manages available tools and generates JSON Schema for the agent."""

import json
from typing import Any, Callable, Awaitable
from dataclasses import dataclass, field


@dataclass
class AgentTool:
    """A tool callable by the ReAct agent.

    Each tool has a name, description, JSON Schema for parameters,
    and an async callable that executes the tool.
    """

    name: str
    description: str
    parameters_schema: dict  # JSON Schema for the parameters object
    func: Callable[..., Awaitable[Any]]  # async (**_kwargs) -> Any

    def to_openai_spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def describe(self) -> str:
        props = self.parameters_schema.get("properties", {})
        required = self.parameters_schema.get("required", [])
        param_lines = []
        for pname, pinfo in props.items():
            req = " (required)" if pname in required else ""
            param_lines.append(f"    {pname}: {pinfo.get('description', pinfo.get('type', ''))}{req}")
        params_block = "\n".join(param_lines) if param_lines else "    (no parameters)"
        return f"{self.name}\n  {self.description}\n  Parameters:\n{params_block}"


class ToolRegistry:
    """Registry of all tools available to the agent."""

    def __init__(self):
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[AgentTool]:
        return list(self._tools.values())

    def get_descriptions(self) -> str:
        """Generate a text block describing all tools for the ReAct system prompt."""
        blocks = [t.describe() for t in self._tools.values()]
        return "\n\n".join(blocks)

    def get_openai_specs(self) -> list[dict]:
        return [t.to_openai_spec() for t in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
