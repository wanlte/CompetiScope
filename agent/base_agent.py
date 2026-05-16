"""ReAct Agent loop engine — hand-written, no LangChain AgentExecutor.

Implements the core Thought → Action → Observation → Thought cycle
that defines the ReAct (Reasoning + Acting) agent pattern.

Design decisions (from plan):
- Hand-written loop, NOT LangChain AgentExecutor (black-box avoidance)
- Tool descriptions injected into system prompt
- Structured response parsing with regex + brace-counting JSON extraction
- Max-steps guardrail to prevent infinite loops
"""

import json
import re
import asyncio
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from llm.provider import BaseLLMProvider
from agent.tool_registry import ToolRegistry
from agent.memory import ConversationMemory, WorkingMemory


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """One step of the agent loop."""
    step_num: int
    thought: str = ""
    action: str = ""
    action_input: dict = field(default_factory=dict)
    observation: str = ""


@dataclass
class AgentResult:
    """Result returned by the agent after completing (or failing) a task."""
    success: bool
    answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    total_steps: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# ReAct system prompt template
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """You are an autonomous competitive-analysis agent. You use the **ReAct** (Reasoning + Acting) framework to complete tasks.

## How You Work

You operate in a loop: **Thought → Action → Observation → Thought → ...**

### Response Format

When you want to use a tool, respond EXACTLY like this:

```
Thought: <your reasoning — what you need to find out and why>
Action: <tool_name>
Action Input: <JSON object>
```

When you have gathered enough information and are ready to answer, respond with:

```
Thought: I now have sufficient information to complete the task.
Final Answer:
<your complete answer in markdown>
```

### Critical Rules

1. **Always start with Thought** — explain your reasoning before acting.
2. **One action per response** — do NOT chain multiple actions.
3. **Action Input must be valid JSON** on a single line or multiple lines.
4. **Read observations carefully** — they contain the data you need.
5. **Don't hallucinate** — if a tool returns no data, admit it and try another approach.
6. **Be thorough** — search for every competitor using multiple queries before writing the final report.
7. **Use the available tools** — they are your only way to gather information.

## Available Tools

{tool_descriptions}

## Task

{task}

---

Begin now. Start with your first Thought and Action."""


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

class ReActAgent:
    """Hand-written ReAct agent loop.

    Key properties:
    - max_steps: safety limit (default 12)
    - provider: LLM provider for reasoning
    - tools: ToolRegistry of available tools
    - memory: ConversationMemory (sliding window)
    - working: WorkingMemory (structured intermediate storage)
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        tool_registry: ToolRegistry,
        memory: Optional[ConversationMemory] = None,
        working_memory: Optional[WorkingMemory] = None,
        max_steps: int = 12,
        verbose: bool = True,
    ):
        self.provider = provider
        self.tools = tool_registry
        self.memory = memory or ConversationMemory()
        self.working = working_memory or WorkingMemory()
        self.max_steps = max_steps
        self.verbose = verbose

    # ---- public API ----

    async def run(self, task: str) -> AgentResult:
        """Execute the agent loop for a given task.

        Returns AgentResult with the final answer, step history, and status.
        """
        steps: list[AgentStep] = []

        system_prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=self.tools.get_descriptions(),
            task=task,
        )

        messages: list = [SystemMessage(content=system_prompt)]

        for step_num in range(1, self.max_steps + 1):
            if self.verbose:
                logger.info(f"━━━ Agent Step {step_num}/{self.max_steps} ━━━")

            # 1. REASONING — call LLM
            try:
                response = await self.provider.ainvoke(messages)
                content = response.content.strip()
            except Exception as exc:
                logger.error(f"LLM call failed at step {step_num}: {exc}")
                return AgentResult(
                    success=False, steps=steps,
                    total_steps=step_num, error=str(exc),
                )

            # 2. PARSE — extract Thought / Action / Final Answer
            parsed = self._parse(content)

            if parsed["type"] == "final_answer":
                step = AgentStep(step_num=step_num, thought=parsed.get("thought", ""))
                steps.append(step)
                if self.verbose:
                    logger.info("  ✅ Agent finished (Final Answer)")
                return AgentResult(
                    success=True, answer=parsed["content"],
                    steps=steps, total_steps=step_num,
                )

            if parsed["type"] == "action":
                action = parsed["action"]
                action_input = parsed.get("action_input", {})

                if self.verbose:
                    thought_preview = parsed.get("thought", "")[:120]
                    logger.info(f"  💭 {thought_preview}")
                    logger.info(f"  🔧 {action}({json.dumps(action_input, ensure_ascii=False)})")

                # 3. ACT — execute the tool
                observation = await self._execute(action, action_input)

                if self.verbose:
                    obs_preview = observation[:250] + ("..." if len(observation) > 250 else "")
                    logger.info(f"  👁 {obs_preview}")

                step = AgentStep(
                    step_num=step_num,
                    thought=parsed.get("thought", ""),
                    action=action,
                    action_input=action_input,
                    observation=observation,
                )
                steps.append(step)

                # 4. OBSERVE — feed result back into conversation
                messages.append(AIMessage(content=content))
                messages.append(ToolMessage(content=observation, tool_call_id=action))
                continue

            # Parse failure — give the LLM corrective feedback
            if self.verbose:
                logger.warning(f"  ⚠ Could not parse LLM response, feeding back error")
            messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=(
                "I could not parse your last response. You MUST use one of these formats:\n\n"
                "To use a tool:\n"
                "Thought: <reasoning>\n"
                "Action: <tool_name>\n"
                "Action Input: <JSON>\n\n"
                "To finish:\n"
                "Thought: I have enough information\n"
                "Final Answer:\n<your complete answer>"
            )))

        # ---- max steps reached — force final answer ----
        logger.warning(f"Max steps ({self.max_steps}) reached — requesting forced summary")
        messages.append(HumanMessage(content=(
            "You have reached the maximum number of steps. "
            "Based on everything you have gathered so far, provide your Final Answer now."
        )))
        try:
            response = await self.provider.ainvoke(messages)
            return AgentResult(
                success=True, answer=response.content,
                steps=steps, total_steps=len(steps),
            )
        except Exception as exc:
            return AgentResult(
                success=False, steps=steps,
                total_steps=len(steps), error=str(exc),
            )

    # ---- parsing ----

    def _parse(self, content: str) -> dict:
        """Parse an LLM response into {'type': 'action'|'final_answer'|'unknown', ...}."""
        result: dict = {"type": "unknown", "content": content}

        # --- Final Answer ---
        fa_match = re.search(
            r'Final Answer:\s*\n?(.*)',
            content, re.DOTALL | re.IGNORECASE,
        )
        if fa_match:
            result["type"] = "final_answer"
            result["content"] = fa_match.group(1).strip()
            t_match = re.search(
                r'Thought:\s*(.*?)(?=Final Answer:)',
                content, re.DOTALL | re.IGNORECASE,
            )
            if t_match:
                result["thought"] = t_match.group(1).strip()
            return result

        # --- Action ---
        action_match = re.search(r'Action:\s*(\S+)', content, re.IGNORECASE)
        if action_match:
            result["type"] = "action"
            result["action"] = action_match.group(1).strip()

            t_match = re.search(
                r'Thought:\s*(.*?)(?=Action:)',
                content, re.DOTALL | re.IGNORECASE,
            )
            if t_match:
                result["thought"] = t_match.group(1).strip()

            # Extract JSON from Action Input using brace counting
            result["action_input"] = {}
            ai_match = re.search(r'Action Input:\s*', content, re.IGNORECASE)
            if ai_match:
                json_text = content[ai_match.end():].strip()
                extracted = self._extract_json(json_text)
                if extracted:
                    try:
                        result["action_input"] = json.loads(extracted)
                    except json.JSONDecodeError:
                        result["action_input"] = {"_raw": json_text[:200]}
                else:
                    result["action_input"] = {"_raw": json_text[:200]}

        return result

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract a complete JSON object/array from text using brace/bracket counting."""
        text = text.strip()
        if not text:
            return None

        open_ch = text[0]
        if open_ch not in ('{', '['):
            return None

        close_ch = '}' if open_ch == '{' else ']'
        depth = 0
        in_string = False
        escape = False

        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[:i + 1]

        return None

    # ---- tool execution ----

    async def _execute(self, tool_name: str, params: dict) -> str:
        """Execute a tool by name and return the observation string."""
        tool = self.tools.get(tool_name)
        if tool is None:
            available = ", ".join(self.tools.tool_names)
            return f"Error: unknown tool '{tool_name}'. Available: {available}"

        try:
            result = await tool.func(**params)
        except TypeError as exc:
            return f"Error: invalid parameters for '{tool_name}': {exc}"
        except Exception as exc:
            logger.error(f"Tool '{tool_name}' failed: {exc}")
            return f"Error executing '{tool_name}': {exc}"

        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, indent=2)
