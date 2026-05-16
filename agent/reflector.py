"""Self-reflection module — agent reviews and improves its own output."""

from dataclasses import dataclass, field
from loguru import logger

from langchain_core.messages import SystemMessage, HumanMessage

from llm.provider import BaseLLMProvider


REFLECTION_SYSTEM_PROMPT = """You are a quality-review agent. Your job is to critique a competitive analysis report and suggest improvements.

Evaluate the report on:
1. **Completeness** — Are all competitors covered? Are all dimensions analyzed?
2. **Accuracy** — Are claims backed by data? Is anything speculative?
3. **Actionability** — Are recommendations specific and executable?
4. **Structure** — Is the report well-organized and easy to read?
5. **Depth** — Is the analysis surface-level or does it dig into root causes?

Respond in JSON:
{
  "score": <1-10>,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggestions": ["..."],
  "missing_areas": ["..."],
  "revised_section": "optional — a rewritten version of the weakest section"
}
"""


@dataclass
class ReflectionResult:
    """The output of a self-reflection pass."""

    score: int
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    missing_areas: list[str] = field(default_factory=list)
    revised_section: str = ""

    @property
    def is_good(self) -> bool:
        return self.score >= 7


class Reflector:
    """Reviews agent output and suggests improvements.

    After the agent produces a report, the Reflector critiques it
    and optionally triggers a revision round. This is the
    "Self-Critique" pattern common in modern agent architectures.
    """

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    async def reflect(
        self,
        task: str,
        answer: str,
    ) -> ReflectionResult:
        """Critique the agent's answer and return a reflection.

        Args:
            task: The original task given to the agent.
            answer: The agent's final answer to critique.

        Returns:
            ReflectionResult with score, strengths, weaknesses, and suggestions.
        """
        import json as _json

        prompt = f"""Original Task:
{task}

Report to Review:
{answer}

Please review the report above and provide your assessment in JSON format."""

        messages = [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.provider.ainvoke(messages)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = _json.loads(content)
        except Exception as e:
            logger.warning(f"Reflection parse failed, using defaults: {e}")
            data = {}

        return ReflectionResult(
            score=int(data.get("score", 5)),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            suggestions=data.get("suggestions", []),
            missing_areas=data.get("missing_areas", []),
            revised_section=data.get("revised_section", ""),
        )

    async def reflect_and_revise(
        self,
        task: str,
        answer: str,
        max_rounds: int = 2,
    ) -> tuple[str, list[ReflectionResult]]:
        """Reflect on the answer and iteratively revise it.

        Args:
            task: The original task.
            answer: The agent's initial answer.
            max_rounds: Maximum revision rounds.

        Returns:
            (final_answer, [reflection_results])
        """
        reflections: list[ReflectionResult] = []
        current = answer

        for round_num in range(1, max_rounds + 1):
            logger.info(f"Reflection round {round_num}/{max_rounds}")
            reflection = await self.reflect(task, current)
            reflections.append(reflection)

            if reflection.is_good:
                logger.info(f"  Score {reflection.score}/10 — good enough, stopping")
                break

            logger.info(f"  Score {reflection.score}/10 — weaknesses: {reflection.weaknesses}")

            # If the reflector provided a revision, use it; otherwise stop
            if reflection.revised_section:
                # Append the revision
                current = current + "\n\n---\n## 🔄 Revised Section (Reflection Round {})\n\n".format(round_num) + reflection.revised_section
            else:
                # No revision provided, build a revision prompt
                revised = await self._generate_revision(task, current, reflection)
                if revised:
                    current = revised

        return current, reflections

    async def _generate_revision(
        self,
        task: str,
        current_answer: str,
        reflection: ReflectionResult,
    ) -> str:
        """Generate a revised version incorporating the reflection feedback."""
        import json as _json

        prompt = f"""Original Task:
{task}

Current Report:
{current_answer}

Review Feedback:
- Weaknesses: {_json.dumps(reflection.weaknesses, ensure_ascii=False)}
- Suggestions: {_json.dumps(reflection.suggestions, ensure_ascii=False)}
- Missing areas: {_json.dumps(reflection.missing_areas, ensure_ascii=False)}

Please rewrite the report incorporating all the feedback. Produce the complete revised report."""

        messages = [
            SystemMessage(content="You are a professional report editor. Rewrite the report based on the feedback provided."),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.provider.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Revision generation failed: {e}")
            return ""
