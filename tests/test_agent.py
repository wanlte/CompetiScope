"""Tests for Phase 2 — Agent ReAct Loop, tools, memory, and reflection."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from agent.tool_registry import ToolRegistry, AgentTool
from agent.memory import ConversationMemory, WorkingMemory
from agent.base_agent import ReActAgent, AgentResult, AgentStep
from agent.reflector import Reflector, ReflectionResult
from tools.tool_adapter import create_agent_tools, _search_web, _scrape_page


# ============================================================================
# Tool Registry Tests
# ============================================================================

class TestToolRegistry:

    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = AgentTool(
            name="test_tool",
            description="A test tool",
            parameters_schema={"type": "object", "properties": {}},
            func=AsyncMock(return_value="ok"),
        )
        registry.register(tool)
        assert registry.get("test_tool") is tool
        assert registry.get("nonexistent") is None

    def test_duplicate_register_raises(self):
        registry = ToolRegistry()
        tool = AgentTool(
            name="test_tool",
            description="A test tool",
            parameters_schema={"type": "object", "properties": {}},
            func=AsyncMock(return_value="ok"),
        )
        registry.register(tool)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool)

    def test_list_tools(self):
        registry = ToolRegistry()
        t1 = AgentTool(name="a", description="", parameters_schema={}, func=AsyncMock())
        t2 = AgentTool(name="b", description="", parameters_schema={}, func=AsyncMock())
        registry.register(t1)
        registry.register(t2)
        assert len(registry.list_tools()) == 2

    def test_get_descriptions(self):
        registry = ToolRegistry()
        tool = AgentTool(
            name="search",
            description="Search the web",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
            func=AsyncMock(),
        )
        registry.register(tool)
        desc = registry.get_descriptions()
        assert "search" in desc
        assert "Search the web" in desc
        assert "query" in desc

    def test_get_openai_specs(self):
        registry = ToolRegistry()
        tool = AgentTool(
            name="test",
            description="Test tool",
            parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            func=AsyncMock(),
        )
        registry.register(tool)
        specs = registry.get_openai_specs()
        assert len(specs) == 1
        assert specs[0]["type"] == "function"
        assert specs[0]["function"]["name"] == "test"

    def test_tool_names_property(self):
        registry = ToolRegistry()
        registry.register(AgentTool(name="a", description="", parameters_schema={}, func=AsyncMock()))
        registry.register(AgentTool(name="b", description="", parameters_schema={}, func=AsyncMock()))
        assert registry.tool_names == ["a", "b"]


# ============================================================================
# Memory Tests
# ============================================================================

class TestConversationMemory:

    def test_add_and_retrieve_messages(self):
        memory = ConversationMemory(max_messages=100)
        memory.add_system("system prompt")
        memory.add_user("hello")
        memory.add_assistant("hi there")
        memory.add_tool_result("search", "results here")

        msgs = memory.get_messages()
        assert len(msgs) == 4
        assert msgs[0].content == "system prompt"
        assert msgs[1].content == "hello"

    def test_sliding_window_truncation(self):
        memory = ConversationMemory(max_messages=5)
        memory.add_system("system")
        for i in range(20):
            memory.add_user(f"msg {i}")
            memory.add_assistant(f"reply {i}")

        msgs = memory.get_messages()
        # system + up to 4 recent = max 5
        assert len(msgs) <= 5
        assert msgs[0].content == "system"

    def test_summary_injection(self):
        memory = ConversationMemory(max_messages=5)
        memory.set_summary("previously discussed X, Y, Z")
        memory.add_system("system")
        for i in range(20):
            memory.add_user(f"msg {i}")
            memory.add_assistant(f"reply {i}")

        msgs = memory.get_messages()
        summaries = [m for m in msgs if "previously discussed" in str(m.content)]
        assert len(summaries) >= 1

    def test_clear(self):
        memory = ConversationMemory()
        memory.add_user("test")
        memory.set_summary("summary")
        memory.clear()
        assert memory.message_count == 0
        assert memory._summary == ""


class TestWorkingMemory:

    def test_search_cache(self):
        wm = WorkingMemory()
        wm.store_search("query1", [{"title": "result"}])
        assert wm.get_search("query1") == [{"title": "result"}]
        assert wm.get_search("nonexistent") is None

    def test_collected_data(self):
        wm = WorkingMemory()
        wm.store_collected("Notion", {"founded": "2016"})
        assert wm.get_collected("Notion") == {"founded": "2016"}
        assert wm.all_competitors == ["Notion"]

    def test_analysis_storage(self):
        wm = WorkingMemory()
        wm.store_analysis("Notion", {"swot": "..."})
        assert wm.get_analysis("Notion") == {"swot": "..."}

    def test_snapshot(self):
        wm = WorkingMemory()
        wm.store_collected("A", {})
        wm.store_search("q1", [])
        wm.store_analysis("A", {})
        snap = wm.snapshot()
        assert snap["competitors_collected"] == ["A"]
        assert "q1" in snap["search_queries"]

    def test_clear(self):
        wm = WorkingMemory()
        wm.store_collected("A", {})
        wm.clear()
        assert wm.all_competitors == []


# ============================================================================
# ReAct Agent Tests
# ============================================================================

class TestReActAgentParsing:

    def test_parse_final_answer(self):
        agent = ReActAgent(
            provider=MagicMock(),
            tool_registry=ToolRegistry(),
            verbose=False,
        )
        content = """Thought: I have gathered sufficient information about the competitors.

Final Answer:
# Competitive Analysis Report

This is the full report content."""
        parsed = agent._parse(content)
        assert parsed["type"] == "final_answer"
        assert "Competitive Analysis Report" in parsed["content"]

    def test_parse_action_with_json(self):
        agent = ReActAgent(
            provider=MagicMock(),
            tool_registry=ToolRegistry(),
            verbose=False,
        )
        content = """Thought: I need to search for information about Notion.

Action: search_web
Action Input: {"query": "Notion company overview", "max_results": 5}"""
        parsed = agent._parse(content)
        assert parsed["type"] == "action"
        assert parsed["action"] == "search_web"
        assert parsed["action_input"] == {"query": "Notion company overview", "max_results": 5}

    def test_parse_action_with_nested_json(self):
        agent = ReActAgent(
            provider=MagicMock(),
            tool_registry=ToolRegistry(),
            verbose=False,
        )
        content = """Thought: Let me analyze the competitor data.

Action: analyze_competitor_data
Action Input: {"competitor_name": "Notion", "data_json": "{\\"key\\": \\"value\\"}"}"""
        parsed = agent._parse(content)
        assert parsed["type"] == "action"
        assert parsed["action"] == "analyze_competitor_data"
        assert "competitor_name" in parsed["action_input"]

    def test_extract_json_object(self):
        text = '{"name": "test", "nested": {"a": 1}}'
        result = ReActAgent._extract_json(text)
        assert result == text

    def test_extract_json_array(self):
        text = '[{"a": 1}, {"b": 2}]'
        result = ReActAgent._extract_json(text)
        assert result == text

    def test_extract_json_unclosed(self):
        text = '{"name": "test"'
        result = ReActAgent._extract_json(text)
        assert result is None

    def test_extract_json_not_json(self):
        text = 'just some text'
        result = ReActAgent._extract_json(text)
        assert result is None

    def test_parse_without_action_input(self):
        agent = ReActAgent(
            provider=MagicMock(),
            tool_registry=ToolRegistry(),
            verbose=False,
        )
        content = """Thought: Let me search.

Action: search_web"""
        parsed = agent._parse(content)
        assert parsed["type"] == "action"
        assert parsed["action"] == "search_web"
        assert parsed["action_input"] == {}


class TestReActAgentExecution:

    @pytest.mark.asyncio
    async def test_single_step_then_final_answer(self, mock_provider):
        """Agent should finish in one step when LLM returns Final Answer."""
        mock_provider.responses = [
            """Thought: I can answer this directly.
Final Answer:
# Analysis
Simple answer here.""",
        ]

        async def dummy_tool(**kwargs):
            return "tool result"

        registry = ToolRegistry()
        registry.register(AgentTool(
            name="search_web",
            description="Search",
            parameters_schema={"type": "object", "properties": {}, "required": []},
            func=dummy_tool,
        ))

        agent = ReActAgent(
            provider=mock_provider,
            tool_registry=registry,
            verbose=False,
        )

        result = await agent.run("Simple task")
        assert result.success
        assert "Analysis" in result.answer
        assert result.total_steps == 1

    @pytest.mark.asyncio
    async def test_tool_call_then_final_answer(self, mock_provider):
        """Agent should call a tool, observe, then finish."""
        mock_provider.responses = [
            # Step 1: call tool
            """Thought: I need to search first.
Action: search_web
Action Input: {"query": "test query"}""",
            # Step 2: final answer
            """Thought: Got the data I need.
Final Answer:
# Report
Based on search results, here is the analysis.""",
        ]

        call_log = []

        async def search_tool(query: str, **_kwargs):
            call_log.append(query)
            return '[{"title": "Result", "snippet": "Info"}]'

        registry = ToolRegistry()
        registry.register(AgentTool(
            name="search_web",
            description="Search the web",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            func=search_tool,
        ))

        agent = ReActAgent(
            provider=mock_provider,
            tool_registry=registry,
            verbose=False,
        )

        result = await agent.run("Analyze test company")
        assert result.success
        assert result.total_steps == 2
        assert len(result.steps) == 2
        assert result.steps[0].action == "search_web"
        assert result.steps[0].observation.startswith("[")
        assert "Report" in result.answer
        assert call_log == ["test query"]

    @pytest.mark.asyncio
    async def test_max_steps_reached(self, mock_provider):
        """Agent should stop after max_steps and request final answer."""
        # Always return an action — agent should hit max_steps
        mock_provider.responses = [
            """Thought: Need to search.
Action: search_web
Action Input: {"query": "test"}""",
        ] * 5 + [
            """Thought: Out of steps.
Final Answer:
Limited analysis due to step limit.""",
        ]

        async def search_tool(**kwargs):
            return "[]"

        registry = ToolRegistry()
        registry.register(AgentTool(
            name="search_web",
            description="Search",
            parameters_schema={"type": "object", "properties": {}, "required": []},
            func=search_tool,
        ))

        agent = ReActAgent(
            provider=mock_provider,
            tool_registry=registry,
            max_steps=3,
            verbose=False,
        )

        result = await agent.run("Test task")
        # With max_steps=3, each step is an action, so at step 4 it hits the limit
        # But we only have 5 action responses + 1 final — the final answer happens at step 5
        # Actually max_steps=3 means steps 1,2,3 will all be actions, then it'll force final
        assert result.total_steps == 3
        # The result should still succeed (forced final answer)
        assert result.success

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, mock_provider):
        """Calling an unregistered tool should produce an error observation."""
        mock_provider.responses = [
            """Thought: Let me try this.
Action: nonexistent_tool
Action Input: {}""",
            """Thought: That didn't work. Let me finish.
Final Answer:
Could not complete analysis.""",
        ]

        agent = ReActAgent(
            provider=mock_provider,
            tool_registry=ToolRegistry(),
            verbose=False,
        )

        result = await agent.run("Test")
        assert result.success
        obs = result.steps[0].observation
        assert "unknown tool" in obs.lower() or "Error" in obs

    @pytest.mark.asyncio
    async def test_tool_exception_handling(self, mock_provider):
        """Tool that raises should not crash the agent."""
        mock_provider.responses = [
            """Thought: Try the tool.
Action: broken_tool
Action Input: {}""",
            """Thought: Tool failed, finishing.
Final Answer:
Partial analysis.""",
        ]

        async def broken(**kwargs):
            raise RuntimeError("Tool is broken!")

        registry = ToolRegistry()
        registry.register(AgentTool(
            name="broken_tool",
            description="Will break",
            parameters_schema={"type": "object", "properties": {}, "required": []},
            func=broken,
        ))

        agent = ReActAgent(
            provider=mock_provider,
            tool_registry=registry,
            verbose=False,
        )

        result = await agent.run("Test")
        assert result.success  # should still succeed (covers with final answer)
        assert "Error" in result.steps[0].observation

    @pytest.mark.asyncio
    async def test_parse_failure_feedback(self, mock_provider):
        """When LLM response can't be parsed, agent should give feedback and continue."""
        mock_provider.responses = [
            "Just some unstructured text without proper format",
            """Thought: Got feedback about format.
Action: search_web
Action Input: {"query": "test"}""",
            """Thought: Done.
Final Answer:
Report after correction.""",
        ]

        async def search_tool(**kwargs):
            return "[]"

        registry = ToolRegistry()
        registry.register(AgentTool(
            name="search_web",
            description="Search",
            parameters_schema={"type": "object", "properties": {}, "required": []},
            func=search_tool,
        ))

        agent = ReActAgent(
            provider=mock_provider,
            tool_registry=registry,
            verbose=False,
        )

        result = await agent.run("Test")
        assert result.success
        # Step 1 failed to parse, step 2 was action, step 3 was final
        assert result.total_steps == 3

    @pytest.mark.asyncio
    async def test_llm_call_failure(self, mock_provider):
        """When LLM returns an error, agent should report failure."""
        # Mock provider's ainvoke to raise an exception
        mock_provider.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))

        agent = ReActAgent(
            provider=mock_provider,
            tool_registry=ToolRegistry(),
            verbose=False,
        )

        result = await agent.run("Test")
        assert not result.success
        assert "API error" in result.error


# ============================================================================
# Reflector Tests
# ============================================================================

class TestReflector:

    @pytest.mark.asyncio
    async def test_reflect_good_report(self, mock_provider):
        """Reflector should parse the critique JSON."""
        mock_provider.responses = [
            json.dumps({
                "score": 8,
                "strengths": ["Well structured", "Data-driven"],
                "weaknesses": ["Missing pricing comparison"],
                "suggestions": ["Add pricing table"],
                "missing_areas": ["Pricing"],
                "revised_section": "## Pricing Comparison\n...",
            }),
        ]

        reflector = Reflector(mock_provider)
        result = await reflector.reflect("Analyze A and B", "# Report\n...")
        assert isinstance(result, ReflectionResult)
        assert result.score == 8
        assert result.is_good
        assert len(result.strengths) == 2

    @pytest.mark.asyncio
    async def test_reflect_poor_report(self, mock_provider):
        """Low score should not be 'good'."""
        mock_provider.responses = [
            json.dumps({
                "score": 4,
                "strengths": ["Has a title"],
                "weaknesses": ["Too short", "No data", "Vague"],
                "suggestions": ["Add specifics", "Include SWOT"],
                "missing_areas": ["SWOT", "Pricing", "Market share"],
                "revised_section": "",
            }),
        ]

        reflector = Reflector(mock_provider)
        result = await reflector.reflect("Analyze", "Short report.")
        assert result.score == 4
        assert not result.is_good

    @pytest.mark.asyncio
    async def test_reflect_and_revise_stops_when_good(self, mock_provider):
        """reflect_and_revise should stop after a good score."""
        mock_provider.responses = [
            json.dumps({
                "score": 5,
                "strengths": [],
                "weaknesses": ["Thin"],
                "suggestions": ["Add more"],
                "missing_areas": ["Details"],
                "revised_section": "Added more details here.",
            }),
            json.dumps({
                "score": 8,
                "strengths": ["Improved"],
                "weaknesses": [],
                "suggestions": [],
                "missing_areas": [],
                "revised_section": "",
            }),
        ]

        reflector = Reflector(mock_provider)
        final, reflections = await reflector.reflect_and_revise(
            "Task", "Initial answer.", max_rounds=3,
        )
        assert len(reflections) == 2
        assert reflections[-1].score == 8

    @pytest.mark.asyncio
    async def test_reflect_parse_failure(self, mock_provider):
        """When JSON parsing fails, should return defaults."""
        mock_provider.responses = ["Not valid JSON at all"]

        reflector = Reflector(mock_provider)
        result = await reflector.reflect("Task", "Answer")
        assert result.score == 5  # default
        assert result.strengths == []


# ============================================================================
# Tool Adapter Tests
# ============================================================================

class TestToolAdapter:

    def test_create_agent_tools_without_provider(self):
        """Creating tools without provider should only return search + scrape."""
        tools = create_agent_tools(provider=None)
        tool_names = {t.name for t in tools}
        assert "search_web" in tool_names
        assert "scrape_page" in tool_names
        assert "analyze_competitor_data" not in tool_names
        assert "generate_report_section" not in tool_names

    def test_create_agent_tools_with_provider(self, mock_provider):
        """With provider, all 4 tools should be available."""
        tools = create_agent_tools(provider=mock_provider)
        tool_names = {t.name for t in tools}
        assert "search_web" in tool_names
        assert "scrape_page" in tool_names
        assert "analyze_competitor_data" in tool_names
        assert "generate_report_section" in tool_names

    def test_tool_schemas_have_required_format(self):
        """Each tool should have valid JSON Schema."""
        tools = create_agent_tools(provider=None)
        for tool in tools:
            assert "type" in tool.parameters_schema
            assert tool.parameters_schema["type"] == "object"
            assert "properties" in tool.parameters_schema
            assert tool.name
            assert tool.description
            assert callable(tool.func)

    def test_tool_openai_spec_format(self):
        """Each tool should produce valid OpenAI function spec."""
        tools = create_agent_tools(provider=None)
        for tool in tools:
            spec = tool.to_openai_spec()
            assert spec["type"] == "function"
            assert "name" in spec["function"]
            assert "description" in spec["function"]
            assert "parameters" in spec["function"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestAgentIntegration:

    @pytest.mark.asyncio
    async def test_full_agent_flow_with_mock_tools(self, mock_provider):
        """Simulate a complete agent flow: search → analyze → write."""
        mock_provider.responses = [
            # Search for Notion
            """Thought: I'll start by searching for Notion.
Action: search_web
Action Input: {"query": "Notion company overview"}""",
            # Search for Linear
            """Thought: Now let me search for Linear.
Action: search_web
Action Input: {"query": "Linear app features"}""",
            # Generate report
            """Thought: I have enough search data. Let me write the report.
Action: generate_report_section
Action Input: {"instructions": "Write a full competitive analysis comparing Notion and Linear", "context": "Search results gathered"}""",
            # Final answer
            """Thought: The report section is complete.
Final Answer:
# Competitive Analysis: Notion vs Linear

## Overview
Both are productivity tools but target different segments.

## Comparison
- Notion: all-in-one workspace
- Linear: focused project management for developers

## Recommendations
1. If targeting knowledge workers, Notion is the main competitor
2. Linear's speed and keyboard-first design is a differentiator""",
        ]

        call_log = []

        async def mock_search(query: str, max_results: int = 5):
            call_log.append(("search", query))
            return json.dumps([{"title": f"Result for {query}", "snippet": "Info"}])

        async def mock_generate(instructions: str, context: str):
            call_log.append(("generate", instructions[:50]))
            return "Generated report section."

        registry = ToolRegistry()
        registry.register(AgentTool(
            name="search_web",
            description="Search the web",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
            func=mock_search,
        ))
        registry.register(AgentTool(
            name="generate_report_section",
            description="Generate report section",
            parameters_schema={
                "type": "object",
                "properties": {
                    "instructions": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["instructions", "context"],
            },
            func=mock_generate,
        ))

        agent = ReActAgent(
            provider=mock_provider,
            tool_registry=registry,
            verbose=False,
        )

        result = await agent.run("Compare Notion and Linear")
        assert result.success
        assert result.total_steps == 4
        assert "Notion" in result.answer
        assert "Linear" in result.answer
        assert len(call_log) == 3  # 2 searches + 1 generate
