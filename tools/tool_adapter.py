"""Tool adapter — wraps existing SearchTool/WebScraper as AgentTools for the ReAct agent."""

import json
import asyncio
from typing import Optional
from loguru import logger

from agent.tool_registry import AgentTool
from tools.search_tool import SearchTool
from tools.web_scraper import WebScraper


# ---------------------------------------------------------------------------
# Async wrappers for sync tools
# ---------------------------------------------------------------------------

async def _search_web(
    query: str,
    max_results: int = 5,
    _tool: SearchTool | None = None,
) -> str:
    """Search the web and return structured results."""
    search = _tool or SearchTool(timeout=30, max_results=10)
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: search.search(query, max_results))
    data = [
        {"title": r.title, "url": r.url, "snippet": r.snippet}
        for r in results
    ]
    logger.info(f"search_web('{query}') → {len(data)} results")
    return json.dumps(data, ensure_ascii=False, indent=2)


async def _scrape_page(url: str, _tool: WebScraper | None = None) -> str:
    """Scrape a web page and return its content."""
    scraper = _tool or WebScraper(timeout=30)
    loop = asyncio.get_event_loop()
    try:
        content = await loop.run_in_executor(None, scraper.scrape, url)
        return json.dumps({
            "url": content.url,
            "title": content.title,
            "content": content.content[:3000],
            "summary": content.summary,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"scrape_page('{url}') failed: {e}")
        return json.dumps({"error": str(e), "url": url})


# ---------------------------------------------------------------------------
# Factory — build the tool set for a given provider
# ---------------------------------------------------------------------------

def create_agent_tools(
    provider=None,  # BaseLLMProvider, used by analysis/report tools
    search_tool: SearchTool | None = None,
    scraper: WebScraper | None = None,
) -> list[AgentTool]:
    """Create the standard set of agent tools.

    Returns tools that the ReAct agent can call autonomously:
    - search_web
    - scrape_page
    - analyze_competitor_data
    - generate_report_section
    """

    tools: list[AgentTool] = []

    # ---- search_web ----
    tools.append(AgentTool(
        name="search_web",
        description="Search the web for competitor information. Use multiple queries to gather comprehensive data about a company.",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'Notion company overview' or 'Linear pricing features'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5)",
                },
            },
            "required": ["query"],
        },
        func=lambda query, max_results=5: _search_web(query, max_results, search_tool),
    ))

    # ---- scrape_page ----
    tools.append(AgentTool(
        name="scrape_page",
        description="Scrape the content of a specific web page. Use to get detailed information from a URL found via search.",
        parameters_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the page to scrape",
                },
            },
            "required": ["url"],
        },
        func=lambda url: _scrape_page(url, scraper),
    ))

    # ---- analyze_competitor_data (LLM-powered) ----
    if provider:

        async def _analyze_data(competitor_name: str, data_json: str) -> str:
            """Use LLM to analyze collected data for a competitor."""
            from langchain_core.messages import SystemMessage, HumanMessage
            from config.prompts import ANALYZER_SYSTEM_PROMPT

            prompt = f"""Analyze the following data about {competitor_name} and produce:

1. SWOT analysis (strengths, weaknesses, opportunities, threats)
2. Key differentiators vs competitors
3. Market position assessment

Data:
{data_json}

Respond in JSON format:
{{
  "competitor": "{competitor_name}",
  "swot": {{"strengths": [...], "weaknesses": [...], "opportunities": [...], "threats": [...]}},
  "differentiators": [...],
  "market_position": "...",
  "key_insights": [...]
}}
"""
            messages = [
                SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            try:
                resp = await provider.ainvoke(messages)
                return resp.content
            except Exception as e:
                logger.error(f"analyze_competitor_data failed: {e}")
                return json.dumps({"error": str(e)})

        tools.append(AgentTool(
            name="analyze_competitor_data",
            description="Analyze collected data for a specific competitor. Produces SWOT, differentiators, and market position. Use AFTER gathering sufficient search data.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "competitor_name": {
                        "type": "string",
                        "description": "Name of the competitor to analyze",
                    },
                    "data_json": {
                        "type": "string",
                        "description": "JSON string of collected data about this competitor (search results, scraped content, etc.)",
                    },
                },
                "required": ["competitor_name", "data_json"],
            },
            func=_analyze_data,
        ))

        # ---- generate_report_section (LLM-powered) ----
        async def _generate_report(instructions: str, context: str) -> str:
            """Generate a section of the final report."""
            from langchain_core.messages import SystemMessage, HumanMessage
            from config.prompts import WRITER_SYSTEM_PROMPT

            prompt = f"""Write a section of a competitive analysis report.

Instructions: {instructions}

Context/Data:
{context}

Write in professional markdown with clear headings. Be specific, data-driven, and actionable.
"""
            messages = [
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            try:
                resp = await provider.ainvoke(messages)
                return resp.content
            except Exception as e:
                logger.error(f"generate_report_section failed: {e}")
                return f"Error generating report: {e}"

        tools.append(AgentTool(
            name="generate_report_section",
            description="Generate a section of the final competitive analysis report using LLM. Use to write the report after all analysis is complete.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": "What section to write and what it should contain",
                    },
                    "context": {
                        "type": "string",
                        "description": "The analysis data and insights to base the section on",
                    },
                },
                "required": ["instructions", "context"],
            },
            func=_generate_report,
        ))

    return tools
