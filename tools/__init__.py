"""
工具模块

提供竞品分析Agent所需的各种工具：
- search_tool: DuckDuckGo搜索工具
- web_scraper: 网页内容抓取工具
"""

from .search_tool import (
    SearchTool,
    SearchResult,
    SearchToolError,
    SearchTimeoutError,
    SearchRateLimitError,
)

from .web_scraper import (
    WebScraper,
    ScrapedContent,
    WebScraperError,
    ScrapingTimeoutError,
    InvalidURLError,
    ContentExtractionError,
)

__all__ = [
    # 搜索工具
    "SearchTool",
    "SearchResult",
    "SearchToolError",
    "SearchTimeoutError",
    "SearchRateLimitError",
    # 网页抓取工具
    "WebScraper",
    "ScrapedContent",
    "WebScraperError",
    "ScrapingTimeoutError",
    "InvalidURLError",
    "ContentExtractionError",
]
