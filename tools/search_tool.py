"""
DuckDuckGo搜索工具模块

提供基于DuckDuckGo的搜索功能，包含：
- 重试机制
- 超时处理
- 搜索结果格式化
- 异常处理
"""

import time
from typing import Optional
from dataclasses import dataclass
from loguru import logger

try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    logger.warning("duckduckgo-search 未安装，将使用 httpx 备用方案")

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@dataclass
class SearchResult:
    """搜索结果数据类"""
    title: str           # 搜索结果标题
    url: str             # 链接地址
    snippet: str         # 摘要内容
    source: str = "duckduckgo"  # 数据来源

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source
        }


class SearchToolError(Exception):
    """搜索工具异常基类"""
    pass


class SearchTimeoutError(SearchToolError):
    """搜索超时异常"""
    pass


class SearchRateLimitError(SearchToolError):
    """搜索频率限制异常"""
    pass


class SearchTool:
    """
    DuckDuckGo搜索工具类

    提供可靠的搜索功能，支持：
    - 自动重试
    - 超时控制
    - 结果去重
    - 格式化输出
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout: int = 30,
        request_interval: float = 1.0,
        max_results: int = 10
    ):
        """
        初始化搜索工具

        Args:
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            request_interval: 请求间隔（秒）
            max_results: 最大返回结果数
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.request_interval = request_interval
        self.max_results = max_results

        # 检查duckduckgo-search是否可用
        if not DUCKDUCKGO_AVAILABLE:
            logger.info("使用 httpx 备用方案进行搜索")

        logger.info(
            f"SearchTool初始化完成: max_retries={max_retries}, "
            f"timeout={timeout}s, max_results={max_results}"
        )

    def _rate_limit(self):
        """请求频率限制"""
        time.sleep(self.request_interval)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True
    )
    def search(self, query: str, max_results: Optional[int] = None) -> list[SearchResult]:
        """
        执行搜索

        Args:
            query: 搜索关键词
            max_results: 最大结果数（默认使用初始化时的值）

        Returns:
            搜索结果列表

        Raises:
            SearchToolError: 搜索失败
            SearchTimeoutError: 搜索超时
        """
        if not query or not query.strip():
            raise ValueError("搜索关键词不能为空")

        query = query.strip()
        results_count = max_results or self.max_results

        logger.info(f"执行搜索: '{query}', 目标结果数: {results_count}")

        try:
            if DUCKDUCKGO_AVAILABLE:
                results = self._search_with_ddgs(query, results_count)
            else:
                results = self._search_with_httpx(query, results_count)

            logger.info(f"搜索完成，获取到 {len(results)} 条结果")
            return results

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise SearchToolError(f"搜索执行失败: {e}")

    def _search_with_ddgs(self, query: str, max_results: int) -> list[SearchResult]:
        """
        使用duckduckgo-search库进行搜索

        Args:
            query: 搜索关键词
            max_results: 最大结果数

        Returns:
            搜索结果列表
        """
        results = []

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                # 使用text方法进行文本搜索
                for i, result in enumerate(ddgs.text(query, max_results=max_results)):
                    if i >= max_results:
                        break

                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("href", ""),
                        snippet=result.get("body", "")
                    )
                    results.append(search_result)

                    # 频率限制
                    if i < max_results - 1:
                        self._rate_limit()

        except Exception as e:
            logger.error(f"DuckDuckGo搜索出错: {e}")
            raise

        return results

    def _search_with_httpx(self, query: str, max_results: int) -> list[SearchResult]:
        """
        使用httpx备用方案进行搜索
        通过DuckDuckGo的HTML页面抓取搜索结果

        Args:
            query: 搜索关键词
            max_results: 最大结果数

        Returns:
            搜索结果列表
        """
        results = []

        try:
            # 构建DuckDuckGo搜索URL
            encoded_query = httpx.utils.encode_url(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }

            response = httpx.get(
                url,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            # 解析HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # 提取搜索结果
            result_divs = soup.select(".result")

            for i, div in enumerate(result_divs[:max_results]):
                title_elem = div.select_one(".result__title a")
                snippet_elem = div.select_one(".result__snippet")

                if title_elem:
                    search_result = SearchResult(
                        title=title_elem.get_text(strip=True),
                        url=title_elem.get("href", ""),
                        snippet=snippet_elem.get_text(strip=True) if snippet_elem else ""
                    )
                    results.append(search_result)

                # 频率限制
                if i < len(result_divs) - 1:
                    self._rate_limit()

        except httpx.TimeoutException:
            logger.error("搜索请求超时")
            raise SearchTimeoutError("搜索请求超时")
        except httpx.HTTPError as e:
            logger.error(f"HTTP请求错误: {e}")
            raise SearchToolError(f"HTTP请求错误: {e}")
        except Exception as e:
            logger.error(f"备用搜索方案出错: {e}")
            raise SearchToolError(f"备用搜索方案出错: {e}")

        return results

    def search_news(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """
        搜索新闻

        Args:
            query: 搜索关键词
            max_results: 最大结果数

        Returns:
            新闻搜索结果列表
        """
        logger.info(f"搜索新闻: '{query}'")

        try:
            results = []

            if DUCKDUCKGO_AVAILABLE:
                with DDGS(timeout=self.timeout) as ddgs:
                    for i, result in enumerate(ddgs.news(query, max_results=max_results)):
                        if i >= max_results:
                            break

                        search_result = SearchResult(
                            title=result.get("title", ""),
                            url=result.get("url", ""),
                            snippet=result.get("description", ""),
                            source="duckduckgo_news"
                        )
                        results.append(search_result)
            else:
                # 备用方案：使用普通搜索并过滤新闻
                all_results = self.search(query, max_results=max_results)
                results = all_results

            logger.info(f"新闻搜索完成，获取到 {len(results)} 条结果")
            return results

        except Exception as e:
            logger.error(f"新闻搜索失败: {e}")
            return []

    def search_with_variations(
        self,
        base_query: str,
        variations: list[str],
        max_results_per_search: int = 5
    ) -> list[SearchResult]:
        """
        使用多个关键词变体进行搜索，结果合并去重

        Args:
            base_query: 基础搜索词
            variations: 关键词变体列表（如 ["功能", "产品", "融资"]）
            max_results_per_search: 每次搜索的最大结果数

        Returns:
            合并后的搜索结果列表（去重）
        """
        all_results: list[SearchResult] = []
        seen_urls: set[str] = set()

        logger.info(
            f"执行变体搜索: 基础词='{base_query}', "
            f"变体={variations}, 每词结果数={max_results_per_search}"
        )

        # 先用基础词搜索
        base_results = self.search(base_query, max_results=max_results_per_search * 2)
        for result in base_results:
            if result.url not in seen_urls:
                all_results.append(result)
                seen_urls.add(result.url)

        # 使用变体搜索
        for variation in variations:
            query = f"{base_query} {variation}"
            try:
                results = self.search(query, max_results=max_results_per_search)
                for result in results:
                    if result.url not in seen_urls:
                        all_results.append(result)
                        seen_urls.add(result.url)
            except SearchToolError as e:
                logger.warning(f"变体搜索失败 '{query}': {e}")
                continue

        logger.info(f"变体搜索完成，总共获取 {len(all_results)} 条不重复结果")
        return all_results


# 导出类
__all__ = [
    "SearchResult",
    "SearchTool",
    "SearchToolError",
    "SearchTimeoutError",
    "SearchRateLimitError",
]
