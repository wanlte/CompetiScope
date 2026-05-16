"""
采集Agent模块 (Async v2)

负责从多渠道并发收集竞品信息，包括：
- 公司基本信息
- 产品功能
- 市场表现
- 用户评价
- 战略动态

Phase 1 升级: 全面异步化 + asyncio.gather 并发 + Provider 模式
"""

import json
import asyncio
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger

from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import COLLECTOR_SYSTEM_PROMPT
from config.settings import (
    LLMConfig, LLMRequestConfig, ProviderConfig as SettingsProvider, CacheConfig,
    KnowledgeBaseConfig,
)
from llm.provider import BaseLLMProvider
from llm.openai_provider import OpenAIProvider
from llm.types import ProviderConfig as LLMProviderConfig
from cache.llm_cache import LLMCache
from tools.search_tool import SearchTool, SearchResult
from tools.web_scraper import WebScraper, ScrapedContent
from rag.knowledge_base import KnowledgeBase


@dataclass
class CollectedData:
    """采集数据容器"""
    competitor: str
    basic_info: dict = field(default_factory=dict)
    product_features: dict = field(default_factory=dict)
    market_performance: dict = field(default_factory=dict)
    user_reviews: dict = field(default_factory=dict)
    strategic_news: dict = field(default_factory=dict)
    raw_sources: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "competitor": self.competitor,
            "basic_info": self.basic_info,
            "product_features": self.product_features,
            "market_performance": self.market_performance,
            "user_reviews": self.user_reviews,
            "strategic_news": self.strategic_news,
            "raw_sources": self.raw_sources,
        }


class CollectorAgent:
    """
    采集Agent (Async v2)

    职责：
    - 从搜索引擎并发获取竞品相关信息
    - 抓取网页详细内容
    - 整理和结构化采集数据
    - 使用 LLM Provider 提取关键信息点
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        search_engine: Optional[SearchTool] = None,
        web_scraper: Optional[WebScraper] = None,
        cache: Optional[LLMCache] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        # 初始化 Provider
        if provider:
            self.provider = provider
        else:
            provider_config = LLMProviderConfig(**SettingsProvider.to_dict())
            self.provider = OpenAIProvider(provider_config)

        # 初始化缓存
        self.cache = cache or LLMCache(
            cache_dir=CacheConfig.CACHE_DIR,
            ttl_seconds=CacheConfig.TTL_SECONDS,
            enabled=CacheConfig.ENABLED,
        )

        # 初始化工具
        self.search_engine = search_engine or SearchTool(timeout=30, max_results=10)
        self.web_scraper = web_scraper or WebScraper(timeout=30)

        # 初始化知识库 (RAG)
        self.kb = knowledge_base
        if self.kb is None and KnowledgeBaseConfig.ENABLED:
            try:
                self.kb = KnowledgeBase(
                    persist_dir=KnowledgeBaseConfig.PERSIST_DIR,
                    enabled=True,
                )
                logger.info("KnowledgeBase (RAG) 已启用")
            except Exception as exc:
                logger.warning(f"KnowledgeBase 初始化失败，RAG 已停用: {exc}")
                self.kb = KnowledgeBase(enabled=False)

        logger.info("CollectorAgent (async v2) 初始化完成")

    # ==================== 公共接口 ====================

    async def collect_batch(self, competitors: list[str]) -> list[CollectedData]:
        """
        批量并发采集多个竞品

        Args:
            competitors: 竞品列表

        Returns:
            采集数据列表
        """
        logger.info(f"开始并发采集 {len(competitors)} 个竞品")
        tasks = [self._collect_all_async(c) for c in competitors]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        collected = []
        for competitor, result in zip(competitors, results):
            if isinstance(result, Exception):
                logger.error(f"采集【{competitor}】失败: {result}")
            else:
                collected.append(result)

        logger.info(f"并发采集完成: 成功 {len(collected)}/{len(competitors)}")
        return collected

    # ==================== 核心并发采集 ====================

    async def _collect_all_async(self, competitor: str) -> CollectedData:
        """并发采集单个竞品的全部维度"""
        logger.info(f"开始并发采集【{competitor}】全维度信息")

        # RAG: check knowledge base for existing data
        kb_hit = await self._check_kb_before_dimension(
            competitor, "all", []
        )

        # 5 个维度并发执行
        basic_info, product_features, market_perf, user_reviews, strategic_news = (
            await asyncio.gather(
                self._collect_basic_info_async(competitor, kb_hit),
                self._collect_product_features_async(competitor),
                self._collect_market_performance_async(competitor),
                self._collect_user_reviews_async(competitor),
                self._collect_strategic_news_async(competitor),
                return_exceptions=True,
            )
        )

        collected = CollectedData(competitor=competitor)
        collected.basic_info = basic_info if not isinstance(basic_info, Exception) else {}
        collected.product_features = product_features if not isinstance(product_features, Exception) else {}
        collected.market_performance = market_perf if not isinstance(market_perf, Exception) else {}
        collected.user_reviews = user_reviews if not isinstance(user_reviews, Exception) else {}
        collected.strategic_news = strategic_news if not isinstance(strategic_news, Exception) else {}
        collected.raw_sources = self._collect_all_sources(collected)

        # RAG: ingest collected data into knowledge base
        await self._ingest_to_kb(competitor, collected)

        logger.info(f"【{competitor}】全维度并发采集完成")
        return collected

    # ==================== 各维度采集（异步） ====================

    async def _collect_basic_info_async(self, competitor: str, kb_hit: Optional[dict] = None) -> dict:
        """异步采集基本信息"""
        logger.info(f"采集【{competitor}】基本信息")

        if kb_hit and KnowledgeBaseConfig.CHECK_SEARCH_CACHE:
            logger.info(f"  📚 使用 KB 缓存数据 for '{competitor}'")

        queries = [
            f"{competitor} 公司介绍",
            f"{competitor} 融资情况",
            f"{competitor} 团队 创始人",
            f"{competitor} 发展历程",
        ]

        # 并发搜索
        all_results = await self._concurrent_search(queries, max_results=5)

        info_data = {"company_name": competitor, "search_results": all_results}

        if all_results:
            extracted = await self._extract_basic_info_with_llm_async(competitor, all_results)
            info_data.update(extracted)

        return info_data

    async def _collect_product_features_async(self, competitor: str) -> dict:
        """异步采集产品功能"""
        logger.info(f"采集【{competitor}】产品功能")

        queries = [
            f"{competitor} 功能特点",
            f"{competitor} 产品优势",
            f"{competitor} 核心功能",
            f"{competitor} pricing 价格",
        ]

        all_results = await self._concurrent_search(queries, max_results=5)

        features_data = {"competitor": competitor, "features": all_results, "pricing": {}}

        # 异步抓取官网
        official_info = await self._scrape_official_site_async(competitor, "product")
        if official_info:
            features_data["official_info"] = official_info

        return features_data

    async def _collect_market_performance_async(self, competitor: str) -> dict:
        """异步采集市场表现"""
        logger.info(f"采集【{competitor}】市场表现")

        queries = [
            f"{competitor} 用户数量",
            f"{competitor} 市场份额",
            f"{competitor} 营收 收入",
            f"{competitor} 增长 融资",
            f"{competitor} market share",
        ]

        search_results = await self._concurrent_search(queries, max_results=5)
        news_results = await self._search_news_async(competitor, max_results=10)

        return {
            "competitor": competitor,
            "metrics": {},
            "growth_trends": search_results,
            "news": news_results,
        }

    async def _collect_user_reviews_async(self, competitor: str) -> dict:
        """异步采集用户评价"""
        logger.info(f"采集【{competitor}】用户评价")

        queries = [
            f"{competitor} 用户评价",
            f"{competitor} review",
            f"{competitor} 口碑",
            f"{competitor} vs 对比",
        ]

        all_results = await self._concurrent_search(queries, max_results=8)

        return {
            "competitor": competitor,
            "positive": [],
            "negative": [],
            "comparisons": all_results,
        }

    async def _collect_strategic_news_async(self, competitor: str) -> dict:
        """异步采集战略动态"""
        logger.info(f"采集【{competitor}】战略动态")

        queries = [
            f"{competitor} 战略",
            f"{competitor} 合作 partnership",
            f"{competitor} 收购 acquisition",
            f"{competitor} 投资 funding",
        ]

        news = await self._search_news_async(competitor, max_results=15)
        strategic_results = await self._concurrent_search(queries, max_results=5)

        return {
            "competitor": competitor,
            "latest_news": news,
            "strategic_moves": strategic_results,
        }

    # ==================== 异步工具方法 ====================

    async def _concurrent_search(self, queries: list[str], max_results: int = 5) -> list[dict]:
        """并发执行多个搜索查询"""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, lambda q=q: self.search_engine.search(q, max_results))
            for q in queries
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for results in results_list:
            if isinstance(results, Exception):
                continue
            all_results.extend([r.to_dict() for r in results])

        return all_results

    async def _search_news_async(self, competitor: str, max_results: int = 5) -> list[dict]:
        """异步搜索新闻"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: self.search_engine.search_news(competitor, max_results)
        )
        return [r.to_dict() for r in results]

    async def _extract_basic_info_with_llm_async(self, competitor: str, search_results: list) -> dict:
        """使用 LLM 异步提取结构化基本信息"""
        prompt = f"""从以下搜索结果中提取{competitor}的基本信息，包括：
1. 公司成立时间
2. 创始人/CEO
3. 公司规模（员工数）
4. 融资阶段和金额
5. 总部位置

搜索结果：
{json.dumps(search_results, ensure_ascii=False, indent=2)}

请以JSON格式输出，字段使用英文: company_name, founded, founders, ceo, headquarters, employee_count, funding_stage, funding_amount, description
"""

        messages = [
            SystemMessage(content=COLLECTOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # Check cache first
        cached = self.cache.get(self.provider.config.model, self.provider._convert_messages(messages))
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass

        try:
            response = await self.provider.ainvoke(messages)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)

            # Cache the response
            self.cache.set(self.provider.config.model, self.provider._convert_messages(messages), json.dumps(result, ensure_ascii=False))
            return result
        except Exception as e:
            logger.warning(f"LLM 提取失败: {e}")
            return {}

    async def _scrape_official_site_async(self, competitor: str, info_type: str) -> Optional[dict]:
        """异步抓取官方网站"""
        possible_urls = [
            f"https://{competitor.lower().replace(' ', '')}.com",
            f"https://www.{competitor.lower().replace(' ', '')}.com",
        ]

        loop = asyncio.get_event_loop()
        for url in possible_urls:
            try:
                content = await loop.run_in_executor(None, self.web_scraper.scrape, url)
                if content.content and len(content.content) > 100:
                    return {
                        "title": content.title,
                        "url": content.url,
                        "summary": content.summary,
                    }
            except Exception as e:
                logger.debug(f"抓取 {url} 失败: {e}")
                continue

        return None

    # ==================== 同步兼容接口（供旧代码过渡） ====================

    def collect_batch_sync(self, competitors: list[str]) -> list[CollectedData]:
        """同步兼容接口"""
        return asyncio.run(self.collect_batch(competitors))

    def collect_all_sync(self, competitor: str) -> CollectedData:
        """同步兼容接口"""
        return asyncio.run(self._collect_all_async(competitor))

    def collect_basic_info(self, competitor: str) -> dict:
        return asyncio.run(self._collect_basic_info_async(competitor))

    def collect_product_features(self, competitor: str) -> dict:
        return asyncio.run(self._collect_product_features_async(competitor))

    def collect_market_performance(self, competitor: str) -> dict:
        return asyncio.run(self._collect_market_performance_async(competitor))

    def collect_user_reviews(self, competitor: str) -> dict:
        return asyncio.run(self._collect_user_reviews_async(competitor))

    def collect_strategic_news(self, competitor: str) -> dict:
        return asyncio.run(self._collect_strategic_news_async(competitor))

    # ==================== 辅助方法 ====================

    async def _check_kb_before_dimension(
        self, competitor: str, dimension: str, queries: list[str]
    ) -> Optional[dict]:
        """Check if recent data exists in knowledge base before searching."""
        if not self.kb or not self.kb.enabled:
            return None

        query = f"{competitor} {dimension}"
        try:
            history = self.kb.get_competitor_history(competitor, n_results=3)
            if history.get("collected_data"):
                logger.info(f"  📚 KB hit for '{competitor}/{dimension}' — reusing cached data")
                return {"kb_source": True, "data": history["collected_data"]}
        except Exception as exc:
            logger.debug(f"KB check failed: {exc}")

        return None

    async def _ingest_to_kb(self, competitor: str, collected: CollectedData):
        """Ingest collected data into knowledge base."""
        if not self.kb or not self.kb.enabled:
            return
        try:
            await self.kb.ingest_collected_data(competitor, collected.to_dict())
        except Exception as exc:
            logger.debug(f"KB ingest failed: {exc}")

    def _collect_all_sources(self, collected: CollectedData) -> list[dict]:
        sources = []

        def extract_urls(data):
            if isinstance(data, dict):
                for v in data.values():
                    extract_urls(v)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "url" in item:
                        sources.append({"url": item["url"], "competitor": collected.competitor})

        extract_urls(collected.to_dict())
        return list({s["url"]: s for s in sources}.values())


__all__ = ["CollectorAgent", "CollectedData"]
