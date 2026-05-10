"""
采集Agent模块

负责从多渠道收集竞品信息，包括：
- 公司基本信息
- 产品功能
- 市场表现
- 用户评价
- 战略动态
"""

import json
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import COLLECTOR_SYSTEM_PROMPT
from config.settings import LLMConfig, LLMRequestConfig
from tools.search_tool import SearchTool, SearchResult
from tools.web_scraper import WebScraper, ScrapedContent


@dataclass
class CollectedData:
    """采集数据容器"""
    competitor: str                           # 竞品名称
    basic_info: dict = field(default_factory=dict)    # 基本信息
    product_features: dict = field(default_factory=dict)  # 产品功能
    market_performance: dict = field(default_factory=dict)  # 市场表现
    user_reviews: list = field(default_factory=list)    # 用户评价
    strategic_news: list = field(default_factory=list)  # 战略动态
    raw_sources: list = field(default_factory=list)     # 原始数据源

    def to_dict(self) -> dict:
        """转换为字典"""
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
    采集Agent

    职责：
    - 从搜索引擎获取竞品相关信息
    - 抓取网页详细内容
    - 整理和结构化采集数据
    - 提取关键信息点
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        search_engine: Optional[SearchTool] = None,
        web_scraper: Optional[WebScraper] = None,
    ):
        """
        初始化采集Agent

        Args:
            model: 模型名称（默认使用配置）
            api_key: API密钥（默认使用配置）
            base_url: API基础URL（默认使用配置）
            search_engine: 搜索引擎实例（可选）
            web_scraper: 网页抓取器实例（可选）
        """
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=model or LLMConfig.get_model(),
            api_key=api_key or LLMConfig.get_api_key(),
            base_url=base_url or LLMConfig.get_base_url(),
            temperature=LLMRequestConfig.TEMPERATURE,
            max_tokens=LLMRequestConfig.MAX_TOKENS,
        )

        # 初始化工具
        self.search_engine = search_engine or SearchTool(
            timeout=30,
            max_results=10,
        )
        self.web_scraper = web_scraper or WebScraper(
            timeout=30,
        )

        logger.info("CollectorAgent初始化完成")

    def collect_basic_info(self, competitor: str) -> dict:
        """
        采集竞品基本信息

        Args:
            competitor: 竞品名称

        Returns:
            基本信息字典
        """
        logger.info(f"采集【{competitor}】基本信息")

        # 搜索关键词组合
        search_queries = [
            f"{competitor} 公司介绍",
            f"{competitor} 融资情况",
            f"{competitor} 团队 创始人",
            f"{competitor} 发展历程",
        ]

        info_data = {
            "company_name": competitor,
            "search_results": [],
        }

        # 执行搜索
        for query in search_queries:
            results = self.search_engine.search(query, max_results=5)
            info_data["search_results"].extend([r.to_dict() for r in results])

        # 使用LLM提取关键信息
        if info_data["search_results"]:
            extracted_info = self._extract_basic_info_with_llm(
                competitor, info_data["search_results"]
            )
            info_data.update(extracted_info)

        return info_data

    def collect_product_features(self, competitor: str) -> dict:
        """
        采集竞品产品功能

        Args:
            competitor: 竞品名称

        Returns:
            产品功能字典
        """
        logger.info(f"采集【{competitor}】产品功能")

        search_queries = [
            f"{competitor} 功能特点",
            f"{competitor} 产品优势",
            f"{competitor} 核心功能",
            f"{competitor} pricing 价格",
        ]

        features_data = {
            "competitor": competitor,
            "features": [],
            "pricing": {},
        }

        for query in search_queries:
            results = self.search_engine.search(query, max_results=5)
            features_data["features"].extend([r.to_dict() for r in results])

        # 抓取官方网站获取产品详情
        official_info = self._scrape_official_site(competitor, "product")
        if official_info:
            features_data["official_info"] = official_info

        return features_data

    def collect_market_performance(self, competitor: str) -> dict:
        """
        采集竞品市场表现

        Args:
            competitor: 竞品名称

        Returns:
            市场表现字典
        """
        logger.info(f"采集【{competitor}】市场表现")

        search_queries = [
            f"{competitor} 用户数量",
            f"{competitor} 市场份额",
            f"{competitor} 营收 收入",
            f"{competitor} 增长 融资",
            f"{competitor} market share",
        ]

        market_data = {
            "competitor": competitor,
            "metrics": {},
            "growth_trends": [],
        }

        for query in search_queries:
            results = self.search_engine.search(query, max_results=5)
            market_data["growth_trends"].extend([r.to_dict() for r in results])

        # 新闻搜索
        news_results = self.search_engine.search_news(competitor, max_results=10)
        market_data["news"] = [r.to_dict() for r in news_results]

        return market_data

    def collect_user_reviews(self, competitor: str) -> dict:
        """
        采集竞品用户评价

        Args:
            competitor: 竞品名称

        Returns:
            用户评价字典
        """
        logger.info(f"采集【{competitor}】用户评价")

        search_queries = [
            f"{competitor} 用户评价",
            f"{competitor} review",
            f"{competitor} 口碑",
            f"{competitor} vs 对比",
        ]

        reviews_data = {
            "competitor": competitor,
            "positive": [],
            "negative": [],
            "comparisons": [],
        }

        for query in search_queries:
            results = self.search_engine.search(query, max_results=8)
            reviews_data["comparisons"].extend([r.to_dict() for r in results])

        return reviews_data

    def collect_strategic_news(self, competitor: str) -> dict:
        """
        采集竞品战略动态

        Args:
            competitor: 竞品名称

        Returns:
            战略动态字典
        """
        logger.info(f"采集【{competitor}】战略动态")

        # 新闻搜索
        news_results = self.search_engine.search_news(competitor, max_results=15)

        # 搜索战略相关关键词
        strategic_queries = [
            f"{competitor} 战略",
            f"{competitor} 合作  partnership",
            f"{competitor} 收购 acquisition",
            f"{competitor} 投资 funding",
        ]

        strategic_data = {
            "competitor": competitor,
            "latest_news": [r.to_dict() for r in news_results],
            "strategic_moves": [],
        }

        for query in strategic_queries:
            results = self.search_engine.search(query, max_results=5)
            strategic_data["strategic_moves"].extend([r.to_dict() for r in results])

        return strategic_data

    def collect_all(self, competitor: str) -> CollectedData:
        """
        采集竞品全维度信息

        Args:
            competitor: 竞品名称

        Returns:
            完整的采集数据
        """
        logger.info(f"开始采集【{competitor}】全维度信息")

        collected = CollectedData(competitor=competitor)

        # 依次采集各维度信息
        collected.basic_info = self.collect_basic_info(competitor)
        collected.product_features = self.collect_product_features(competitor)
        collected.market_performance = self.collect_market_performance(competitor)
        collected.user_reviews = self.collect_user_reviews(competitor)
        collected.strategic_news = self.collect_strategic_news(competitor)

        # 记录原始数据源
        collected.raw_sources = self._collect_all_sources(collected)

        logger.info(f"【{competitor}】信息采集完成")
        return collected

    def collect_batch(self, competitors: list[str]) -> list[CollectedData]:
        """
        批量采集多个竞品

        Args:
            competitors: 竞品列表

        Returns:
            采集数据列表
        """
        logger.info(f"开始批量采集 {len(competitors)} 个竞品")

        results = []
        for competitor in competitors:
            try:
                data = self.collect_all(competitor)
                results.append(data)
            except Exception as e:
                logger.error(f"采集【{competitor}】失败: {e}")
                continue

        logger.info(f"批量采集完成: 成功 {len(results)}/{len(competitors)}")
        return results

    def _extract_basic_info_with_llm(self, competitor: str, search_results: list) -> dict:
        """
        使用LLM从搜索结果中提取基本信息

        Args:
            competitor: 竞品名称
            search_results: 搜索结果列表

        Returns:
            提取的信息字典
        """
        prompt = f"""从以下搜索结果中提取{competitor}的基本信息，包括：
1. 公司成立时间
2. 创始人/CEO
3. 公司规模（员工数）
4. 融资阶段和金额
5. 总部位置

搜索结果：
{json.dumps(search_results, ensure_ascii=False, indent=2)}

请以JSON格式输出，字段使用英文。
"""

        messages = [
            SystemMessage(content=COLLECTOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            # 尝试解析JSON
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except Exception as e:
            logger.warning(f"LLM提取失败: {e}")
            return {}

    def _scrape_official_site(self, competitor: str, info_type: str) -> Optional[dict]:
        """
        抓取官方网站获取信息

        Args:
            competitor: 竞品名称
            info_type: 信息类型

        Returns:
            抓取的内容
        """
        # 常见的官方网站URL模式
        possible_urls = [
            f"https://{competitor.lower().replace(' ', '')}.com",
            f"https://www.{competitor.lower().replace(' ', '')}.com",
        ]

        for url in possible_urls:
            try:
                content = self.web_scraper.scrape(url)
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

    def _collect_all_sources(self, collected: CollectedData) -> list[dict]:
        """
        收集所有原始数据源URL

        Args:
            collected: 采集数据

        Returns:
            数据源列表
        """
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


# 导出
__all__ = ["CollectorAgent", "CollectedData"]
