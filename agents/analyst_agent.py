"""
分析Agent模块

负责对采集的竞品数据进行深度分析，包括：
- 功能对比矩阵
- SWOT分析
- 竞争格局分析
- 趋势预测
"""

import json
from typing import Optional, Literal
from dataclasses import dataclass, field
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import ANALYZER_SYSTEM_PROMPT
from config.settings import LLMConfig, LLMRequestConfig
from agents.collector_agent import CollectedData


@dataclass
class FeatureComparison:
    """功能对比项"""
    feature_name: str           # 功能名称
    description: str = ""       # 功能描述
    competitor_scores: dict = field(default_factory=dict)  # 各竞品评分(1-5)
    our_score: int = 0          # 我们产品的评分


@dataclass
class SWOTAnalysis:
    """SWOT分析结果"""
    competitor: str
    strengths: list[str] = field(default_factory=list)   # 优势
    weaknesses: list[str] = field(default_factory=list)  # 劣势
    opportunities: list[str] = field(default_factory=list)  # 机会
    threats: list[str] = field(default_factory=list)     # 威胁


@dataclass
class AnalysisReport:
    """分析报告"""
    competitors: list[str]
    feature_comparison: list[FeatureComparison]  # 功能对比
    swot_analysis: list[SWOTAnalysis]            # SWOT分析
    competitive_landscape: dict = field(default_factory=dict)  # 竞争格局
    key_insights: list[str] = field(default_factory=list)     # 关键洞察
    risks: list[str] = field(default_factory=list)            # 风险
    opportunities: list[str] = field(default_factory=list)    # 机会

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "competitors": self.competitors,
            "feature_comparison": [
                {
                    "feature_name": f.feature_name,
                    "description": f.description,
                    "competitor_scores": f.competitor_scores,
                    "our_score": f.our_score,
                }
                for f in self.feature_comparison
            ],
            "swot_analysis": [
                {
                    "competitor": s.competitor,
                    "strengths": s.strengths,
                    "weaknesses": s.weaknesses,
                    "opportunities": s.opportunities,
                    "threats": s.threats,
                }
                for s in self.swot_analysis
            ],
            "competitive_landscape": self.competitive_landscape,
            "key_insights": self.key_insights,
            "risks": self.risks,
            "opportunities": self.opportunities,
        }


class AnalystAgent:
    """
    分析Agent

    职责：
    - 对采集数据进行深度分析
    - 生成功能对比矩阵
    - 进行SWOT分析
    - 识别竞争格局和关键洞察
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        初始化分析Agent

        Args:
            model: 模型名称（默认使用配置）
            api_key: API密钥（默认使用配置）
            base_url: API基础URL（默认使用配置）
        """
        self.llm = ChatOpenAI(
            model=model or LLMConfig.get_model(),
            api_key=api_key or LLMConfig.get_api_key(),
            base_url=base_url or LLMConfig.get_base_url(),
            temperature=LLMRequestConfig.TEMPERATURE,
            max_tokens=LLMRequestConfig.MAX_TOKENS,
        )

        logger.info("AnalystAgent初始化完成")

    def analyze_feature_comparison(
        self,
        collected_data: list[CollectedData],
        our_features: Optional[dict] = None
    ) -> list[FeatureComparison]:
        """
        分析功能对比

        Args:
            collected_data: 采集的数据列表
            our_features: 我们产品的功能（可选）

        Returns:
            功能对比列表
        """
        logger.info(f"开始功能对比分析，共 {len(collected_data)} 个竞品")

        # 构建分析提示
        competitors_info = []
        for data in collected_data:
            competitors_info.append({
                "name": data.competitor,
                "features": data.product_features.get("features", [])[:5],
            })

        prompt = f"""基于以下竞品信息，生成功能对比矩阵。

竞品信息：
{json.dumps(competitors_info, ensure_ascii=False, indent=2)}

请分析以下核心功能维度的对比：
1. 用户体验/界面设计
2. 核心功能完备度
3. 性能/稳定性
4. 价格/性价比
5. 客户服务
6. 第三方集成
7. 数据安全/隐私

输出格式（JSON数组）：
[
  {{
    "feature_name": "功能名称",
    "description": "功能描述",
    "competitor_scores": {{"竞品A": 4, "竞品B": 3}}
  }}
]

评分标准：1=差，2=一般，3=良好，4=优秀，5=卓越
"""

        messages = [
            SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # 解析JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            comparisons = json.loads(content)

            # 转换为FeatureComparison对象
            result = []
            for item in comparisons:
                result.append(FeatureComparison(
                    feature_name=item.get("feature_name", ""),
                    description=item.get("description", ""),
                    competitor_scores=item.get("competitor_scores", {}),
                    our_score=item.get("our_score", 0),
                ))

            logger.info(f"功能对比分析完成，共 {len(result)} 个维度")
            return result

        except Exception as e:
            logger.error(f"功能对比分析失败: {e}")
            return []

    def analyze_swot(
        self,
        collected_data: list[CollectedData]
    ) -> list[SWOTAnalysis]:
        """
        进行SWOT分析

        Args:
            collected_data: 采集的数据列表

        Returns:
            SWOT分析结果列表
        """
        logger.info(f"开始SWOT分析，共 {len(collected_data)} 个竞品")

        swot_results = []

        for data in collected_data:
            logger.info(f"分析【{data.competitor}】SWOT")

            # 构建分析提示
            prompt = f"""对{data.competitor}进行SWOT分析。

公司基本信息：
{json.dumps(data.basic_info, ensure_ascii=False, indent=2)}

产品功能：
{json.dumps(data.product_features, ensure_ascii=False, indent=2)}

市场表现：
{json.dumps(data.market_performance, ensure_ascii=False, indent=2)}

用户评价：
{json.dumps(data.user_reviews, ensure_ascii=False, indent=2)}

战略动态：
{json.dumps(data.strategic_news, ensure_ascii=False, indent=2)}

请输出JSON格式的SWOT分析：
{{
  "competitor": "{data.competitor}",
  "strengths": ["优势1", "优势2"],
  "weaknesses": ["劣势1", "劣势2"],
  "opportunities": ["机会1", "机会2"],
  "threats": ["威胁1", "威胁2"]
}}

要求：
- 每个类别至少3条
- 描述要具体，基于提供的数据
- 使用中文
"""

            messages = [
                SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            try:
                response = self.llm.invoke(messages)
                content = response.content.strip()

                # 解析JSON
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]

                swot_data = json.loads(content)

                swot_results.append(SWOTAnalysis(
                    competitor=swot_data.get("competitor", data.competitor),
                    strengths=swot_data.get("strengths", []),
                    weaknesses=swot_data.get("weaknesses", []),
                    opportunities=swot_data.get("opportunities", []),
                    threats=swot_data.get("threats", []),
                ))

            except Exception as e:
                logger.error(f"【{data.competitor}】SWOT分析失败: {e}")
                # 创建空结果
                swot_results.append(SWOTAnalysis(competitor=data.competitor))

        logger.info(f"SWOT分析完成")
        return swot_results

    def analyze_competitive_landscape(
        self,
        collected_data: list[CollectedData],
        feature_comparison: list[FeatureComparison]
    ) -> dict:
        """
        分析竞争格局

        Args:
            collected_data: 采集的数据列表
            feature_comparison: 功能对比结果

        Returns:
            竞争格局分析结果
        """
        logger.info("开始竞争格局分析")

        # 构建分析数据
        analysis_data = {
            "competitors": [d.competitor for d in collected_data],
            "market_data": [
                {
                    "name": d.competitor,
                    "metrics": d.market_performance.get("metrics", {}),
                }
                for d in collected_data
            ],
            "features": [
                {
                    "name": f.feature_name,
                    "scores": f.competitor_scores,
                }
                for f in feature_comparison[:5]
            ],
        }

        prompt = f"""分析以下竞品的竞争格局：

{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

请从以下维度进行分析：

1. **市场定位**：各竞品的目标用户和市场定位有何不同？
2. **竞争强度**：哪些竞品之间竞争最激烈？
3. **差异化因素**：各竞品的核心差异化优势是什么？
4. **市场空白**：哪些细分市场或需求还未被充分满足？

输出JSON格式：
{{
  "market_positions": {{"竞品名": "定位描述"}},
  "competitive_intensity": "竞争强度描述",
  "differentiation_factors": ["因素1", "因素2"],
  "market_gaps": ["空白1", "空白2"],
  "leader": "市场领导者",
  "challengers": ["挑战者列表"],
  "niche_players": ["细分玩家列表"]
}}
"""

        messages = [
            SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)
            logger.info("竞争格局分析完成")
            return result

        except Exception as e:
            logger.error(f"竞争格局分析失败: {e}")
            return {}

    def extract_key_insights(
        self,
        collected_data: list[CollectedData],
        swot_analysis: list[SWOTAnalysis],
        competitive_landscape: dict
    ) -> tuple[list[str], list[str], list[str]]:
        """
        提取关键洞察

        Args:
            collected_data: 采集的数据列表
            swot_analysis: SWOT分析结果
            competitive_landscape: 竞争格局分析

        Returns:
            (关键洞察, 风险列表, 机会列表)
        """
        logger.info("提取关键洞察")

        prompt = f"""基于以下分析结果，提取关键洞察、风险和机会：

竞品列表：{[d.competitor for d in collected_data]}

SWOT分析：
{json.dumps([s.to_dict() if hasattr(s, 'to_dict') else s for s in swot_analysis], ensure_ascii=False, indent=2)}

竞争格局：
{json.dumps(competitive_landscape, ensure_ascii=False, indent=2)}

请输出JSON格式：
{{
  "key_insights": ["洞察1", "洞察2", "洞察3"],
  "risks": ["风险1", "风险2"],
  "opportunities": ["机会1", "机会2"]
}}

要求：
- 关键洞察3-5条，每条要具体、有价值
- 风险要可量化、可应对
- 机会要可执行、有前景
- 使用中文
"""

        messages = [
            SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)

            insights = result.get("key_insights", [])
            risks = result.get("risks", [])
            opportunities = result.get("opportunities", [])

            logger.info(f"提取洞察完成：{len(insights)}条洞察，{len(risks)}个风险，{len(opportunities)}个机会")
            return insights, risks, opportunities

        except Exception as e:
            logger.error(f"提取洞察失败: {e}")
            return [], [], []

    def analyze_all(
        self,
        collected_data: list[CollectedData],
        our_features: Optional[dict] = None
    ) -> AnalysisReport:
        """
        执行完整的分析流程

        Args:
            collected_data: 采集的数据列表
            our_features: 我们产品的功能（可选）

        Returns:
            完整的分析报告
        """
        logger.info("开始完整的竞品分析流程")

        competitors = [d.competitor for d in collected_data]

        # 1. 功能对比分析
        feature_comparison = self.analyze_feature_comparison(collected_data, our_features)

        # 2. SWOT分析
        swot_analysis = self.analyze_swot(collected_data)

        # 3. 竞争格局分析
        competitive_landscape = self.analyze_competitive_landscape(
            collected_data, feature_comparison
        )

        # 4. 提取关键洞察
        key_insights, risks, opportunities = self.extract_key_insights(
            collected_data, swot_analysis, competitive_landscape
        )

        report = AnalysisReport(
            competitors=competitors,
            feature_comparison=feature_comparison,
            swot_analysis=swot_analysis,
            competitive_landscape=competitive_landscape,
            key_insights=key_insights,
            risks=risks,
            opportunities=opportunities,
        )

        logger.info("竞品分析完成")
        return report

    def generate_feature_table(
        self,
        feature_comparison: list[FeatureComparison]
    ) -> str:
        """
        生成功能对比表格（Markdown格式）

        Args:
            feature_comparison: 功能对比列表

        Returns:
            Markdown格式的表格
        """
        if not feature_comparison:
            return ""

        # 获取所有竞品
        all_competitors = set()
        for f in feature_comparison:
            all_competitors.update(f.competitor_scores.keys())

        competitors = sorted(all_competitors)

        # 构建表格
        lines = [
            "| 功能维度 | " + " | ".join(competitors) + " |",
            "|" + "---|" * (len(competitors) + 1),
        ]

        for f in feature_comparison:
            scores = []
            for c in competitors:
                score = f.competitor_scores.get(c, "-")
                if isinstance(score, int):
                    stars = "⭐" * score
                    scores.append(stars if stars else "-")
                else:
                    scores.append(str(score))
            lines.append(f"| {f.feature_name} | " + " | ".join(scores) + " |")

        return "\n".join(lines)


# 导出
__all__ = ["AnalystAgent", "AnalysisReport", "FeatureComparison", "SWOTAnalysis"]
