"""
撰写Agent模块 (v2)

负责将分析结果转化为专业的竞品分析报告，支持：
- 完整分析报告
- 执行摘要
- 快速概览
- SSE 流式生成

Phase 1 升级: Provider 模式
Phase 4 升级: SSE Streaming
"""

import json
import asyncio
from typing import Optional, Literal, AsyncIterator
from datetime import datetime
from dataclasses import dataclass
from loguru import logger

from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import WRITER_SYSTEM_PROMPT
from config.settings import (
    LLMConfig, LLMRequestConfig, OutputConfig, ProviderConfig as SettingsProvider,
    KnowledgeBaseConfig,
)
from llm.provider import BaseLLMProvider
from llm.openai_provider import OpenAIProvider
from llm.types import ProviderConfig as LLMProviderConfig
from agents.analyst_agent import AnalysisReport, FeatureComparison, SWOTAnalysis
from agents.collector_agent import CollectedData
from rag.knowledge_base import KnowledgeBase


@dataclass
class ReportMetadata:
    """报告元数据"""
    title: str = ""
    author: str = "竞品分析系统"
    created_at: str = ""
    report_type: str = "full"  # full, summary, snapshot
    language: str = "zh-CN"


class WriterAgent:
    """
    撰写Agent

    职责：
    - 根据分析结果生成结构化报告
    - 支持多种报告格式（完整报告、执行摘要、快速概览）
    - 输出Markdown/HTML/JSON格式
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        """
        初始化撰写Agent

        Args:
            provider: LLM Provider（优先使用，推荐）
            model: 模型名称（回退兼容）
            api_key: API密钥（回退兼容）
            base_url: API基础URL（回退兼容）
            knowledge_base: 知识库 (RAG)
        """
        if provider:
            self.provider = provider
        else:
            provider_config = LLMProviderConfig(**SettingsProvider.to_dict())
            if model:
                provider_config.model = model
            if api_key:
                provider_config.api_key = api_key
            if base_url:
                provider_config.base_url = base_url
            self.provider = OpenAIProvider(provider_config)

        # 向后兼容
        self.llm = self.provider

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

        logger.info("WriterAgent (v2) 初始化完成")

    def _invoke(self, messages: list) -> str:
        """同步 LLM 调用辅助方法"""
        return asyncio.run(self.provider.ainvoke(messages)).content

    def write_full_report(
        self,
        analysis_report: AnalysisReport,
        collected_data: list[CollectedData],
        metadata: Optional[ReportMetadata] = None
    ) -> str:
        """
        生成完整的竞品分析报告

        Args:
            analysis_report: 分析报告
            collected_data: 原始采集数据
            metadata: 报告元数据

        Returns:
            Markdown格式的完整报告
        """
        logger.info("生成完整竞品分析报告")

        if metadata is None:
            metadata = ReportMetadata(
                title="竞品分析报告",
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                report_type="full",
            )

        # 生成各部分内容
        sections = []

        # 1. 报告封面
        sections.append(self._generate_cover(analysis_report, metadata))

        # 2. 执行摘要
        sections.append(self._generate_executive_summary(analysis_report))

        # 3. 市场概览
        sections.append(self._generate_market_overview(analysis_report))

        # 4. 竞品全景分析
        sections.append(self._generate_competitor_overview(analysis_report))

        # 5. 功能对比
        sections.append(self._generate_feature_comparison_section(analysis_report))

        # 6. SWOT分析
        sections.append(self._generate_swot_section(analysis_report))

        # 7. 竞争格局
        sections.append(self._generate_competitive_landscape_section(analysis_report))

        # 8. 关键洞察
        sections.append(self._generate_insights_section(analysis_report))

        # 9. 风险与机会
        sections.append(self._generate_risk_opportunity_section(analysis_report))

        # 10. 战略建议
        sections.append(self._generate_strategy_section(analysis_report))

        # 11. 附录
        sections.append(self._generate_appendix(collected_data))

        report = "\n\n".join(sections)

        logger.info(f"完整报告生成完成，共 {len(report)} 字符")
        return report

    def write_summary_report(
        self,
        analysis_report: AnalysisReport,
        metadata: Optional[ReportMetadata] = None
    ) -> str:
        """
        生成执行摘要报告

        Args:
            analysis_report: 分析报告
            metadata: 报告元数据

        Returns:
            Markdown格式的执行摘要
        """
        logger.info("生成执行摘要报告")

        if metadata is None:
            metadata = ReportMetadata(
                title="竞品分析执行摘要",
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                report_type="summary",
            )

        sections = [
            self._generate_cover(analysis_report, metadata),
            self._generate_executive_summary(analysis_report, detailed=False),
            self._generate_key_findings(analysis_report),
        ]

        report = "\n\n".join(sections)

        logger.info(f"执行摘要生成完成，共 {len(report)} 字符")
        return report

    def write_snapshot_report(
        self,
        analysis_report: AnalysisReport,
        metadata: Optional[ReportMetadata] = None
    ) -> str:
        """
        生成快速概览报告

        Args:
            analysis_report: 分析报告
            metadata: 报告元数据

        Returns:
            Markdown格式的快速概览
        """
        logger.info("生成快速概览报告")

        if metadata is None:
            metadata = ReportMetadata(
                title="竞品分析快速概览",
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                report_type="snapshot",
            )

        lines = [
            "# 竞品快照",
            "",
            f"**生成时间**: {metadata.created_at}",
            "",
            "## 关键发现",
        ]

        # 关键发现
        for i, insight in enumerate(analysis_report.key_insights[:5], 1):
            lines.append(f"{i}. {insight}")

        # 竞品排名
        lines.extend([
            "",
            "## 竞品概览",
            "",
            "| 竞品 | 核心优势 | 主要劣势 |",
            "|------|----------|----------|",
        ])

        for swot in analysis_report.swot_analysis:
            strengths = ", ".join(swot.strengths[:2]) if swot.strengths else "-"
            weaknesses = ", ".join(swot.weaknesses[:2]) if swot.weaknesses else "-"
            lines.append(f"| {swot.competitor} | {strengths} | {weaknesses} |")

        # 行动建议
        lines.extend([
            "",
            "## 行动建议",
            "",
        ])

        for i, opportunity in enumerate(analysis_report.opportunities[:3], 1):
            lines.append(f"{i}. {opportunity}")

        report = "\n".join(lines)

        logger.info(f"快速概览生成完成，共 {len(report)} 字符")
        return report

    def save_report(
        self,
        report_content: str,
        filename: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        保存报告到文件

        Args:
            report_content: 报告内容
            filename: 文件名
            output_dir: 输出目录（默认使用配置）

        Returns:
            保存的文件路径
        """
        import os
        from pathlib import Path

        output_path = Path(output_dir) if output_dir else OutputConfig.OUTPUT_DIR
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"报告已保存: {file_path}")
        return str(file_path)

    # ==================== Phase 4: SSE Streaming ====================

    async def astream_full_report(
        self,
        analysis_report: AnalysisReport,
        collected_data: list[CollectedData],
        metadata: Optional[ReportMetadata] = None,
    ) -> AsyncIterator[str]:
        """Stream full report generation — yields sections as they complete.

        Usage (FastAPI SSE):
            async for chunk in writer.astream_full_report(report, data):
                yield {"event": "section", "data": chunk}
        """
        if metadata is None:
            metadata = ReportMetadata(
                title="竞品分析报告",
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                report_type="full",
            )

        sections = [
            ("cover", self._generate_cover(analysis_report, metadata)),
            ("executive_summary", self._generate_executive_summary(analysis_report)),
            ("market_overview", self._generate_market_overview(analysis_report)),
            ("competitor_overview", self._generate_competitor_overview(analysis_report)),
            ("feature_comparison", self._generate_feature_comparison_section(analysis_report)),
            ("swot", self._generate_swot_section(analysis_report)),
            ("competitive_landscape", self._generate_competitive_landscape_section(analysis_report)),
            ("insights", self._generate_insights_section(analysis_report)),
            ("risk_opportunity", self._generate_risk_opportunity_section(analysis_report)),
            ("strategy", self._generate_strategy_section(analysis_report)),
            ("appendix", self._generate_appendix(collected_data)),
        ]

        for section_name, section_content in sections:
            yield json.dumps({"section": section_name, "content": section_content}, ensure_ascii=False)
            await asyncio.sleep(0.01)  # yield control

    async def awrite_full_report_async(
        self,
        analysis_report: AnalysisReport,
        collected_data: list[CollectedData],
        metadata: Optional[ReportMetadata] = None,
    ) -> str:
        """Async version of write_full_report — same result, non-blocking."""
        return await asyncio.to_thread(
            self.write_full_report, analysis_report, collected_data, metadata
        )

    # ==================== 私有方法 ====================

    def _get_historical_reports(self, competitors: list[str]) -> str:
        """Retrieve historical report snippets for style and content reference."""
        if not self.kb or not self.kb.enabled:
            return ""

        parts = []
        for competitor in competitors:
            try:
                history = self.kb.get_competitor_history(competitor, n_results=2)
                if history.get("reports"):
                    parts.append(f"### {competitor} 历史报告参考")
                    for r in history["reports"][:2]:
                        parts.append(r.get("content", "")[:400])
            except Exception as exc:
                logger.debug(f"RAG report retrieval failed: {exc}")

        return "\n\n".join(parts) if parts else ""

    async def _ingest_report_to_kb(self, competitors: list[str], report: str, report_type: str):
        """Store generated report in knowledge base."""
        if not self.kb or not self.kb.enabled:
            return
        try:
            await self.kb.ingest_report(competitors, report, report_type)
        except Exception as exc:
            logger.debug(f"KB report ingestion failed: {exc}")

    def _generate_cover(
        self,
        analysis_report: AnalysisReport,
        metadata: ReportMetadata
    ) -> str:
        """生成报告封面"""
        competitors_str = "、".join(analysis_report.competitors)

        return f"""# {metadata.title}

**分析竞品**: {competitors_str}

**生成时间**: {metadata.created_at}

**分析机构**: {metadata.author}

---

*本报告由竞品分析系统自动生成*
"""

    def _generate_executive_summary(
        self,
        analysis_report: AnalysisReport,
        detailed: bool = True
    ) -> str:
        """生成执行摘要"""
        landscape = analysis_report.competitive_landscape

        sections = [
            "## 执行摘要",
            "",
            f"本报告对 **{', '.join(analysis_report.competitors)}** 进行了全面的竞品分析。",
            "",
        ]

        # 市场地位
        if "leader" in landscape:
            sections.append(f"**市场领导者**: {landscape['leader']}")

        if "challengers" in landscape and landscape["challengers"]:
            sections.append(f"**主要挑战者**: {', '.join(landscape['challengers'])}")

        sections.append("")

        # 核心发现
        sections.append("### 核心发现")

        if detailed:
            for i, insight in enumerate(analysis_report.key_insights[:5], 1):
                sections.append(f"{i}. {insight}")
        else:
            for i, insight in enumerate(analysis_report.key_insights[:3], 1):
                sections.append(f"- {insight}")

        return "\n".join(sections)

    def _generate_market_overview(self, analysis_report: AnalysisReport) -> str:
        """生成市场概览章节"""
        landscape = analysis_report.competitive_landscape

        sections = [
            "## 市场概览",
            "",
        ]

        if "market_positions" in landscape:
            sections.append("### 市场定位分布")
            for competitor, position in landscape["market_positions"].items():
                sections.append(f"- **{competitor}**: {position}")

        return "\n".join(sections)

    def _generate_competitor_overview(self, analysis_report: AnalysisReport) -> str:
        """生成竞品概览章节"""
        sections = [
            "## 竞品全景分析",
            "",
            f"本次分析涵盖 **{len(analysis_report.competitors)}** 个主要竞品：",
            "",
        ]

        for competitor in analysis_report.competitors:
            sections.append(f"- **{competitor}**")

        return "\n".join(sections)

    def _generate_feature_comparison_section(
        self,
        analysis_report: AnalysisReport
    ) -> str:
        """生成功能对比章节"""
        sections = [
            "## 功能对比分析",
            "",
            "### 核心功能对比矩阵",
            "",
        ]

        if analysis_report.feature_comparison:
            # 构建Markdown表格
            all_competitors = set()
            for f in analysis_report.feature_comparison:
                all_competitors.update(f.competitor_scores.keys())

            competitors = sorted(all_competitors)

            # 表头
            header = "| 功能维度 | " + " | ".join(competitors) + " |"
            separator = "|" + "---|" * (len(competitors) + 1)

            sections.append(header)
            sections.append(separator)

            # 数据行
            for f in analysis_report.feature_comparison:
                scores = []
                for c in competitors:
                    score = f.competitor_scores.get(c, "-")
                    if isinstance(score, int):
                        stars = "⭐" * score
                        scores.append(stars if stars else "-")
                    else:
                        scores.append(str(score))
                sections.append(f"| {f.feature_name} | " + " | ".join(scores) + " |")
        else:
            sections.append("*暂无功能对比数据*")

        return "\n".join(sections)

    def _generate_swot_section(self, analysis_report: AnalysisReport) -> str:
        """生成SWOT分析章节"""
        sections = [
            "## SWOT分析",
            "",
        ]

        for swot in analysis_report.swot_analysis:
            sections.extend([
                f"### {swot.competitor}",
                "",
                "| 维度 | 内容 |",
                "|-----|-----|",
                f"| **优势 (S)** | {', '.join(swot.strengths) if swot.strengths else '-'} |",
                f"| **劣势 (W)** | {', '.join(swot.weaknesses) if swot.weaknesses else '-'} |",
                f"| **机会 (O)** | {', '.join(swot.opportunities) if swot.opportunities else '-'} |",
                f"| **威胁 (T)** | {', '.join(swot.threats) if swot.threats else '-'} |",
                "",
            ])

        return "\n".join(sections)

    def _generate_competitive_landscape_section(
        self,
        analysis_report: AnalysisReport
    ) -> str:
        """生成竞争格局章节"""
        landscape = analysis_report.competitive_landscape

        sections = [
            "## 竞争格局分析",
            "",
        ]

        if "market_gaps" in landscape and landscape["market_gaps"]:
            sections.append("### 市场空白")
            for gap in landscape["market_gaps"]:
                sections.append(f"- {gap}")
            sections.append("")

        if "differentiation_factors" in landscape and landscape["differentiation_factors"]:
            sections.append("### 差异化因素")
            for factor in landscape["differentiation_factors"]:
                sections.append(f"- {factor}")

        return "\n".join(sections)

    def _generate_insights_section(self, analysis_report: AnalysisReport) -> str:
        """生成关键洞察章节"""
        sections = [
            "## 关键洞察",
            "",
        ]

        for i, insight in enumerate(analysis_report.key_insights, 1):
            sections.append(f"**{i}. {insight}**")
            sections.append("")

        return "\n".join(sections)

    def _generate_risk_opportunity_section(
        self,
        analysis_report: AnalysisReport
    ) -> str:
        """生成风险与机会章节"""
        sections = [
            "## 风险与机会",
            "",
            "### 主要风险",
            "",
        ]

        if analysis_report.risks:
            for risk in analysis_report.risks:
                sections.append(f"- {risk}")
        else:
            sections.append("*暂无风险数据*")

        sections.extend(["", "### 核心机会", ""])

        if analysis_report.opportunities:
            for opportunity in analysis_report.opportunities:
                sections.append(f"- {opportunity}")
        else:
            sections.append("*暂无机会数据*")

        return "\n".join(sections)

    def _generate_strategy_section(self, analysis_report: AnalysisReport) -> str:
        """生成战略建议章节"""
        # RAG: retrieve historical reports for context
        historical_ref = self._get_historical_reports(analysis_report.competitors)
        rag_part = ""
        if historical_ref:
            rag_part = f"""

历史分析参考（请参考其风格和深度，但不抄袭）：
{historical_ref[:800]}
"""

        prompt = f"""基于以下竞品分析结果，生成3-5条战略建议：

竞品：{', '.join(analysis_report.competitors)}

关键洞察：
{chr(10).join(analysis_report.key_insights)}

机会：
{chr(10).join(analysis_report.opportunities)}

风险：
{chr(10).join(analysis_report.risks)}
{rag_part}

请以Markdown列表格式输出战略建议，每条建议要具体、可执行。
"""

        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            strategy = self._invoke(messages).strip()
        except Exception as e:
            logger.error(f"生成战略建议失败: {e}")
            strategy = "*建议生成失败*"

        sections = [
            "## 战略建议",
            "",
            strategy,
        ]

        return "\n".join(sections)

    def _generate_key_findings(self, analysis_report: AnalysisReport) -> str:
        """生成关键发现章节（用于摘要）"""
        sections = [
            "## 关键发现",
            "",
        ]

        # Top 3洞察
        sections.append("### 最重要发现")
        for i, insight in enumerate(analysis_report.key_insights[:3], 1):
            sections.append(f"{i}. {insight}")

        # Top 3机会
        sections.extend(["", "### 优先机会"])
        for i, opportunity in enumerate(analysis_report.opportunities[:3], 1):
            sections.append(f"{i}. {opportunity}")

        return "\n".join(sections)

    def _generate_appendix(self, collected_data: list[CollectedData]) -> str:
        """生成附录章节"""
        sections = [
            "## 附录",
            "",
            "### 数据来源",
            "",
        ]

        # 收集所有数据源
        sources = []
        for data in collected_data:
            sources.extend(data.raw_sources)

        # 去重
        unique_sources = list({s.get("url", ""): s for s in sources}.values())

        if unique_sources:
            for source in unique_sources[:20]:  # 限制数量
                url = source.get("url", "")
                if url:
                    sections.append(f"- {url}")
        else:
            sections.append("*暂无数据源信息*")

        sections.extend([
            "",
            "---",
            f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ])

        return "\n".join(sections)


# 导出
__all__ = ["WriterAgent", "ReportMetadata"]
