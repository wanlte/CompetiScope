"""
主管Agent模块

负责协调整个竞品分析流程，包括：
- 任务规划和分配
- 流程控制和状态跟踪
- 异常处理和重试
- 结果整合和输出
"""

import json
import time
from typing import Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import SUPERVISOR_SYSTEM_PROMPT
from config.settings import LLMConfig, LLMRequestConfig, AgentConfig
from agents.collector_agent import CollectorAgent, CollectedData
from agents.analyst_agent import AnalystAgent, AnalysisReport
from agents.writer_agent import WriterAgent, ReportMetadata


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"      # 待执行
    RUNNING = "running"      # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    RETRYING = "retrying"    # 重试中


class TaskPhase(Enum):
    """任务阶段枚举"""
    PLANNING = "planning"        # 规划阶段
    COLLECTION = "collection"    # 采集阶段
    ANALYSIS = "analysis"        # 分析阶段
    WRITING = "writing"          # 撰写阶段
    FINALIZING = "finalizing"    # 收尾阶段


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    phase: TaskPhase
    status: TaskStatus = TaskStatus.PENDING
    message: str = ""
    created_at: str = ""
    updated_at: str = ""
    error: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "phase": self.phase.value,
            "status": self.status.value,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class AnalysisTask:
    """竞品分析任务"""
    task_id: str
    competitors: list[str]           # 竞品列表
    analysis_dimensions: list[str]   # 分析维度
    report_type: str = "full"        # 报告类型
    our_product: Optional[str] = None  # 我们产品（用于对比）

    # 执行状态
    phase: TaskPhase = TaskPhase.PLANNING
    status: TaskStatus = TaskStatus.PENDING

    # 各阶段结果
    collected_data: list[CollectedData] = field(default_factory=list)
    analysis_report: Optional[AnalysisReport] = None
    final_report: Optional[str] = None

    # 任务信息
    tasks: list[TaskInfo] = field(default_factory=list)
    created_at: str = ""
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ManagerAgent:
    """
    主管Agent

    职责：
    - 接收用户分析需求
    - 规划和分配任务给子Agent
    - 协调整个分析流程
    - 处理异常和重试
    - 整合最终输出
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        初始化主管Agent

        Args:
            model: 模型名称（默认使用配置）
            api_key: API密钥（默认使用配置）
            base_url: API基础URL（默认使用配置）
        """
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=model or LLMConfig.get_model(),
            api_key=api_key or LLMConfig.get_api_key(),
            base_url=base_url or LLMConfig.get_base_url(),
            temperature=LLMRequestConfig.TEMPERATURE,
            max_tokens=LLMRequestConfig.MAX_TOKENS,
        )

        # 初始化子Agent
        self.collector = CollectorAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()

        # 配置参数
        self.max_iterations = AgentConfig.SUPERVISOR_MAX_ITERATIONS
        self.timeout = AgentConfig.SUPERVISOR_TIMEOUT

        logger.info("ManagerAgent初始化完成")

    def analyze(
        self,
        competitors: list[str],
        analysis_dimensions: Optional[list[str]] = None,
        report_type: str = "full",
        our_product: Optional[str] = None,
        show_progress: bool = True
    ) -> dict:
        """
        执行竞品分析

        Args:
            competitors: 竞品列表
            analysis_dimensions: 分析维度（可选）
            report_type: 报告类型（full/summary/snapshot）
            our_product: 我们产品名称（用于对比）
            show_progress: 是否显示进度

        Returns:
            分析结果字典
        """
        logger.info(f"开始竞品分析任务: {competitors}")

        # 创建任务
        task = AnalysisTask(
            task_id=self._generate_task_id(),
            competitors=competitors,
            analysis_dimensions=analysis_dimensions or [
                "产品功能", "市场表现", "用户评价", "战略动态", "商业模式"
            ],
            report_type=report_type,
            our_product=our_product,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        try:
            # 执行各阶段
            self._run_collection_phase(task, show_progress)
            self._run_analysis_phase(task, show_progress)
            self._run_writing_phase(task, show_progress)

            # 完成任务
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            logger.info("竞品分析任务完成")

            return {
                "success": True,
                "task_id": task.task_id,
                "report": task.final_report,
                "collected_data": [d.to_dict() for d in task.collected_data],
                "analysis_report": task.analysis_report.to_dict() if task.analysis_report else None,
                "completed_at": task.completed_at,
            }

        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)

            return {
                "success": False,
                "task_id": task.task_id,
                "error": str(e),
            }

    def analyze_with_planning(
        self,
        target_company: str,
        report_type: str = "full",
        show_progress: bool = True
    ) -> dict:
        """
        带智能规划的竞品分析

        Args:
            target_company: 目标公司/产品
            report_type: 报告类型
            show_progress: 是否显示进度

        Returns:
            分析结果
        """
        logger.info(f"智能规划模式分析: {target_company}")

        # 使用LLM规划分析任务
        plan = self._create_analysis_plan(target_company)

        if show_progress:
            print(f"\n📋 分析规划:")
            print(f"   目标: {target_company}")
            print(f"   竞品: {', '.join(plan['competitors'])}")
            print(f"   维度: {', '.join(plan['dimensions'])}")
            print()

        # 执行分析
        return self.analyze(
            competitors=plan["competitors"],
            analysis_dimensions=plan["dimensions"],
            report_type=report_type,
            our_product=plan.get("our_product"),
            show_progress=show_progress,
        )

    # ==================== 私有方法 ====================

    def _run_collection_phase(
        self,
        task: AnalysisTask,
        show_progress: bool
    ):
        """
        执行采集阶段

        Args:
            task: 分析任务
            show_progress: 是否显示进度
        """
        task.phase = TaskPhase.COLLECTION

        if show_progress:
            print(f"\n🔍 阶段1: 数据采集中...")
            print(f"   竞品数量: {len(task.competitors)}")

        # 创建采集任务
        collection_task = TaskInfo(
            task_id=f"{task.task_id}_collection",
            phase=TaskPhase.COLLECTION,
            status=TaskStatus.RUNNING,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        task.tasks.append(collection_task)

        start_time = time.time()

        try:
            # 批量采集
            collected_data = self.collector.collect_batch(task.competitors)

            task.collected_data = collected_data

            # 更新任务状态
            collection_task.status = TaskStatus.COMPLETED
            collection_task.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            collection_task.message = f"成功采集 {len(collected_data)} 个竞品数据"

            if show_progress:
                elapsed = time.time() - start_time
                print(f"   ✅ 采集完成 ({elapsed:.1f}秒)")
                print(f"   成功: {len(collected_data)}/{len(task.competitors)} 个竞品")

        except Exception as e:
            logger.error(f"采集阶段失败: {e}")
            collection_task.status = TaskStatus.FAILED
            collection_task.error = str(e)
            raise

    def _run_analysis_phase(
        self,
        task: AnalysisTask,
        show_progress: bool
    ):
        """
        执行分析阶段

        Args:
            task: 分析任务
            show_progress: 是否显示进度
        """
        task.phase = TaskPhase.ANALYSIS

        if show_progress:
            print(f"\n📊 阶段2: 数据分析中...")

        # 创建分析任务
        analysis_task = TaskInfo(
            task_id=f"{task.task_id}_analysis",
            phase=TaskPhase.ANALYSIS,
            status=TaskStatus.RUNNING,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        task.tasks.append(analysis_task)

        start_time = time.time()

        try:
            # 执行分析
            analysis_report = self.analyst.analyze_all(task.collected_data)

            task.analysis_report = analysis_report

            # 更新任务状态
            analysis_task.status = TaskStatus.COMPLETED
            analysis_task.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            analysis_task.message = "分析完成"

            if show_progress:
                elapsed = time.time() - start_time
                print(f"   ✅ 分析完成 ({elapsed:.1f}秒)")
                print(f"   发现: {len(analysis_report.key_insights)} 条关键洞察")
                print(f"   识别: {len(analysis_report.opportunities)} 个机会")

        except Exception as e:
            logger.error(f"分析阶段失败: {e}")
            analysis_task.status = TaskStatus.FAILED
            analysis_task.error = str(e)
            raise

    def _run_writing_phase(
        self,
        task: AnalysisTask,
        show_progress: bool
    ):
        """
        执行撰写阶段

        Args:
            task: 分析任务
            show_progress: 是否显示进度
        """
        task.phase = TaskPhase.WRITING

        if show_progress:
            print(f"\n✍️  阶段3: 报告撰写中...")

        # 创建撰写任务
        writing_task = TaskInfo(
            task_id=f"{task.task_id}_writing",
            phase=TaskPhase.WRITING,
            status=TaskStatus.RUNNING,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        task.tasks.append(writing_task)

        start_time = time.time()

        try:
            # 创建报告元数据
            metadata = ReportMetadata(
                title=f"竞品分析报告 - {', '.join(task.competitors)}",
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                report_type=task.report_type,
            )

            # 根据报告类型生成报告
            if task.report_type == "snapshot":
                report = self.writer.write_snapshot_report(task.analysis_report, metadata)
            elif task.report_type == "summary":
                report = self.writer.write_summary_report(task.analysis_report, metadata)
            else:
                report = self.writer.write_full_report(
                    task.analysis_report,
                    task.collected_data,
                    metadata
                )

            task.final_report = report

            # 更新任务状态
            writing_task.status = TaskStatus.COMPLETED
            writing_task.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writing_task.message = f"报告生成完成 ({len(report)} 字符)"

            if show_progress:
                elapsed = time.time() - start_time
                print(f"   ✅ 报告生成完成 ({elapsed:.1f}秒)")
                print(f"   报告长度: {len(report)} 字符")

        except Exception as e:
            logger.error(f"撰写阶段失败: {e}")
            writing_task.status = TaskStatus.FAILED
            writing_task.error = str(e)
            raise

    def _create_analysis_plan(self, target_company: str) -> dict:
        """
        使用LLM创建分析计划

        Args:
            target_company: 目标公司

        Returns:
            分析计划字典
        """
        prompt = f"""分析{target_company}的竞品情况，并规划分析任务。

请输出JSON格式的分析计划：
{{
  "competitors": ["竞品1", "竞品2", "竞品3"],  // 主要竞品列表（3-5个）
  "dimensions": ["产品功能", "市场表现", "用户评价"],  // 分析维度
  "our_product": "对比产品名（如果有）",
  "priority": "高/中/低"
}}

请基于{target_company}的特点，推荐合适的竞品和分析重点。
"""

        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            plan = json.loads(content)
            return plan

        except Exception as e:
            logger.error(f"创建分析计划失败: {e}")
            # 返回默认计划
            return {
                "competitors": [target_company],
                "dimensions": ["产品功能", "市场表现", "用户评价"],
                "our_product": None,
                "priority": "中",
            }

    def _generate_task_id(self) -> str:
        """生成任务ID"""
        import uuid
        return f"task_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            任务状态字典
        """
        # 简化实现，实际应从存储中获取
        return None

    def list_recent_tasks(self, limit: int = 10) -> list[dict]:
        """
        列出最近的任务

        Args:
            limit: 返回数量限制

        Returns:
            任务列表
        """
        # 简化实现，实际应从存储中获取
        return []


# 导出
__all__ = ["ManagerAgent", "AnalysisTask", "TaskStatus", "TaskPhase", "TaskInfo"]
