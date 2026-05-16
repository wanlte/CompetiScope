"""
主管Agent模块 (Phase 2 — ReAct Agent Loop)

职责：协调整个竞品分析流程

Phase 2 升级: 从固定流水线升级为 ReAct Agent 自主循环
- LLM 自主决定何时搜索、分析、撰写
- 不再硬编码 _run_collection → _run_analysis → _run_writing
- 保留旧流水线作为 _legacy_* 方法向后兼容
"""

import json
import time
import asyncio
import uuid
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from loguru import logger

from config.prompts import REACT_AGENT_TASK_TEMPLATE
from config.settings import AgentConfig, AgentLoopConfig, KnowledgeBaseConfig
from llm.provider import BaseLLMProvider
from llm.openai_provider import OpenAIProvider
from llm.types import ProviderConfig as LLMProviderConfig
from config.settings import ProviderConfig as SettingsProvider

from agents.collector_agent import CollectorAgent, CollectedData
from agents.analyst_agent import AnalystAgent, AnalysisReport
from agents.writer_agent import WriterAgent, ReportMetadata

from agent.base_agent import ReActAgent, AgentResult
from agent.tool_registry import ToolRegistry
from agent.memory import ConversationMemory, WorkingMemory
from agent.reflector import Reflector
from tools.tool_adapter import create_agent_tools
from rag.knowledge_base import KnowledgeBase


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class TaskPhase(Enum):
    PLANNING = "planning"
    COLLECTION = "collection"
    ANALYSIS = "analysis"
    WRITING = "writing"
    FINALIZING = "finalizing"


@dataclass
class TaskInfo:
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
    task_id: str
    competitors: list[str]
    analysis_dimensions: list[str]
    report_type: str = "full"
    our_product: Optional[str] = None
    phase: TaskPhase = TaskPhase.PLANNING
    status: TaskStatus = TaskStatus.PENDING
    collected_data: list[CollectedData] = field(default_factory=list)
    analysis_report: Optional[AnalysisReport] = None
    final_report: Optional[str] = None
    tasks: list[TaskInfo] = field(default_factory=list)
    created_at: str = ""
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ManagerAgent:
    """主管Agent (Phase 2 — ReAct Agent Loop)

    职责：
    - 接收用户分析需求
    - 创建 ReAct Agent 并赋予工具
    - Agent 自主决定搜索→分析→撰写流程
    - 可选：自我反思 + 报告修订
    - 整合最终输出
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        use_agent_loop: bool = True,
    ):
        # ---- Provider ----
        if provider:
            self.provider = provider
        else:
            provider_config = LLMProviderConfig(**SettingsProvider.to_dict())
            self.provider = OpenAIProvider(provider_config)

        # ---- KnowledgeBase (RAG) ----
        self.kb = None
        if KnowledgeBaseConfig.ENABLED:
            try:
                self.kb = KnowledgeBase(
                    persist_dir=KnowledgeBaseConfig.PERSIST_DIR,
                    enabled=True,
                )
                logger.info("KnowledgeBase (RAG) 已启用")
            except Exception as exc:
                logger.warning(f"KnowledgeBase 初始化失败，RAG 已停用: {exc}")
                self.kb = KnowledgeBase(enabled=False)

        # ---- 子 Agent (共享 KB) ----
        self.collector = CollectorAgent(provider=self.provider, knowledge_base=self.kb)
        self.analyst = AnalystAgent(provider=self.provider, knowledge_base=self.kb)
        self.writer = WriterAgent(provider=self.provider, knowledge_base=self.kb)

        # ---- 配置 ----
        self.max_iterations = AgentConfig.SUPERVISOR_MAX_ITERATIONS
        self.timeout = AgentConfig.SUPERVISOR_TIMEOUT
        self.use_agent_loop = use_agent_loop

        logger.info("ManagerAgent (Phase 2 — ReAct Loop) 初始化完成")

    # ==================== 主分析接口 (ReAct Agent) ====================

    async def analyze_async(
        self,
        competitors: list[str],
        analysis_dimensions: Optional[list[str]] = None,
        report_type: str = "full",
        our_product: Optional[str] = None,
        show_progress: bool = True,
        enable_reflection: Optional[bool] = None,
    ) -> dict:
        """异步执行竞品分析（Phase 2 — ReAct Agent 自主循环）

        Args:
            competitors: 竞品列表
            analysis_dimensions: 分析维度
            report_type: 报告类型 (full/summary/snapshot)
            our_product: 我们产品
            show_progress: 显示进度
            enable_reflection: 是否启用反思（默认读取配置）

        Returns:
            分析结果字典
        """
        logger.info(f"ReAct Agent 模式启动: {competitors}")

        dimensions = analysis_dimensions or [
            "产品功能", "市场表现", "用户评价", "战略动态", "商业模式"
        ]

        task = AnalysisTask(
            task_id=self._generate_task_id(),
            competitors=competitors,
            analysis_dimensions=dimensions,
            report_type=report_type,
            our_product=our_product,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        try:
            # 1. 构建 ReAct 任务描述
            task_prompt = REACT_AGENT_TASK_TEMPLATE.format(
                competitors="、".join(competitors),
                dimensions="、".join(dimensions),
                our_product=our_product or "未指定",
                report_type=report_type,
            )

            # 2. 创建工具集
            tools = create_agent_tools(provider=self.provider)
            registry = ToolRegistry()
            for tool in tools:
                registry.register(tool)

            # 3. 创建 ReAct Agent
            agent = ReActAgent(
                provider=self.provider,
                tool_registry=registry,
                memory=ConversationMemory(),
                working_memory=WorkingMemory(),
                max_steps=AgentLoopConfig.MAX_STEPS,
                verbose=show_progress,
            )

            if show_progress:
                print(f"\n🤖 ReAct Agent 启动")
                print(f"   竞品: {', '.join(competitors)}")
                print(f"   维度: {', '.join(dimensions)}")
                print(f"   最大步数: {AgentLoopConfig.MAX_STEPS}")
                print(f"   可用工具: {', '.join(registry.tool_names)}")
                print()

            # 4. 运行 Agent 循环
            start_time = time.time()
            agent_result = await agent.run(task_prompt)
            elapsed = time.time() - start_time

            if show_progress:
                print(f"\n⏱ Agent 运行 {agent_result.total_steps} 步, 耗时 {elapsed:.1f}s")

            if not agent_result.success:
                task.status = TaskStatus.FAILED
                task.error = agent_result.error
                return {
                    "success": False,
                    "task_id": task.task_id,
                    "error": agent_result.error,
                    "agent_steps": agent_result.total_steps,
                }

            # 5. 可选：自我反思
            report = agent_result.answer
            reflection_results = []

            should_reflect = enable_reflection
            if should_reflect is None:
                should_reflect = AgentLoopConfig.REFLECTION_ENABLED

            if should_reflect:
                if show_progress:
                    print(f"\n🔍 自我反思中... (最多 {AgentLoopConfig.REFLECTION_ROUNDS} 轮)")

                reflector = Reflector(self.provider)
                report, reflection_results = await reflector.reflect_and_revise(
                    task=task_prompt,
                    answer=report,
                    max_rounds=AgentLoopConfig.REFLECTION_ROUNDS,
                )

                if show_progress and reflection_results:
                    final_score = reflection_results[-1].score
                    print(f"   最终评分: {final_score}/10")

            # 6. RAG: 将报告写入知识库
            if self.kb and self.kb.enabled:
                try:
                    await self.kb.ingest_report(competitors, report, report_type)
                    if show_progress:
                        print(f"   📚 报告已存入知识库")
                except Exception as exc:
                    logger.debug(f"KB ingest failed: {exc}")

            # 7. 组装结果
            task.status = TaskStatus.COMPLETED
            task.final_report = report
            task.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cost = self.provider.get_cost_summary()

            if show_progress:
                print(f"\n✅ 分析完成")
                print(f"   报告长度: {len(report)} 字符")
                print(f"   Agent 步数: {agent_result.total_steps}")
                print(f"   💰 API费用: ${cost['total_cost_usd']:.6f} ({cost['total_tokens']} tokens)")

            return {
                "success": True,
                "task_id": task.task_id,
                "report": report,
                "agent_steps": agent_result.total_steps,
                "agent_step_details": [
                    {
                        "step": s.step_num,
                        "thought": s.thought[:200] if s.thought else "",
                        "action": s.action,
                        "action_input": s.action_input,
                    }
                    for s in agent_result.steps
                ],
                "reflection_rounds": len(reflection_results),
                "reflection_scores": [r.score for r in reflection_results],
                "completed_at": task.completed_at,
                "cost": cost,
            }

        except Exception as e:
            logger.error(f"Agent 执行失败: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return {"success": False, "task_id": task.task_id, "error": str(e)}

    async def analyze_with_planning_async(
        self,
        target_company: str,
        report_type: str = "full",
        show_progress: bool = True,
    ) -> dict:
        """带智能规划的异步竞品分析"""
        logger.info(f"智能规划模式: {target_company}")

        plan = await self._create_analysis_plan_async(target_company)

        if show_progress:
            print(f"\n📋 分析规划:")
            print(f"   目标: {target_company}")
            print(f"   竞品: {', '.join(plan['competitors'])}")
            print(f"   维度: {', '.join(plan['dimensions'])}")
            print()

        return await self.analyze_async(
            competitors=plan["competitors"],
            analysis_dimensions=plan["dimensions"],
            report_type=report_type,
            our_product=plan.get("our_product"),
            show_progress=show_progress,
        )

    # ==================== 旧流水线 (向后兼容) ====================

    async def analyze_legacy_async(
        self,
        competitors: list[str],
        analysis_dimensions: Optional[list[str]] = None,
        report_type: str = "full",
        our_product: Optional[str] = None,
        show_progress: bool = True,
    ) -> dict:
        """旧版固定流水线分析（向后兼容）"""
        logger.info(f"Legacy 流水线模式: {competitors}")

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
            await self._run_collection_phase_async(task, show_progress)
            await self._run_analysis_phase_async(task, show_progress)
            await self._run_writing_phase_async(task, show_progress)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cost = self.provider.get_cost_summary()

            return {
                "success": True,
                "task_id": task.task_id,
                "report": task.final_report,
                "collected_data": [d.to_dict() for d in task.collected_data],
                "analysis_report": task.analysis_report.to_dict() if task.analysis_report else None,
                "completed_at": task.completed_at,
                "cost": cost,
            }

        except Exception as e:
            logger.error(f"Legacy 任务执行失败: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return {"success": False, "task_id": task.task_id, "error": str(e)}

    # ==================== 同步兼容接口 ====================

    def analyze(self, **kwargs) -> dict:
        """同步兼容接口"""
        return asyncio.run(self.analyze_async(**kwargs))

    def analyze_with_planning(self, **kwargs) -> dict:
        return asyncio.run(self.analyze_with_planning_async(**kwargs))

    def analyze_legacy(self, **kwargs) -> dict:
        return asyncio.run(self.analyze_legacy_async(**kwargs))

    # ==================== 旧流水线阶段方法 ====================

    async def _run_collection_phase_async(self, task: AnalysisTask, show_progress: bool):
        """旧版 — 异步采集阶段"""
        task.phase = TaskPhase.COLLECTION
        if show_progress:
            print(f"\n🔍 阶段1: 数据采集中... (并发模式)")

        collection_task = TaskInfo(
            task_id=f"{task.task_id}_collection",
            phase=TaskPhase.COLLECTION,
            status=TaskStatus.RUNNING,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        task.tasks.append(collection_task)
        start_time = time.time()

        try:
            collected_data = await self.collector.collect_batch(task.competitors)
            task.collected_data = collected_data
            collection_task.status = TaskStatus.COMPLETED
            collection_task.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            collection_task.message = f"并发采集 {len(collected_data)} 个竞品数据"

            if show_progress:
                elapsed = time.time() - start_time
                print(f"   ✅ 采集完成 ({elapsed:.1f}秒) [并发]")
                print(f"   成功: {len(collected_data)}/{len(task.competitors)} 个竞品")
        except Exception as e:
            logger.error(f"采集阶段失败: {e}")
            collection_task.status = TaskStatus.FAILED
            collection_task.error = str(e)
            raise

    async def _run_analysis_phase_async(self, task: AnalysisTask, show_progress: bool):
        """旧版 — 异步分析阶段"""
        task.phase = TaskPhase.ANALYSIS
        if show_progress:
            print(f"\n📊 阶段2: 数据分析中...")

        analysis_task = TaskInfo(
            task_id=f"{task.task_id}_analysis",
            phase=TaskPhase.ANALYSIS,
            status=TaskStatus.RUNNING,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        task.tasks.append(analysis_task)
        start_time = time.time()

        try:
            analysis_report = self.analyst.analyze_all(task.collected_data)
            task.analysis_report = analysis_report
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

    async def _run_writing_phase_async(self, task: AnalysisTask, show_progress: bool):
        """旧版 — 异步撰写阶段"""
        task.phase = TaskPhase.WRITING
        if show_progress:
            print(f"\n✍️  阶段3: 报告撰写中...")

        writing_task = TaskInfo(
            task_id=f"{task.task_id}_writing",
            phase=TaskPhase.WRITING,
            status=TaskStatus.RUNNING,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        task.tasks.append(writing_task)
        start_time = time.time()

        try:
            metadata = ReportMetadata(
                title=f"竞品分析报告 - {', '.join(task.competitors)}",
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                report_type=task.report_type,
            )

            if task.report_type == "snapshot":
                report = self.writer.write_snapshot_report(task.analysis_report, metadata)
            elif task.report_type == "summary":
                report = self.writer.write_summary_report(task.analysis_report, metadata)
            else:
                report = self.writer.write_full_report(task.analysis_report, task.collected_data, metadata)

            task.final_report = report
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

    # ==================== 智能规划 ====================

    async def _create_analysis_plan_async(self, target_company: str) -> dict:
        """使用 LLM 异步创建分析计划"""
        from langchain_core.messages import HumanMessage, SystemMessage
        from config.prompts import SUPERVISOR_SYSTEM_PROMPT

        prompt = f"""分析{target_company}的竞品情况，并规划分析任务。

请输出JSON格式的分析计划：
{{
  "competitors": ["竞品1", "竞品2", "竞品3"],
  "dimensions": ["产品功能", "市场表现", "用户评价"],
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
            response = await self.provider.ainvoke(messages)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except Exception as e:
            logger.error(f"创建分析计划失败: {e}")
            return {
                "competitors": [target_company],
                "dimensions": ["产品功能", "市场表现", "用户评价"],
                "our_product": None,
                "priority": "中",
            }

    def _generate_task_id(self) -> str:
        return f"task_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    def get_cost_summary(self) -> dict:
        return self.provider.get_cost_summary()


__all__ = ["ManagerAgent", "AnalysisTask", "TaskStatus", "TaskPhase", "TaskInfo"]
