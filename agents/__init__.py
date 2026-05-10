"""
Agent模块

包含四个Agent：
- collector: 采集Agent - 从多渠道收集竞品信息
- analyst: 分析Agent - 深度分析和洞察提炼
- writer: 撰写Agent - 生成专业分析报告
- manager: 主管Agent - 协调整个分析流程
"""

from .collector_agent import CollectorAgent, CollectedData
from .analyst_agent import AnalystAgent, AnalysisReport, FeatureComparison, SWOTAnalysis
from .writer_agent import WriterAgent, ReportMetadata
from .manager_agent import ManagerAgent, AnalysisTask, TaskStatus, TaskPhase, TaskInfo

__all__ = [
    # 采集Agent
    "CollectorAgent",
    "CollectedData",
    # 分析Agent
    "AnalystAgent",
    "AnalysisReport",
    "FeatureComparison",
    "SWOTAnalysis",
    # 撰写Agent
    "WriterAgent",
    "ReportMetadata",
    # 主管Agent
    "ManagerAgent",
    "AnalysisTask",
    "TaskStatus",
    "TaskPhase",
    "TaskInfo",
]
