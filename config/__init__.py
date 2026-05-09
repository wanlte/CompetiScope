"""
配置模块

包含：
- settings: 系统配置
- prompts: Agent提示词
"""

from .settings import (
    LLMConfig,
    LLMRequestConfig,
    SearchConfig,
    AgentConfig,
    OutputConfig,
    LogConfig,
    KnowledgeBaseConfig,
    get_settings,
    validate_config,
)

from .prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    COLLECTOR_SYSTEM_PROMPT,
    ANALYZER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    get_agent_prompt,
    format_task_message,
)

__all__ = [
    # 配置类
    "LLMConfig",
    "LLMRequestConfig",
    "SearchConfig",
    "AgentConfig",
    "OutputConfig",
    "LogConfig",
    "KnowledgeBaseConfig",
    "get_settings",
    "validate_config",
    # 提示词
    "SUPERVISOR_SYSTEM_PROMPT",
    "COLLECTOR_SYSTEM_PROMPT",
    "ANALYZER_SYSTEM_PROMPT",
    "WRITER_SYSTEM_PROMPT",
    "get_agent_prompt",
    "format_task_message",
]
