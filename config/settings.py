"""
竞品分析Agent系统配置文件
包含LLM配置、搜索配置、Agent配置等
"""
import os
from typing import Literal
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# LLM 配置
# ============================================================

class LLMConfig:
    """LLM配置类，支持多种模型提供商"""

    # 支持的模型提供商
    PROVIDERS = Literal["deepseek", "openai"]

    # DeepSeek 配置
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # OpenAI 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # 当前使用的提供商
    ACTIVE_PROVIDER: PROVIDERS = os.getenv("LLM_PROVIDER", "deepseek")

    @classmethod
    def get_api_key(cls) -> str:
        """获取当前提供商的API密钥"""
        if cls.ACTIVE_PROVIDER == "deepseek":
            return cls.DEEPSEEK_API_KEY
        return cls.OPENAI_API_KEY

    @classmethod
    def get_base_url(cls) -> str:
        """获取当前提供商的API基础URL"""
        if cls.ACTIVE_PROVIDER == "deepseek":
            return cls.DEEPSEEK_BASE_URL
        return cls.OPENAI_BASE_URL

    @classmethod
    def get_model(cls) -> str:
        """获取当前使用的模型"""
        if cls.ACTIVE_PROVIDER == "deepseek":
            return cls.DEEPSEEK_MODEL
        return cls.OPENAI_MODEL

    @classmethod
    def switch_provider(cls, provider: PROVIDERS) -> None:
        """切换LLM提供商"""
        cls.ACTIVE_PROVIDER = provider


# ============================================================
# LLM 请求配置
# ============================================================

class LLMRequestConfig:
    """LLM请求相关配置"""

    # 请求超时时间（秒）
    TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))

    # 最大重试次数
    MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # 重试间隔（秒）
    RETRY_DELAY: float = float(os.getenv("LLM_RETRY_DELAY", "2.0"))

    # 最大token数
    MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    # Temperature配置
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # 请求间隔（秒），避免API限流
    REQUEST_INTERVAL: float = float(os.getenv("LLM_REQUEST_INTERVAL", "1.0"))


# ============================================================
# 搜索配置
# ============================================================

class SearchConfig:
    """搜索相关配置"""

    # 搜索引擎配置
    SEARCH_ENGINE: str = os.getenv("SEARCH_ENGINE", "serper")  # 支持: serper, duckduckgo, bing

    # Serper API配置
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")

    # DuckDuckGo配置（免费，无需API Key）
    DUCKDUCKGO_MAXRESULTS: int = int(os.getenv("DUCKDUCKGO_MAXRESULTS", "10"))

    # Bing搜索配置
    BING_API_KEY: str = os.getenv("BING_API_KEY", "")

    # 搜索结果数量
    DEFAULT_SEARCH_RESULTS: int = int(os.getenv("DEFAULT_SEARCH_RESULTS", "10"))

    # 搜索超时（秒）
    SEARCH_TIMEOUT: int = int(os.getenv("SEARCH_TIMEOUT", "30"))

    # 每个关键词搜索次数
    SEARCH_ITERATIONS: int = int(os.getenv("SEARCH_ITERATIONS", "2"))


# ============================================================
# Agent 配置
# ============================================================

class AgentConfig:
    """Agent相关配置"""

    # 主管Agent配置
    SUPERVISOR_MAX_ITERATIONS: int = int(os.getenv("SUPERVISOR_MAX_ITERATIONS", "10"))
    SUPERVISOR_TIMEOUT: int = int(os.getenv("SUPERVISOR_TIMEOUT", "300"))

    # 采集Agent配置
    COLLECTOR_MAX_RETRIES: int = int(os.getenv("COLLECTOR_MAX_RETRIES", "3"))
    COLLECTOR_TIMEOUT: int = int(os.getenv("COLLECTOR_TIMEOUT", "180"))

    # 分析Agent配置
    ANALYZER_MAX_RETRIES: int = int(os.getenv("ANALYZER_MAX_RETRIES", "3"))
    ANALYZER_TIMEOUT: int = int(os.getenv("ANALYZER_TIMEOUT", "180"))

    # 撰写Agent配置
    WRITER_MAX_RETRIES: int = int(os.getenv("WRITER_MAX_RETRIES", "3"))
    WRITER_TIMEOUT: int = int(os.getenv("WRITER_TIMEOUT", "180"))

    # Agent间消息传递的最大token限制
    MESSAGE_TOKEN_LIMIT: int = int(os.getenv("MESSAGE_TOKEN_LIMIT", "16000"))


# ============================================================
# 输出配置
# ============================================================

class OutputConfig:
    """输出相关配置"""

    # 输出目录
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "output")))

    # 输出格式
    OUTPUT_FORMAT: str = os.getenv("OUTPUT_FORMAT", "markdown")  # markdown, json, html

    # 是否保存中间结果
    SAVE_INTERMEDIATE: bool = os.getenv("SAVE_INTERMEDIATE", "true").lower() == "true"

    # 中间结果保存目录
    INTERMEDIATE_DIR: Path = OUTPUT_DIR / "intermediate"

    # 是否生成可视化图表
    GENERATE_CHARTS: bool = os.getenv("GENERATE_CHARTS", "false").lower() == "true"


# ============================================================
# 日志配置
# ============================================================

class LogConfig:
    """日志相关配置"""

    # 日志级别
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR

    # 日志目录
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs")))

    # 日志文件名格式
    LOG_FILE_FORMAT: str = os.getenv("LOG_FILE_FORMAT", "competiscope_{date}.log")

    # 是否打印到控制台
    LOG_TO_CONSOLE: bool = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"

    # 日志格式
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ============================================================
# 知识库配置（可选）
# ============================================================

class KnowledgeBaseConfig:
    """知识库配置（用于存储历史分析结果）"""

    # 是否启用知识库
    ENABLE_KNOWLEDGE_BASE: bool = os.getenv("ENABLE_KNOWLEDGE_BASE", "false").lower() == "true"

    # 知识库存储路径
    KNOWLEDGE_BASE_DIR: Path = Path(os.getenv("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "knowledge_base")))

    # 知识库向量维度
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1536"))

    # 相似度阈值
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))


# ============================================================
# 全局配置实例
# ============================================================

def get_settings():
    """获取完整配置字典"""
    return {
        "llm": {
            "provider": LLMConfig.ACTIVE_PROVIDER,
            "api_key": LLMConfig.get_api_key(),
            "base_url": LLMConfig.get_base_url(),
            "model": LLMConfig.get_model(),
            "request": {
                "timeout": LLMRequestConfig.TIMEOUT,
                "max_retries": LLMRequestConfig.MAX_RETRIES,
                "retry_delay": LLMRequestConfig.RETRY_DELAY,
                "max_tokens": LLMRequestConfig.MAX_TOKENS,
                "temperature": LLMRequestConfig.TEMPERATURE,
            }
        },
        "search": {
            "engine": SearchConfig.SEARCH_ENGINE,
            "default_results": SearchConfig.DEFAULT_SEARCH_RESULTS,
            "timeout": SearchConfig.SEARCH_TIMEOUT,
        },
        "agents": {
            "supervisor": {
                "max_iterations": AgentConfig.SUPERVISOR_MAX_ITERATIONS,
                "timeout": AgentConfig.SUPERVISOR_TIMEOUT,
            },
            "collector": {
                "max_retries": AgentConfig.COLLECTOR_MAX_RETRIES,
                "timeout": AgentConfig.COLLECTOR_TIMEOUT,
            },
            "analyzer": {
                "max_retries": AgentConfig.ANALYZER_MAX_RETRIES,
                "timeout": AgentConfig.ANALYZER_TIMEOUT,
            },
            "writer": {
                "max_retries": AgentConfig.WRITER_MAX_RETRIES,
                "timeout": AgentConfig.WRITER_TIMEOUT,
            },
        },
        "output": {
            "dir": str(OutputConfig.OUTPUT_DIR),
            "format": OutputConfig.OUTPUT_FORMAT,
            "save_intermediate": OutputConfig.SAVE_INTERMEDIATE,
        },
        "log": {
            "level": LogConfig.LOG_LEVEL,
            "dir": str(LogConfig.LOG_DIR),
        }
    }


def validate_config() -> tuple[bool, list[str]]:
    """
    验证配置是否完整
    返回: (是否通过, 错误信息列表)
    """
    errors = []

    # 检查LLM配置
    if not LLMConfig.get_api_key():
        errors.append("LLM API密钥未配置")

    # 检查搜索配置
    if SearchConfig.SEARCH_ENGINE == "serper" and not SearchConfig.SERPER_API_KEY:
        errors.append("Serper API密钥未配置")

    # 检查目录
    try:
        OutputConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"目录创建失败: {e}")

    return len(errors) == 0, errors
