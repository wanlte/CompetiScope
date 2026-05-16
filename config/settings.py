"""
竞品分析Agent系统配置文件
包含LLM配置、搜索配置、Agent配置等
"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# LLM 配置
# ============================================================

class LLMConfig:
    """LLM配置类，支持多种模型提供商"""

    # DeepSeek 配置
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    @classmethod
    def get_api_key(cls) -> str:
        return cls.DEEPSEEK_API_KEY

    @classmethod
    def get_base_url(cls) -> str:
        return cls.DEEPSEEK_BASE_URL

    @classmethod
    def get_model(cls) -> str:
        return cls.DEEPSEEK_MODEL


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
# LLM Provider 配置
# ============================================================

class ProviderConfig:
    """LLM Provider 配置（lazy — reads env at access time to support testing）"""

    @staticmethod
    def _api_key() -> str:
        return os.getenv("DEEPSEEK_API_KEY", "")

    @staticmethod
    def _base_url() -> str:
        return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    @staticmethod
    def _model() -> str:
        return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    @classmethod
    def to_dict(cls) -> dict:
        return {
            "api_key": cls._api_key(),
            "base_url": cls._base_url(),
            "model": cls._model(),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
            "timeout": int(os.getenv("LLM_TIMEOUT", "60")),
            "max_retries": int(os.getenv("LLM_MAX_RETRIES", "3")),
            "retry_delay": float(os.getenv("LLM_RETRY_DELAY", "2.0")),
            "input_price_per_1k": float(os.getenv("LLM_INPUT_PRICE", "0.00014")),
            "output_price_per_1k": float(os.getenv("LLM_OUTPUT_PRICE", "0.00028")),
        }


# ============================================================
# 缓存配置
# ============================================================

class CacheConfig:
    """LLM 响应缓存配置"""

    # 是否启用缓存
    ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"

    # 缓存目录
    CACHE_DIR: str = os.getenv("CACHE_DIR", str(BASE_DIR / ".cache" / "llm"))

    # 缓存过期时间（秒），默认 24 小时
    TTL_SECONDS: int = int(os.getenv("CACHE_TTL", "86400"))


# ============================================================
# Agent 循环配置 (Phase 2 — ReAct Loop)
# ============================================================

class AgentLoopConfig:
    """ReAct Agent 循环配置"""

    # 最大推理步数
    MAX_STEPS: int = int(os.getenv("AGENT_MAX_STEPS", "12"))

    # 反思轮数
    REFLECTION_ROUNDS: int = int(os.getenv("AGENT_REFLECTION_ROUNDS", "2"))

    # 是否启用反思
    REFLECTION_ENABLED: bool = os.getenv("AGENT_REFLECTION_ENABLED", "true").lower() == "true"

    # 反思触发阈值（分数低于此值触发修订）
    REFLECTION_SCORE_THRESHOLD: int = int(os.getenv("AGENT_REFLECTION_THRESHOLD", "7"))


# ============================================================
# Embedding 配置 (Phase 3 — RAG)
# ============================================================

class EmbedderConfig:
    """Embedding 服务配置"""

    # Provider: "local" (SentenceTransformer, free) or "openai" (API-based)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")

    # Local model name (SentenceTransformer)
    LOCAL_EMBEDDING_MODEL: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # OpenAI embedding model
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # OpenAI API key (uses DEEPSEEK_API_KEY if not set, for convenience)
    OPENAI_EMBEDDING_API_KEY: str = os.getenv(
        "OPENAI_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")
    )

    # OpenAI base URL
    OPENAI_EMBEDDING_BASE_URL: str = os.getenv(
        "OPENAI_EMBEDDING_BASE_URL", "https://api.openai.com/v1"
    )


# ============================================================
# 知识库配置 (Phase 3 — RAG)
# ============================================================

class KnowledgeBaseConfig:
    """知识库配置（向量存储 + RAG 检索）"""

    # 是否启用知识库
    ENABLED: bool = os.getenv("ENABLE_KNOWLEDGE_BASE", "true").lower() == "true"

    # 知识库持久化目录
    PERSIST_DIR: str = os.getenv("KB_PERSIST_DIR", str(BASE_DIR / ".chromadb"))

    # RAG 检索时返回的结果数
    RAG_N_RESULTS: int = int(os.getenv("RAG_N_RESULTS", "5"))

    # 向量搜索权重 (vs 关键词)
    VECTOR_WEIGHT: float = float(os.getenv("RAG_VECTOR_WEIGHT", "0.6"))
    KEYWORD_WEIGHT: float = float(os.getenv("RAG_KEYWORD_WEIGHT", "0.4"))

    # 分块大小
    CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

    # 是否在采集前检查缓存
    CHECK_SEARCH_CACHE: bool = os.getenv("RAG_CHECK_SEARCH_CACHE", "true").lower() == "true"

    # 是否在分析时注入历史数据
    INJECT_HISTORY_IN_ANALYSIS: bool = os.getenv("RAG_INJECT_HISTORY", "true").lower() == "true"


# ============================================================
# 全局配置实例
# ============================================================

def get_provider_config() -> dict:
    """获取 LLM Provider 配置字典"""
    return ProviderConfig.to_dict()


def get_cache_config() -> dict:
    """获取缓存配置字典"""
    return {
        "enabled": CacheConfig.ENABLED,
        "cache_dir": CacheConfig.CACHE_DIR,
        "ttl_seconds": CacheConfig.TTL_SECONDS,
    }


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
