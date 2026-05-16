"""CompetiScope unified exception hierarchy."""


class CompetiScopeError(Exception):
    """Base exception for all CompetiScope errors."""


class ConfigError(CompetiScopeError):
    """Configuration-related errors (missing keys, invalid values)."""


class AgentError(CompetiScopeError):
    """Agent execution errors (loop failure, tool failure, reflection failure)."""


class ToolError(AgentError):
    """Tool execution errors (search timeout, scrape failure)."""


class APIError(CompetiScopeError):
    """API-level errors (invalid request, task not found)."""


class TaskNotFoundError(APIError):
    """Requested task does not exist."""
