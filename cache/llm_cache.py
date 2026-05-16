import hashlib
import json
import time
from pathlib import Path
from typing import Optional
from loguru import logger


class LLMCache:
    """File-based cache for LLM responses to avoid duplicate API costs.

    Cache key: MD5(model + sorted_messages_json)
    Cache TTL: configurable, default 24 hours.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/llm",
        ttl_seconds: int = 86400,  # 24 hours
        enabled: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl_seconds
        self.enabled = enabled
        self._hits = 0
        self._misses = 0

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"LLMCache initialized: dir={self.cache_dir}, ttl={ttl_seconds}s")

    def _make_key(self, model: str, messages: list[dict]) -> str:
        """Generate a deterministic cache key."""
        content = json.dumps({"model": model, "messages": messages}, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, model: str, messages: list[dict]) -> Optional[str]:
        """Retrieve cached response. Returns None if miss or expired."""
        if not self.enabled:
            self._misses += 1
            return None

        key = self._make_key(model, messages)
        path = self._cache_path(key)

        if not path.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - data["timestamp"] > self.ttl:
                path.unlink()
                self._misses += 1
                return None
            self._hits += 1
            logger.debug(f"Cache hit: {key[:8]}... (hits={self._hits}, misses={self._misses})")
            return data["content"]
        except Exception:
            self._misses += 1
            return None

    def set(self, model: str, messages: list[dict], content: str):
        """Store response in cache."""
        if not self.enabled:
            return

        key = self._make_key(model, messages)
        path = self._cache_path(key)
        data = {"key": key, "model": model, "timestamp": time.time(), "content": content}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def clear(self):
        """Clear all cache entries."""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
            logger.info(f"Cache cleared: {self.cache_dir}")

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "hit_rate": self._hits / max(self._hits + self._misses, 1)}
