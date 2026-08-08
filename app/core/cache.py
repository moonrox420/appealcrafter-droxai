"""Redis cache service for donor profiles, predictions, and templates.

Provides a thin async-friendly wrapper around redis-py with configurable
TTL and hit-rate metrics tracking.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

HIT_COUNT = 0
MISS_COUNT = 0


def _increment_hit() -> None:
    """Increment the cache hit counter."""
    global HIT_COUNT
    HIT_COUNT += 1


def _increment_miss() -> None:
    """Increment the cache miss counter."""
    global MISS_COUNT
    MISS_COUNT += 1


class RedisCacheService:
    """Redis-backed cache with JSON serialization and hit-rate metrics."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: redis.Redis | None = None
        if self.settings.redis.cache_enabled:
            self._client = redis.Redis.from_url(
                self.settings.redis.url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=3,
            )

    @property
    def is_enabled(self) -> bool:
        """Return whether the cache is enabled and connected."""
        return self._client is not None

    def get(self, key: str) -> Any | None:
        """Return the cached JSON value for a key, or None on miss."""
        if self._client is None:
            _increment_miss()
            return None
        try:
            raw_value = self._client.get(key)
            if raw_value is None:
                _increment_miss()
                return None
            _increment_hit()
            return json.loads(raw_value)
        except (redis.RedisError, json.JSONDecodeError) as exc:
            logger.warning("Cache get failed", extra={"key": key, "error": str(exc)})
            _increment_miss()
            return None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Store a JSON-serializable value in the cache with TTL."""
        if self._client is None:
            return False
        ttl = ttl_seconds or self.settings.redis.cache_ttl_seconds
        try:
            serialized = json.dumps(value)
            self._client.set(key, serialized, ex=ttl)
            return True
        except (redis.RedisError, TypeError) as exc:
            logger.warning("Cache set failed", extra={"key": key, "error": str(exc)})
            return False

    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        if self._client is None:
            return False
        try:
            self._client.delete(key)
            return True
        except redis.RedisError as exc:
            logger.warning("Cache delete failed", extra={"key": key, "error": str(exc)})
            return False

    def increment(self, key: str, amount: int = 1) -> int | None:
        """Increment a counter in the cache, returning the new value."""
        if self._client is None:
            return None
        try:
            return int(self._client.incrby(key, amount))
        except redis.RedisError as exc:
            logger.warning("Cache increment failed", extra={"key": key, "error": str(exc)})
            return None

    def get_hit_rate(self) -> float:
        """Return the aggregate cache hit rate as a percentage."""
        total = HIT_COUNT + MISS_COUNT
        if total == 0:
            return 0.0
        return round(HIT_COUNT / total * 100.0, 2)

    def get_hit_count(self) -> int:
        """Return the total cache hits."""
        return HIT_COUNT

    def get_miss_count(self) -> int:
        """Return the total cache misses."""
        return MISS_COUNT


def get_cache_service() -> RedisCacheService:
    """Return a configured Redis cache service instance."""
    return RedisCacheService()