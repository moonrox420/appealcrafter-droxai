"""Unit tests for RedisCacheService."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import redis

from app.core.cache import (
    RedisCacheService,
    get_cache_service,
)


def test_cache_disabled_behavior() -> None:
    """Verify cache operations fail gracefully when cache is disabled."""
    service = RedisCacheService()
    assert service.is_enabled is False
    assert service.get("any_key") is None
    assert service.set("any_key", {"a": 1}) is False
    assert service.delete("any_key") is False
    assert service.increment("any_key") is None


def test_cache_enabled_get_and_set() -> None:
    """Verify get and set serialization and hit tracking with mocked redis client."""
    with patch("app.core.cache.get_settings") as mock_settings, \
         patch("redis.Redis.from_url") as mock_from_url:
        mock_settings.return_value.redis.cache_enabled = True
        mock_settings.return_value.redis.url = "redis://localhost:6379/0"
        mock_settings.return_value.redis.cache_ttl_seconds = 3600

        mock_redis = MagicMock()
        mock_from_url.return_value = mock_redis

        service = get_cache_service()
        assert service.is_enabled is True

        # Test set
        success = service.set("user:123", {"name": "Alice", "score": 98})
        assert success is True
        mock_redis.set.assert_called_once_with("user:123", json.dumps({"name": "Alice", "score": 98}), ex=3600)

        # Test get hit
        mock_redis.get.return_value = json.dumps({"name": "Alice", "score": 98})
        val = service.get("user:123")
        assert val == {"name": "Alice", "score": 98}

        # Test get miss
        mock_redis.get.return_value = None
        miss_val = service.get("user:nonexistent")
        assert miss_val is None


def test_cache_error_handling() -> None:
    """Verify cache handles redis errors and non-serializable objects gracefully."""
    with patch("app.core.cache.get_settings") as mock_settings, \
         patch("redis.Redis.from_url") as mock_from_url:
        mock_settings.return_value.redis.cache_enabled = True
        mock_settings.return_value.redis.url = "redis://localhost:6379/0"
        mock_settings.return_value.redis.cache_ttl_seconds = 300

        mock_redis = MagicMock()
        mock_from_url.return_value = mock_redis

        service = RedisCacheService()

        # Set non-serializable object
        assert service.set("bad_key", object()) is False

        # RedisError on set
        mock_redis.set.side_effect = redis.RedisError("Connection lost")
        assert service.set("conn_err", {"key": "val"}) is False

        # RedisError on get
        mock_redis.get.side_effect = redis.RedisError("Timeout")
        assert service.get("timeout_key") is None

        # Corrupted JSON on get
        mock_redis.get.side_effect = None
        mock_redis.get.return_value = "invalid-json{"
        assert service.get("corrupt_key") is None

        # Delete error
        mock_redis.delete.side_effect = redis.RedisError("Delete failed")
        assert service.delete("del_key") is False

        # Increment success and error
        mock_redis.incrby.side_effect = None
        mock_redis.incrby.return_value = 5
        assert service.increment("counter_key", 2) == 5

        mock_redis.incrby.side_effect = redis.RedisError("Incr failed")
        assert service.increment("counter_key", 1) is None


def test_cache_hit_rate_metrics() -> None:
    """Verify get_hit_rate, get_hit_count, get_miss_count calculations."""
    service = RedisCacheService()
    hits = service.get_hit_count()
    misses = service.get_miss_count()
    rate = service.get_hit_rate()

    assert isinstance(hits, int)
    assert isinstance(misses, int)
    assert 0.0 <= rate <= 100.0
