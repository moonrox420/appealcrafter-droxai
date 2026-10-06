"""Unit tests for sliding-window rate limiter."""

from __future__ import annotations

import time
from app.core.rate_limit import InMemoryRateLimiter


def test_in_memory_rate_limiter_allows_under_limit() -> None:
    """Verify rate limiter allows requests up to max_requests."""
    limiter = InMemoryRateLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        allowed, retry_after = limiter.check_request("client_1")
        assert allowed is True
        assert retry_after == 0


def test_in_memory_rate_limiter_blocks_over_limit() -> None:
    """Verify rate limiter blocks request when limit is exceeded and provides positive retry-after."""
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=10)
    for _ in range(3):
        allowed, _ = limiter.check_request("client_2")
        assert allowed is True

    blocked, retry_after = limiter.check_request("client_2")
    assert blocked is False
    assert retry_after > 0


def test_in_memory_rate_limiter_resets_state() -> None:
    """Verify reset clears tracked requests."""
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=10)
    limiter.check_request("client_3")
    limiter.check_request("client_3")
    blocked, _ = limiter.check_request("client_3")
    assert blocked is False

    limiter.reset()
    allowed_again, _ = limiter.check_request("client_3")
    assert allowed_again is True
