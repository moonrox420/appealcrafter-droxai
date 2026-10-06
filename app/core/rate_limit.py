"""Global and per-endpoint rate limiting with 429 + Retry-After."""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import Request, Response, status


class InMemoryRateLimiter:
    """Sliding-window in-memory rate limiter per client IP."""

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def check_request(self, client_key: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds) for a request from a client."""
        current_time = time.monotonic()
        cutoff = current_time - self.window_seconds
        request_times = [
            timestamp
            for timestamp in self._requests.get(client_key, [])
            if timestamp >= cutoff
        ]
        if len(request_times) >= self.max_requests:
            oldest = min(request_times)
            retry_after = max(1, int(self.window_seconds - (current_time - oldest)))
            return False, retry_after
        request_times.append(current_time)
        self._requests[client_key] = request_times
        return True, 0

    def reset(self) -> None:
        """Clear all rate limit state."""
        self._requests.clear()


class RateLimitMiddleware:
    """ASGI middleware enforcing the global rate limit."""

    def __init__(
        self,
        app: Callable,
        max_requests_per_minute: int = 120,
        max_burst: int = 200,
    ) -> None:
        self.app = app
        self.limiter = InMemoryRateLimiter(max_requests=max_requests_per_minute)
        self.burst_limiter = InMemoryRateLimiter(
            max_requests=max_burst, window_seconds=10
        )

    async def __call__(self, scope, receive, send) -> None:
        """Enforce rate limits for each incoming request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        client_ip = scope.get("client", ("unknown", 0))[0]
        allowed, retry_after = self.limiter.check_request(str(client_ip))
        if not allowed:
            response = Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        burst_allowed, burst_retry_after = self.burst_limiter.check_request(
            str(client_ip)
        )
        if not burst_allowed:
            response = Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={"Retry-After": str(burst_retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def build_rate_limit_middleware(
    app: Callable, max_per_minute: int, max_burst: int
) -> RateLimitMiddleware:
    """Construct a configured rate limit middleware instance."""
    return RateLimitMiddleware(
        app=app,
        max_requests_per_minute=max_per_minute,
        max_burst=max_burst,
    )
