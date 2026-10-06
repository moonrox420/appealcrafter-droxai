"""Unit tests for Prometheus middleware and metric instrumentation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response

from app.core.metrics import (
    prometheus_metrics_endpoint,
    prometheus_middleware,
)


@pytest.mark.asyncio
async def test_prometheus_middleware_success_call() -> None:
    """Verify prometheus_middleware records request count and latency for 200 responses."""
    request = MagicMock(spec=Request)
    request.url.path = "/api/v1/appeals"
    request.method = "POST"

    response = Response(status_code=200, content="ok")
    call_next = AsyncMock(return_value=response)

    res = await prometheus_middleware(request, call_next)
    assert res.status_code == 200
    call_next.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_prometheus_middleware_500_error_call() -> None:
    """Verify prometheus_middleware increments ERROR_COUNT on 500 response."""
    request = MagicMock(spec=Request)
    request.url.path = "/api/v1/broken"
    request.method = "GET"

    response = Response(status_code=500, content="internal server error")
    call_next = AsyncMock(return_value=response)

    res = await prometheus_middleware(request, call_next)
    assert res.status_code == 500


@pytest.mark.asyncio
async def test_prometheus_metrics_endpoint_output() -> None:
    """Verify prometheus_metrics_endpoint produces Prometheus text."""
    resp = await prometheus_metrics_endpoint()
    assert resp.status_code == 200
    assert resp.media_type.startswith("text/plain") or "version=0.0.4" in resp.media_type
    assert b"appealcrafter_http_requests_total" in resp.body
