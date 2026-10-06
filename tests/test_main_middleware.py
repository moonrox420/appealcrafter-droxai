"""Unit tests for FastAPI security headers and global exception handler."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from fastapi import Request
from app.main import unhandled_exception_handler


@pytest.mark.asyncio
async def test_security_headers_present_on_all_responses(async_client: AsyncClient) -> None:
    """Verify HSTS, nosniff, DENY, CSP, and X-Trace-Id headers are present."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert "x-trace-id" in response.headers or "X-Trace-Id" in response.headers
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("content-security-policy") == "default-src 'none'"
    assert "max-age=31536000" in response.headers.get("strict-transport-security", "")


@pytest.mark.asyncio
async def test_unhandled_exception_handler_returns_safe_500() -> None:
    """Verify unhandled_exception_handler returns generic JSON error without leaking trace."""
    mock_request = Request({"type": "http", "method": "GET", "path": "/crash", "headers": []})
    mock_exc = RuntimeError("Secret database connection string leaked!")

    resp = await unhandled_exception_handler(mock_request, mock_exc)
    assert resp.status_code == 500
    assert resp.body == b'{"detail":"Internal server error"}'
