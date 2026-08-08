"""Pytest configuration and fixtures."""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True, scope="session")
def set_test_environment() -> None:
    """Set environment variables for testing before app loads."""
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("DATABASE_HOST", "localhost")
    os.environ.setdefault("DATABASE_NAME", "appealcrafter_test")
    os.environ.setdefault("DATABASE_USER", "test")
    os.environ.setdefault("DATABASE_PASSWORD", "test")
    os.environ.setdefault("SECURITY_JWT_SECRET", "a" * 32)
    os.environ.setdefault("SECURITY_ENCRYPTION_KEY", "b" * 32)
    os.environ.setdefault("EMAIL_FROM_ADDRESS", "test@example.com")
    os.environ.setdefault("EMAIL_PHYSICAL_ADDRESS", "123 Test St")
    os.environ.setdefault("EMAIL_WEBHOOK_SECRET", "c" * 16)
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
    os.environ.setdefault("REDIS_CACHE_ENABLED", "false")
    os.environ.setdefault("LLM_RAG_ENABLED", "false")
    os.environ.setdefault("LLM_GUARDRAILS_ENABLED", "true")
    os.environ.setdefault("OBSERVABILITY_PROMETHEUS_ENABLED", "false")


@pytest.fixture()
def test_app() -> FastAPI:
    """Return the FastAPI application instance."""
    from app.main import app

    return app


@pytest.fixture()
async def async_client(test_app: FastAPI) -> AsyncClient:
    """Return an async HTTP client for integration tests."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        yield client