"""Tests for health, readiness, and metrics endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session


@pytest.mark.anyio
async def test_health_endpoint(async_client: AsyncClient) -> None:
    """Test /health returns 200 ok for liveness checks."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_readiness_endpoint_healthy(async_client: AsyncClient) -> None:
    """Test /ready returns 200 when database and broker are connected."""
    with patch("app.api.health.verify_database_connection", return_value=True), patch(
        "app.api.health.celery_app.connection"
    ) as mock_conn:
        mock_conn.return_value = MagicMock(connected=True)
        response = await async_client.get("/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["database"] is True
    assert data["broker"] is True


@pytest.mark.anyio
async def test_readiness_endpoint_unhealthy_broker(async_client: AsyncClient) -> None:
    """Test /ready returns 503 when broker is unreachable."""
    with patch("app.api.health.verify_database_connection", return_value=True), patch(
        "app.api.health.celery_app.connection"
    ) as mock_conn:
        mock_conn.return_value = MagicMock(connected=False)
        response = await async_client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["broker"] is False


@pytest.mark.anyio
async def test_prometheus_metrics_endpoint(async_client: AsyncClient) -> None:
    """Test /metrics returns Prometheus formatted plaintext metrics."""
    response = await async_client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "appealcrafter_" in content


@pytest.mark.anyio
async def test_operational_metrics_endpoint(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test /metrics/summary returns structured JSON operational metrics."""
    response = await async_client.get("/metrics/summary", headers=operator_headers)
    assert response.status_code == 200
    data = response.json()
    assert "total_deliveries" in data
    assert "sent_count" in data
    assert "delivered_count" in data
    assert "error_rate_percent" in data

    # Unauthorized returns 401
    unauth_resp = await async_client.get("/metrics/summary")
    assert unauth_resp.status_code == 401
