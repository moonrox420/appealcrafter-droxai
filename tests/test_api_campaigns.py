"""Tests for /campaigns API endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from app.models.entities import Campaign


@pytest.mark.anyio
async def test_create_campaign(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
) -> None:
    """Test creating a new campaign via API."""
    response = await async_client.post(
        "/campaigns",
        json={
            "name": "Annual Spring Gala",
            "description": "Annual fundraising gala campaign for regional donors.",
        },
        headers=operator_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Annual Spring Gala"
    assert (
        data["description"] == "Annual fundraising gala campaign for regional donors."
    )
    assert data["is_active"] is True
    assert "id" in data
    assert "created_at" in data


@pytest.mark.anyio
async def test_list_campaigns(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test listing all campaigns."""
    campaign = Campaign(name="Winter Appeal 2026", description="End of year drive")
    db_session.add(campaign)
    db_session.commit()

    response = await async_client.get(
        "/campaigns",
        headers=operator_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert any(c["name"] == "Winter Appeal 2026" for c in data)


@pytest.mark.anyio
async def test_campaigns_unauthorized(async_client: AsyncClient) -> None:
    """Test unauthorized access to campaigns returns 401."""
    response = await async_client.get("/campaigns")
    assert response.status_code == 401
