"""Tests for /generate-appeal API endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from app.models.entities import Campaign, Donor


@pytest.mark.anyio
async def test_generate_appeal_endpoint(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test generating appeals for eligible donors."""
    donor = Donor(
        email="appeal-donor@example.com",
        first_name="Jane",
        last_name="Doe",
        channel="email",
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
    )
    db_session.add(donor)

    campaign = Campaign(name="Appeal Campaign", description="Test appeal generation")
    db_session.add(campaign)
    db_session.commit()

    with patch("app.api.appeals.send_campaign_task.delay") as mock_delay:
        mock_delay.return_value = MagicMock(id="celery-task-12345")
        response = await async_client.post(
            "/generate-appeal",
            json={
                "tone": "inspiring",
                "campaign_id": campaign.id,
                "limit": 10,
            },
            headers=operator_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["generated_count"] >= 1
    assert "celery-task-12345" in data["task_ids"]
    assert len(data["appeals"]) >= 1
    assert data["appeals"][0]["tone"] == "inspiring"


@pytest.mark.anyio
async def test_generate_appeal_invalid_tone(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
) -> None:
    """Test generating appeal with invalid tone is rejected with 422 Unprocessable Entity."""
    response = await async_client.post(
        "/generate-appeal",
        json={
            "tone": "invalid_tone_choice",
            "limit": 10,
        },
        headers=operator_headers,
    )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_generate_appeal_unauthorized(async_client: AsyncClient) -> None:
    """Test unauthorized request to /generate-appeal returns 401."""
    response = await async_client.post(
        "/generate-appeal",
        json={"tone": "inspiring", "limit": 10},
    )
    assert response.status_code == 401
