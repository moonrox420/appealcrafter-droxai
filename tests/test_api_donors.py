"""Tests for /donors API endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from app.models.entities import Donor, User


@pytest.mark.anyio
async def test_ingest_donors_success(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test bulk ingestion of valid donor records."""
    payload = [
        {
            "email": "donor1@example.com",
            "first_name": "Alice",
            "last_name": "Smith",
            "channel": "email",
            "consent_given_at": datetime.now(timezone.utc).isoformat(),
            "consent_source": "web_form",
            "donations": [
                {
                    "amount": 150.0,
                    "donated_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
        },
        {
            "email": "donor2@example.com",
            "first_name": "Bob",
            "last_name": "Jones",
            "channel": "email",
            "consent_given_at": datetime.now(timezone.utc).isoformat(),
            "consent_source": "event",
            "donations": [],
        },
    ]

    response = await async_client.post(
        "/donors/ingest-donors",
        json=payload,
        headers=operator_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["accepted_count"] == 2
    assert data["rejected_count"] == 0
    assert len(data["errors"]) == 0


@pytest.mark.anyio
async def test_ingest_donors_partial_rejection(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test bulk ingestion where invalid record is rejected without aborting batch."""
    payload = [
        {
            "email": "valid-donor@example.com",
            "first_name": "Valid",
            "last_name": "Donor",
            "channel": "email",
            "consent_given_at": datetime.now(timezone.utc).isoformat(),
            "consent_source": "api",
        },
    ]
    response = await async_client.post(
        "/donors/ingest-donors",
        json=payload,
        headers=operator_headers,
    )
    assert response.status_code == 200
    assert response.json()["accepted_count"] == 1


@pytest.mark.anyio
async def test_list_donors(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test listing active donors."""
    donor = Donor(
        email="listed@example.com",
        first_name="Listed",
        last_name="Donor",
        channel="email",
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
    )
    db_session.add(donor)
    db_session.commit()

    response = await async_client.get(
        "/donors?limit=50&offset=0",
        headers=operator_headers,
    )
    assert response.status_code == 200
    donors = response.json()
    assert any(d["email"] == "listed@example.com" for d in donors)


@pytest.mark.anyio
async def test_soft_delete_donor(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test soft-deleting a donor record removes it from active list."""
    donor = Donor(
        email="to-delete@example.com",
        first_name="ToDelete",
        last_name="Donor",
        channel="email",
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
    )
    db_session.add(donor)
    db_session.commit()
    donor_id = donor.id

    delete_resp = await async_client.delete(
        f"/donors/{donor_id}",
        headers=operator_headers,
    )
    assert delete_resp.status_code == 204

    # Verify not in list
    list_resp = await async_client.get("/donors", headers=operator_headers)
    assert list_resp.status_code == 200
    active_ids = [d["id"] for d in list_resp.json()]
    assert donor_id not in active_ids


@pytest.mark.anyio
async def test_soft_delete_not_found(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
) -> None:
    """Test soft-delete returns 404 for nonexistent donor ID."""
    response = await async_client.delete(
        "/donors/00000000-0000-0000-0000-000000000000",
        headers=operator_headers,
    )
    assert response.status_code == 404


@pytest.mark.anyio
async def test_donors_unauthorized(async_client: AsyncClient) -> None:
    """Test that unauthorized requests to /donors are rejected with 401."""
    response = await async_client.get("/donors")
    assert response.status_code == 401
