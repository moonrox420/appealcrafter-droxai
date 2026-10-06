"""Tests for /webhooks API endpoints with HMAC signature verification."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.entities import Appeal, Delivery, DeliveryStatus, Donor


@pytest.mark.anyio
async def test_webhook_processed_with_valid_signature(
    async_client: AsyncClient,
    db_session: Session,
) -> None:
    """Test webhook event processing with valid HMAC-SHA256 signature."""
    settings = get_settings()
    donor = Donor(
        email="webhook-test@example.com",
        first_name="Webhook",
        last_name="Test",
        channel="email",
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
    )
    db_session.add(donor)

    appeal = Appeal(
        donor=donor,
        subject="Support Our Cause",
        body="Help us today",
        cta="Donate",
        tone="inspiring",
    )
    db_session.add(appeal)

    delivery = Delivery(
        appeal=appeal,
        recipient_email=donor.email,
        provider="email",
        provider_message_id="msg-unique-12345",
        status=DeliveryStatus.SENT,
    )
    db_session.add(delivery)
    db_session.commit()

    payload_dict = {
        "event_type": "delivered",
        "message_id": "msg-unique-12345",
        "recipient_email": "webhook-test@example.com",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload_bytes = json.dumps(payload_dict).encode("utf-8")
    signature = hmac.new(
        settings.email.webhook_secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    response = await async_client.post(
        "/webhooks/email-events",
        content=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "x-webhook-signature": signature,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processed"
    assert data["delivery_id"] == delivery.id

    db_session.refresh(delivery)
    assert delivery.status == DeliveryStatus.DELIVERED


@pytest.mark.anyio
async def test_webhook_missing_signature(async_client: AsyncClient) -> None:
    """Test webhook rejection when signature header is missing."""
    response = await async_client.post(
        "/webhooks/email-events",
        json={
            "event_type": "delivered",
            "message_id": "any-id",
            "recipient_email": "test@example.com",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    assert response.status_code == 401
    assert "Missing webhook signature" in response.json()["detail"]


@pytest.mark.anyio
async def test_webhook_invalid_signature(async_client: AsyncClient) -> None:
    """Test webhook rejection when signature is invalid."""
    response = await async_client.post(
        "/webhooks/email-events",
        json={
            "event_type": "delivered",
            "message_id": "any-id",
            "recipient_email": "test@example.com",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        headers={"x-webhook-signature": "bad_signature_hex_value"},
    )
    assert response.status_code == 401
    assert "Invalid webhook signature" in response.json()["detail"]


@pytest.mark.anyio
async def test_webhook_unknown_message(async_client: AsyncClient) -> None:
    """Test webhook ignoring unknown message_id with valid signature."""
    settings = get_settings()
    payload_dict = {
        "event_type": "opened",
        "message_id": "unknown-nonexistent-msg-id",
        "recipient_email": "nobody@example.com",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload_bytes = json.dumps(payload_dict).encode("utf-8")
    signature = hmac.new(
        settings.email.webhook_secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    response = await async_client.post(
        "/webhooks/email-events",
        content=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "x-webhook-signature": signature,
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
