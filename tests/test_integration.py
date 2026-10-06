"""Integration tests covering ingest → generate → queue → status flow."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.entities import Delivery, DeliveryStatus
from app.schemas.donor import DonationCreate, DonorCreate
from app.services.appeal import AppealService
from app.services.delivery import DeliveryService
from app.services.donor import DonorService


def test_ingest_generate_queue_status_flow(db_session: Session) -> None:
    """Verify the full donor ingest to delivery status pipeline."""
    donor_service = DonorService(db_session)
    donor_payload = DonorCreate(
        email="flow@example.com",
        first_name="Flow",
        last_name="Test",
        channel="email",
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
        donations=[
            DonationCreate(amount=100.0, donated_at=datetime.now(timezone.utc)),
            DonationCreate(amount=250.0, donated_at=datetime.now(timezone.utc)),
        ],
    )
    donor = donor_service.create_donor(donor_payload)
    assert donor.id is not None
    assert len(donor.donation_history) == 2

    appeal_service = AppealService(db_session)
    appeals = appeal_service.generate_appeals(tone="inspiring", limit=10)
    assert len(appeals) == 1
    assert appeals[0].donor_id == donor.id
    assert appeals[0].is_template_fallback is True

    delivery_service = DeliveryService(db_session)
    delivery = delivery_service.create_delivery(
        appeal_id=appeals[0].id,
        recipient_email=donor.email,
        provider="email",
    )
    assert delivery.status == DeliveryStatus.QUEUED

    delivery = delivery_service.send_delivery(
        delivery, appeals[0].subject, appeals[0].body
    )
    assert delivery.status in (DeliveryStatus.SENT, DeliveryStatus.FAILED)

    updated_delivery = db_session.get(Delivery, delivery.id)
    assert updated_delivery is not None
    assert updated_delivery.status == delivery.status
