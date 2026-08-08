"""Unit tests for suppression and frequency capping logic."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.session import Base
from app.models.entities import Appeal, Delivery, DeliveryStatus, Donor, Unsubscribe
from app.services.suppression import SuppressionService


@pytest.fixture()
def db_session() -> Session:
    """Create a PostgreSQL test database session."""
    settings = get_settings()
    engine = create_engine(settings.database.sqlalchemy_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


def test_suppressed_email_blocked(db_session: Session) -> None:
    """Suppressed emails are never eligible for sending."""
    donor = Donor(email="suppressed@example.com", channel="email")
    db_session.add(donor)
    db_session.flush()
    db_session.add(Unsubscribe(email="suppressed@example.com", source="webhook"))
    db_session.commit()

    service = SuppressionService(db_session)
    allowed, reason = service.can_send_to_donor(donor)
    assert not allowed
    assert reason == "suppressed"


def test_soft_deleted_donor_blocked(db_session: Session) -> None:
    """Soft-deleted donors are excluded from generation."""
    donor = Donor(email="deleted@example.com", channel="email", is_deleted=True)
    db_session.add(donor)
    db_session.commit()

    service = SuppressionService(db_session)
    allowed, reason = service.can_send_to_donor(donor)
    assert not allowed
    assert reason == "donor_soft_deleted"


def test_frequency_cap_blocks_recent_send(db_session: Session) -> None:
    """Donors within the cooldown window are blocked."""
    donor = Donor(email="capped@example.com", channel="email")
    db_session.add(donor)
    db_session.flush()
    appeal = Appeal(
        donor_id=donor.id,
        subject="Test",
        body="Test body",
        cta="Donate",
    )
    db_session.add(appeal)
    db_session.flush()
    db_session.add(
        Delivery(
            appeal_id=appeal.id,
            recipient_email=donor.email,
            provider="email",
            status=DeliveryStatus.SENT,
            sent_at=datetime.now(timezone.utc),
        )
    )
    db_session.commit()

    service = SuppressionService(db_session)
    allowed, reason = service.can_send_to_donor(donor)
    assert not allowed
    assert reason == "frequency_capped"


def test_eligible_donor_allowed(db_session: Session) -> None:
    """Eligible donors pass all checks."""
    donor = Donor(email="eligible@example.com", channel="email")
    db_session.add(donor)
    db_session.commit()

    service = SuppressionService(db_session)
    allowed, reason = service.can_send_to_donor(donor)
    assert allowed
    assert reason is None