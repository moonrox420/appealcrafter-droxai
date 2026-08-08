"""Suppression list and frequency capping service.

Enforces unsubscribe records, API-managed suppression entries, and
frequency capping before every send.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.entities import (
    Delivery,
    DeliveryStatus,
    Donor,
    SuppressionEntry,
    Unsubscribe,
)


class SuppressionService:
    """Enforce unsubscribe and frequency cap rules before every send."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.settings = get_settings()

    def is_suppressed(self, email: str) -> bool:
        """Return True if the email is on the suppression list."""
        statement = select(Unsubscribe).where(Unsubscribe.email == email)
        if self.db_session.execute(statement).scalar_one_or_none() is not None:
            return True

        now = datetime.now(timezone.utc)
        entry_statement = select(SuppressionEntry).where(
            SuppressionEntry.email == email,
            (SuppressionEntry.expires_at.is_(None)) | (SuppressionEntry.expires_at > now),
        )
        return self.db_session.execute(entry_statement).scalar_one_or_none() is not None

    def is_donor_deleted(self, donor: Donor) -> bool:
        """Return True if the donor is soft-deleted."""
        return donor.is_deleted

    def is_within_frequency_cap(self, donor_id: str) -> bool:
        """Return True if the donor has received email within the cooldown window."""
        cooldown_cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.settings.email.frequency_cap_days
        )
        statement = (
            select(Delivery)
            .join(Delivery.appeal)
            .where(
                Delivery.appeal.has(donor_id=donor_id),
                Delivery.sent_at >= cooldown_cutoff,
                Delivery.status.in_(
                    [
                        DeliveryStatus.SENT,
                        DeliveryStatus.DELIVERED,
                        DeliveryStatus.OPENED,
                        DeliveryStatus.CLICKED,
                        DeliveryStatus.CONVERTED,
                    ]
                ),
            )
        )
        recent_delivery = self.db_session.execute(statement).scalars().first()
        return recent_delivery is not None

    def can_send_to_donor(self, donor: Donor) -> tuple[bool, str | None]:
        """Return (allowed, reason) for whether a donor can receive email."""
        if self.is_donor_deleted(donor):
            return False, "donor_soft_deleted"
        if self.is_suppressed(donor.email):
            return False, "suppressed"
        if self.is_within_frequency_cap(donor.id):
            return False, "frequency_capped"
        return True, None

    def add_suppression_entry(
        self,
        email: str,
        reason: str | None = None,
        source: str = "api",
        created_by_user_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> SuppressionEntry:
        """Add an API-managed suppression list entry."""
        existing = self.db_session.execute(
            select(SuppressionEntry).where(SuppressionEntry.email == email)
        ).scalars().first()
        if existing is not None:
            existing.reason = reason or existing.reason
            existing.source = source
            existing.expires_at = expires_at
            self.db_session.commit()
            self.db_session.refresh(existing)
            return existing

        entry = SuppressionEntry(
            email=email,
            reason=reason,
            source=source,
            created_by_user_id=created_by_user_id,
            expires_at=expires_at,
        )
        self.db_session.add(entry)
        self.db_session.commit()
        self.db_session.refresh(entry)
        return entry

    def remove_suppression_entry(self, email: str) -> bool:
        """Remove a suppression entry by email."""
        existing = self.db_session.execute(
            select(SuppressionEntry).where(SuppressionEntry.email == email)
        ).scalars().first()
        if existing is None:
            return False
        self.db_session.delete(existing)
        self.db_session.commit()
        return True

    def list_suppression_entries(self) -> list[SuppressionEntry]:
        """Return all current suppression entries."""
        return list(self.db_session.execute(
            select(SuppressionEntry).order_by(SuppressionEntry.suppressed_at.desc())
        ).scalars().all())