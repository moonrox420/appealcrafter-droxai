"""Donor ingestion and management service."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.entities import DonationHistory, Donor
from app.schemas.donor import DonorCreate

logger = logging.getLogger(__name__)


class DonorService:
    """Persist and query donor records with per-record validation."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def find_by_email(self, email: str) -> Donor | None:
        """Return the active donor with the given email, if any."""
        statement = select(Donor).where(
            Donor.email == email, Donor.is_deleted.is_(False)
        )
        return self.db_session.execute(statement).scalar_one_or_none()

    def create_donor(self, donor_payload: DonorCreate) -> Donor:
        """Create a donor record with donation history."""
        existing_donor = self.find_by_email(donor_payload.email)
        if existing_donor is not None:
            raise ValueError(f"Donor with email {donor_payload.email} already exists.")
        donor = Donor(
            external_id=donor_payload.external_id,
            email=donor_payload.email,
            first_name=donor_payload.first_name,
            last_name=donor_payload.last_name,
            interests=donor_payload.interests,
            channel=donor_payload.channel,
            consent_given_at=donor_payload.consent_given_at,
            consent_source=donor_payload.consent_source,
        )
        for donation_payload in donor_payload.donations:
            donor.donation_history.append(
                DonationHistory(
                    amount=donation_payload.amount,
                    donated_at=donation_payload.donated_at,
                )
            )
        self.db_session.add(donor)
        self.db_session.commit()
        self.db_session.refresh(donor)
        return donor

    def soft_delete_donor(self, donor_id: str) -> Donor:
        """Soft-delete a donor record."""
        donor = self.db_session.get(Donor, donor_id)
        if donor is None:
            raise ValueError(f"Donor {donor_id} not found.")
        donor.is_deleted = True
        donor.deleted_at = donor.updated_at
        self.db_session.commit()
        self.db_session.refresh(donor)
        return donor

    def list_active_donors(self, limit: int = 100, offset: int = 0) -> list[Donor]:
        """Return active (non-deleted) donors."""
        statement = (
            select(Donor)
            .where(Donor.is_deleted.is_(False))
            .order_by(Donor.created_at)
            .limit(limit)
            .offset(offset)
        )
        return list(self.db_session.execute(statement).scalars().all())
