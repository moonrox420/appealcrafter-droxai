"""GDPR/CCPA compliance service for data export, deletion, and anonymization.

Implements tested export and delete/anonymize flows to satisfy
GDPR Article 17 (right to erasure) and CCPA deletion requests.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.encryption import PiiEncryptionService
from app.models.entities import (
    Appeal,
    Delivery,
    DonationHistory,
    Donor,
    DonorPreference,
    GuardrailDecision,
    PredictionRecord,
    SuppressionEntry,
    Unsubscribe,
)

logger = logging.getLogger(__name__)


class ComplianceService:
    """Handle GDPR/CCPA data subject rights requests."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.encryption_service = PiiEncryptionService()

    def export_donor_data(self, donor_id: str) -> dict[str, Any]:
        """Export all personal data held for a donor in a portable format."""
        donor = self.db_session.get(Donor, donor_id)
        if donor is None:
            raise ValueError(f"Donor {donor_id} not found.")

        donations = list(self.db_session.execute(
            select(DonationHistory).where(DonationHistory.donor_id == donor_id)
        ).scalars().all())

        appeals = list(self.db_session.execute(
            select(Appeal).where(Appeal.donor_id == donor_id)
        ).scalars().all())

        deliveries = list(self.db_session.execute(
            select(Delivery).join(Appeal, Delivery.appeal_id == Appeal.id).where(
                Appeal.donor_id == donor_id
            )
        ).scalars().all())

        unsubscribes = list(self.db_session.execute(
            select(Unsubscribe).where(Unsubscribe.email == donor.email)
        ).scalars().all())

        preferences = list(self.db_session.execute(
            select(DonorPreference).where(DonorPreference.donor_id == donor_id)
        ).scalars().all())

        predictions = list(self.db_session.execute(
            select(PredictionRecord).where(PredictionRecord.donor_id == donor_id)
        ).scalars().all())

        guardrail_entries = list(self.db_session.execute(
            select(GuardrailDecision).where(GuardrailDecision.donor_id == donor_id)
        ).scalars().all())

        email = self.encryption_service.decrypt_field(donor.email)

        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "donor": {
                "id": donor.id,
                "email": email,
                "first_name": donor.first_name,
                "last_name": donor.last_name,
                "interests": donor.interests,
                "channel": donor.channel,
                "capacity_score": donor.capacity_score,
                "propensity_score": donor.propensity_score,
                "engagement_score": donor.engagement_score,
                "consent_given_at": donor.consent_given_at.isoformat() if donor.consent_given_at else None,
                "consent_source": donor.consent_source,
                "rfm_recency_days": donor.rfm_recency_days,
                "rfm_frequency_count": donor.rfm_frequency_count,
                "rfm_monetary_value": donor.rfm_monetary_value,
            },
            "donation_history": [
                {"amount": donation.amount, "donated_at": donation.donated_at.isoformat()}
                for donation in donations
            ],
            "appeals": [
                {"id": appeal.id, "subject": appeal.subject, "created_at": appeal.created_at.isoformat()}
                for appeal in appeals
            ],
            "deliveries": [
                {
                    "id": delivery.id,
                    "status": delivery.status.value,
                    "sent_at": delivery.sent_at.isoformat() if delivery.sent_at else None,
                    "opened_at": delivery.opened_at.isoformat() if delivery.opened_at else None,
                    "clicked_at": delivery.clicked_at.isoformat() if delivery.clicked_at else None,
                }
                for delivery in deliveries
            ],
            "unsubscribes": [
                {
                    "email": unsubscribe.email,
                    "reason": unsubscribe.reason,
                    "unsubscribed_at": unsubscribe.unsubscribed_at.isoformat(),
                }
                for unsubscribe in unsubscribes
            ],
            "preferences": [
                {
                    "channel": preference.channel,
                    "subscribed": preference.subscribed,
                    "frequency_preference": preference.frequency_preference,
                }
                for preference in preferences
            ],
            "predictions": [
                {
                    "propensity_score": prediction.propensity_score,
                    "capacity_score": prediction.capacity_score,
                    "engagement_score": prediction.engagement_score,
                    "predicted_at": prediction.predicted_at.isoformat(),
                }
                for prediction in predictions
            ],
            "guardrail_decisions": [
                {
                    "risk_score": decision.risk_score,
                    "action": decision.action.value,
                    "created_at": decision.created_at.isoformat(),
                }
                for decision in guardrail_entries
            ],
        }

    def anonymize_donor(self, donor_id: str) -> Donor:
        """Anonymize all personal data for a donor while preserving analytics."""
        donor = self.db_session.get(Donor, donor_id)
        if donor is None:
            raise ValueError(f"Donor {donor_id} not found.")

        donor_email = self.encryption_service.decrypt_field(donor.email)

        donor.first_name = "[ANONYMIZED]"
        donor.last_name = "[ANONYMIZED]"
        donor.email = f"anonymized-{donor.id}@anonymized.invalid"
        donor.interests = None
        donor.email_encrypted = False

        anonymized_subject = "[Anonymized]"
        anonymized_body = "[Anonymized content per GDPR/CCPA request]"

        appeals = list(self.db_session.execute(
            select(Appeal).where(Appeal.donor_id == donor_id)
        ).scalars().all())
        for appeal in appeals:
            appeal.subject = anonymized_subject
            appeal.body = anonymized_body

        unsubscribes = list(self.db_session.execute(
            select(Unsubscribe).where(Unsubscribe.email == donor_email)
        ).scalars().all())
        for unsubscribe in unsubscribes:
            unsubscribe.email = f"anonymized-{donor.id}@anonymized.invalid"

        suppression_entries = list(self.db_session.execute(
            select(SuppressionEntry).where(SuppressionEntry.email == donor_email)
        ).scalars().all())
        for entry in suppression_entries:
            entry.email = f"anonymized-{donor.id}@anonymized.invalid"

        self.db_session.commit()
        self.db_session.refresh(donor)
        logger.info(
            "Donor data anonymized",
            extra={
                "donor_id": donor.id,
                "appeal_count": len(appeals),
                "unsubscribe_count": len(unsubscribes),
            },
        )
        return donor

    def delete_donor(self, donor_id: str) -> None:
        """Delete a donor and all associated records."""
        donor = self.db_session.get(Donor, donor_id)
        if donor is None:
            raise ValueError(f"Donor {donor_id} not found.")

        donor_email = self.encryption_service.decrypt_field(donor.email)

        deliveries = list(self.db_session.execute(
            select(Delivery).join(Appeal, Delivery.appeal_id == Appeal.id).where(
                Appeal.donor_id == donor_id
            )
        ).scalars().all())
        for delivery in deliveries:
            delivery.recipient_email = f"deleted-{donor.id}@deleted.invalid"

        unsubscribes = list(self.db_session.execute(
            select(Unsubscribe).where(Unsubscribe.email == donor_email)
        ).scalars().all())
        for unsubscribe in unsubscribes:
            self.db_session.delete(unsubscribe)

        self.db_session.delete(donor)
        self.db_session.commit()
        logger.info("Donor data deleted", extra={"donor_id": donor_id})


def get_compliance_service(db_session: Session) -> ComplianceService:
    """Return a configured compliance service instance."""
    return ComplianceService(db_session)