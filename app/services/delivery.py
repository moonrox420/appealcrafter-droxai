"""Delivery record management and webhook processing service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.metrics import DELIVERY_RATE
from app.models.entities import Delivery, DeliveryStatus, Unsubscribe
from app.services.email_provider import (
    EmailMessage,
    EmailProviderService,
    ProviderSendResult,
)

logger = logging.getLogger(__name__)


class DeliveryService:
    """Create and update delivery records through the email lifecycle."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.email_provider = EmailProviderService()

    def create_delivery(
        self, appeal_id: str, recipient_email: str, provider: str
    ) -> Delivery:
        """Create a queued delivery record."""
        delivery = Delivery(
            appeal_id=appeal_id,
            recipient_email=recipient_email,
            provider=provider,
            status=DeliveryStatus.QUEUED,
        )
        self.db_session.add(delivery)
        self.db_session.commit()
        self.db_session.refresh(delivery)
        return delivery

    def send_delivery(self, delivery: Delivery, subject: str, body: str) -> Delivery:
        """Send a delivery through the email provider."""
        settings = self.email_provider.settings
        message = EmailMessage(
            to_address=delivery.recipient_email,
            subject=subject,
            body=body,
            from_address=settings.email.from_address,
            from_name=settings.email.from_name,
        )
        try:
            result: ProviderSendResult = self.email_provider.send(message)
            delivery.provider_message_id = result.provider_message_id
            delivery.status = DeliveryStatus.SENT
            delivery.sent_at = datetime.now(timezone.utc)
        except Exception as exc:
            delivery.status = DeliveryStatus.FAILED
            delivery.error_detail = str(exc)
            logger.error(
                "Email send failed",
                extra={"delivery_id": delivery.id, "error": str(exc)},
            )
        self.db_session.commit()
        self.db_session.refresh(delivery)
        status_val = (
            delivery.status.value
            if hasattr(delivery.status, "value")
            else str(delivery.status)
        )
        DELIVERY_RATE.labels(status=status_val).inc()
        return delivery

    def process_webhook_event(
        self,
        event_type: str,
        message_id: str,
        recipient_email: str,
        timestamp: datetime,
    ) -> Delivery | None:
        """Update a delivery record based on a provider webhook event."""
        statement = select(Delivery).where(Delivery.provider_message_id == message_id)
        delivery = self.db_session.execute(statement).scalar_one_or_none()
        if delivery is None:
            logger.warning(
                "Webhook event for unknown message",
                extra={"message_id": message_id, "event_type": event_type},
            )
            return None

        if event_type == "delivered":
            delivery.status = DeliveryStatus.DELIVERED
            delivery.delivered_at = timestamp
        elif event_type == "opened":
            delivery.status = DeliveryStatus.OPENED
            delivery.opened_at = timestamp
        elif event_type == "clicked":
            delivery.status = DeliveryStatus.CLICKED
            delivery.clicked_at = timestamp
        elif event_type == "converted":
            delivery.status = DeliveryStatus.CONVERTED
            delivery.converted_at = timestamp
        elif event_type == "bounced":
            delivery.status = DeliveryStatus.BOUNCED
            delivery.bounced_at = timestamp
        elif event_type == "complained":
            delivery.status = DeliveryStatus.COMPLAINED
            delivery.complained_at = timestamp
            self._record_unsubscribe(recipient_email, "complaint")
        elif event_type == "unsubscribed":
            delivery.status = DeliveryStatus.UNSUBSCRIBED
            self._record_unsubscribe(recipient_email, "unsubscribe_link")
        else:
            logger.warning(
                "Unknown webhook event type",
                extra={"event_type": event_type, "message_id": message_id},
            )
            return delivery

        self.db_session.commit()
        self.db_session.refresh(delivery)
        status_val = (
            delivery.status.value
            if hasattr(delivery.status, "value")
            else str(delivery.status)
        )
        DELIVERY_RATE.labels(status=status_val).inc()
        return delivery

    def _record_unsubscribe(self, email: str, source: str) -> None:
        """Persist an unsubscribe record for the given email."""
        existing = self.db_session.execute(
            select(Unsubscribe).where(Unsubscribe.email == email)
        ).scalar_one_or_none()
        if existing is None:
            self.db_session.add(
                Unsubscribe(
                    email=email,
                    source=source,
                    unsubscribed_at=datetime.now(timezone.utc),
                )
            )
