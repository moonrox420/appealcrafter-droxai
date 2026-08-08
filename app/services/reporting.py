"""Campaign reporting service with full delivery lifecycle metrics."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.entities import Appeal, Campaign, Delivery, DeliveryStatus

logger = logging.getLogger(__name__)


class ReportingService:
    """Compute campaign and delivery lifecycle reports."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def get_campaign_report(self, campaign_id: str) -> dict:
        """Return a full report for a campaign with lifecycle counts."""
        campaign = self.db_session.get(Campaign, campaign_id)
        if campaign is None:
            raise ValueError(f"Campaign {campaign_id} not found.")

        appeal_ids = self.db_session.execute(
            select(Appeal.id).where(Appeal.campaign_id == campaign_id)
        ).scalars().all()

        delivery_query = select(Delivery).where(Delivery.appeal_id.in_(appeal_ids)) if appeal_ids else select(Delivery).where(False)
        deliveries = list(self.db_session.execute(delivery_query).scalars().all())

        status_counts = {status: 0 for status in DeliveryStatus}
        for delivery in deliveries:
            status_counts[delivery.status] = status_counts.get(delivery.status, 0) + 1

        total_sends = status_counts[DeliveryStatus.SENT]
        total_delivered = status_counts[DeliveryStatus.DELIVERED]
        total_opened = status_counts[DeliveryStatus.OPENED]
        total_clicked = status_counts[DeliveryStatus.CLICKED]
        total_converted = status_counts[DeliveryStatus.CONVERTED]
        total_bounced = status_counts[DeliveryStatus.BOUNCED]
        total_unsubscribed = status_counts[DeliveryStatus.UNSUBSCRIBED]

        return {
            "campaign_id": campaign_id,
            "campaign_name": campaign.name,
            "total_appeals": len(appeal_ids),
            "total_deliveries": len(deliveries),
            "sent": total_sends,
            "delivered": total_delivered,
            "opened": total_opened,
            "clicked": total_clicked,
            "converted": total_converted,
            "bounced": total_bounced,
            "unsubscribed": total_unsubscribed,
            "delivery_rate": round((total_delivered / total_sends) * 100.0, 2) if total_sends > 0 else 0.0,
            "open_rate": round((total_opened / total_sends) * 100.0, 2) if total_sends > 0 else 0.0,
            "click_rate": round((total_clicked / total_sends) * 100.0, 2) if total_sends > 0 else 0.0,
            "conversion_rate": round((total_converted / total_sends) * 100.0, 2) if total_sends > 0 else 0.0,
            "bounce_rate": round((total_bounced / total_sends) * 100.0, 2) if total_sends > 0 else 0.0,
            "unsubscribe_rate": round((total_unsubscribed / total_sends) * 100.0, 2) if total_sends > 0 else 0.0,
        }

    def get_delivery_analytics(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict:
        """Return time-bucketed delivery analytics."""
        if start_date is None:
            start_date = datetime.now(timezone.utc).replace(day=1)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        statement = select(Delivery).where(
            Delivery.created_at >= start_date,
            Delivery.created_at <= end_date,
        )
        deliveries = list(self.db_session.execute(statement).scalars().all())

        daily_counts: dict[str, dict[str, int]] = {}
        for delivery in deliveries:
            day_key = delivery.created_at.date().isoformat()
            if day_key not in daily_counts:
                daily_counts[day_key] = {"sent": 0, "delivered": 0, "opened": 0, "clicked": 0, "converted": 0, "bounced": 0, "unsubscribed": 0}
            daily_counts[day_key][delivery.status.value] = daily_counts[day_key].get(delivery.status.value, 0) + 1

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_deliveries": len(deliveries),
            "daily_breakdown": daily_counts,
        }


def get_reporting_service(db_session: Session) -> ReportingService:
    """Return a configured reporting service instance."""
    return ReportingService(db_session)