"""Basic metrics API route."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db_session
from app.models.entities import Delivery, DeliveryStatus, User

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
def get_metrics(
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Return basic operational metrics."""
    total_deliveries = db_session.execute(
        select(func.count(Delivery.id))
    ).scalar_one()
    sent_count = db_session.execute(
        select(func.count(Delivery.id)).where(Delivery.status == DeliveryStatus.SENT)
    ).scalar_one()
    delivered_count = db_session.execute(
        select(func.count(Delivery.id)).where(Delivery.status == DeliveryStatus.DELIVERED)
    ).scalar_one()
    bounced_count = db_session.execute(
        select(func.count(Delivery.id)).where(Delivery.status == DeliveryStatus.BOUNCED)
    ).scalar_one()
    failed_count = db_session.execute(
        select(func.count(Delivery.id)).where(Delivery.status == DeliveryStatus.FAILED)
    ).scalar_one()

    error_rate = (failed_count / total_deliveries * 100.0) if total_deliveries > 0 else 0.0

    return {
        "total_deliveries": total_deliveries,
        "sent_count": sent_count,
        "delivered_count": delivered_count,
        "bounced_count": bounced_count,
        "failed_count": failed_count,
        "error_rate_percent": round(error_rate, 2),
    }