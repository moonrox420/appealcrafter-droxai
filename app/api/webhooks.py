"""Provider webhook API routes with signature verification."""

from __future__ import annotations

import hashlib
import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db_session
from app.schemas.delivery import WebhookEvent
from app.services.delivery import DeliveryService

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


async def verify_webhook_signature(
    request: Request,
    x_webhook_signature: Annotated[str | None, Header()] = None,
) -> None:
    """Verify the provider webhook signature using HMAC-SHA256."""
    settings = get_settings()
    if x_webhook_signature is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing webhook signature",
        )
    body = await request.body()
    expected_signature = hmac.new(
        settings.email.webhook_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_signature, x_webhook_signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )


@router.post("/email-events", status_code=status.HTTP_200_OK)
async def process_email_event(
    payload: WebhookEvent,
    db_session: Annotated[Session, Depends(get_db_session)],
    _signature_verified: Annotated[None, Depends(verify_webhook_signature)],
) -> dict:
    """Process email delivery events from the provider."""
    delivery_service = DeliveryService(db_session)
    delivery = delivery_service.process_webhook_event(
        event_type=payload.event_type,
        message_id=payload.message_id,
        recipient_email=payload.recipient_email,
        timestamp=payload.timestamp,
    )
    if delivery is None:
        return {"status": "ignored", "reason": "unknown_message"}
    return {"status": "processed", "delivery_id": delivery.id}
