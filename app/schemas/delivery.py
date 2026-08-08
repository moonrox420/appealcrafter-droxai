"""Delivery and webhook event schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DeliveryResponse(BaseModel):
    """Delivery record response payload."""

    id: str
    appeal_id: str
    recipient_email: str
    provider_message_id: str | None
    status: str
    provider: str
    error_detail: str | None
    sent_at: datetime | None
    delivered_at: datetime | None
    opened_at: datetime | None
    clicked_at: datetime | None
    bounced_at: datetime | None
    complained_at: datetime | None
    created_at: datetime


class WebhookEvent(BaseModel):
    """Provider webhook event payload."""

    event_type: str = Field(..., description="Provider event type (delivered, opened, clicked, bounced, complained).")
    message_id: str = Field(..., description="Provider message identifier.")
    recipient_email: str = Field(..., description="Recipient email address.")
    timestamp: datetime = Field(..., description="Event timestamp.")
    metadata: dict = Field(default_factory=dict, description="Additional provider metadata.")