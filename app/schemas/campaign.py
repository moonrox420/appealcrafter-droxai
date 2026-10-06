"""Campaign request and response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CampaignCreate(BaseModel):
    """Campaign creation payload."""

    name: str = Field(..., min_length=1, max_length=255, description="Campaign name.")
    description: str | None = Field(default=None, description="Campaign description.")


class CampaignResponse(BaseModel):
    """Campaign response payload."""

    id: str
    name: str
    description: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime
