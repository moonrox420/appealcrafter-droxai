"""Appeal generation request and response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class AppealGenerateRequest(BaseModel):
    """Payload for generating appeals for a campaign."""

    campaign_id: str | None = Field(default=None, description="Optional campaign identifier.")
    tone: str = Field(default="inspiring", pattern="^(inspiring|urgent|grateful|hopeful)$", description="Appeal tone.")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of appeals to generate.")


class AppealResponse(BaseModel):
    """Generated appeal response payload."""

    id: str
    donor_id: str
    campaign_id: str | None
    subject: str
    body: str
    cta: str
    tone: str
    capacity_score: float | None
    is_template_fallback: bool
    created_at: datetime


class GenerateAppealResponse(BaseModel):
    """Bulk appeal generation result."""

    generated_count: int = Field(..., description="Number of appeals generated.")
    task_ids: list[str] = Field(default_factory=list, description="Celery task IDs for queued sends.")
    appeals: list[AppealResponse] = Field(default_factory=list, description="Generated appeal records.")