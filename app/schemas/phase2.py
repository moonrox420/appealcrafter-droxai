"""Phase 2 schemas for knowledge, templates, experiments, reporting, and more."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class KnowledgeDocumentCreate(BaseModel):
    """Knowledge document creation payload."""

    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    source_url: str | None = Field(default=None, max_length=500)
    is_approved: bool = Field(default=False)


class KnowledgeDocumentResponse(BaseModel):
    """Knowledge document response payload."""

    id: str
    title: str
    source_url: str | None
    is_approved: bool
    created_at: datetime


class TemplateCreate(BaseModel):
    """Template creation payload."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(default=None)
    subject_template: str = Field(..., min_length=1, max_length=255)
    body_template: str = Field(..., min_length=1)
    cta_template: str = Field(..., min_length=1, max_length=100)
    tone: str = Field(
        default="inspiring", pattern="^(inspiring|urgent|grateful|hopeful)$"
    )
    merge_tags: dict = Field(default_factory=dict)


class TemplateVersionCreate(BaseModel):
    """Template version creation payload."""

    subject_template: str = Field(..., min_length=1, max_length=255)
    body_template: str = Field(..., min_length=1)
    cta_template: str = Field(..., min_length=1, max_length=100)
    tone: str = Field(
        default="inspiring", pattern="^(inspiring|urgent|grateful|hopeful)$"
    )
    change_note: str | None = Field(default=None)


class TemplateResponse(BaseModel):
    """Template response payload."""

    id: str
    name: str
    description: str | None
    is_active: bool
    current_version_id: str | None
    created_at: datetime


class TemplateVersionResponse(BaseModel):
    """Template version response payload."""

    id: str
    template_id: str
    version_number: int
    subject_template: str
    body_template: str
    cta_template: str
    tone: str
    change_note: str | None
    created_at: datetime


class ExperimentCreate(BaseModel):
    """Experiment creation payload."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(default=None)
    hypothesis: str | None = Field(default=None)
    assignment_key: str = Field(default="donor_id", max_length=100)
    variants: list[dict] = Field(default_factory=list)


class ExperimentResponse(BaseModel):
    """Experiment response payload."""

    id: str
    name: str
    description: str | None
    hypothesis: str | None
    status: str
    assignment_key: str
    traffic_allocation_percent: float
    started_at: datetime | None
    completed_at: datetime | None
    promoted_variant_id: str | None
    confidence_level: float | None
    p_value: float | None


class SuppressionEntryCreate(BaseModel):
    """Suppression list entry creation payload."""

    email: EmailStr = Field(..., description="Email to suppress.")
    reason: str | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)


class SuppressionEntryResponse(BaseModel):
    """Suppression list entry response payload."""

    id: str
    email: str
    reason: str | None
    source: str
    suppressed_at: datetime
    expires_at: datetime | None


class FeatureFlagCreate(BaseModel):
    """Feature flag creation payload."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(default=None)
    status: str = Field(default="disabled", pattern="^(enabled|disabled|rollout)$")
    rollout_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    rules: dict | None = Field(default=None)


class FeatureFlagResponse(BaseModel):
    """Feature flag response payload."""

    id: str
    name: str
    description: str | None
    status: str
    rollout_percent: float
    updated_at: datetime


class JourneyCreate(BaseModel):
    """Journey creation payload."""

    name: str = Field(..., min_length=1, max_length=255)
    campaign_id: str | None = Field(default=None)
    trigger_type: str = Field(default="scheduled", max_length=50)
    config: dict = Field(default_factory=dict)
    steps: list[dict] = Field(default_factory=list)


class JourneyResponse(BaseModel):
    """Journey response payload."""

    id: str
    campaign_id: str | None
    name: str
    trigger_type: str
    is_active: bool
    created_at: datetime


class AsyncJobResponse(BaseModel):
    """Async job status response payload."""

    id: str
    job_type: str
    status: str
    total_items: int | None
    completed_items: int | None
    result_summary: dict | None
    error_detail: str | None
    created_at: datetime


class ComplianceActionResponse(BaseModel):
    """Compliance action response payload."""

    status: str = "completed"
    donor_id: str | None = None


class ModelTrainResponse(BaseModel):
    """Model training response payload."""

    model_version_id: str
    model_name: str
    version_number: str
    status: str
    metrics: dict | None
