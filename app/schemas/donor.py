"""Donor and donation request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class DonationCreate(BaseModel):
    """Single donation record payload."""

    amount: float = Field(..., gt=0, description="Donation amount.")
    donated_at: datetime = Field(..., description="Donation timestamp.")


class DonorCreate(BaseModel):
    """Donor creation payload."""

    external_id: str | None = Field(default=None, max_length=255, description="External donor identifier.")
    email: EmailStr = Field(..., description="Donor email address.")
    first_name: str | None = Field(default=None, max_length=100, description="Donor first name.")
    last_name: str | None = Field(default=None, max_length=100, description="Donor last name.")
    interests: str | None = Field(default=None, description="Donor interest keywords.")
    channel: str = Field(default="email", max_length=20, description="Preferred contact channel.")
    consent_given_at: datetime | None = Field(default=None, description="Consent timestamp.")
    consent_source: str | None = Field(default=None, max_length=100, description="Consent source.")
    donations: list[DonationCreate] = Field(default_factory=list, description="Donation history records.")


class DonorResponse(BaseModel):
    """Donor response payload."""

    id: str
    external_id: str | None
    email: EmailStr
    first_name: str | None
    last_name: str | None
    interests: str | None
    channel: str
    capacity_score: float | None
    consent_given_at: datetime | None
    created_at: datetime


class DonorIngestResponse(BaseModel):
    """Bulk donor ingestion result."""

    accepted_count: int = Field(..., description="Number of valid donor records persisted.")
    rejected_count: int = Field(..., description="Number of invalid donor records.")
    errors: list[dict] = Field(default_factory=list, description="Per-record validation errors.")