"""Pydantic schemas for API request and response validation."""

from app.schemas.appeal import (
    AppealGenerateRequest,
    AppealResponse,
    GenerateAppealResponse,
)
from app.schemas.auth import LoginRequest, RefreshRequest, TokenResponse, UserCreate
from app.schemas.campaign import CampaignCreate, CampaignResponse
from app.schemas.delivery import DeliveryResponse, WebhookEvent
from app.schemas.donor import (
    DonationCreate,
    DonorCreate,
    DonorIngestResponse,
    DonorResponse,
)

__all__ = [
    "AppealGenerateRequest",
    "AppealResponse",
    "CampaignCreate",
    "CampaignResponse",
    "DeliveryResponse",
    "DonationCreate",
    "DonorCreate",
    "DonorIngestResponse",
    "DonorResponse",
    "GenerateAppealResponse",
    "LoginRequest",
    "RefreshRequest",
    "TokenResponse",
    "UserCreate",
    "WebhookEvent",
]
