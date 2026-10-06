"""Campaign management API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db_session
from app.models.entities import Campaign, User
from app.schemas.campaign import CampaignCreate, CampaignResponse

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


@router.post("", response_model=CampaignResponse, status_code=status.HTTP_201_CREATED)
def create_campaign(
    payload: CampaignCreate,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> CampaignResponse:
    """Create a new campaign."""
    campaign = Campaign(name=payload.name, description=payload.description)
    db_session.add(campaign)
    db_session.commit()
    db_session.refresh(campaign)
    return CampaignResponse(
        id=campaign.id,
        name=campaign.name,
        description=campaign.description,
        is_active=campaign.is_active,
        created_at=campaign.created_at,
        updated_at=campaign.updated_at,
    )


@router.get("", response_model=list[CampaignResponse])
def list_campaigns(
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[CampaignResponse]:
    """List all campaigns."""
    statement = select(Campaign).order_by(Campaign.created_at)
    campaigns = list(db_session.execute(statement).scalars().all())
    return [
        CampaignResponse(
            id=campaign.id,
            name=campaign.name,
            description=campaign.description,
            is_active=campaign.is_active,
            created_at=campaign.created_at,
            updated_at=campaign.updated_at,
        )
        for campaign in campaigns
    ]
