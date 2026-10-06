"""Appeal generation API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db_session
from app.models.entities import AppealTone, User
from app.schemas.appeal import (
    AppealGenerateRequest,
    AppealResponse,
    GenerateAppealResponse,
)
from app.services.appeal import AppealService
from app.workers.tasks import send_campaign_task

router = APIRouter(prefix="/generate-appeal", tags=["appeals"])


@router.post("", response_model=GenerateAppealResponse)
def generate_appeal(
    payload: AppealGenerateRequest,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> GenerateAppealResponse:
    """Generate appeals, persist records, and queue real sends."""
    appeal_service = AppealService(db_session)
    try:
        appeals = appeal_service.generate_appeals(
            tone=AppealTone(payload.tone),
            campaign_id=payload.campaign_id,
            limit=payload.limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    task_ids: list[str] = []
    appeal_responses: list[AppealResponse] = []
    for appeal in appeals:
        task_result = send_campaign_task.delay(appeal.id)
        task_ids.append(task_result.id)
        appeal_responses.append(
            AppealResponse(
                id=appeal.id,
                donor_id=appeal.donor_id,
                campaign_id=appeal.campaign_id,
                subject=appeal.subject,
                body=appeal.body,
                cta=appeal.cta,
                tone=(
                    appeal.tone.value
                    if hasattr(appeal.tone, "value")
                    else str(appeal.tone)
                ),
                capacity_score=appeal.capacity_score,
                is_template_fallback=appeal.is_template_fallback,
                created_at=appeal.created_at,
            )
        )

    return GenerateAppealResponse(
        generated_count=len(appeal_responses),
        task_ids=task_ids,
        appeals=appeal_responses,
    )
