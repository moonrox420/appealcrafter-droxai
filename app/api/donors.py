"""Donor management API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db_session
from app.models.entities import User
from app.schemas.donor import DonorCreate, DonorIngestResponse, DonorResponse
from app.services.donor import DonorService

router = APIRouter(prefix="/donors", tags=["donors"])


@router.post("/ingest-donors", response_model=DonorIngestResponse)
def ingest_donors(
    payload: list[DonorCreate],
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> DonorIngestResponse:
    """Bulk ingest donor records with per-record validation."""
    donor_service = DonorService(db_session)
    accepted_count = 0
    rejected_count = 0
    errors: list[dict] = []

    for index, donor_payload in enumerate(payload):
        try:
            donor_service.create_donor(donor_payload)
            accepted_count += 1
        except Exception as exc:
            rejected_count += 1
            errors.append(
                {"index": index, "email": donor_payload.email, "error": str(exc)}
            )

    return DonorIngestResponse(
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        errors=errors,
    )


@router.get("", response_model=list[DonorResponse])
def list_donors(
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = 100,
    offset: int = 0,
) -> list[DonorResponse]:
    """List active donors."""
    donor_service = DonorService(db_session)
    donors = donor_service.list_active_donors(limit=limit, offset=offset)
    return [
        DonorResponse(
            id=donor.id,
            external_id=donor.external_id,
            email=donor.email,
            first_name=donor.first_name,
            last_name=donor.last_name,
            interests=donor.interests,
            channel=donor.channel,
            capacity_score=donor.capacity_score,
            consent_given_at=donor.consent_given_at,
            created_at=donor.created_at,
        )
        for donor in donors
    ]


@router.delete("/{donor_id}", status_code=status.HTTP_204_NO_CONTENT)
def soft_delete_donor(
    donor_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> None:
    """Soft-delete a donor record."""
    donor_service = DonorService(db_session)
    try:
        donor_service.soft_delete_donor(donor_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
