"""Operations API routers: suppression, reporting, jobs, compliance, ML."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_role
from app.db.session import get_db_session
from app.models.entities import User, UserRole
from app.schemas.phase2 import (
    AsyncJobResponse,
    ComplianceActionResponse,
    ModelTrainResponse,
    SuppressionEntryCreate,
    SuppressionEntryResponse,
)
from app.services.async_jobs import AsyncJobService
from app.services.compliance import ComplianceService
from app.services.ml import PredictionService
from app.services.reporting import ReportingService
from app.services.suppression import SuppressionService

router = APIRouter(tags=["operations"])

suppression_router = APIRouter(prefix="/suppression", tags=["suppression"])
reporting_router = APIRouter(prefix="/reporting", tags=["reporting"])
jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])
compliance_router = APIRouter(prefix="/compliance", tags=["compliance"])
ml_router = APIRouter(prefix="/ml", tags=["ml"])

admin_required = require_role(UserRole.ADMIN)


def _val(x: Any) -> str:
    """Safely return value of enum or string."""
    return x.value if hasattr(x, "value") else str(x)


@suppression_router.post(
    "", response_model=SuppressionEntryResponse, status_code=status.HTTP_201_CREATED
)
def add_suppression_entry(
    payload: SuppressionEntryCreate,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> SuppressionEntryResponse:
    """Add an API-managed suppression list entry."""
    service = SuppressionService(db_session)
    entry = service.add_suppression_entry(
        email=payload.email,
        reason=payload.reason,
        source="api",
        created_by_user_id=current_user.id,
        expires_at=payload.expires_at,
    )
    return SuppressionEntryResponse(
        id=entry.id,
        email=entry.email,
        reason=entry.reason,
        source=entry.source,
        suppressed_at=entry.suppressed_at,
        expires_at=entry.expires_at,
    )


@suppression_router.get("", response_model=list[SuppressionEntryResponse])
def list_suppression_entries(
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[SuppressionEntryResponse]:
    """List all suppression entries."""
    service = SuppressionService(db_session)
    entries = service.list_suppression_entries()
    return [
        SuppressionEntryResponse(
            id=entry.id,
            email=entry.email,
            reason=entry.reason,
            source=entry.source,
            suppressed_at=entry.suppressed_at,
            expires_at=entry.expires_at,
        )
        for entry in entries
    ]


@suppression_router.delete("/{email}", status_code=status.HTTP_204_NO_CONTENT)
def remove_suppression_entry(
    email: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> None:
    """Remove a suppression entry by email."""
    service = SuppressionService(db_session)
    removed = service.remove_suppression_entry(email)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Suppression entry not found."
        )


@reporting_router.get("/campaigns/{campaign_id}")
def get_campaign_report(
    campaign_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Return a full campaign report with lifecycle metrics."""
    service = ReportingService(db_session)
    try:
        return service.get_campaign_report(campaign_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@reporting_router.get("/deliveries")
def get_delivery_analytics(
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Return time-bucketed delivery analytics."""
    from datetime import datetime

    service = ReportingService(db_session)
    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None
    return service.get_delivery_analytics(start_date=start, end_date=end)


@jobs_router.post(
    "/{job_type}", response_model=AsyncJobResponse, status_code=status.HTTP_202_ACCEPTED
)
def create_async_job(
    job_type: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> AsyncJobResponse:
    """Create an async bulk job and return its ID immediately."""
    service = AsyncJobService(db_session)
    job = service.create_job(
        job_type=job_type,
        created_by_user_id=current_user.id,
    )
    return AsyncJobResponse(
        id=job.id,
        job_type=job.job_type,
        status=_val(job.status),
        total_items=job.total_items,
        completed_items=job.completed_items,
        result_summary=job.result_summary,
        error_detail=job.error_detail,
        created_at=job.created_at,
    )


@jobs_router.get("/{job_id}", response_model=AsyncJobResponse)
def get_async_job(
    job_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> AsyncJobResponse:
    """Return the status of an async job."""
    service = AsyncJobService(db_session)
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found."
        )
    return AsyncJobResponse(
        id=job.id,
        job_type=job.job_type,
        status=_val(job.status),
        total_items=job.total_items,
        completed_items=job.completed_items,
        result_summary=job.result_summary,
        error_detail=job.error_detail,
        created_at=job.created_at,
    )


@jobs_router.get("", response_model=list[AsyncJobResponse])
def list_async_jobs(
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[AsyncJobResponse]:
    """List recent async jobs."""
    service = AsyncJobService(db_session)
    jobs = service.list_jobs()
    return [
        AsyncJobResponse(
            id=job.id,
            job_type=job.job_type,
            status=_val(job.status),
            total_items=job.total_items,
            completed_items=job.completed_items,
            result_summary=job.result_summary,
            error_detail=job.error_detail,
            created_at=job.created_at,
        )
        for job in jobs
    ]


@compliance_router.get("/donors/{donor_id}/export")
def export_donor_data(
    donor_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Export all personal data for a donor (GDPR/CCPA)."""
    service = ComplianceService(db_session)
    try:
        return service.export_donor_data(donor_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@compliance_router.post(
    "/donors/{donor_id}/anonymize", response_model=ComplianceActionResponse
)
def anonymize_donor_data(
    donor_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(admin_required)],
) -> ComplianceActionResponse:
    """Anonymize all personal data for a donor (GDPR/CCPA)."""
    service = ComplianceService(db_session)
    try:
        service.anonymize_donor(donor_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return ComplianceActionResponse(status="completed", donor_id=donor_id)


@compliance_router.delete("/donors/{donor_id}", response_model=ComplianceActionResponse)
def delete_donor_data(
    donor_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(admin_required)],
) -> ComplianceActionResponse:
    """Delete all personal data for a donor (GDPR/CCPA)."""
    service = ComplianceService(db_session)
    try:
        service.delete_donor(donor_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return ComplianceActionResponse(status="completed", donor_id=donor_id)


@ml_router.post(
    "/train/{model_name}/{version_number}", response_model=ModelTrainResponse
)
def train_model(
    model_name: str,
    version_number: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(admin_required)],
) -> ModelTrainResponse:
    """Train a new model version with metric-gated promotion."""
    service = PredictionService(db_session)
    model_version = service.train_retrain_model(model_name, version_number)
    service.model_registry.promote_if_passes_gate(model_version)
    return ModelTrainResponse(
        model_version_id=model_version.id,
        model_name=model_version.model_name,
        version_number=model_version.version_number,
        status=_val(model_version.status),
        metrics=model_version.metrics,
    )


@ml_router.get("/models/{model_name}/drift")
def check_model_drift(
    model_name: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Check prediction drift for a model."""
    service = PredictionService(db_session)
    model_version = service.model_registry.get_promoted_model(model_name)
    if model_version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No promoted model found."
        )
    return service.check_drift(model_version)
