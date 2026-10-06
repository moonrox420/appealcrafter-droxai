"""Async bulk job tracking service for ingest, generation, and reporting."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.entities import AsyncJob, AsyncJobStatus

logger = logging.getLogger(__name__)


class AsyncJobService:
    """Create and track long-running bulk jobs."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def create_job(
        self,
        job_type: str,
        payload: dict[str, Any] | None = None,
        created_by_user_id: str | None = None,
        total_items: int | None = None,
    ) -> AsyncJob:
        """Create a pending async job and return it immediately."""
        job = AsyncJob(
            job_type=job_type,
            status=AsyncJobStatus.PENDING,
            payload=payload,
            created_by_user_id=created_by_user_id,
            total_items=total_items,
            completed_items=0,
        )
        self.db_session.add(job)
        self.db_session.commit()
        self.db_session.refresh(job)
        return job

    def mark_running(self, job_id: str) -> AsyncJob:
        """Mark a job as running."""
        job = self.db_session.get(AsyncJob, job_id)
        if job is None:
            raise ValueError(f"Async job {job_id} not found.")
        job.status = AsyncJobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        self.db_session.commit()
        self.db_session.refresh(job)
        return job

    def mark_completed(
        self,
        job_id: str,
        result_summary: dict[str, Any] | None = None,
        completed_items: int | None = None,
    ) -> AsyncJob:
        """Mark a job as completed."""
        job = self.db_session.get(AsyncJob, job_id)
        if job is None:
            raise ValueError(f"Async job {job_id} not found.")
        job.status = AsyncJobStatus.COMPLETED
        job.result_summary = result_summary
        job.completed_items = (
            completed_items if completed_items is not None else job.total_items
        )
        job.completed_at = datetime.now(timezone.utc)
        self.db_session.commit()
        self.db_session.refresh(job)
        return job

    def mark_failed(self, job_id: str, error_detail: str) -> AsyncJob:
        """Mark a job as failed."""
        job = self.db_session.get(AsyncJob, job_id)
        if job is None:
            raise ValueError(f"Async job {job_id} not found.")
        job.status = AsyncJobStatus.FAILED
        job.error_detail = error_detail
        job.completed_at = datetime.now(timezone.utc)
        self.db_session.commit()
        self.db_session.refresh(job)
        return job

    def increment_progress(self, job_id: str, completed_items: int) -> AsyncJob:
        """Update the completed item count for a running job."""
        job = self.db_session.get(AsyncJob, job_id)
        if job is None:
            raise ValueError(f"Async job {job_id} not found.")
        job.completed_items = completed_items
        self.db_session.commit()
        self.db_session.refresh(job)
        return job

    def get_job(self, job_id: str) -> AsyncJob | None:
        """Return a job record by ID."""
        return self.db_session.get(AsyncJob, job_id)

    def list_jobs(self, limit: int = 50, offset: int = 0) -> list[AsyncJob]:
        """Return recent async jobs."""
        return list(
            self.db_session.execute(
                select(AsyncJob)
                .order_by(AsyncJob.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            .scalars()
            .all()
        )


def get_async_job_service(db_session: Session) -> AsyncJobService:
    """Return a configured async job service instance."""
    return AsyncJobService(db_session)
