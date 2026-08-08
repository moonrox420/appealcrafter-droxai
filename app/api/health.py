"""Health and readiness API routes."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.db.session import verify_database_connection
from app.workers.celery_app import celery_app

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    """Return 200 if the process is alive."""
    return {"status": "ok"}


@router.get("/ready")
def readiness_check(response: Response) -> dict:
    """Return 200 only when database and broker are reachable."""
    database_ok = verify_database_connection()
    broker_ok = celery_app.connection().connected
    if not database_ok or not broker_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "database": database_ok,
            "broker": broker_ok,
        }
    return {"status": "ready", "database": True, "broker": True}