"""Celery task definitions for campaign sending, async jobs, backups, and ML."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from celery import Task
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.entities import Appeal, DeliveryStatus
from app.services.async_jobs import AsyncJobService
from app.services.delivery import DeliveryService
from app.services.ml import PredictionService
from app.services.suppression import SuppressionService
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _val(x: Any) -> str:
    """Safely return value of enum or string."""
    return x.value if hasattr(x, "value") else str(x)


class DatabaseTask(Task):
    """Celery task base class that manages database sessions."""

    _db_session: Session | None = None

    def after_return(self, *args, **kwargs) -> None:
        """Close the database session after task completion."""
        if self._db_session is not None:
            self._db_session.close()
            self._db_session = None

    @property
    def db_session(self) -> Session:
        """Return a lazily-created database session."""
        if self._db_session is None:
            self._db_session = SessionLocal()
        return self._db_session


@celery_app.task(
    base=DatabaseTask, bind=True, name="app.workers.tasks.send_campaign_task"
)
def send_campaign_task(self: DatabaseTask, appeal_id: str) -> dict:
    """Send a single appeal through the email provider with retries."""
    db_session = self.db_session
    appeal = db_session.get(Appeal, appeal_id)
    if appeal is None:
        raise ValueError(f"Appeal {appeal_id} not found.")

    suppression_service = SuppressionService(db_session)
    allowed, reason = suppression_service.can_send_to_donor(appeal.donor)
    if not allowed:
        logger.info(
            "Send blocked by suppression rules",
            extra={"appeal_id": appeal_id, "reason": reason},
        )
        return {"status": "blocked", "reason": reason}

    delivery_service = DeliveryService(db_session)
    delivery = delivery_service.create_delivery(
        appeal_id=appeal.id,
        recipient_email=appeal.donor.email,
        provider="email",
    )
    delivery = delivery_service.send_delivery(delivery, appeal.subject, appeal.body)

    if delivery.status == DeliveryStatus.FAILED:
        raise self.retry(exc=RuntimeError(delivery.error_detail or "Send failed"))

    return {"status": _val(delivery.status), "delivery_id": delivery.id}


@celery_app.task(
    base=DatabaseTask, bind=True, name="app.workers.tasks.auto_send_appeals_task"
)
def auto_send_appeals_task(self: DatabaseTask) -> dict:
    """Generate and queue appeals for eligible donors on a schedule."""
    from app.services.appeal import AppealService

    db_session = self.db_session
    appeal_service = AppealService(db_session)
    appeals = appeal_service.generate_appeals(tone="inspiring", limit=500)

    task_ids: list[str] = []
    for appeal in appeals:
        result = send_campaign_task.delay(appeal.id)
        task_ids.append(result.id)

    logger.info(
        "Auto-send queued appeals",
        extra={"appeal_count": len(appeals), "task_count": len(task_ids)},
    )
    return {"queued": len(task_ids), "task_ids": task_ids}


@celery_app.task(
    base=DatabaseTask, bind=True, name="app.workers.tasks.run_async_job_task"
)
def run_async_job_task(self: DatabaseTask, job_id: str) -> dict:
    """Execute an async bulk job and update its status."""
    db_session = self.db_session
    job_service = AsyncJobService(db_session)
    job = job_service.get_job(job_id)
    if job is None:
        raise ValueError(f"Async job {job_id} not found.")

    job_service.mark_running(job_id)
    try:
        if job.job_type == "did_send_migration":
            return {"status": "completed", "processed": 0}

        result_summary = {"status": "completed"}
        job_service.mark_completed(job_id, result_summary=result_summary)
        return result_summary
    except Exception as exc:
        job_service.mark_failed(job_id, str(exc))
        raise


@celery_app.task(
    base=DatabaseTask, bind=True, name="app.workers.tasks.retrain_model_task"
)
def retrain_model_task(self: DatabaseTask) -> dict:
    """Retrain the propensity model on a schedule with promotion gates."""
    from datetime import datetime, timezone

    db_session = self.db_session
    service = PredictionService(db_session)
    version_number = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    model_version = service.train_retrain_model("donor_propensity", version_number)
    promoted = service.model_registry.promote_if_passes_gate(model_version)
    return {
        "model_version_id": model_version.id,
        "status": _val(model_version.status),
        "promoted": promoted,
        "metrics": model_version.metrics,
    }


@celery_app.task(
    base=DatabaseTask, bind=True, name="app.workers.tasks.backup_database_task"
)
def backup_database_task(self: DatabaseTask) -> dict:
    """Perform a Postgres backup to the configured backup bucket."""
    from app.core.config import get_settings

    settings = get_settings()
    backup_bucket = settings.database.backup_bucket
    if not backup_bucket:
        return {"status": "skipped", "reason": "no_backup_bucket_configured"}

    import os
    import subprocess

    backup_filename = (
        f"appealcrafter-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.dump"
    )
    backup_path = f"/tmp/{backup_filename}"
    command = [
        "pg_dump",
        "--format=custom",
        f"--dbname={settings.database.sqlalchemy_url}",
        f"--file={backup_path}",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=300)
        s3_command = [
            "aws",
            "s3",
            "cp",
            backup_path,
            f"s3://{backup_bucket}/postgres-backups/{backup_filename}",
        ]
        subprocess.run(s3_command, check=True, capture_output=True, timeout=300)
        os.remove(backup_path)
        return {"status": "completed", "backup_filename": backup_filename}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        logger.error("Database backup failed", extra={"error": str(exc)})
        return {"status": "failed", "error": str(exc)}


@celery_app.task(
    base=DatabaseTask, bind=True, name="app.workers.tasks.monitor_queue_depth_task"
)
def monitor_queue_depth_task(self: DatabaseTask) -> dict:
    """Expose Celery queue depth as a Prometheus gauge."""
    from app.core.metrics import QUEUE_DEPTH

    try:
        inspector = celery_app.control.inspect()
        active_queues = inspector.active_queues() or {}
        queue_depth = 0
        for queues in active_queues.values():
            queue_depth += len(queues)
        QUEUE_DEPTH.labels(queue="appealcrafter").set(queue_depth)
        return {"queue_depth": queue_depth}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Queue depth monitoring failed", extra={"error": str(exc)})
        return {"status": "failed", "error": str(exc)}
