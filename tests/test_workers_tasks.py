"""Unit tests for Celery background tasks."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from sqlalchemy.orm import Session

from app.models.entities import Appeal, AsyncJob, AsyncJobStatus, Campaign, Delivery, DeliveryStatus, Donor, SuppressionEntry
from app.workers.tasks import (
    DatabaseTask,
    auto_send_appeals_task,
    backup_database_task,
    monitor_queue_depth_task,
    retrain_model_task,
    run_async_job_task,
    send_campaign_task,
)




def test_send_campaign_task_success(db_session: Session) -> None:
    """Verify send_campaign_task executes email delivery for an appeal."""
    donor = Donor(
        email="task-donor@example.com",
        first_name="Task",
        last_name="Donor",
        channel="email",
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
    )
    db_session.add(donor)

    appeal = Appeal(
        donor=donor,
        subject="Spring Appeal",
        body="Dear Donor, please support us.",
        cta="Donate",
        tone="inspiring",
    )
    db_session.add(appeal)
    db_session.commit()

    res = send_campaign_task.apply(args=[appeal.id])
    assert res.state == "SUCCESS"
    result = res.result
    assert result["status"] in ("sent", "delivered", "queued")
    assert "delivery_id" in result


def test_send_campaign_task_blocked_by_suppression(db_session: Session) -> None:
    """Verify send_campaign_task blocks delivery when donor is suppressed."""
    donor = Donor(
        email="suppressed-task@example.com",
        first_name="Suppressed",
        last_name="Donor",
        channel="email",
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
    )
    db_session.add(donor)

    suppression = SuppressionEntry(
        email=donor.email,
        reason="unsubscribe",
        source="api",
    )
    db_session.add(suppression)

    appeal = Appeal(
        donor=donor,
        subject="Blocked Appeal",
        body="This should not be delivered.",
        cta="Donate",
        tone="urgent",
    )
    db_session.add(appeal)
    db_session.commit()

    res = send_campaign_task.apply(args=[appeal.id])
    assert res.state == "SUCCESS"
    result = res.result
    assert result["status"] == "blocked"
    assert "suppress" in result["reason"].lower() or "unsubscribe" in result["reason"].lower()


def test_run_async_job_task(db_session: Session) -> None:
    """Verify run_async_job_task executes and marks job completed."""
    job = AsyncJob(
        job_type="general_maintenance",
        status=AsyncJobStatus.PENDING,
        total_items=10,
    )
    db_session.add(job)
    db_session.commit()

    res = run_async_job_task.apply(args=[job.id])
    assert res.state == "SUCCESS"
    result = res.result
    assert result["status"] == "completed"

    db_session.refresh(job)
    assert job.status == AsyncJobStatus.COMPLETED


def test_auto_send_appeals_task(db_session: Session) -> None:
    """Verify auto_send_appeals_task generates appeals and enqueues tasks."""
    donor = Donor(
        email="autosend-donor@example.com",
        first_name="Auto",
        last_name="Send",
        channel="email",
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
    )
    db_session.add(donor)
    db_session.commit()

    with patch("app.workers.tasks.send_campaign_task.delay") as mock_delay:
        mock_delay.return_value = MagicMock(id="auto-task-999")
        res = auto_send_appeals_task.apply()

    assert res.state == "SUCCESS"
    result = res.result
    assert "queued" in result
    assert "task_ids" in result


def test_retrain_model_task(db_session: Session) -> None:
    """Verify retrain_model_task runs training and returns model version."""
    res = retrain_model_task.apply()
    assert res.state == "SUCCESS"
    result = res.result

    assert "model_version_id" in result
    assert "status" in result
    assert "promoted" in result


def test_backup_database_task_skipped_when_no_bucket() -> None:
    """Verify backup_database_task skips when no S3 bucket is configured."""
    with patch("app.core.config.get_settings") as mock_settings:
        mock_settings.return_value.database.backup_bucket = None
        res = backup_database_task.apply()

    assert res.state == "SUCCESS"
    result = res.result
    assert result["status"] == "skipped"
    assert result["reason"] == "no_backup_bucket_configured"


def test_backup_database_task_success() -> None:
    """Verify backup_database_task runs pg_dump and aws s3 cp when bucket is configured."""
    with patch("app.core.config.get_settings") as mock_settings, \
         patch("subprocess.run") as mock_subproc, \
         patch("os.remove") as mock_remove:
        mock_settings.return_value.database.backup_bucket = "my-backup-bucket"
        mock_settings.return_value.database.sqlalchemy_url = "postgresql://localhost/db"
        mock_subproc.return_value = MagicMock(returncode=0)

        res = backup_database_task.apply()

    assert res.state == "SUCCESS"
    result = res.result
    assert result["status"] == "completed"
    assert "backup_filename" in result
    assert mock_subproc.call_count == 2


def test_monitor_queue_depth_task() -> None:
    """Verify monitor_queue_depth_task inspects active queues."""
    with patch("app.workers.tasks.celery_app.control.inspect") as mock_inspect:
        mock_inspect.return_value.active_queues.return_value = {
            "worker1@host": [{"name": "appealcrafter"}],
            "worker2@host": [{"name": "appealcrafter"}],
        }
        res = monitor_queue_depth_task.apply()

    assert res.state == "SUCCESS"
    result = res.result
    assert result["queue_depth"] == 2
