"""Celery application configuration with retries and dead-letter queue."""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "appealcrafter",
    broker=settings.redis.url,
    backend=settings.redis.url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_default_queue="appealcrafter",
    task_routes={
        "app.workers.tasks.send_campaign_task": {"queue": "appealcrafter"},
        "app.workers.tasks.auto_send_appeals_task": {"queue": "appealcrafter"},
        "app.workers.tasks.run_async_job_task": {"queue": "appealcrafter"},
        "app.workers.tasks.retrain_model_task": {"queue": "appealcrafter"},
        "app.workers.tasks.backup_database_task": {"queue": "appealcrafter"},
        "app.workers.tasks.monitor_queue_depth_task": {"queue": "appealcrafter"},
    },
    task_annotations={
        "app.workers.tasks.send_campaign_task": {
            "max_retries": 3,
            "default_retry_delay": 60,
            "retry_backoff": True,
            "retry_backoff_max": 600,
            "retry_jitter": True,
        }
    },
)

celery_app.conf.beat_schedule = {
    "auto-send-appeals-daily": {
        "task": "app.workers.tasks.auto_send_appeals_task",
        "schedule": crontab(hour=9, minute=0),
    },
    "retrain-model-weekly": {
        "task": "app.workers.tasks.retrain_model_task",
        "schedule": crontab(hour=2, minute=0, day_of_week="sunday"),
    },
    "backup-database-daily": {
        "task": "app.workers.tasks.backup_database_task",
        "schedule": crontab(hour=1, minute=30),
    },
    "monitor-queue-depth": {
        "task": "app.workers.tasks.monitor_queue_depth_task",
        "schedule": crontab(minute="*/5"),
    },
}