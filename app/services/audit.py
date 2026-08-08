"""Administrative audit logging service for protected actions."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_trace_id
from app.models.entities import AuditLog, User

logger = logging.getLogger(__name__)


class AuditLogService:
    """Record before/after state for administrative actions."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def record_action(
        self,
        actor_user: User | None,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        before_state: dict[str, Any] | None = None,
        after_state: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        ip_address: str | None = None,
    ) -> AuditLog:
        """Record an auditable administrative action."""
        audit_log = AuditLog(
            actor_user_id=actor_user.id if actor_user else None,
            tenant_id=tenant_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            before_state=before_state,
            after_state=after_state,
            ip_address=ip_address,
            trace_id=get_trace_id(),
        )
        self.db_session.add(audit_log)
        self.db_session.commit()
        self.db_session.refresh(audit_log)
        logger.info(
            "Audit action recorded",
            extra={
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "actor_user_id": audit_log.actor_user_id,
            },
        )
        return audit_log


def get_audit_log_service(db_session: Session) -> AuditLogService:
    """Return a configured audit log service instance."""
    return AuditLogService(db_session)