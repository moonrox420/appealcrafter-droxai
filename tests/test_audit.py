"""Unit tests for administrative audit logging."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.models.entities import AuditLog, User, UserRole
from app.services.audit import AuditLogService, get_audit_log_service


def test_audit_log_record_action_with_user(db_session: Session) -> None:
    """Verify recording an administrative action by an authenticated user."""
    user = User(
        id=str(uuid.uuid4()),
        email="auditor@example.com",
        password_hash="hash",
        role=UserRole.ADMIN,
        is_active=True,
    )
    from app.models.entities import Tenant

    tenant = Tenant(
        id="tenant-abc",
        name="Test Nonprofit",
    )
    db_session.add(tenant)
    db_session.add(user)
    db_session.commit()

    service = get_audit_log_service(db_session)
    log_entry = service.record_action(
        actor_user=user,
        action="update_template",
        resource_type="template",
        resource_id="tpl-123",
        before_state={"tone": "formal"},
        after_state={"tone": "inspiring"},
        tenant_id="tenant-abc",
        ip_address="192.168.1.1",
    )

    assert log_entry.id is not None
    assert log_entry.actor_user_id == user.id
    assert log_entry.action == "update_template"
    assert log_entry.resource_type == "template"
    assert log_entry.resource_id == "tpl-123"
    assert log_entry.before_state == {"tone": "formal"}
    assert log_entry.after_state == {"tone": "inspiring"}
    assert log_entry.tenant_id == "tenant-abc"
    assert log_entry.ip_address == "192.168.1.1"

    # Query from db
    saved = db_session.get(AuditLog, log_entry.id)
    assert saved is not None
    assert saved.action == "update_template"


def test_audit_log_record_action_anonymous_or_system(db_session: Session) -> None:
    """Verify recording a system action without an actor user."""
    service = AuditLogService(db_session)
    log_entry = service.record_action(
        actor_user=None,
        action="system_purge",
        resource_type="retention_policy",
        resource_id=None,
        before_state=None,
        after_state={"purged_count": 42},
    )

    assert log_entry.id is not None
    assert log_entry.actor_user_id is None
    assert log_entry.action == "system_purge"
    assert log_entry.resource_type == "retention_policy"
    assert log_entry.after_state == {"purged_count": 42}
