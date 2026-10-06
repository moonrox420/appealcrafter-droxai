"""Unit tests for FeatureFlagService percentage rollouts, tenant overrides, and hash assignment."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.entities import FeatureFlag, FeatureFlagStatus, Tenant
from app.services.feature_flags import FeatureFlagService


def test_feature_flag_enabled_and_disabled(db_session: Session) -> None:
    """Verify ENABLED and DISABLED status evaluation."""
    service = FeatureFlagService(db_session)

    flag_enabled = FeatureFlag(
        name="test_flag_enabled",
        status=FeatureFlagStatus.ENABLED,
        rollout_percent=0.0,
    )
    flag_disabled = FeatureFlag(
        name="test_flag_disabled",
        status=FeatureFlagStatus.DISABLED,
        rollout_percent=100.0,
    )
    db_session.add_all([flag_enabled, flag_disabled])
    db_session.commit()

    assert service.is_enabled("test_flag_enabled") is True
    assert service.is_enabled("test_flag_disabled") is False
    assert service.is_enabled("nonexistent_flag") is False


def test_feature_flag_rollout_percentages(db_session: Session) -> None:
    """Verify 0%, 100%, and deterministic hash evaluation across subjects."""
    service = FeatureFlagService(db_session)

    flag_zero = FeatureFlag(
        name="flag_zero_pct",
        status=FeatureFlagStatus.ROLLOUT,
        rollout_percent=0.0,
    )
    flag_hundred = FeatureFlag(
        name="flag_hundred_pct",
        status=FeatureFlagStatus.ROLLOUT,
        rollout_percent=100.0,
    )
    flag_fifty = FeatureFlag(
        name="flag_fifty_pct",
        status=FeatureFlagStatus.ROLLOUT,
        rollout_percent=50.0,
    )
    db_session.add_all([flag_zero, flag_hundred, flag_fifty])
    db_session.commit()

    # 0% rollout is always False
    assert service.is_enabled("flag_zero_pct", subject_key="user-123") is False

    # 100% rollout is always True
    assert service.is_enabled("flag_hundred_pct", subject_key="user-123") is True

    # No subject key for rollout flag returns False
    assert service.is_enabled("flag_fifty_pct", subject_key=None) is False

    # Hash consistency: same subject always gets same decision
    res1 = service.is_enabled("flag_fifty_pct", subject_key="consistent-subject-uuid")
    res2 = service.is_enabled("flag_fifty_pct", subject_key="consistent-subject-uuid")
    assert res1 == res2


def test_feature_flag_tenant_override(db_session: Session) -> None:
    """Verify that tenant-specific flags override global flags."""
    service = FeatureFlagService(db_session)

    tenant = Tenant(name="Acme Non-Profit")
    db_session.add(tenant)
    db_session.flush()

    global_flag = FeatureFlag(
        name="ai_v2_pilot",
        status=FeatureFlagStatus.DISABLED,
        tenant_id=None,
    )
    tenant_flag = FeatureFlag(
        name="ai_v2_pilot",
        status=FeatureFlagStatus.ENABLED,
        tenant_id=tenant.id,
    )
    db_session.add_all([global_flag, tenant_flag])
    db_session.commit()

    # Global context gets global flag (disabled)
    assert service.is_enabled("ai_v2_pilot") is False

    # Tenant context gets tenant flag (enabled)
    assert service.is_enabled("ai_v2_pilot", tenant_id=tenant.id) is True

    # Retrieve flag directly
    retrieved = service.get_flag("ai_v2_pilot", tenant_id=tenant.id)
    assert retrieved is not None
    assert retrieved.tenant_id == tenant.id
