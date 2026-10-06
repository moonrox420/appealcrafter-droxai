"""Feature flag service for controlled rollout and canary/blue-green support."""

from __future__ import annotations

import hashlib
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.cache import RedisCacheService
from app.models.entities import FeatureFlag, FeatureFlagStatus

logger = logging.getLogger(__name__)


class FeatureFlagService:
    """Evaluate feature flags with percentage-based rollout."""

    def __init__(
        self, db_session: Session, cache_service: RedisCacheService | None = None
    ) -> None:
        self.db_session = db_session
        self.cache_service = cache_service or RedisCacheService()

    def is_enabled(
        self,
        flag_name: str,
        tenant_id: str | None = None,
        subject_key: str | None = None,
    ) -> bool:
        """Return whether a feature flag is enabled for the given context."""
        cache_key = f"feature_flag:{tenant_id or 'global'}:{flag_name}"
        cached_value = self.cache_service.get(cache_key)
        if cached_value is not None:
            return bool(cached_value)

        statement = select(FeatureFlag).where(FeatureFlag.name == flag_name)
        if tenant_id is not None:
            statement = statement.where(
                (FeatureFlag.tenant_id == tenant_id) | (FeatureFlag.tenant_id.is_(None))
            )
        else:
            statement = statement.where(FeatureFlag.tenant_id.is_(None))
        flag = (
            self.db_session.execute(
                statement.order_by(FeatureFlag.tenant_id.desc().nulls_last())
            )
            .scalars()
            .first()
        )

        if flag is None:
            return False

        if flag.status == FeatureFlagStatus.ENABLED:
            self.cache_service.set(cache_key, True)
            return True
        if flag.status == FeatureFlagStatus.DISABLED:
            self.cache_service.set(cache_key, False)
            return False
        if flag.status == FeatureFlagStatus.ROLLOUT:
            if flag.rollout_percent >= 100.0:
                self.cache_service.set(cache_key, True)
                return True
            if flag.rollout_percent <= 0.0:
                self.cache_service.set(cache_key, False)
                return False
            if not subject_key:
                self.cache_service.set(cache_key, False)
                return False
            hash_digest = hashlib.sha256(
                f"{flag.id}:{subject_key}".encode()
            ).hexdigest()
            hash_value = int(hash_digest[:16], 16) / 0xFFFFFFFFFFFFFFFF
            enabled = (hash_value * 100.0) < flag.rollout_percent
            self.cache_service.set(cache_key, enabled)
            return enabled

        return False

    def get_flag(
        self, flag_name: str, tenant_id: str | None = None
    ) -> FeatureFlag | None:
        """Return a feature flag record."""
        statement = select(FeatureFlag).where(FeatureFlag.name == flag_name)
        if tenant_id is not None:
            statement = statement.where(
                (FeatureFlag.tenant_id == tenant_id) | (FeatureFlag.tenant_id.is_(None))
            )
        return (
            self.db_session.execute(
                statement.order_by(FeatureFlag.tenant_id.desc().nulls_last())
            )
            .scalars()
            .first()
        )


def get_feature_flag_service(db_session: Session) -> FeatureFlagService:
    """Return a configured feature flag service instance."""
    return FeatureFlagService(db_session)
