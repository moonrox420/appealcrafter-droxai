"""SQLAlchemy ORM models for the normalized AppealCrafter schema.

Tables: users, tenants, donors, donation_history, campaigns, appeals,
deliveries, unsubscribes, suppression_entries, templates, template_versions,
knowledge_documents, document_chunks, audit_logs, experiments,
experiment_variants, experiment_assignments, feature_flags, async_jobs,
model_versions, predictions, preferences, journeys, journey_steps,
guardrail_decisions.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def generate_uuid() -> str:
    """Generate a UUID string for primary keys."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class UserRole(str, Enum):
    """Role identifiers for RBAC."""

    ADMIN = "admin"
    OPERATOR = "operator"


class DeliveryStatus(str, Enum):
    """Delivery lifecycle status values."""

    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    BOUNCED = "bounced"
    FAILED = "failed"
    OPENED = "opened"
    CLICKED = "clicked"
    CONVERTED = "converted"
    COMPLAINED = "complained"
    UNSUBSCRIBED = "unsubscribed"


class AppealTone(str, Enum):
    """Supported appeal tone identifiers."""

    INSPIRING = "inspiring"
    URGENT = "urgent"
    GRATEFUL = "grateful"
    HOPEFUL = "hopeful"


class RiskLevel(str, Enum):
    """Guardrail risk classification levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailAction(str, Enum):
    """Actions taken by the guardrail pipeline."""

    AUTO_APPROVE = "auto_approve"
    HUMAN_REVIEW = "human_review"
    REJECT = "reject"
    REGENERATE = "regenerate"


class AsyncJobStatus(str, Enum):
    """Async bulk job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentStatus(str, Enum):
    """A/B experiment lifecycle states."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    PROMOTED = "promoted"
    ARCHIVED = "archived"


class FeatureFlagStatus(str, Enum):
    """Feature flag toggle states."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLOUT = "rollout"


class ModelVersionStatus(str, Enum):
    """ML model registry lifecycle states."""

    TRAINING = "training"
    VALIDATING = "validating"
    PROMOTED = "promoted"
    RETIRED = "retired"


class Tenant(Base):
    """Multi-tenant isolation root record."""

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    users: Mapped[list[User]] = relationship(back_populates="tenant")
    donors: Mapped[list[Donor]] = relationship(back_populates="tenant")
    campaigns: Mapped[list[Campaign]] = relationship(back_populates="tenant")
    templates: Mapped[list[Template]] = relationship(back_populates="tenant")
    experiments: Mapped[list[Experiment]] = relationship(back_populates="tenant")
    feature_flags: Mapped[list[FeatureFlag]] = relationship(back_populates="tenant")


class User(Base):
    """Authenticated application user with RBAC role and tenant scope."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenants.id"), index=True, nullable=True
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        String(20), nullable=False, default=UserRole.OPERATOR
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    tenant: Mapped[Tenant | None] = relationship(back_populates="users")


class Donor(Base):
    """Donor record with soft-delete support and tenant isolation."""

    __tablename__ = "donors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenants.id"), index=True, nullable=True
    )
    external_id: Mapped[str | None] = mapped_column(
        String(255), index=True, nullable=True
    )
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    first_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    interests: Mapped[str | None] = mapped_column(Text, nullable=True)
    channel: Mapped[str] = mapped_column(String(20), default="email", nullable=False)
    capacity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    propensity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    engagement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rfm_recency_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    rfm_frequency_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rfm_monetary_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    consent_given_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    consent_source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    email_encrypted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    tenant: Mapped[Tenant | None] = relationship(back_populates="donors")
    donation_history: Mapped[list[DonationHistory]] = relationship(
        back_populates="donor", cascade="all, delete-orphan"
    )
    appeals: Mapped[list[Appeal]] = relationship(back_populates="donor")
    unsubscribes: Mapped[list[Unsubscribe]] = relationship(back_populates="donor")
    preferences: Mapped[list[DonorPreference]] = relationship(
        back_populates="donor", cascade="all, delete-orphan"
    )
    predictions: Mapped[list[PredictionRecord]] = relationship(back_populates="donor")

    __table_args__ = (
        Index("ix_donors_email_active", "email", "is_deleted"),
        Index("ix_donors_tenant_email", "tenant_id", "email"),
    )


class DonationHistory(Base):
    """Individual donation record for a donor."""

    __tablename__ = "donation_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    donor_id: Mapped[str] = mapped_column(
        ForeignKey("donors.id", ondelete="CASCADE"), index=True, nullable=False
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    donated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    campaign_id: Mapped[str | None] = mapped_column(
        ForeignKey("campaigns.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    donor: Mapped[Donor] = relationship(back_populates="donation_history")
    campaign: Mapped[Campaign | None] = relationship(back_populates="donations")

    __table_args__ = (
        Index("ix_donation_history_donor_date", "donor_id", "donated_at"),
    )


class Campaign(Base):
    """A fundraising campaign that groups appeals and deliveries."""

    __tablename__ = "campaigns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenants.id"), index=True, nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    tenant: Mapped[Tenant | None] = relationship(back_populates="campaigns")
    appeals: Mapped[list[Appeal]] = relationship(back_populates="campaign")
    donations: Mapped[list[DonationHistory]] = relationship(back_populates="campaign")
    journeys: Mapped[list[Journey]] = relationship(back_populates="campaign")


class Appeal(Base):
    """A generated appeal message for a specific donor."""

    __tablename__ = "appeals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    donor_id: Mapped[str] = mapped_column(
        ForeignKey("donors.id"), index=True, nullable=False
    )
    campaign_id: Mapped[str | None] = mapped_column(
        ForeignKey("campaigns.id"), nullable=True
    )
    template_id: Mapped[str | None] = mapped_column(
        ForeignKey("templates.id"), nullable=True
    )
    experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiments.id"), nullable=True
    )
    experiment_variant_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiment_variants.id"), nullable=True
    )
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    cta: Mapped[str] = mapped_column(String(100), nullable=False)
    tone: Mapped[AppealTone] = mapped_column(
        String(20), nullable=False, default=AppealTone.INSPIRING
    )
    capacity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_template_fallback: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    guardrail_decision_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "guardrail_decisions.id",
            use_alter=True,
            name="fk_appeals_guardrail_decision_id",
        ),
        nullable=True,
    )
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(
        JSON, default=list, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    donor: Mapped[Donor] = relationship(back_populates="appeals")
    campaign: Mapped[Campaign | None] = relationship(back_populates="appeals")
    template: Mapped[Template | None] = relationship(back_populates="appeals")
    experiment: Mapped[Experiment | None] = relationship(back_populates="appeals")
    experiment_variant: Mapped[ExperimentVariant | None] = relationship(
        back_populates="appeals"
    )
    guardrail_decision: Mapped[GuardrailDecision | None] = relationship(
        foreign_keys=[guardrail_decision_id]
    )
    deliveries: Mapped[list[Delivery]] = relationship(back_populates="appeal")

    __table_args__ = (
        Index("ix_appeals_donor_created", "donor_id", "created_at"),
        Index("ix_appeals_campaign_id", "campaign_id"),
    )


class Delivery(Base):
    """Delivery record tracking email lifecycle status."""

    __tablename__ = "deliveries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    appeal_id: Mapped[str] = mapped_column(
        ForeignKey("appeals.id"), index=True, nullable=False
    )
    recipient_email: Mapped[str] = mapped_column(
        String(255), index=True, nullable=False
    )
    provider_message_id: Mapped[str | None] = mapped_column(
        String(255), index=True, nullable=True
    )
    status: Mapped[DeliveryStatus] = mapped_column(
        String(20), nullable=False, default=DeliveryStatus.QUEUED
    )
    provider: Mapped[str] = mapped_column(String(20), nullable=False)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    delivered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    opened_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    clicked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    converted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    bounced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    complained_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    appeal: Mapped[Appeal] = relationship(back_populates="deliveries")

    __table_args__ = (
        Index("ix_deliveries_status_created", "status", "created_at"),
        Index("ix_deliveries_donor_email", "recipient_email", "status"),
    )


class Unsubscribe(Base):
    """Unsubscribe and suppression record for a donor."""

    __tablename__ = "unsubscribes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    donor_id: Mapped[str | None] = mapped_column(
        ForeignKey("donors.id"), index=True, nullable=True
    )
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="webhook")
    unsubscribed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    donor: Mapped[Donor | None] = relationship(back_populates="unsubscribes")

    __table_args__ = (UniqueConstraint("email", name="uq_unsubscribes_email"),)


class SuppressionEntry(Base):
    """API-manageable suppression list entry."""

    __tablename__ = "suppression_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="api")
    created_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    suppressed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (UniqueConstraint("email", name="uq_suppression_entries_email"),)


class Template(Base):
    """Versioned email template with rollback support."""

    __tablename__ = "templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenants.id"), index=True, nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    tenant: Mapped[Tenant | None] = relationship(back_populates="templates")
    versions: Mapped[list[TemplateVersion]] = relationship(
        back_populates="template", cascade="all, delete-orphan"
    )
    appeals: Mapped[list[Appeal]] = relationship(back_populates="template")

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_template_per_tenant"),
    )


class TemplateVersion(Base):
    """Immutable snapshot of a template at a specific version."""

    __tablename__ = "template_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    template_id: Mapped[str] = mapped_column(
        ForeignKey("templates.id", ondelete="CASCADE"), index=True, nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    subject_template: Mapped[str] = mapped_column(String(255), nullable=False)
    body_template: Mapped[str] = mapped_column(Text, nullable=False)
    cta_template: Mapped[str] = mapped_column(String(100), nullable=False)
    tone: Mapped[AppealTone] = mapped_column(
        String(20), nullable=False, default=AppealTone.INSPIRING
    )
    merge_tags: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    change_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    template: Mapped[Template] = relationship(back_populates="versions")

    __table_args__ = (
        UniqueConstraint(
            "template_id", "version_number", name="uq_template_version_number"
        ),
    )


class KnowledgeDocument(Base):
    """Approved knowledge document for RAG ingestion."""

    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_approved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """Semantic chunk of an approved knowledge document with pgvector embedding."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_vector: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    document: Mapped[KnowledgeDocument] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
    )


class AuditLog(Base):
    """Immutable audit trail for administrative actions."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    actor_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=True
    )
    tenant_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenants.id"), index=True, nullable=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    before_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    after_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )

    __table_args__ = (
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
        Index("ix_audit_logs_created_resource", "created_at", "resource_type"),
    )


class Experiment(Base):
    """A/B experiment lifecycle entity."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenants.id"), index=True, nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ExperimentStatus] = mapped_column(
        String(20), nullable=False, default=ExperimentStatus.DRAFT
    )
    assignment_key: Mapped[str] = mapped_column(
        String(100), nullable=False, default="donor_id"
    )
    traffic_allocation_percent: Mapped[float] = mapped_column(
        Float, nullable=False, default=100.0
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    promoted_variant_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    confidence_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    tenant: Mapped[Tenant | None] = relationship(back_populates="experiments")
    variants: Mapped[list[ExperimentVariant]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    assignments: Mapped[list[ExperimentAssignment]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    appeals: Mapped[list[Appeal]] = relationship(back_populates="experiment")


class ExperimentVariant(Base):
    """A variant within an A/B experiment."""

    __tablename__ = "experiment_variants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    template_id: Mapped[str | None] = mapped_column(
        ForeignKey("templates.id"), nullable=True
    )
    is_control: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    experiment: Mapped[Experiment] = relationship(back_populates="variants")
    template: Mapped[Template | None] = relationship()
    appeals: Mapped[list[Appeal]] = relationship(back_populates="experiment_variant")


class ExperimentAssignment(Base):
    """Sticky assignment of a subject to an experiment variant."""

    __tablename__ = "experiment_assignments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True, nullable=False
    )
    variant_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_variants.id", ondelete="CASCADE"), nullable=False
    )
    subject_key: Mapped[str] = mapped_column(String(255), nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    experiment: Mapped[Experiment] = relationship(back_populates="assignments")
    variant: Mapped[ExperimentVariant] = relationship()

    __table_args__ = (
        UniqueConstraint("experiment_id", "subject_key", name="uq_experiment_subject"),
    )


class FeatureFlag(Base):
    """Feature flag for controlled rollout."""

    __tablename__ = "feature_flags"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenants.id"), index=True, nullable=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[FeatureFlagStatus] = mapped_column(
        String(20), nullable=False, default=FeatureFlagStatus.DISABLED
    )
    rollout_percent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rules: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    tenant: Mapped[Tenant | None] = relationship(back_populates="feature_flags")

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_feature_flag_per_tenant"),
    )


class AsyncJob(Base):
    """Async bulk job tracking record."""

    __tablename__ = "async_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    job_type: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[AsyncJobStatus] = mapped_column(
        String(20), nullable=False, default=AsyncJobStatus.PENDING
    )
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    total_items: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completed_items: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=0
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (Index("ix_async_jobs_status_created", "status", "created_at"),)


class ModelVersion(Base):
    """ML model registry for versioned propensity/capacity models."""

    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version_number: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[ModelVersionStatus] = mapped_column(
        String(20), nullable=False, default=ModelVersionStatus.TRAINING
    )
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    feature_importance: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    training_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    training_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("model_name", "version_number", name="uq_model_version"),
    )


class PredictionRecord(Base):
    """Cached donor prediction scores."""

    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    donor_id: Mapped[str] = mapped_column(
        ForeignKey("donors.id", ondelete="CASCADE"), index=True, nullable=False
    )
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    propensity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    capacity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    engagement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    donor: Mapped[Donor] = relationship(back_populates="predictions")
    model_version: Mapped[ModelVersion | None] = relationship()

    __table_args__ = (
        Index("ix_predictions_donor_created", "donor_id", "predicted_at"),
    )


class DonorPreference(Base):
    """Advanced preference/suppression center record."""

    __tablename__ = "preferences"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    donor_id: Mapped[str] = mapped_column(
        ForeignKey("donors.id", ondelete="CASCADE"), index=True, nullable=False
    )
    channel: Mapped[str] = mapped_column(String(20), nullable=False, default="email")
    subscribed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    frequency_preference: Mapped[str] = mapped_column(
        String(50), nullable=False, default="standard"
    )
    topics: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    donor: Mapped[Donor] = relationship(back_populates="preferences")

    __table_args__ = (
        UniqueConstraint("donor_id", "channel", name="uq_preference_donor_channel"),
    )


class Journey(Base):
    """Multi-step donor journey with conditional logic."""

    __tablename__ = "journeys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    campaign_id: Mapped[str | None] = mapped_column(
        ForeignKey("campaigns.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    trigger_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="scheduled"
    )
    config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    campaign: Mapped[Campaign | None] = relationship(back_populates="journeys")
    steps: Mapped[list[JourneyStep]] = relationship(
        back_populates="journey", cascade="all, delete-orphan"
    )


class JourneyStep(Base):
    """A single step within a donor journey."""

    __tablename__ = "journey_steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    journey_id: Mapped[str] = mapped_column(
        ForeignKey("journeys.id", ondelete="CASCADE"), index=True, nullable=False
    )
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    step_type: Mapped[str] = mapped_column(String(50), nullable=False)
    channel: Mapped[str] = mapped_column(String(20), nullable=False, default="email")
    delay_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    condition: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    template_id: Mapped[str | None] = mapped_column(
        ForeignKey("templates.id"), nullable=True
    )
    config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    journey: Mapped[Journey] = relationship(back_populates="steps")
    template: Mapped[Template | None] = relationship()

    __table_args__ = (
        UniqueConstraint("journey_id", "step_order", name="uq_journey_step_order"),
    )


class GuardrailDecision(Base):
    """Audit record for every LLM guardrail decision."""

    __tablename__ = "guardrail_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    appeal_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "appeals.id",
            use_alter=True,
            name="fk_guardrail_decisions_appeal_id",
        ),
        nullable=True,
    )
    donor_id: Mapped[str | None] = mapped_column(ForeignKey("donors.id"), nullable=True)
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk_level: Mapped[RiskLevel] = mapped_column(
        String(20), nullable=False, default=RiskLevel.LOW
    )
    action: Mapped[GuardrailAction] = mapped_column(
        String(20), nullable=False, default=GuardrailAction.AUTO_APPROVE
    )
    stage_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    final_reasons: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    candidate_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    validation_failure_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    circuit_breaker_open: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    appeal: Mapped[Appeal | None] = relationship(foreign_keys=[appeal_id])

    __table_args__ = (
        Index("ix_guardrail_decisions_created", "created_at"),
        Index("ix_guardrail_decisions_donor", "donor_id"),
    )
