"""Phase 2 schema expansion.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add Phase 2 tables and columns."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tenants",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.add_column(
        "users",
        sa.Column(
            "tenant_id", sa.String(36), sa.ForeignKey("tenants.id"), nullable=True
        ),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])

    op.add_column(
        "donors",
        sa.Column(
            "tenant_id", sa.String(36), sa.ForeignKey("tenants.id"), nullable=True
        ),
    )
    op.add_column("donors", sa.Column("propensity_score", sa.Float(), nullable=True))
    op.add_column("donors", sa.Column("engagement_score", sa.Float(), nullable=True))
    op.add_column("donors", sa.Column("rfm_recency_days", sa.Float(), nullable=True))
    op.add_column(
        "donors", sa.Column("rfm_frequency_count", sa.Integer(), nullable=True)
    )
    op.add_column("donors", sa.Column("rfm_monetary_value", sa.Float(), nullable=True))
    op.add_column(
        "donors",
        sa.Column(
            "email_encrypted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index("ix_donors_tenant_id", "donors", ["tenant_id"])
    op.create_index("ix_donors_tenant_email", "donors", ["tenant_id", "email"])

    op.add_column(
        "campaigns",
        sa.Column(
            "tenant_id", sa.String(36), sa.ForeignKey("tenants.id"), nullable=True
        ),
    )
    op.create_index("ix_campaigns_tenant_id", "campaigns", ["tenant_id"])

    op.create_table(
        "suppression_entries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default="api"),
        sa.Column(
            "created_by_user_id",
            sa.String(36),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("suppressed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("email", name="uq_suppression_entries_email"),
    )
    op.create_index("ix_suppression_entries_email", "suppression_entries", ["email"])

    op.create_table(
        "templates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id", sa.String(36), sa.ForeignKey("tenants.id"), nullable=True
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column("current_version_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("tenant_id", "name", name="uq_template_per_tenant"),
    )
    op.create_index("ix_templates_tenant_id", "templates", ["tenant_id"])

    op.create_table(
        "template_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "template_id",
            sa.String(36),
            sa.ForeignKey("templates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("subject_template", sa.String(255), nullable=False),
        sa.Column("body_template", sa.Text(), nullable=False),
        sa.Column("cta_template", sa.String(100), nullable=False),
        sa.Column("tone", sa.String(20), nullable=False, server_default="inspiring"),
        sa.Column("merge_tags", sa.JSON(), nullable=False),
        sa.Column("change_note", sa.Text(), nullable=True),
        sa.Column(
            "created_by_user_id",
            sa.String(36),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "template_id", "version_number", name="uq_template_version_number"
        ),
    )
    op.create_index(
        "ix_template_versions_template_id", "template_versions", ["template_id"]
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "actor_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True
        ),
        sa.Column(
            "tenant_id", sa.String(36), sa.ForeignKey("tenants.id"), nullable=True
        ),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(100), nullable=False),
        sa.Column("resource_id", sa.String(36), nullable=True),
        sa.Column("before_state", sa.JSON(), nullable=True),
        sa.Column("after_state", sa.JSON(), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("trace_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_logs_actor_user_id", "audit_logs", ["actor_user_id"])
    op.create_index("ix_audit_logs_tenant_id", "audit_logs", ["tenant_id"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])
    op.create_index(
        "ix_audit_logs_resource", "audit_logs", ["resource_type", "resource_id"]
    )
    op.create_index(
        "ix_audit_logs_created_resource", "audit_logs", ["created_at", "resource_type"]
    )

    op.create_table(
        "experiments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id", sa.String(36), sa.ForeignKey("tenants.id"), nullable=True
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("hypothesis", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="draft"),
        sa.Column(
            "assignment_key", sa.String(100), nullable=False, server_default="donor_id"
        ),
        sa.Column(
            "traffic_allocation_percent",
            sa.Float(),
            nullable=False,
            server_default="100",
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("promoted_variant_id", sa.String(36), nullable=True),
        sa.Column("confidence_level", sa.Float(), nullable=True),
        sa.Column("p_value", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_experiments_tenant_id", "experiments", ["tenant_id"])

    op.create_table(
        "experiment_variants",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "experiment_id",
            sa.String(36),
            sa.ForeignKey("experiments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1"),
        sa.Column(
            "template_id", sa.String(36), sa.ForeignKey("templates.id"), nullable=True
        ),
        sa.Column(
            "is_control", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_experiment_variants_experiment_id", "experiment_variants", ["experiment_id"]
    )

    op.create_table(
        "experiment_assignments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "experiment_id",
            sa.String(36),
            sa.ForeignKey("experiments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "variant_id",
            sa.String(36),
            sa.ForeignKey("experiment_variants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("subject_key", sa.String(255), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "experiment_id", "subject_key", name="uq_experiment_subject"
        ),
    )
    op.create_index(
        "ix_experiment_assignments_experiment_id",
        "experiment_assignments",
        ["experiment_id"],
    )

    op.create_table(
        "feature_flags",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id", sa.String(36), sa.ForeignKey("tenants.id"), nullable=True
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="disabled"),
        sa.Column("rollout_percent", sa.Float(), nullable=False, server_default="0"),
        sa.Column("rules", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("tenant_id", "name", name="uq_feature_flag_per_tenant"),
    )
    op.create_index("ix_feature_flags_tenant_id", "feature_flags", ["tenant_id"])

    op.create_table(
        "async_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_type", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("result_summary", sa.JSON(), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column(
            "created_by_user_id",
            sa.String(36),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("total_items", sa.Integer(), nullable=True),
        sa.Column("completed_items", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_async_jobs_status_created", "async_jobs", ["status", "created_at"]
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("version_number", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="training"),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("feature_importance", sa.JSON(), nullable=True),
        sa.Column("training_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("training_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("model_name", "version_number", name="uq_model_version"),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "donor_id",
            sa.String(36),
            sa.ForeignKey("donors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "model_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id"),
            nullable=True,
        ),
        sa.Column("propensity_score", sa.Float(), nullable=True),
        sa.Column("capacity_score", sa.Float(), nullable=True),
        sa.Column("engagement_score", sa.Float(), nullable=True),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "fallback_used",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index("ix_predictions_donor_id", "predictions", ["donor_id"])
    op.create_index(
        "ix_predictions_donor_created", "predictions", ["donor_id", "predicted_at"]
    )

    op.create_table(
        "preferences",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "donor_id",
            sa.String(36),
            sa.ForeignKey("donors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("channel", sa.String(20), nullable=False, server_default="email"),
        sa.Column(
            "subscribed", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column(
            "frequency_preference",
            sa.String(50),
            nullable=False,
            server_default="standard",
        ),
        sa.Column("topics", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("donor_id", "channel", name="uq_preference_donor_channel"),
    )
    op.create_index("ix_preferences_donor_id", "preferences", ["donor_id"])

    op.create_table(
        "journeys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "campaign_id", sa.String(36), sa.ForeignKey("campaigns.id"), nullable=True
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column(
            "trigger_type", sa.String(50), nullable=False, server_default="scheduled"
        ),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "journey_steps",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "journey_id",
            sa.String(36),
            sa.ForeignKey("journeys.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step_order", sa.Integer(), nullable=False),
        sa.Column("step_type", sa.String(50), nullable=False),
        sa.Column("channel", sa.String(20), nullable=False, server_default="email"),
        sa.Column("delay_days", sa.Integer(), nullable=True),
        sa.Column("condition", sa.JSON(), nullable=True),
        sa.Column(
            "template_id", sa.String(36), sa.ForeignKey("templates.id"), nullable=True
        ),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("journey_id", "step_order", name="uq_journey_step_order"),
    )
    op.create_index("ix_journey_steps_journey_id", "journey_steps", ["journey_id"])

    op.create_table(
        "guardrail_decisions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "appeal_id", sa.String(36), sa.ForeignKey("appeals.id"), nullable=True
        ),
        sa.Column("donor_id", sa.String(36), sa.ForeignKey("donors.id"), nullable=True),
        sa.Column("approved", sa.Boolean(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("risk_level", sa.String(20), nullable=False, server_default="low"),
        sa.Column(
            "action", sa.String(20), nullable=False, server_default="auto_approve"
        ),
        sa.Column("stage_results", sa.JSON(), nullable=True),
        sa.Column("final_reasons", sa.JSON(), nullable=False),
        sa.Column("candidate_snapshot", sa.JSON(), nullable=True),
        sa.Column("validation_failure_rate", sa.Float(), nullable=True),
        sa.Column(
            "circuit_breaker_open",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_guardrail_decisions_created_at", "guardrail_decisions", ["created_at"]
    )
    op.create_index(
        "ix_guardrail_decisions_donor_id", "guardrail_decisions", ["donor_id"]
    )

    op.add_column(
        "appeals",
        sa.Column(
            "template_id", sa.String(36), sa.ForeignKey("templates.id"), nullable=True
        ),
    )
    op.add_column(
        "appeals",
        sa.Column(
            "experiment_id",
            sa.String(36),
            sa.ForeignKey("experiments.id"),
            nullable=True,
        ),
    )
    op.add_column(
        "appeals",
        sa.Column(
            "experiment_variant_id",
            sa.String(36),
            sa.ForeignKey("experiment_variants.id"),
            nullable=True,
        ),
    )
    op.add_column(
        "appeals",
        sa.Column(
            "guardrail_decision_id",
            sa.String(36),
            sa.ForeignKey("guardrail_decisions.id"),
            nullable=True,
        ),
    )
    op.add_column(
        "appeals",
        sa.Column(
            "retrieved_chunk_ids", sa.JSON(), nullable=False, server_default="[]"
        ),
    )
    op.create_index("ix_appeals_campaign_id", "appeals", ["campaign_id"])

    op.add_column(
        "deliveries",
        sa.Column("converted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_deliveries_donor_email", "deliveries", ["recipient_email", "status"]
    )

    op.alter_column(
        "document_chunks",
        "embedding_vector",
        type_=Vector(384),
        existing_type=sa.Text(),
        nullable=True,
        postgresql_using="embedding_vector::vector(384)",
    )


def downgrade() -> None:
    """Drop Phase 2 tables and columns in reverse dependency order."""
    op.drop_index("ix_deliveries_donor_email", table_name="deliveries")
    op.drop_column("deliveries", "converted_at")

    op.drop_index("ix_appeals_campaign_id", table_name="appeals")
    op.drop_column("appeals", "retrieved_chunk_ids")
    op.drop_column("appeals", "guardrail_decision_id")
    op.drop_column("appeals", "experiment_variant_id")
    op.drop_column("appeals", "experiment_id")
    op.drop_column("appeals", "template_id")

    op.drop_index("ix_guardrail_decisions_donor_id", table_name="guardrail_decisions")
    op.drop_index("ix_guardrail_decisions_created_at", table_name="guardrail_decisions")
    op.drop_table("guardrail_decisions")

    op.drop_index("ix_journey_steps_journey_id", table_name="journey_steps")
    op.drop_table("journey_steps")
    op.drop_table("journeys")

    op.drop_index("ix_preferences_donor_id", table_name="preferences")
    op.drop_table("preferences")

    op.drop_index("ix_predictions_donor_created", table_name="predictions")
    op.drop_index("ix_predictions_donor_id", table_name="predictions")
    op.drop_table("predictions")
    op.drop_table("model_versions")

    op.drop_index("ix_async_jobs_status_created", table_name="async_jobs")
    op.drop_table("async_jobs")

    op.drop_index("ix_feature_flags_tenant_id", table_name="feature_flags")
    op.drop_table("feature_flags")

    op.drop_index(
        "ix_experiment_assignments_experiment_id", table_name="experiment_assignments"
    )
    op.drop_table("experiment_assignments")
    op.drop_index(
        "ix_experiment_variants_experiment_id", table_name="experiment_variants"
    )
    op.drop_table("experiment_variants")
    op.drop_index("ix_experiments_tenant_id", table_name="experiments")
    op.drop_table("experiments")

    op.drop_index("ix_audit_logs_created_resource", table_name="audit_logs")
    op.drop_index("ix_audit_logs_resource", table_name="audit_logs")
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_tenant_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_actor_user_id", table_name="audit_logs")
    op.drop_table("audit_logs")

    op.drop_index("ix_template_versions_template_id", table_name="template_versions")
    op.drop_table("template_versions")
    op.drop_index("ix_templates_tenant_id", table_name="templates")
    op.drop_table("templates")

    op.drop_index("ix_suppression_entries_email", table_name="suppression_entries")
    op.drop_table("suppression_entries")

    op.drop_index("ix_campaigns_tenant_id", table_name="campaigns")
    op.drop_column("campaigns", "tenant_id")

    op.drop_index("ix_donors_tenant_email", table_name="donors")
    op.drop_index("ix_donors_tenant_id", table_name="donors")
    op.drop_column("donors", "email_encrypted")
    op.drop_column("donors", "rfm_monetary_value")
    op.drop_column("donors", "rfm_frequency_count")
    op.drop_column("donors", "rfm_recency_days")
    op.drop_column("donors", "engagement_score")
    op.drop_column("donors", "propensity_score")
    op.drop_column("donors", "tenant_id")

    op.drop_index("ix_users_tenant_id", table_name="users")
    op.drop_column("users", "tenant_id")

    op.drop_table("tenants")
