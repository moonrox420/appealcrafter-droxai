"""Initial normalized schema.

Revision ID: 0001
Revises:
Create Date: 2026-08-08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all initial tables."""
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "campaigns",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "donors",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("external_id", sa.String(255), nullable=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("first_name", sa.String(100), nullable=True),
        sa.Column("last_name", sa.String(100), nullable=True),
        sa.Column("interests", sa.Text(), nullable=True),
        sa.Column("channel", sa.String(20), nullable=False, server_default="email"),
        sa.Column("capacity_score", sa.Float(), nullable=True),
        sa.Column("consent_given_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consent_source", sa.String(100), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_donors_email", "donors", ["email"])
    op.create_index("ix_donors_external_id", "donors", ["external_id"])
    op.create_index("ix_donors_email_active", "donors", ["email", "is_deleted"])

    op.create_table(
        "donation_history",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("donor_id", sa.String(36), sa.ForeignKey("donors.id", ondelete="CASCADE"), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("donated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("campaign_id", sa.String(36), sa.ForeignKey("campaigns.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_donation_history_donor_id", "donation_history", ["donor_id"])
    op.create_index("ix_donation_history_donor_date", "donation_history", ["donor_id", "donated_at"])

    op.create_table(
        "appeals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("donor_id", sa.String(36), sa.ForeignKey("donors.id"), nullable=False),
        sa.Column("campaign_id", sa.String(36), sa.ForeignKey("campaigns.id"), nullable=True),
        sa.Column("subject", sa.String(255), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("cta", sa.String(100), nullable=False),
        sa.Column("tone", sa.String(20), nullable=False, server_default="inspiring"),
        sa.Column("capacity_score", sa.Float(), nullable=True),
        sa.Column("is_template_fallback", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_appeals_donor_id", "appeals", ["donor_id"])
    op.create_index("ix_appeals_donor_created", "appeals", ["donor_id", "created_at"])

    op.create_table(
        "deliveries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("appeal_id", sa.String(36), sa.ForeignKey("appeals.id"), nullable=False),
        sa.Column("recipient_email", sa.String(255), nullable=False),
        sa.Column("provider_message_id", sa.String(255), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("provider", sa.String(20), nullable=False),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("clicked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("bounced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("complained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_deliveries_appeal_id", "deliveries", ["appeal_id"])
    op.create_index("ix_deliveries_recipient_email", "deliveries", ["recipient_email"])
    op.create_index("ix_deliveries_provider_message_id", "deliveries", ["provider_message_id"])
    op.create_index("ix_deliveries_status_created", "deliveries", ["status", "created_at"])

    op.create_table(
        "unsubscribes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("donor_id", sa.String(36), sa.ForeignKey("donors.id"), nullable=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default="webhook"),
        sa.Column("unsubscribed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("email", name="uq_unsubscribes_email"),
    )
    op.create_index("ix_unsubscribes_donor_id", "unsubscribes", ["donor_id"])
    op.create_index("ix_unsubscribes_email", "unsubscribes", ["email"])

    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_url", sa.String(500), nullable=True),
        sa.Column("is_approved", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding_vector", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])


def downgrade() -> None:
    """Drop all tables in reverse dependency order."""
    op.drop_table("document_chunks")
    op.drop_table("knowledge_documents")
    op.drop_table("unsubscribes")
    op.drop_table("deliveries")
    op.drop_table("appeals")
    op.drop_table("donation_history")
    op.drop_table("donors")
    op.drop_table("campaigns")
    op.drop_table("users")