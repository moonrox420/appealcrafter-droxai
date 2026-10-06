"""Phase 2 tests: guardrails, RAG, experiments, compliance, ML, rate limiting."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.session import Base
from app.models.entities import (
    AppealTone,
    Donor,
    Experiment,
    ExperimentStatus,
    ExperimentVariant,
    GuardrailAction,
    KnowledgeDocument,
    RiskLevel,
)
from app.schemas.donor import DonationCreate, DonorCreate
from app.services.compliance import ComplianceService
from app.services.donor import DonorService
from app.services.experiments import ExperimentService
from app.services.guardrails import AppealCandidate, GuardrailPipeline
from app.services.ml import PredictionService
from app.services.rag import RagPipeline, SemanticChunker
from app.services.suppression import SuppressionService
from app.services.template import TemplateManagementService




def _create_test_donor(db_session: Session) -> Donor:
    """Create a test donor with donation history."""
    donor_service = DonorService(db_session)
    donor_payload = DonorCreate(
        email="phase2@example.com",
        first_name="Phase",
        last_name="Two",
        channel="email",
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
        donations=[
            DonationCreate(amount=100.0, donated_at=datetime.now(timezone.utc)),
            DonationCreate(amount=250.0, donated_at=datetime.now(timezone.utc)),
        ],
    )
    return donor_service.create_donor(donor_payload)


def test_guardrail_pipeline_approves_valid_candidate(db_session: Session) -> None:
    """Verify the guardrail pipeline approves a valid appeal candidate."""
    pipeline = GuardrailPipeline(db_session)
    candidate = AppealCandidate(
        subject="Your support can change lives this month",
        body=(
            "Dear friend, thanks to donors like you we reached 12,000 families last year. "
            "Will you help us continue this work? "
            "You are receiving this email because you have supported our cause. "
            "To unsubscribe, click the unsubscribe link. 123 Test St"
        ),
        cta="Donate now",
        tone=AppealTone.INSPIRING,
        donor_id="donor-1",
        capacity=250.0,
        retrieved_chunk_ids=["IR-2025-14"],
        retrieved_chunks=["Approved impact text about reaching families."],
    )
    decision = pipeline.validate(candidate)
    assert decision.approved is True
    assert decision.action == GuardrailAction.AUTO_APPROVE
    assert decision.risk_level == RiskLevel.LOW


def test_guardrail_pipeline_rejects_forbidden_pattern(db_session: Session) -> None:
    """Verify the guardrail pipeline rejects forbidden patterns."""
    pipeline = GuardrailPipeline(db_session)
    candidate = AppealCandidate(
        subject="Guaranteed results for your donation",
        body=(
            "Act now or children will die. This is your last chance. "
            "100% of your donation goes to the cause. "
            "You are receiving this email because you have supported our cause. "
            "To unsubscribe, click the unsubscribe link. 123 Test St"
        ),
        cta="Donate now",
        tone=AppealTone.URGENT,
        donor_id="donor-2",
        capacity=100.0,
    )
    decision = pipeline.validate(candidate)
    assert decision.approved is False
    assert decision.action == GuardrailAction.REJECT
    assert decision.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)


def test_guardrail_pipeline_routes_high_value_donor_to_human_review(
    db_session: Session,
) -> None:
    """Verify high-value donors always route to human review."""
    pipeline = GuardrailPipeline(db_session, high_value_threshold=5000.0)
    candidate = AppealCandidate(
        subject="Your support can change lives this month",
        body=(
            "Dear friend, thanks to donors like you we reached 12,000 families last year. "
            "Will you help us continue this work? "
            "You are receiving this email because you have supported our cause. "
            "To unsubscribe, click the unsubscribe link. 123 Test St"
        ),
        cta="Donate now",
        tone=AppealTone.INSPIRING,
        donor_id="donor-3",
        capacity=10000.0,
    )
    decision = pipeline.validate(candidate)
    assert decision.approved is False
    assert decision.action == GuardrailAction.HUMAN_REVIEW


def test_guardrail_pipeline_persists_decision(db_session: Session) -> None:
    """Verify guardrail decisions are persisted for audit."""
    donor = _create_test_donor(db_session)
    pipeline = GuardrailPipeline(db_session)
    candidate = AppealCandidate(
        subject="Your support can change lives this month",
        body=(
            "Dear friend, thanks to donors like you we reached 12,000 families last year. "
            "Will you help us continue this work? "
            "You are receiving this email because you have supported our cause. "
            "To unsubscribe, click the unsubscribe link. 123 Test St"
        ),
        cta="Donate now",
        tone=AppealTone.INSPIRING,
        donor_id=donor.id,
        capacity=100.0,
        retrieved_chunk_ids=["IR-2025-14"],
        retrieved_chunks=["Approved impact text about reaching families."],
    )
    decision = pipeline.validate(candidate)
    persisted = pipeline.persist_decision(candidate, decision)
    assert persisted.id is not None
    assert persisted.approved is True
    assert persisted.risk_level == RiskLevel.LOW


def test_semantic_chunker_splits_large_document() -> None:
    """Verify the semantic chunker splits large documents with overlap."""
    chunker = SemanticChunker()
    content = " ".join(["word"] * 500)
    chunks = chunker.chunk_document(content)
    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 200 for chunk in chunks)


def test_rag_pipeline_rejects_unapproved_document(db_session: Session) -> None:
    """Verify RAG ingestion rejects unapproved documents."""
    document = KnowledgeDocument(
        title="Unapproved",
        content="This document is not approved for RAG.",
        is_approved=False,
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    rag_pipeline = RagPipeline(db_session)
    with pytest.raises(ValueError):
        rag_pipeline.ingest_document(document.id)


def test_experiment_sticky_assignment(db_session: Session) -> None:
    """Verify experiment sticky assignment returns the same variant."""
    experiment = Experiment(name="Test Experiment", status=ExperimentStatus.DRAFT)
    db_session.add(experiment)
    db_session.flush()

    variant_a = ExperimentVariant(
        experiment_id=experiment.id, name="control", weight=1.0, is_control=True
    )
    variant_b = ExperimentVariant(
        experiment_id=experiment.id, name="variant", weight=1.0
    )
    db_session.add_all([variant_a, variant_b])
    db_session.commit()

    service = ExperimentService(db_session)
    first_assignment = service.assign_variant(experiment, "donor-100")
    second_assignment = service.assign_variant(experiment, "donor-100")
    assert first_assignment.id == second_assignment.id


def test_experiment_statistical_results(db_session: Session) -> None:
    """Verify experiment statistical results are computed."""
    experiment = Experiment(name="Stats Experiment", status=ExperimentStatus.RUNNING)
    db_session.add(experiment)
    db_session.flush()

    variant_a = ExperimentVariant(
        experiment_id=experiment.id, name="control", weight=1.0, is_control=True
    )
    variant_b = ExperimentVariant(
        experiment_id=experiment.id, name="variant", weight=1.0
    )
    db_session.add_all([variant_a, variant_b])
    db_session.commit()

    service = ExperimentService(db_session)
    results = service.compute_statistical_results(experiment)
    assert "variants" in results
    assert len(results["variants"]) == 2


def test_compliance_export_donor_data(db_session: Session) -> None:
    """Verify GDPR/CCPA export returns donor data."""
    donor = _create_test_donor(db_session)
    service = ComplianceService(db_session)
    exported = service.export_donor_data(donor.id)
    assert exported["donor"]["email"] == "phase2@example.com"
    assert len(exported["donation_history"]) == 2


def test_compliance_anonymize_donor(db_session: Session) -> None:
    """Verify GDPR/CCPA anonymization removes PII."""
    donor = _create_test_donor(db_session)
    service = ComplianceService(db_session)
    anonymized = service.anonymize_donor(donor.id)
    assert anonymized.first_name == "[ANONYMIZED]"
    assert "anonymized" in anonymized.email


def test_compliance_delete_donor(db_session: Session) -> None:
    """Verify GDPR/CCPA deletion removes donor records."""
    donor = _create_test_donor(db_session)
    service = ComplianceService(db_session)
    service.delete_donor(donor.id)
    assert db_session.get(Donor, donor.id) is None


def test_ml_prediction_generates_scores(db_session: Session) -> None:
    """Verify ML prediction generates propensity/capacity scores."""
    donor = _create_test_donor(db_session)
    service = PredictionService(db_session)
    prediction = service.predict_for_donor(donor)
    assert prediction.propensity_score is not None
    assert prediction.capacity_score is not None
    assert prediction.engagement_score is not None


def test_ml_rfm_features_computed(db_session: Session) -> None:
    """Verify RFM features are computed for a donor."""
    donor = _create_test_donor(db_session)
    service = PredictionService(db_session)
    features = service.feature_engine.compute_rfm_features(donor)
    assert features["rfm_frequency_count"] == 2
    assert features["rfm_monetary_value"] == 350.0
    assert features["rfm_recency_days"] >= 0


def test_suppression_api_entry(db_session: Session) -> None:
    """Verify API-managed suppression entries block sends."""
    donor = _create_test_donor(db_session)
    service = SuppressionService(db_session)
    service.add_suppression_entry(email=donor.email, reason="test")
    allowed, reason = service.can_send_to_donor(donor)
    assert allowed is False
    assert reason == "suppressed"


def test_template_versioning_and_rollback(db_session: Session) -> None:
    """Verify template versioning and rollback."""
    service = TemplateManagementService(db_session)
    template = service.create_template(
        name="Test Template",
        subject_template="Hello {{first_name}}",
        body_template="Dear {{first_name}}, please support us.",
        cta_template="Donate",
        tone=AppealTone.INSPIRING,
    )
    assert template.current_version_id is not None

    version = service.create_new_version(
        template_id=template.id,
        subject_template="New subject {{first_name}}",
        body_template="New body {{first_name}}",
        cta_template="Donate now",
        tone=AppealTone.INSPIRING,
        change_note="Updated copy",
    )
    assert version.version_number == 2

    rolled_back = service.rollback_to_version(template.id, 1)
    assert rolled_back.current_version_id is not None
    assert rolled_back.current_version_id != version.id


def test_rate_limiter_returns_429() -> None:
    """Verify the rate limiter returns 429 with Retry-After."""
    from app.core.rate_limit import InMemoryRateLimiter

    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
    assert limiter.check_request("client-1")[0] is True
    assert limiter.check_request("client-1")[0] is True
    allowed, retry_after = limiter.check_request("client-1")
    assert allowed is False
    assert retry_after >= 1
