"""Appeal generation and persistence service with RAG + guardrails."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.entities import Appeal, AppealTone, Campaign, Donor
from app.services.guardrails import AppealCandidate, GuardrailPipeline
from app.services.llm_generation import LlmAppealGenerationService
from app.services.ml import PredictionService
from app.services.rag import RagPipeline
from app.services.suppression import SuppressionService
from app.services.template import TemplateService

logger = logging.getLogger(__name__)


class AppealService:
    """Generate and persist appeal records for eligible donors."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.template_service = TemplateService(db_session)
        self.suppression_service = SuppressionService(db_session)
        self.llm_generation_service = LlmAppealGenerationService()
        self.rag_pipeline = RagPipeline(db_session)
        self.guardrail_pipeline = GuardrailPipeline(db_session)
        self.prediction_service = PredictionService(db_session)

    def generate_appeals(
        self,
        tone: AppealTone | str,
        campaign_id: str | None = None,
        limit: int = 100,
    ) -> list[Appeal]:
        """Generate appeals for eligible donors and persist them."""
        resolved_tone = AppealTone(tone) if isinstance(tone, str) else tone
        campaign = None
        if campaign_id is not None:
            campaign = self.db_session.get(Campaign, campaign_id)
            if campaign is None:
                raise ValueError(f"Campaign {campaign_id} not found.")

        statement = (
            select(Donor)
            .where(Donor.is_deleted.is_(False))
            .order_by(Donor.created_at)
            .limit(limit)
        )
        donors = list(self.db_session.execute(statement).scalars().all())

        generated_appeals: list[Appeal] = []
        for donor in donors:
            allowed, reason = self.suppression_service.can_send_to_donor(donor)
            if not allowed:
                logger.info(
                    "Donor skipped for appeal generation",
                    extra={"donor_id": donor.id, "reason": reason},
                )
                continue

            self.prediction_service.predict_for_donor(donor)

            generation_result = self.llm_generation_service.generate(
                donor=donor,
                tone=resolved_tone,
                rag_pipeline=self.rag_pipeline,
            )

            appeal = Appeal(
                donor_id=donor.id,
                campaign_id=campaign.id if campaign else None,
                subject=generation_result.subject,
                body=generation_result.body,
                cta=generation_result.cta,
                tone=generation_result.tone,
                capacity_score=donor.capacity_score,
                is_template_fallback=generation_result.is_template_fallback,
                retrieved_chunk_ids=generation_result.retrieved_chunk_ids,
            )
            self.db_session.add(appeal)
            self.db_session.flush()

            candidate = AppealCandidate(
                subject=appeal.subject,
                body=appeal.body,
                cta=appeal.cta,
                tone=appeal.tone,
                donor_id=donor.id,
                capacity=donor.capacity_score,
                retrieved_chunk_ids=appeal.retrieved_chunk_ids,
                retrieved_chunks=[],
            )
            decision = self.guardrail_pipeline.validate(candidate)
            guardrail_decision = self.guardrail_pipeline.persist_decision(
                candidate, decision, appeal
            )
            appeal.guardrail_decision_id = guardrail_decision.id

            if not decision.approved:
                logger.info(
                    "Appeal not approved by guardrails; using template fallback",
                    extra={
                        "donor_id": donor.id,
                        "action": decision.action.value,
                        "risk_score": decision.risk_score,
                    },
                )
                template_content = self.template_service.generate(donor, tone)
                appeal.subject = template_content.subject
                appeal.body = template_content.body
                appeal.cta = template_content.cta
                appeal.is_template_fallback = True

            generated_appeals.append(appeal)

        self.db_session.commit()
        for appeal in generated_appeals:
            self.db_session.refresh(appeal)
        return generated_appeals
