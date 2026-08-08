"""Multi-step donor journey orchestration with conditionals and branching."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.entities import (
    Appeal,
    Campaign,
    Delivery,
    DeliveryStatus,
    Donor,
    Journey,
    JourneyStep,
    Template,
)

logger = logging.getLogger(__name__)


class JourneyService:
    """Execute multi-step donor journeys with conditional logic."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def create_journey(
        self,
        name: str,
        campaign_id: str | None,
        trigger_type: str,
        config: dict,
        steps: list[dict],
    ) -> Journey:
        """Create a journey with ordered steps."""
        journey = Journey(
            name=name,
            campaign_id=campaign_id,
            trigger_type=trigger_type,
            config=config,
        )
        self.db_session.add(journey)
        self.db_session.flush()

        for index, step_payload in enumerate(steps):
            step = JourneyStep(
                journey_id=journey.id,
                step_order=index,
                step_type=step_payload.get("step_type", "send_email"),
                channel=step_payload.get("channel", "email"),
                delay_days=step_payload.get("delay_days"),
                condition=step_payload.get("condition"),
                template_id=step_payload.get("template_id"),
                config=step_payload.get("config", {}),
            )
            self.db_session.add(step)

        self.db_session.commit()
        self.db_session.refresh(journey)
        return journey

    def evaluate_condition(self, condition: dict | None, donor: Donor) -> bool:
        """Evaluate a journey step condition against a donor."""
        if condition is None:
            return True
        field_name = condition.get("field")
        operator = condition.get("operator", "eq")
        expected_value = condition.get("value")

        if field_name == "capacity_score":
            actual_value = donor.capacity_score
        elif field_name == "propensity_score":
            actual_value = donor.propensity_score
        elif field_name == "engagement_score":
            actual_value = donor.engagement_score
        elif field_name == "has_donated":
            actual_value = len(donor.donation_history) > 0
        else:
            actual_value = getattr(donor, field_name, None)

        if operator == "eq":
            return actual_value == expected_value
        if operator == "gt":
            return actual_value is not None and actual_value > expected_value
        if operator == "gte":
            return actual_value is not None and actual_value >= expected_value
        if operator == "lt":
            return actual_value is not None and actual_value < expected_value
        if operator == "lte":
            return actual_value is not None and actual_value <= expected_value
        if operator == "in":
            return actual_value in (expected_value or [])
        return False

    def get_due_steps(self, journey: Journey, donor: Donor) -> list[JourneyStep]:
        """Return journey steps that are due for a donor."""
        steps = list(self.db_session.execute(
            select(JourneyStep)
            .where(JourneyStep.journey_id == journey.id)
            .order_by(JourneyStep.step_order)
        ).scalars().all())

        due_steps: list[JourneyStep] = []
        for step in steps:
            if not self.evaluate_condition(step.condition, donor):
                continue
            if step.delay_days:
                last_delivery = self.db_session.execute(
                    select(Delivery)
                    .join(Appeal, Delivery.appeal_id == Appeal.id)
                    .where(Appeal.donor_id == donor.id)
                    .order_by(Delivery.created_at.desc())
                ).scalars().first()
                if last_delivery is not None:
                    eligible_at = last_delivery.created_at + timedelta(days=step.delay_days)
                    if datetime.now(timezone.utc) < eligible_at:
                        continue
            due_steps.append(step)
        return due_steps

    def execute_step(self, step: JourneyStep, donor: Donor) -> Appeal | None:
        """Execute a single journey step, returning the created appeal if any."""
        if step.step_type != "send_email":
            logger.info(
                "Journey step type not yet supported",
                extra={"step_type": step.step_type, "journey_id": step.journey_id},
            )
            return None

        template = self.db_session.get(Template, step.template_id) if step.template_id else None
        if template is None:
            logger.warning(
                "Journey step has no template",
                extra={"step_id": step.id, "journey_id": step.journey_id},
            )
            return None

        current_version = None
        if template.current_version_id:
            from app.models.entities import TemplateVersion

            current_version = self.db_session.get(TemplateVersion, template.current_version_id)

        if current_version is None:
            logger.warning(
                "Journey template has no current version",
                extra={"template_id": template.id},
            )
            return None

        first_name = donor.first_name or "Friend"
        subject = current_version.subject_template.replace("{{first_name}}", first_name)
        body = current_version.body_template.replace("{{first_name}}", first_name)
        cta = current_version.cta_template

        appeal = Appeal(
            donor_id=donor.id,
            campaign_id=step.journey.campaign_id if step.journey.campaign_id else None,
            template_id=template.id,
            subject=subject,
            body=body,
            cta=cta,
            tone=current_version.tone,
            capacity_score=donor.capacity_score,
            is_template_fallback=True,
        )
        self.db_session.add(appeal)
        self.db_session.commit()
        self.db_session.refresh(appeal)
        return appeal


def get_journey_service(db_session: Session) -> JourneyService:
    """Return a configured journey service instance."""
    return JourneyService(db_session)