"""Unit tests for JourneyService conditional evaluations, delay tracking, and execution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy.orm import Session

from app.models.entities import Appeal, Campaign, Delivery, DeliveryStatus, Donor, Journey, JourneyStep, Template, TemplateVersion
from app.services.journeys import JourneyService


def test_journey_condition_evaluations(db_session: Session) -> None:
    """Verify condition evaluator across all supported operators and fields."""
    service = JourneyService(db_session)
    donor = Donor(
        email="journey-donor@example.com",
        first_name="Sam",
        last_name="Taylor",
        channel="email",
        capacity_score=500.0,
        propensity_score=0.85,
        engagement_score=0.9,
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
    )
    db_session.add(donor)
    db_session.commit()

    # None condition is always True
    assert service.evaluate_condition(None, donor) is True

    # eq operator
    assert service.evaluate_condition({"field": "channel", "operator": "eq", "value": "email"}, donor) is True
    assert service.evaluate_condition({"field": "channel", "operator": "eq", "value": "sms"}, donor) is False

    # gt and gte operators on capacity_score
    assert service.evaluate_condition({"field": "capacity_score", "operator": "gt", "value": 400.0}, donor) is True
    assert service.evaluate_condition({"field": "capacity_score", "operator": "gt", "value": 600.0}, donor) is False
    assert service.evaluate_condition({"field": "capacity_score", "operator": "gte", "value": 500.0}, donor) is True

    # lt and lte operators on propensity_score
    assert service.evaluate_condition({"field": "propensity_score", "operator": "lt", "value": 0.9}, donor) is True
    assert service.evaluate_condition({"field": "propensity_score", "operator": "lte", "value": 0.85}, donor) is True

    # in operator
    assert service.evaluate_condition({"field": "channel", "operator": "in", "value": ["email", "direct_mail"]}, donor) is True
    assert service.evaluate_condition({"field": "channel", "operator": "in", "value": ["phone", "sms"]}, donor) is False

    # has_donated field
    assert service.evaluate_condition({"field": "has_donated", "operator": "eq", "value": False}, donor) is True

    # Unknown operator returns False
    assert service.evaluate_condition({"field": "channel", "operator": "unknown_op", "value": "email"}, donor) is False


def test_journey_due_steps_and_delays(db_session: Session) -> None:
    """Verify that steps with delays are only due when the required delay has elapsed."""
    service = JourneyService(db_session)
    donor = Donor(
        email="due-test@example.com",
        first_name="Due",
        last_name="Tester",
        channel="email",
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
    )
    db_session.add(donor)

    campaign = Campaign(name="Delay Campaign", description="Testing delay intervals")
    db_session.add(campaign)
    db_session.flush()

    journey = service.create_journey(
        name="Delay Test Journey",
        campaign_id=campaign.id,
        trigger_type="scheduled",
        config={},
        steps=[
            {
                "step_type": "send_email",
                "channel": "email",
                "delay_days": 0,
            },
            {
                "step_type": "send_email",
                "channel": "email",
                "delay_days": 7,
            },
        ],
    )

    # With no past deliveries, both steps are considered due
    due_steps = service.get_due_steps(journey, donor)
    assert len(due_steps) == 2

    # Now simulate a delivery that happened 2 days ago (less than 7 days)
    appeal = Appeal(
        donor=donor,
        campaign=campaign,
        subject="Past Appeal",
        body="Past Body",
        cta="Donate",
        tone="inspiring",
    )
    db_session.add(appeal)
    db_session.flush()

    recent_delivery = Delivery(
        appeal=appeal,
        recipient_email=donor.email,
        provider="email",
        status=DeliveryStatus.DELIVERED,
        created_at=datetime.now(timezone.utc) - timedelta(days=2),
    )
    db_session.add(recent_delivery)
    db_session.commit()

    # Step with delay_days=7 should now be excluded because only 2 days have elapsed
    due_steps_after_delivery = service.get_due_steps(journey, donor)
    assert len(due_steps_after_delivery) == 1
    assert due_steps_after_delivery[0].delay_days == 0


def test_journey_execute_step(db_session: Session) -> None:
    """Verify executing a journey step creates and persists an appeal."""
    service = JourneyService(db_session)
    donor = Donor(
        email="exec-test@example.com",
        first_name="Execution",
        last_name="Tester",
        channel="email",
        capacity_score=250.0,
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
    )
    db_session.add(donor)

    campaign = Campaign(name="Exec Campaign", description="Step execution test")
    db_session.add(campaign)
    db_session.flush()

    template = Template(
        name="Journey Welcome Email",
        description="Onboarding template",
    )
    db_session.add(template)
    db_session.flush()

    version = TemplateVersion(
        template_id=template.id,
        version_number=1,
        subject_template="Welcome {{first_name}}!",
        body_template="Hello {{first_name}}, thank you for joining.",
        cta_template="Learn More",
        tone="inspiring",
    )
    db_session.add(version)
    db_session.flush()

    template.current_version_id = version.id
    db_session.commit()

    journey = service.create_journey(
        name="Exec Journey",
        campaign_id=campaign.id,
        trigger_type="scheduled",
        config={},
        steps=[
            {
                "step_type": "send_email",
                "template_id": template.id,
            }
        ],
    )

    steps = service.get_due_steps(journey, donor)
    assert len(steps) == 1

    appeal = service.execute_step(steps[0], donor)
    assert appeal is not None
    assert appeal.donor_id == donor.id
    assert appeal.subject == "Welcome Execution!"
    assert appeal.body == "Hello Execution, thank you for joining."
    assert appeal.cta == "Learn More"
    assert appeal.is_template_fallback is True
