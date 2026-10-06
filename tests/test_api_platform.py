"""Tests for /platform API routers (knowledge, templates, experiments, journeys, feature-flags)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from app.models.entities import Campaign


@pytest.mark.anyio
async def test_knowledge_api_lifecycle(
    async_client: AsyncClient,
    admin_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test full knowledge document lifecycle: create, list, get, ingest."""
    create_resp = await async_client.post(
        "/knowledge",
        json={
            "title": "Clean Water Initiative Q1 Report",
            "content": "Our clean water project delivered over 50,000 liters of potable water to rural regions in Guatemala.",
            "source_url": "https://example.org/reports/guatemala-2026.pdf",
            "is_approved": True,
        },
        headers=admin_headers,
    )
    assert create_resp.status_code == 201
    doc_data = create_resp.json()
    assert doc_data["title"] == "Clean Water Initiative Q1 Report"
    doc_id = doc_data["id"]

    list_resp = await async_client.get("/knowledge", headers=admin_headers)
    assert list_resp.status_code == 200
    assert any(d["id"] == doc_id for d in list_resp.json())

    get_resp = await async_client.get(f"/knowledge/{doc_id}", headers=admin_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["content"].startswith("Our clean water")

    ingest_resp = await async_client.post(
        f"/knowledge/{doc_id}/ingest", headers=admin_headers
    )
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["chunk_count"] >= 1

    not_found_resp = await async_client.get(
        "/knowledge/00000000-0000-0000-0000-000000000000", headers=admin_headers
    )
    assert not_found_resp.status_code == 404


@pytest.mark.anyio
async def test_templates_api_lifecycle(
    async_client: AsyncClient,
    admin_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test template creation, versioning, rollback, and retrieval."""
    create_resp = await async_client.post(
        "/templates",
        json={
            "name": "Standard Urgent Appeal",
            "description": "Used when emergency response is triggered.",
            "subject_template": "Urgent update for {{first_name}}",
            "body_template": "Dear {{first_name}}, urgent support is needed right now.",
            "cta_template": "Donate Now",
            "tone": "urgent",
        },
        headers=admin_headers,
    )
    assert create_resp.status_code == 201
    tpl_data = create_resp.json()
    tpl_id = tpl_data["id"]

    list_resp = await async_client.get("/templates", headers=admin_headers)
    assert list_resp.status_code == 200
    assert any(t["id"] == tpl_id for t in list_resp.json())

    new_ver_resp = await async_client.post(
        f"/templates/{tpl_id}/versions",
        json={
            "subject_template": "Immediate help needed, {{first_name}}!",
            "body_template": "Dear {{first_name}}, families need your urgent help today.",
            "cta_template": "Support Families",
            "tone": "urgent",
            "change_note": "Sharpened CTA and subject",
        },
        headers=admin_headers,
    )
    assert new_ver_resp.status_code == 201
    assert new_ver_resp.json()["version_number"] == 2

    versions_resp = await async_client.get(
        f"/templates/{tpl_id}/versions", headers=admin_headers
    )
    assert versions_resp.status_code == 200
    assert len(versions_resp.json()) == 2

    rollback_resp = await async_client.post(
        f"/templates/{tpl_id}/rollback/1", headers=admin_headers
    )
    assert rollback_resp.status_code == 200


@pytest.mark.anyio
async def test_experiments_api_lifecycle(
    async_client: AsyncClient,
    admin_headers: dict[str, str],
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test A/B experiment creation, start, results, and promotion."""
    create_resp = await async_client.post(
        "/experiments",
        json={
            "name": "Subject Line Optimization 2026",
            "description": "Testing urgent vs hopeful subject tone.",
            "hypothesis": "Hopeful subjects yield 15% higher open rates.",
            "assignment_key": "donor_id",
            "variants": [
                {"name": "Control-Urgent", "weight": 1.0},
                {"name": "Treatment-Hopeful", "weight": 1.0},
            ],
        },
        headers=operator_headers,
    )
    assert create_resp.status_code == 201
    exp_data = create_resp.json()
    exp_id = exp_data["id"]

    list_resp = await async_client.get("/experiments", headers=operator_headers)
    assert list_resp.status_code == 200
    assert any(e["id"] == exp_id for e in list_resp.json())

    # Starting experiment requires admin role
    forbidden_resp = await async_client.post(
        f"/experiments/{exp_id}/start", headers=operator_headers
    )
    assert forbidden_resp.status_code == 403

    start_resp = await async_client.post(
        f"/experiments/{exp_id}/start", headers=admin_headers
    )
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "running"

    results_resp = await async_client.get(
        f"/experiments/{exp_id}/results", headers=operator_headers
    )
    assert results_resp.status_code == 200
    assert "variants" in results_resp.json()


@pytest.mark.anyio
async def test_journeys_api_lifecycle(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test creating and listing multi-step donor journeys."""
    campaign = Campaign(name="Welcome Campaign", description="Onboarding donor journey")
    db_session.add(campaign)
    db_session.commit()

    create_resp = await async_client.post(
        "/journeys",
        json={
            "name": "First-Time Donor Welcome Sequence",
            "campaign_id": campaign.id,
            "trigger_type": "donor_created",
            "config": {"max_delay_days": 30},
            "steps": [
                {
                    "step_type": "send_email",
                    "channel": "email",
                    "delay_days": 0,
                    "condition": None,
                },
                {
                    "step_type": "send_email",
                    "channel": "email",
                    "delay_days": 7,
                    "condition": {
                        "field": "has_donated",
                        "operator": "eq",
                        "value": True,
                    },
                },
            ],
        },
        headers=operator_headers,
    )
    assert create_resp.status_code == 201
    journey_data = create_resp.json()
    assert journey_data["name"] == "First-Time Donor Welcome Sequence"

    list_resp = await async_client.get("/journeys", headers=operator_headers)
    assert list_resp.status_code == 200
    assert any(j["id"] == journey_data["id"] for j in list_resp.json())


@pytest.mark.anyio
async def test_feature_flags_api_lifecycle(
    async_client: AsyncClient,
    admin_headers: dict[str, str],
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test feature flag creation, listing, and evaluation."""
    # Creating flag requires admin role
    forbidden_resp = await async_client.post(
        "/feature-flags",
        json={
            "name": "enable_ai_guardrails_v2",
            "description": "Beta 7-stage guardrails cascade",
            "status": "rollout",
            "rollout_percent": 50.0,
            "rules": {},
        },
        headers=operator_headers,
    )
    assert forbidden_resp.status_code == 403

    create_resp = await async_client.post(
        "/feature-flags",
        json={
            "name": "enable_ai_guardrails_v2",
            "description": "Beta 7-stage guardrails cascade",
            "status": "rollout",
            "rollout_percent": 50.0,
            "rules": {},
        },
        headers=admin_headers,
    )
    assert create_resp.status_code == 201
    flag_data = create_resp.json()
    assert flag_data["name"] == "enable_ai_guardrails_v2"
    assert flag_data["status"] == "rollout"

    list_resp = await async_client.get("/feature-flags", headers=operator_headers)
    assert list_resp.status_code == 200
    assert any(f["name"] == "enable_ai_guardrails_v2" for f in list_resp.json())

    eval_resp = await async_client.get(
        "/feature-flags/enable_ai_guardrails_v2?subject_key=donor-abc-123",
        headers=operator_headers,
    )
    assert eval_resp.status_code == 200
    assert "enabled" in eval_resp.json()
