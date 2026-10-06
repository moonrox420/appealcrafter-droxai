"""Tests for /operations API endpoints (suppression, reporting, jobs, compliance, ML)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from app.models.entities import Campaign, Donor


@pytest.mark.anyio
async def test_suppression_api_flow(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test suppression list management via API."""
    add_resp = await async_client.post(
        "/suppression",
        json={
            "email": "optout@example.com",
            "reason": "unsubscribe",
        },
        headers=operator_headers,
    )
    assert add_resp.status_code == 201
    entry = add_resp.json()
    assert entry["email"] == "optout@example.com"
    assert entry["reason"] == "unsubscribe"

    list_resp = await async_client.get("/suppression", headers=operator_headers)
    assert list_resp.status_code == 200
    assert any(e["email"] == "optout@example.com" for e in list_resp.json())

    del_resp = await async_client.delete(
        "/suppression/optout@example.com", headers=operator_headers
    )
    assert del_resp.status_code == 204

    not_found_del = await async_client.delete(
        "/suppression/unknown@example.com", headers=operator_headers
    )
    assert not_found_del.status_code == 404


@pytest.mark.anyio
async def test_reporting_api(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test campaign reports and delivery analytics endpoints."""
    campaign = Campaign(name="Reporting Campaign", description="Testing reporting API")
    db_session.add(campaign)
    db_session.commit()

    report_resp = await async_client.get(
        f"/reporting/campaigns/{campaign.id}",
        headers=operator_headers,
    )
    assert report_resp.status_code == 200
    report_data = report_resp.json()
    assert report_data["campaign_name"] == "Reporting Campaign"
    assert "delivery_rate" in report_data

    deliveries_resp = await async_client.get(
        "/reporting/deliveries",
        headers=operator_headers,
    )
    assert deliveries_resp.status_code == 200
    assert "daily_breakdown" in deliveries_resp.json()

    not_found_report = await async_client.get(
        "/reporting/campaigns/00000000-0000-0000-0000-000000000000",
        headers=operator_headers,
    )
    assert not_found_report.status_code == 404


@pytest.mark.anyio
async def test_jobs_api_lifecycle(
    async_client: AsyncClient,
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test async bulk job creation and status polling."""
    create_resp = await async_client.post(
        "/jobs/bulk_donor_import",
        headers=operator_headers,
    )
    assert create_resp.status_code == 202
    job_data = create_resp.json()
    assert job_data["job_type"] == "bulk_donor_import"
    assert job_data["status"] == "pending"
    job_id = job_data["id"]

    get_resp = await async_client.get(f"/jobs/{job_id}", headers=operator_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == job_id

    list_resp = await async_client.get("/jobs", headers=operator_headers)
    assert list_resp.status_code == 200
    assert any(j["id"] == job_id for j in list_resp.json())

    not_found_job = await async_client.get(
        "/jobs/00000000-0000-0000-0000-000000000000", headers=operator_headers
    )
    assert not_found_job.status_code == 404


@pytest.mark.anyio
async def test_compliance_api_flow(
    async_client: AsyncClient,
    admin_headers: dict[str, str],
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test GDPR/CCPA export, anonymize, and delete endpoints."""
    donor = Donor(
        email="compliance-api@example.com",
        first_name="Compliant",
        last_name="User",
        channel="email",
        consent_given_at=datetime.now(timezone.utc),
        consent_source="api",
    )
    db_session.add(donor)
    db_session.commit()
    donor_id = donor.id

    # Export donor data
    export_resp = await async_client.get(
        f"/compliance/donors/{donor_id}/export",
        headers=operator_headers,
    )
    assert export_resp.status_code == 200
    export_data = export_resp.json()
    assert export_data["donor"]["id"] == donor_id
    assert export_data["donor"]["first_name"] == "Compliant"

    # Anonymize requires admin role
    forbidden_anonymize = await async_client.post(
        f"/compliance/donors/{donor_id}/anonymize",
        headers=operator_headers,
    )
    assert forbidden_anonymize.status_code == 403

    anon_resp = await async_client.post(
        f"/compliance/donors/{donor_id}/anonymize",
        headers=admin_headers,
    )
    assert anon_resp.status_code == 200
    assert anon_resp.json()["status"] == "completed"

    # Delete requires admin role
    del_resp = await async_client.delete(
        f"/compliance/donors/{donor_id}",
        headers=admin_headers,
    )
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "completed"

    # Subsequent export returns 404
    not_found_resp = await async_client.get(
        f"/compliance/donors/{donor_id}/export",
        headers=operator_headers,
    )
    assert not_found_resp.status_code == 404


@pytest.mark.anyio
async def test_ml_api_train_and_drift(
    async_client: AsyncClient,
    admin_headers: dict[str, str],
    operator_headers: dict[str, str],
    db_session: Session,
) -> None:
    """Test ML model training and drift check endpoints."""
    from app.models.entities import ModelVersion, ModelVersionStatus

    # Training requires admin role
    forbidden_train = await async_client.post(
        "/ml/train/donor_propensity/v1.0.0",
        headers=operator_headers,
    )
    assert forbidden_train.status_code == 403

    train_resp = await async_client.post(
        "/ml/train/donor_propensity/v1.0.0",
        headers=admin_headers,
    )
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert train_data["model_name"] == "donor_propensity"
    assert "metrics" in train_data

    # Add a promoted model to test drift check
    promoted_model = ModelVersion(
        model_name="donor_propensity",
        version_number="v1.0.0-prod",
        status=ModelVersionStatus.PROMOTED,
        metrics={"auc_roc": 0.88, "precision": 0.85},
    )
    db_session.add(promoted_model)
    db_session.commit()

    drift_resp = await async_client.get(
        "/ml/models/donor_propensity/drift",
        headers=operator_headers,
    )
    assert drift_resp.status_code == 200
    assert "drift_detected" in drift_resp.json()

    not_found_drift = await async_client.get(
        "/ml/models/nonexistent_model/drift",
        headers=operator_headers,
    )
    assert not_found_drift.status_code == 404
