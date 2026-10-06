"""Platform API routers: knowledge, templates, experiments, journeys, feature flags."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_role
from app.db.session import get_db_session
from app.models.entities import (
    AppealTone,
    Experiment,
    ExperimentVariant,
    FeatureFlag,
    FeatureFlagStatus,
    Journey,
    KnowledgeDocument,
    Template,
    User,
    UserRole,
)
from app.schemas.phase2 import (
    ExperimentCreate,
    ExperimentResponse,
    FeatureFlagCreate,
    FeatureFlagResponse,
    JourneyCreate,
    JourneyResponse,
    KnowledgeDocumentCreate,
    KnowledgeDocumentResponse,
    TemplateCreate,
    TemplateResponse,
    TemplateVersionCreate,
    TemplateVersionResponse,
)
from app.services.experiments import ExperimentService
from app.services.feature_flags import FeatureFlagService
from app.services.journeys import JourneyService
from app.services.rag import RagPipeline
from app.services.template import TemplateManagementService

router = APIRouter(tags=["platform"])

knowledge_router = APIRouter(prefix="/knowledge", tags=["knowledge"])
templates_router = APIRouter(prefix="/templates", tags=["templates"])
experiments_router = APIRouter(prefix="/experiments", tags=["experiments"])
journeys_router = APIRouter(prefix="/journeys", tags=["journeys"])
feature_flags_router = APIRouter(prefix="/feature-flags", tags=["feature-flags"])

admin_required = require_role(UserRole.ADMIN)


def _val(x: Any) -> str:
    """Safely return value of enum or string."""
    return x.value if hasattr(x, "value") else str(x)


@knowledge_router.post(
    "", response_model=KnowledgeDocumentResponse, status_code=status.HTTP_201_CREATED
)
def create_knowledge_document(
    payload: KnowledgeDocumentCreate,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> KnowledgeDocumentResponse:
    """Create a knowledge document for RAG ingestion."""
    document = KnowledgeDocument(
        title=payload.title,
        content=payload.content,
        source_url=payload.source_url,
        is_approved=payload.is_approved,
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)
    return KnowledgeDocumentResponse(
        id=document.id,
        title=document.title,
        source_url=document.source_url,
        is_approved=document.is_approved,
        created_at=document.created_at,
    )


@knowledge_router.get("", response_model=list[KnowledgeDocumentResponse])
def list_knowledge_documents(
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[KnowledgeDocumentResponse]:
    """List knowledge documents."""
    documents = list(
        db_session.execute(
            select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [
        KnowledgeDocumentResponse(
            id=doc.id,
            title=doc.title,
            source_url=doc.source_url,
            is_approved=doc.is_approved,
            created_at=doc.created_at,
        )
        for doc in documents
    ]


@knowledge_router.post("/{document_id}/ingest")
def ingest_knowledge_document(
    document_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Chunk and embed an approved knowledge document."""
    rag_pipeline = RagPipeline(db_session)
    try:
        chunk_count = rag_pipeline.ingest_document(document_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return {"document_id": document_id, "chunk_count": chunk_count}


@knowledge_router.get("/{document_id}")
def get_knowledge_document(
    document_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Return the full knowledge document with chunks."""
    document = db_session.get(KnowledgeDocument, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found."
        )
    return {
        "id": document.id,
        "title": document.title,
        "content": document.content,
        "source_url": document.source_url,
        "is_approved": document.is_approved,
        "chunk_count": len(document.chunks),
    }


@templates_router.post(
    "", response_model=TemplateResponse, status_code=status.HTTP_201_CREATED
)
def create_template(
    payload: TemplateCreate,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> TemplateResponse:
    """Create a versioned template."""
    service = TemplateManagementService(db_session)
    template = service.create_template(
        name=payload.name,
        subject_template=payload.subject_template,
        body_template=payload.body_template,
        cta_template=payload.cta_template,
        tone=AppealTone(payload.tone),
        tenant_id=current_user.tenant_id,
        description=payload.description,
        merge_tags=payload.merge_tags,
        created_by_user_id=current_user.id,
    )
    return TemplateResponse(
        id=template.id,
        name=template.name,
        description=template.description,
        is_active=template.is_active,
        current_version_id=template.current_version_id,
        created_at=template.created_at,
    )


@templates_router.get("", response_model=list[TemplateResponse])
def list_templates(
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[TemplateResponse]:
    """List all templates."""
    templates = list(
        db_session.execute(select(Template).order_by(Template.created_at.desc()))
        .scalars()
        .all()
    )
    return [
        TemplateResponse(
            id=template.id,
            name=template.name,
            description=template.description,
            is_active=template.is_active,
            current_version_id=template.current_version_id,
            created_at=template.created_at,
        )
        for template in templates
    ]


@templates_router.post(
    "/{template_id}/versions",
    response_model=TemplateVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_template_version(
    template_id: str,
    payload: TemplateVersionCreate,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> TemplateVersionResponse:
    """Create a new version of a template."""
    service = TemplateManagementService(db_session)
    try:
        version = service.create_new_version(
            template_id=template_id,
            subject_template=payload.subject_template,
            body_template=payload.body_template,
            cta_template=payload.cta_template,
            tone=AppealTone(payload.tone),
            change_note=payload.change_note,
            created_by_user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return TemplateVersionResponse(
        id=version.id,
        template_id=version.template_id,
        version_number=version.version_number,
        subject_template=version.subject_template,
        body_template=version.body_template,
        cta_template=version.cta_template,
        tone=_val(version.tone),
        change_note=version.change_note,
        created_at=version.created_at,
    )


@templates_router.post(
    "/{template_id}/rollback/{version_number}", response_model=TemplateResponse
)
def rollback_template(
    template_id: str,
    version_number: int,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> TemplateResponse:
    """Roll back a template to a previous version."""
    service = TemplateManagementService(db_session)
    try:
        template = service.rollback_to_version(template_id, version_number)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return TemplateResponse(
        id=template.id,
        name=template.name,
        description=template.description,
        is_active=template.is_active,
        current_version_id=template.current_version_id,
        created_at=template.created_at,
    )


@templates_router.get(
    "/{template_id}/versions", response_model=list[TemplateVersionResponse]
)
def list_template_versions(
    template_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[TemplateVersionResponse]:
    """List all versions of a template."""
    service = TemplateManagementService(db_session)
    try:
        versions = service.list_versions(template_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return [
        TemplateVersionResponse(
            id=version.id,
            template_id=version.template_id,
            version_number=version.version_number,
            subject_template=version.subject_template,
            body_template=version.body_template,
            cta_template=version.cta_template,
            tone=_val(version.tone),
            change_note=version.change_note,
            created_at=version.created_at,
        )
        for version in versions
    ]


@experiments_router.post(
    "", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED
)
def create_experiment(
    payload: ExperimentCreate,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> ExperimentResponse:
    """Create an A/B experiment with variants."""
    experiment = Experiment(
        name=payload.name,
        description=payload.description,
        hypothesis=payload.hypothesis,
        assignment_key=payload.assignment_key,
        tenant_id=current_user.tenant_id,
    )
    db_session.add(experiment)
    db_session.flush()

    for index, variant_payload in enumerate(payload.variants):
        variant = ExperimentVariant(
            experiment_id=experiment.id,
            name=str(variant_payload.get("name", f"variant-{index}")),
            weight=float(variant_payload.get("weight", 1.0)),
            template_id=variant_payload.get("template_id"),
            is_control=bool(index == 0),
        )
        db_session.add(variant)

    db_session.commit()
    db_session.refresh(experiment)
    return ExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        hypothesis=experiment.hypothesis,
        status=_val(experiment.status),
        assignment_key=experiment.assignment_key,
        traffic_allocation_percent=experiment.traffic_allocation_percent,
        started_at=experiment.started_at,
        completed_at=experiment.completed_at,
        promoted_variant_id=experiment.promoted_variant_id,
        confidence_level=experiment.confidence_level,
        p_value=experiment.p_value,
    )


@experiments_router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[ExperimentResponse]:
    """List all experiments."""
    experiments = list(
        db_session.execute(select(Experiment).order_by(Experiment.created_at.desc()))
        .scalars()
        .all()
    )
    return [
        ExperimentResponse(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            hypothesis=experiment.hypothesis,
            status=_val(experiment.status),
            assignment_key=experiment.assignment_key,
            traffic_allocation_percent=experiment.traffic_allocation_percent,
            started_at=experiment.started_at,
            completed_at=experiment.completed_at,
            promoted_variant_id=experiment.promoted_variant_id,
            confidence_level=experiment.confidence_level,
            p_value=experiment.p_value,
        )
        for experiment in experiments
    ]


@experiments_router.post("/{experiment_id}/start", response_model=ExperimentResponse)
def start_experiment(
    experiment_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(admin_required)],
) -> ExperimentResponse:
    """Start an experiment."""
    experiment = db_session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found."
        )
    service = ExperimentService(db_session)
    experiment = service.start_experiment(experiment)
    return ExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        hypothesis=experiment.hypothesis,
        status=_val(experiment.status),
        assignment_key=experiment.assignment_key,
        traffic_allocation_percent=experiment.traffic_allocation_percent,
        started_at=experiment.started_at,
        completed_at=experiment.completed_at,
        promoted_variant_id=experiment.promoted_variant_id,
        confidence_level=experiment.confidence_level,
        p_value=experiment.p_value,
    )


@experiments_router.get("/{experiment_id}/results")
def experiment_results(
    experiment_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Compute statistical results for an experiment."""
    experiment = db_session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found."
        )
    service = ExperimentService(db_session)
    return service.compute_statistical_results(experiment)


@experiments_router.post(
    "/{experiment_id}/promote/{variant_id}", response_model=ExperimentResponse
)
def promote_experiment_variant(
    experiment_id: str,
    variant_id: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(admin_required)],
) -> ExperimentResponse:
    """Promote a winning variant with an audit trail."""
    experiment = db_session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found."
        )
    service = ExperimentService(db_session)
    try:
        experiment = service.promote_winning_variant(experiment, variant_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return ExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        hypothesis=experiment.hypothesis,
        status=_val(experiment.status),
        assignment_key=experiment.assignment_key,
        traffic_allocation_percent=experiment.traffic_allocation_percent,
        started_at=experiment.started_at,
        completed_at=experiment.completed_at,
        promoted_variant_id=experiment.promoted_variant_id,
        confidence_level=experiment.confidence_level,
        p_value=experiment.p_value,
    )


@journeys_router.post(
    "", response_model=JourneyResponse, status_code=status.HTTP_201_CREATED
)
def create_journey(
    payload: JourneyCreate,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> JourneyResponse:
    """Create a multi-step donor journey."""
    service = JourneyService(db_session)
    journey = service.create_journey(
        name=payload.name,
        campaign_id=payload.campaign_id,
        trigger_type=payload.trigger_type,
        config=payload.config,
        steps=payload.steps,
    )
    return JourneyResponse(
        id=journey.id,
        campaign_id=journey.campaign_id,
        name=journey.name,
        trigger_type=journey.trigger_type,
        is_active=journey.is_active,
        created_at=journey.created_at,
    )


@journeys_router.get("", response_model=list[JourneyResponse])
def list_journeys(
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[JourneyResponse]:
    """List all journeys."""
    journeys = list(
        db_session.execute(select(Journey).order_by(Journey.created_at.desc()))
        .scalars()
        .all()
    )
    return [
        JourneyResponse(
            id=journey.id,
            campaign_id=journey.campaign_id,
            name=journey.name,
            trigger_type=journey.trigger_type,
            is_active=journey.is_active,
            created_at=journey.created_at,
        )
        for journey in journeys
    ]


@feature_flags_router.post(
    "", response_model=FeatureFlagResponse, status_code=status.HTTP_201_CREATED
)
def create_feature_flag(
    payload: FeatureFlagCreate,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(admin_required)],
) -> FeatureFlagResponse:
    """Create a feature flag."""
    flag = FeatureFlag(
        name=payload.name,
        description=payload.description,
        status=FeatureFlagStatus(payload.status),
        rollout_percent=payload.rollout_percent,
        rules=payload.rules,
        tenant_id=current_user.tenant_id,
    )
    db_session.add(flag)
    db_session.commit()
    db_session.refresh(flag)
    return FeatureFlagResponse(
        id=flag.id,
        name=flag.name,
        description=flag.description,
        status=_val(flag.status),
        rollout_percent=flag.rollout_percent,
        updated_at=flag.updated_at,
    )


@feature_flags_router.get("/{flag_name}")
def evaluate_feature_flag(
    flag_name: str,
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
    subject_key: str | None = None,
) -> dict:
    """Evaluate a feature flag for a subject."""
    service = FeatureFlagService(db_session)
    enabled = service.is_enabled(
        flag_name, tenant_id=current_user.tenant_id, subject_key=subject_key
    )
    return {"name": flag_name, "enabled": enabled}


@feature_flags_router.get("", response_model=list[FeatureFlagResponse])
def list_feature_flags(
    db_session: Annotated[Session, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[FeatureFlagResponse]:
    """List all feature flags."""
    flags = list(
        db_session.execute(select(FeatureFlag).order_by(FeatureFlag.created_at.desc()))
        .scalars()
        .all()
    )
    return [
        FeatureFlagResponse(
            id=flag.id,
            name=flag.name,
            description=flag.description,
            status=_val(flag.status),
            rollout_percent=flag.rollout_percent,
            updated_at=flag.updated_at,
        )
        for flag in flags
    ]
