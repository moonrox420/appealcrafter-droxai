"""Template generation and versioning service with rollback support.

Provides the permanent template fallback path plus versioned template
storage with audit-trailed rollback and rich merge tags with safe fallbacks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.entities import AppealTone, Donor, Template, TemplateVersion

logger = logging.getLogger(__name__)

MERGE_TAG_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@dataclass(frozen=True)
class GeneratedAppealContent:
    """Immutable generated appeal content."""

    subject: str
    body: str
    cta: str
    tone: AppealTone
    is_template_fallback: bool


class TemplateService:
    """Generate appeal content from templates or the built-in fallback."""

    def __init__(self, db_session: Session | None = None) -> None:
        self.db_session = db_session
        self.settings = get_settings()

    def generate(self, donor: Donor, tone: AppealTone) -> GeneratedAppealContent:
        """Generate a personalized appeal using the template fallback path."""
        settings = self.settings
        first_name = donor.first_name or "Friend"
        capacity = donor.capacity_score or 100.0

        if self.db_session is not None:
            active_template = self._find_active_template(tone)
            if active_template is not None:
                content = self._render_template(active_template, donor, tone, first_name, capacity)
                if content is not None:
                    return content

        subject = "Your support can change lives this month"
        body = (
            f"Dear {first_name},\n\n"
            f"Thanks to donors like you, we reached thousands of families last year. "
            f"Your past generosity has made a real difference.\n\n"
            f"Will you help us continue this work? Your gift of ${capacity:.0f} "
            f"can provide critical support to those who need it most.\n\n"
            f"Thank you for being part of our community.\n\n"
            f"---\n"
            f"You are receiving this email because you have supported our cause. "
            f"To unsubscribe, click the unsubscribe link in your email client or "
            f"reply with 'unsubscribe'.\n"
            f"{settings.email.physical_address}"
        )
        cta = "Donate now"
        return GeneratedAppealContent(
            subject=subject,
            body=body,
            cta=cta,
            tone=tone,
            is_template_fallback=True,
        )

    def _find_active_template(self, tone: AppealTone) -> TemplateVersion | None:
        """Find an active template version for the given tone."""
        if self.db_session is None:
            return None
        statement = (
            select(Template)
            .where(Template.is_active.is_(True))
            .order_by(Template.updated_at.desc())
            .limit(10)
        )
        templates = list(self.db_session.execute(statement).scalars().all())
        for template in templates:
            if template.current_version_id:
                version = self.db_session.get(TemplateVersion, template.current_version_id)
                if version is not None and version.tone == tone:
                    return version
        return None

    def _render_template(
        self,
        version: TemplateVersion,
        donor: Donor,
        tone: AppealTone,
        first_name: str,
        capacity: float,
    ) -> GeneratedAppealContent | None:
        """Render a versioned template with merge tag replacements."""
        merge_values: dict[str, str] = {
            "first_name": first_name,
            "last_name": donor.last_name or "",
            "capacity": f"{capacity:.0f}",
            "email": donor.email,
            "interests": donor.interests or "our cause",
            "tone": tone.value,
        }

        def replace_merge_tag(match: re.Match[str]) -> str:
            tag_name = match.group(1)
            return merge_values.get(tag_name, "")

        try:
            subject = MERGE_TAG_PATTERN.sub(replace_merge_tag, version.subject_template)
            body = MERGE_TAG_PATTERN.sub(replace_merge_tag, version.body_template)
            cta = MERGE_TAG_PATTERN.sub(replace_merge_tag, version.cta_template)
        except re.error as exc:
            logger.error("Template rendering failed", extra={"template_id": version.template_id, "error": str(exc)})
            return None

        if self.settings.email.physical_address and self.settings.email.physical_address not in body:
            body = f"{body}\n\n{self.settings.email.physical_address}"
        if "unsubscribe" not in body.lower() and "opt out" not in body.lower():
            body = (
                f"{body}\n\n"
                f"You are receiving this email because you have supported our cause. "
                f"To unsubscribe, click the unsubscribe link or reply with 'unsubscribe'."
            )
        return GeneratedAppealContent(
            subject=subject,
            body=body,
            cta=cta,
            tone=tone,
            is_template_fallback=False,
        )


class TemplateManagementService:
    """Manage versioned templates with rollback support."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def create_template(
        self,
        name: str,
        subject_template: str,
        body_template: str,
        cta_template: str,
        tone: AppealTone,
        tenant_id: str | None = None,
        description: str | None = None,
        merge_tags: dict | None = None,
        created_by_user_id: str | None = None,
    ) -> Template:
        """Create a template with its first version."""
        template = Template(
            name=name,
            tenant_id=tenant_id,
            description=description,
            is_active=True,
        )
        self.db_session.add(template)
        self.db_session.flush()

        version = TemplateVersion(
            template_id=template.id,
            version_number=1,
            subject_template=subject_template,
            body_template=body_template,
            cta_template=cta_template,
            tone=tone,
            merge_tags=merge_tags or {},
            created_by_user_id=created_by_user_id,
        )
        self.db_session.add(version)
        self.db_session.flush()

        template.current_version_id = version.id
        self.db_session.commit()
        self.db_session.refresh(template)
        return template

    def create_new_version(
        self,
        template_id: str,
        subject_template: str,
        body_template: str,
        cta_template: str,
        tone: AppealTone,
        change_note: str | None = None,
        created_by_user_id: str | None = None,
    ) -> TemplateVersion:
        """Create a new immutable version of an existing template."""
        template = self.db_session.get(Template, template_id)
        if template is None:
            raise ValueError(f"Template {template_id} not found.")

        next_version_number = 1
        if template.current_version_id:
            current = self.db_session.get(TemplateVersion, template.current_version_id)
            if current is not None:
                next_version_number = current.version_number + 1

        version = TemplateVersion(
            template_id=template_id,
            version_number=next_version_number,
            subject_template=subject_template,
            body_template=body_template,
            cta_template=cta_template,
            tone=tone,
            change_note=change_note,
            created_by_user_id=created_by_user_id,
        )
        self.db_session.add(version)
        self.db_session.flush()

        template.current_version_id = version.id
        self.db_session.commit()
        self.db_session.refresh(version)
        return version

    def rollback_to_version(self, template_id: str, version_number: int) -> Template:
        """Roll back a template to a previous version."""
        template = self.db_session.get(Template, template_id)
        if template is None:
            raise ValueError(f"Template {template_id} not found.")

        version = self.db_session.execute(
            select(TemplateVersion).where(
                TemplateVersion.template_id == template_id,
                TemplateVersion.version_number == version_number,
            )
        ).scalars().first()
        if version is None:
            raise ValueError(f"Template version {version_number} not found.")

        template.current_version_id = version.id
        self.db_session.commit()
        self.db_session.refresh(template)
        return template

    def list_versions(self, template_id: str) -> list[TemplateVersion]:
        """List all versions of a template in order."""
        return list(self.db_session.execute(
            select(TemplateVersion)
            .where(TemplateVersion.template_id == template_id)
            .order_by(TemplateVersion.version_number)
        ).scalars().all())


def get_template_management_service(db_session: Session) -> TemplateManagementService:
    """Return a configured template management service instance."""
    return TemplateManagementService(db_session)