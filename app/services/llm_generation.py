"""Constrained LLM generation service using retrieved approved chunks.

Generates appeal content only from approved RAG chunks with the permanent
template fallback when RAG is unavailable or retrieval is insufficient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from app.core.config import get_settings
from app.core.metrics import LLM_COST_DOLLARS
from app.models.entities import AppealTone, Donor
from app.services.rag import RagPipeline, RetrievalResult
from app.services.template import GeneratedAppealContent, TemplateService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    """Result of appeal generation with RAG metadata."""

    subject: str
    body: str
    cta: str
    tone: AppealTone
    is_template_fallback: bool
    retrieved_chunk_ids: list[str]


class LlmAppealGenerationService:
    """Generate appeals using retrieved RAG chunks with constrained generation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.template_service = TemplateService()

    def generate(
        self,
        donor: Donor,
        tone: AppealTone,
        rag_pipeline: RagPipeline,
    ) -> GenerationResult:
        """Generate an appeal constrained to approved retrieved chunks."""
        if not self.settings.llm.rag_enabled:
            template_content = self.template_service.generate(donor, tone)
            return GenerationResult(
                subject=template_content.subject,
                body=template_content.body,
                cta=template_content.cta,
                tone=template_content.tone,
                is_template_fallback=True,
                retrieved_chunk_ids=[],
            )

        donor_context = (
            f"Donor name: {donor.first_name or 'Friend'} "
            f"({donor.last_name or ''}). "
            f"Interests: {donor.interests or 'general'}. "
            f"Capacity score: {donor.capacity_score or 100.0:.0f}."
        )
        query = f"Write a {tone.value} fundraising appeal for a donor who is interested in {donor.interests or 'our cause'}."
        retrieval_result = rag_pipeline.retrieve(query, limit=5)

        if not retrieval_result.chunks:
            template_content = self.template_service.generate(donor, tone)
            return GenerationResult(
                subject=template_content.subject,
                body=template_content.body,
                cta=template_content.cta,
                tone=template_content.tone,
                is_template_fallback=True,
                retrieved_chunk_ids=[],
            )

        if self.settings.llm.api_base_url is None or self.settings.llm.api_key is None:
            template_content = self.template_service.generate(donor, tone)
            return GenerationResult(
                subject=template_content.subject,
                body=template_content.body,
                cta=template_content.cta,
                tone=template_content.tone,
                is_template_fallback=True,
                retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieval_result.chunks],
            )

        citation_context = rag_pipeline.build_citation_context(retrieval_result)
        try:
            response = httpx.post(
                f"{self.settings.llm.api_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.llm.api_key}"},
                json={
                    "model": self.settings.llm.model_name,
                    "temperature": self.settings.llm.temperature,
                    "max_tokens": self.settings.llm.max_tokens,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a fundraising appeal writer. Generate a personalized appeal "
                                "for a nonprofit donor. CRITICAL RULES:\n"
                                "1. Use ONLY facts from the provided approved context sections.\n"
                                "2. Do NOT invent statistics, claims, or impact numbers.\n"
                                "3. If a claim is not in the context, do not include it.\n"
                                "4. Match the requested tone.\n"
                                "5. End with a clear call-to-action.\n"
                                "6. Include an unsubscribe notice and the physical mailing address.\n"
                                "7. Keep the subject under 78 characters.\n"
                                "8. Do not use manipulative or high-pressure language.\n"
                                "Return JSON with keys: subject, body, cta."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Donor context: {donor_context}\n\n"
                                f"Approved context (use only these facts):\n{citation_context}\n\n"
                                f"Tone: {tone.value}"
                            ),
                        },
                    ],
                },
                timeout=self.settings.llm.timeout_seconds,
            )
            response.raise_for_status()
            generation_data = response.json()
            generated_content = generation_data["choices"][0]["message"]["content"]
            import json as json_module

            parsed = json_module.loads(generated_content)
            subject = str(parsed.get("subject", ""))[:255]
            body = str(parsed.get("body", ""))
            cta = str(parsed.get("cta", "Donate now"))[:100]
            physical_address = self.settings.email.physical_address
            if physical_address and physical_address not in body:
                body = f"{body}\n\n{physical_address}"
            if "unsubscribe" not in body.lower() and "opt out" not in body.lower():
                body = (
                    f"{body}\n\n"
                    f"You are receiving this email because you have supported our cause. "
                    f"To unsubscribe, click the unsubscribe link or reply with 'unsubscribe'."
                )

            estimated_cost = 0.00001 * (len(body) // 4)
            LLM_COST_DOLLARS.labels(model=self.settings.llm.model_name).inc(estimated_cost)

            return GenerationResult(
                subject=subject,
                body=body,
                cta=cta,
                tone=tone,
                is_template_fallback=False,
                retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieval_result.chunks],
            )
        except Exception as exc:
            logger.error(
                "LLM generation failed; falling back to template",
                extra={"error": str(exc), "donor_id": donor.id},
            )
            template_content = self.template_service.generate(donor, tone)
            return GenerationResult(
                subject=template_content.subject,
                body=template_content.body,
                cta=template_content.cta,
                tone=template_content.tone,
                is_template_fallback=True,
                retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieval_result.chunks],
            )


def get_llm_generation_service() -> LlmAppealGenerationService:
    """Return a configured LLM generation service instance."""
    return LlmAppealGenerationService()