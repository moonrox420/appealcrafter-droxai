"""Unit tests for LlmAppealGenerationService."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.models.entities import AppealTone, Donor
from app.services.llm_generation import (
    GenerationResult,
    LlmAppealGenerationService,
    get_llm_generation_service,
)
from app.services.rag import RetrievalResult, RetrievedChunk


@pytest.fixture
def sample_donor() -> Donor:
    return Donor(
        id="d-123",
        email="donor@example.com",
        first_name="Jane",
        last_name="Doe",
        channel="email",
        interests="wildlife conservation",
        capacity_score=250.0,
    )


def test_generation_rag_disabled_fallback(sample_donor: Donor) -> None:
    """Verify fallback to template when rag_enabled is False."""
    service = get_llm_generation_service()
    mock_rag = MagicMock()

    with patch.object(service.settings.llm, "rag_enabled", False):
        res = service.generate(sample_donor, AppealTone.INSPIRING, mock_rag)
        assert isinstance(res, GenerationResult)
        assert res.is_template_fallback is True
        assert res.retrieved_chunk_ids == []
        assert "Jane" in res.body or "Friend" in res.body


def test_generation_empty_chunks_fallback(sample_donor: Donor) -> None:
    """Verify fallback to template when RAG returns 0 chunks."""
    service = LlmAppealGenerationService()
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = RetrievalResult(chunks=[], query="test")

    with patch.object(service.settings.llm, "rag_enabled", True):
        res = service.generate(sample_donor, AppealTone.URGENT, mock_rag)
        assert res.is_template_fallback is True
        assert res.retrieved_chunk_ids == []


def test_generation_missing_api_key_fallback(sample_donor: Donor) -> None:
    """Verify fallback to template when API key is missing."""
    service = LlmAppealGenerationService()
    mock_rag = MagicMock()
    chunk = RetrievedChunk(
        chunk_id="chk-1",
        document_id="doc-1",
        document_title="Title",
        content="Clean water for all",
        similarity_score=0.92,
        source_url=None,
    )
    mock_rag.retrieve.return_value = RetrievalResult(chunks=[chunk], query="test")

    with patch.object(service.settings.llm, "rag_enabled", True), patch.object(
        service.settings.llm, "api_base_url", "https://api.openai.com/v1"
    ), patch.object(service.settings.llm, "api_key", None):
        res = service.generate(sample_donor, AppealTone.GRATEFUL, mock_rag)
        assert res.is_template_fallback is True
        assert res.retrieved_chunk_ids == ["chk-1"]


def test_generation_successful_llm_call(sample_donor: Donor) -> None:
    """Verify successful LLM generation with CAN-SPAM and address formatting."""
    service = LlmAppealGenerationService()
    mock_rag = MagicMock()
    chunk = RetrievedChunk(
        chunk_id="chk-1",
        document_id="doc-1",
        document_title="Title",
        content="Clean water for all",
        similarity_score=0.92,
        source_url=None,
    )
    mock_rag.retrieve.return_value = RetrievalResult(chunks=[chunk], query="test")
    mock_rag.build_citation_context.return_value = "[1] Clean water for all"

    llm_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "subject": "Help us bring clean water",
                            "body": "Dear Jane, with your help we can provide clean water.",
                            "cta": "Support clean water today",
                        }
                    )
                }
            }
        ]
    }

    with patch.object(service.settings.llm, "rag_enabled", True), patch.object(
        service.settings.llm, "api_base_url", "https://api.openai.com/v1"
    ), patch.object(service.settings.llm, "api_key", "sk-test-key"), patch(
        "httpx.post"
    ) as mock_post:

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = llm_response
        mock_post.return_value = mock_resp

        res = service.generate(sample_donor, AppealTone.INSPIRING, mock_rag)

        assert res.is_template_fallback is False
        assert res.subject == "Help us bring clean water"
        assert "clean water" in res.body.lower()
        assert "unsubscribe" in res.body.lower()
        assert res.cta == "Support clean water today"
        assert res.retrieved_chunk_ids == ["chk-1"]


def test_generation_llm_exception_fallback(sample_donor: Donor) -> None:
    """Verify fallback to template when LLM HTTP call raises an exception."""
    service = LlmAppealGenerationService()
    mock_rag = MagicMock()
    chunk = RetrievedChunk(
        chunk_id="chk-1",
        document_id="doc-1",
        document_title="Title",
        content="Facts",
        similarity_score=0.88,
        source_url=None,
    )
    mock_rag.retrieve.return_value = RetrievalResult(chunks=[chunk], query="test")
    mock_rag.build_citation_context.return_value = "[1] Facts"

    with patch.object(service.settings.llm, "rag_enabled", True), patch.object(
        service.settings.llm, "api_base_url", "https://api.openai.com/v1"
    ), patch.object(service.settings.llm, "api_key", "sk-test-key"), patch(
        "httpx.post", side_effect=RuntimeError("API gateway timeout")
    ):

        res = service.generate(sample_donor, AppealTone.HOPEFUL, mock_rag)
        assert res.is_template_fallback is True
        assert res.retrieved_chunk_ids == ["chk-1"]
