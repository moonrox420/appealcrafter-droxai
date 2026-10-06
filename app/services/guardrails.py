"""Seven-stage LLM guardrail pipeline.

Implements the mandatory guardrail stages in cascade order:
1. Structural/format validation
2. Safety classifiers (fast + LLM-as-Judge)
3. Rule/blocklist checks
4. Factual grounding vs RAG chunks
5. Brand voice/tone validation
6. Compliance elements
7. Risk score + routing

Every decision is persisted for audit. A circuit breaker falls back to
templates if the validation failure rate spikes. Cascade runs cheap checks
first and only invokes the expensive LLM-as-Judge on borderline cases.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.metrics import GUARDRAIL_DECISIONS
from app.models.entities import (
    Appeal,
    AppealTone,
    GuardrailAction,
    GuardrailDecision,
    RiskLevel,
)

logger = logging.getLogger(__name__)

FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"guaranteed?\s+(results?|impact|outcome)",
    r"act\s+now\s+or\s+.+\s+will\s+die",
    r"100%\s+of\s+your\s+donation",
    r"your\s+(last|final)\s+chance",
    r"urgent\s+action\s+required\s+immediately",
    r"you've?\s+been\s+(selected|chosen)\s+to",
    r"claim\s+your\s+(gift|prize|reward)",
)

HIGH_VALUE_DONOR_THRESHOLD: float = 5000.0
REJECT_SCORE_THRESHOLD: float = 70.0
HUMAN_REVIEW_SCORE_THRESHOLD: float = 35.0


class GuardrailStage(str, Enum):
    """Identifiers for the seven guardrail stages."""

    STRUCTURAL = "structural"
    SAFETY = "safety"
    RULE_BASED = "rule_based"
    GROUNDING = "grounding"
    BRAND_VOICE = "brand_voice"
    COMPLIANCE = "compliance"
    RISK_SCORING = "risk_scoring"


@dataclass(frozen=True)
class GuardrailStageResult:
    """Immutable result of a single guardrail stage."""

    stage: GuardrailStage
    passed: bool
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AppealCandidate:
    """Immutable candidate for guardrail validation."""

    subject: str
    body: str
    cta: str
    tone: AppealTone
    donor_id: str
    capacity: float | None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_chunks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GuardrailDecisionResult:
    """Immutable guardrail pipeline decision."""

    approved: bool
    risk_score: float
    risk_level: RiskLevel
    action: GuardrailAction
    stage_results: list[GuardrailStageResult]
    final_reasons: list[str]


class GuardrailPipeline:
    """Coordinate the seven-stage guardrail validation pipeline."""

    def __init__(
        self,
        db_session: Session,
        high_value_threshold: float = HIGH_VALUE_DONOR_THRESHOLD,
    ) -> None:
        self.db_session = db_session
        self.high_value_threshold = high_value_threshold
        self.settings = get_settings()
        self._validation_events: list[tuple[datetime, bool]] = []

    def _record_validation_event(self, passed: bool) -> None:
        """Record a validation outcome in the sliding window."""
        self._validation_events.append((datetime.now(timezone.utc), passed))
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.settings.llm.circuit_breaker_window_seconds
        )
        self._validation_events = [
            (event_time, outcome)
            for event_time, outcome in self._validation_events
            if event_time >= cutoff
        ]

    def is_circuit_breaker_open(self) -> bool:
        """Return True if the guardrail circuit breaker is open."""
        if not self._validation_events:
            return False
        total_events = len(self._validation_events)
        failed_events = sum(1 for _, passed in self._validation_events if not passed)
        failure_rate = failed_events / total_events
        return failure_rate > self.settings.llm.circuit_breaker_threshold

    def _run_llm_judge(self, candidate: AppealCandidate) -> GuardrailStageResult:
        """Invoke LLM-as-Judge using an explicit fundraising rubric."""
        if self.settings.llm.api_base_url is None or self.settings.llm.api_key is None:
            return GuardrailStageResult(
                stage=GuardrailStage.SAFETY,
                passed=True,
                score=5.0,
                details={"judge": "rule_based_fallback"},
            )
        try:
            import httpx

            response = httpx.post(
                f"{self.settings.llm.api_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.llm.api_key}"},
                json={
                    "model": self.settings.llm.judge_model_name,
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a fundraising email safety judge. "
                                "Evaluate the candidate email against this rubric:\n"
                                "1. Toxicity and manipulation (score 0-100)\n"
                                "2. Pressure tactics (score 0-100)\n"
                                "3. Factual claims that could be misleading (score 0-100)\n"
                                "4. Brand voice appropriateness (score 0-100)\n"
                                "Return only JSON with keys: toxicity, manipulation, "
                                "pressure, misleading, brand_voice, reasons."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Subject: {candidate.subject}\n\nBody: {candidate.body}\n\nCTA: {candidate.cta}",
                        },
                    ],
                },
                timeout=self.settings.llm.timeout_seconds,
            )
            response.raise_for_status()
            judge_data = response.json()
            judge_content = judge_data["choices"][0]["message"]["content"]
            import json as json_module

            parsed = json_module.loads(judge_content)
            toxicity_score = float(parsed.get("toxicity", 0.0))
            manipulation_score = float(parsed.get("manipulation", 0.0))
            pressure_score = float(parsed.get("pressure", 0.0))
            misleading_score = float(parsed.get("misleading", 0.0))
            brand_score = float(parsed.get("brand_voice", 0.0))
            judge_reasons: list[str] = parsed.get("reasons", [])
            combined_score = max(
                toxicity_score, manipulation_score, pressure_score, misleading_score
            )
            reasons = [str(reason_item) for reason_item in judge_reasons]
            passed = combined_score < 30.0
            return GuardrailStageResult(
                stage=GuardrailStage.SAFETY,
                passed=passed,
                score=combined_score,
                reasons=reasons,
                details={
                    "toxicity": toxicity_score,
                    "manipulation": manipulation_score,
                    "pressure": pressure_score,
                    "misleading": misleading_score,
                    "brand_voice": brand_score,
                    "judge_model": self.settings.llm.judge_model_name,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM-as-Judge failed", extra={"error": str(exc)})
            return GuardrailStageResult(
                stage=GuardrailStage.SAFETY,
                passed=False,
                score=50.0,
                reasons=["LLM judge unavailable; routing to human review"],
                details={"error": str(exc)},
            )

    def validate(self, candidate: AppealCandidate) -> GuardrailDecisionResult:
        """Run the seven-stage pipeline and return a decision."""
        stage_results: list[GuardrailStageResult] = []

        if self.is_circuit_breaker_open():
            logger.warning("Guardrail circuit breaker open; using template fallback")
            return GuardrailDecisionResult(
                approved=False,
                risk_score=100.0,
                risk_level=RiskLevel.CRITICAL,
                action=GuardrailAction.REGENERATE,
                stage_results=[
                    GuardrailStageResult(
                        stage=GuardrailStage.RISK_SCORING,
                        passed=False,
                        score=100.0,
                        reasons=["Circuit breaker open"],
                    )
                ],
                final_reasons=["Circuit breaker open; template fallback required"],
            )

        stage_results.append(self._stage_structural(candidate))
        if not stage_results[-1].passed:
            return self._decide(stage_results, candidate, short_circuit=True)

        stage_results.append(self._stage_rule_based(candidate))
        stage_results.append(self._stage_safety(candidate))
        stage_results.append(self._stage_grounding(candidate))
        stage_results.append(self._stage_brand_voice(candidate))
        stage_results.append(self._stage_compliance(candidate))
        stage_results.append(self._stage_risk_scoring(candidate))

        return self._decide(stage_results, candidate, short_circuit=False)

    def _stage_structural(self, candidate: AppealCandidate) -> GuardrailStageResult:
        """Stage 1: validate format and structural requirements."""
        reasons: list[str] = []
        if not candidate.subject:
            reasons.append("Subject is missing")
        elif len(candidate.subject) > 78:
            reasons.append("Subject exceeds 78 characters")
        if not candidate.body or len(candidate.body) < 50:
            reasons.append("Body is missing or too short")
        if not candidate.cta:
            reasons.append("CTA is missing")
        passed = len(reasons) == 0
        return GuardrailStageResult(
            stage=GuardrailStage.STRUCTURAL,
            passed=passed,
            score=0.0 if passed else 80.0,
            reasons=reasons,
            details={
                "subject_length": len(candidate.subject),
                "body_length": len(candidate.body),
            },
        )

    def _stage_rule_based(self, candidate: AppealCandidate) -> GuardrailStageResult:
        """Stage 3: check for forbidden patterns and blocklisted phrases."""
        text = f"{candidate.subject} {candidate.body}".lower()
        reasons: list[str] = []
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                reasons.append(f"Forbidden pattern matched: {pattern}")
        score = min(100.0, len(reasons) * 40.0)
        passed = len(reasons) == 0
        return GuardrailStageResult(
            stage=GuardrailStage.RULE_BASED,
            passed=passed,
            score=score,
            reasons=reasons,
        )

    def _stage_safety(self, candidate: AppealCandidate) -> GuardrailStageResult:
        """Stage 2: safety classification using fast checks then LLM judge."""
        text = f"{candidate.subject} {candidate.body}".lower()
        fast_reasons: list[str] = []
        fast_score = 0.0

        urgent_pressure_words = (
            "immediately",
            "act now",
            "right now",
            "today only",
            "last chance",
        )
        if any(word in text for word in urgent_pressure_words):
            fast_score = max(fast_score, 25.0)
            fast_reasons.append("Pressure language detected")

        manipulative_phrases = (
            "you deserve",
            "you've been missing out",
            "hidden benefits",
        )
        if any(phrase in text for phrase in manipulative_phrases):
            fast_score = max(fast_score, 20.0)
            fast_reasons.append("Potential manipulative language")

        if fast_score >= 30.0:
            return GuardrailStageResult(
                stage=GuardrailStage.SAFETY,
                passed=False,
                score=fast_score,
                reasons=fast_reasons,
                details={"judge": "fast_classifier"},
            )

        if fast_score >= 15.0:
            return self._run_llm_judge(candidate)

        return GuardrailStageResult(
            stage=GuardrailStage.SAFETY,
            passed=True,
            score=fast_score,
            reasons=fast_reasons,
            details={"judge": "fast_classifier_passed"},
        )

    def _stage_grounding(self, candidate: AppealCandidate) -> GuardrailStageResult:
        """Stage 4: factual grounding vs retrieved RAG chunks."""
        if not candidate.retrieved_chunk_ids and not candidate.retrieved_chunks:
            return GuardrailStageResult(
                stage=GuardrailStage.GROUNDING,
                passed=False,
                score=60.0,
                reasons=[
                    "No retrieved context provided; LLM claims cannot be verified"
                ],
                details={"chunks_used": 0},
            )

        if self.settings.llm.api_base_url is None or self.settings.llm.api_key is None:
            return GuardrailStageResult(
                stage=GuardrailStage.GROUNDING,
                passed=True,
                score=10.0,
                details={
                    "chunks_used": max(
                        len(candidate.retrieved_chunk_ids),
                        len(candidate.retrieved_chunks),
                    ),
                    "method": "chunks_present",
                },
            )

        context_text = "\n\n".join(candidate.retrieved_chunks[:5])
        try:
            import httpx

            response = httpx.post(
                f"{self.settings.llm.api_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.llm.api_key}"},
                json={
                    "model": self.settings.llm.judge_model_name,
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You verify whether email claims are supported by the provided context. "
                                "Return JSON with keys: supported (bool), ungrounded_claims (list of strings), score (0-100)."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context_text}\n\nEmail:\n{candidate.body}\n\nSubject: {candidate.subject}",
                        },
                    ],
                },
                timeout=self.settings.llm.timeout_seconds,
            )
            response.raise_for_status()
            judge_data = response.json()
            judge_content = judge_data["choices"][0]["message"]["content"]
            import json as json_module

            parsed = json_module.loads(judge_content)
            supported = bool(parsed.get("supported", False))
            ungrounded_claims: list[str] = parsed.get("ungrounded_claims", [])
            grounding_score = float(parsed.get("score", 0.0))
            return GuardrailStageResult(
                stage=GuardrailStage.GROUNDING,
                passed=supported,
                score=grounding_score,
                reasons=ungrounded_claims,
                details={"chunks_used": len(candidate.retrieved_chunks)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Grounding judge failed", extra={"error": str(exc)})
            return GuardrailStageResult(
                stage=GuardrailStage.GROUNDING,
                passed=False,
                score=50.0,
                reasons=["Grounding verification unavailable"],
                details={"error": str(exc)},
            )

    def _stage_brand_voice(self, candidate: AppealCandidate) -> GuardrailStageResult:
        """Stage 5: verify brand voice and tone alignment."""
        voice_concerns: list[str] = []

        if candidate.tone == AppealTone.URGENT:
            if candidate.tone.value not in candidate.body.lower():
                voice_concerns.append("Urgent tone not reflected in body")
        elif (
            candidate.tone == AppealTone.GRATEFUL
            and "thank" not in candidate.body.lower()
            and "grateful" not in candidate.body.lower()
        ):
            voice_concerns.append("Grateful tone not reflected in body")

        if len(candidate.body.split()) > 500:
            voice_concerns.append(
                "Body exceeds 500 words; exceeds brand voice guidelines"
            )

        passed = len(voice_concerns) == 0
        return GuardrailStageResult(
            stage=GuardrailStage.BRAND_VOICE,
            passed=passed,
            score=0.0 if passed else 30.0,
            reasons=voice_concerns,
        )

    def _stage_compliance(self, candidate: AppealCandidate) -> GuardrailStageResult:
        """Stage 6: verify required compliance elements are present."""
        reasons: list[str] = []
        body_lower = candidate.body.lower()
        if "unsubscribe" not in body_lower and "opt out" not in body_lower:
            reasons.append("Missing unsubscribe language")
        if (
            self.settings.email.physical_address
            and self.settings.email.physical_address not in candidate.body
        ):
            reasons.append("Missing physical mailing address")
        score = 40.0 if reasons else 0.0
        return GuardrailStageResult(
            stage=GuardrailStage.COMPLIANCE,
            passed=len(reasons) == 0,
            score=score,
            reasons=reasons,
        )

    def _stage_risk_scoring(self, candidate: AppealCandidate) -> GuardrailStageResult:
        """Stage 7: compute aggregate risk score with high-value donor routing."""
        return GuardrailStageResult(
            stage=GuardrailStage.RISK_SCORING,
            passed=True,
            score=0.0,
            details={
                "high_value_donor": bool(
                    candidate.capacity
                    and candidate.capacity >= self.high_value_threshold
                ),
                "capacity": candidate.capacity,
            },
        )

    def _decide(
        self,
        stage_results: list[GuardrailStageResult],
        candidate: AppealCandidate,
        short_circuit: bool = False,
    ) -> GuardrailDecisionResult:
        """Combine stage results into a final decision with routing."""
        total_score = (
            max(result.score for result in stage_results) if stage_results else 0.0
        )
        all_reasons = [reason for result in stage_results for reason in result.reasons]
        is_high_value = (
            candidate.capacity is not None
            and candidate.capacity >= self.high_value_threshold
        )

        if short_circuit or total_score >= REJECT_SCORE_THRESHOLD:
            action = GuardrailAction.REJECT
            risk_level = RiskLevel.CRITICAL if total_score >= 90.0 else RiskLevel.HIGH
            approved = False
        elif total_score >= HUMAN_REVIEW_SCORE_THRESHOLD or is_high_value:
            action = GuardrailAction.HUMAN_REVIEW
            risk_level = RiskLevel.MEDIUM
            approved = False
        else:
            action = GuardrailAction.AUTO_APPROVE
            risk_level = RiskLevel.LOW
            approved = True

        self._record_validation_event(approved)
        GUARDRAIL_DECISIONS.labels(action=action.value).inc()

        logger.info(
            "Guardrail decision",
            extra={
                "donor_id": candidate.donor_id,
                "action": action.value,
                "risk_score": total_score,
                "risk_level": risk_level.value,
                "approved": approved,
                "reasons": all_reasons,
            },
        )
        return GuardrailDecisionResult(
            approved=approved,
            risk_score=total_score,
            risk_level=risk_level,
            action=action,
            stage_results=stage_results,
            final_reasons=all_reasons,
        )

    def persist_decision(
        self,
        candidate: AppealCandidate,
        decision: GuardrailDecisionResult,
        appeal: Appeal | None = None,
    ) -> GuardrailDecision:
        """Persist a guardrail decision for audit and training data."""
        guardrail_decision = GuardrailDecision(
            appeal_id=appeal.id if appeal else None,
            donor_id=candidate.donor_id,
            approved=decision.approved,
            risk_score=decision.risk_score,
            risk_level=decision.risk_level,
            action=decision.action,
            stage_results=[
                {
                    "stage": result.stage.value,
                    "passed": result.passed,
                    "score": result.score,
                    "reasons": result.reasons,
                    "details": result.details,
                }
                for result in decision.stage_results
            ],
            final_reasons=decision.final_reasons,
            candidate_snapshot={
                "subject": candidate.subject,
                "body": candidate.body,
                "cta": candidate.cta,
                "tone": (
                    candidate.tone.value
                    if hasattr(candidate.tone, "value")
                    else str(candidate.tone)
                ),
                "retrieved_chunk_ids": candidate.retrieved_chunk_ids,
            },
            circuit_breaker_open=self.is_circuit_breaker_open(),
        )
        self.db_session.add(guardrail_decision)
        self.db_session.commit()
        self.db_session.refresh(guardrail_decision)
        return guardrail_decision


def build_candidate_from_appeal(
    appeal: Appeal,
    retrieved_chunks: list[str] | None = None,
) -> AppealCandidate:
    """Build a guardrail candidate from a persisted appeal record."""
    return AppealCandidate(
        subject=appeal.subject,
        body=appeal.body,
        cta=appeal.cta,
        tone=appeal.tone,
        donor_id=appeal.donor_id,
        capacity=appeal.capacity_score,
        retrieved_chunk_ids=appeal.retrieved_chunk_ids,
        retrieved_chunks=retrieved_chunks or [],
    )
