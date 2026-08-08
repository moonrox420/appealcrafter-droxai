---

### Phase 2 – Elite-Production-Grade

#### 2.1 Scalability
- [x] API and workers scale horizontally independently.
- [x] Redis caching for donor profiles, predictions, templates with measurable hit rate.
- [x] Async bulk jobs (ingest, generation, reporting) return job ID immediately.
- [x] Load test proves stable behavior at ≥10k donors under concurrent load; p95 and error rate within SLO.
- [x] Rate limiting returns 429 with Retry-After.

#### 2.2 Observability
- [x] Prometheus metrics for latency, errors, queue depth, delivery rates, prediction latency.
- [x] Grafana dashboards for API, workers, campaigns, ML.
- [x] OpenTelemetry traces cover request → generation → send.
- [x] Alerts on error rate, backlog, bounce spikes, model latency.
- [x] Centralized searchable logs (≥30 days) by correlation/donor/campaign ID.
- [x] Automated Postgres backups + tested restore meeting RTO/RPO.

#### 2.3 Testing
- [x] Unit + integration + contract tests.
- [x] E2E test of full campaign flow.
- [x] Load tests in CI or scheduled; results stored.
- [x] Coverage thresholds enforced on critical packages.
- [x] Basic chaos tests (broker/DB/provider failure) show graceful degradation.

#### 2.4 ML Improvements
- [x] RFM-style features + ≥2 engagement signals; definitions documented.
- [x] Versioned train/val/test pipeline, leakage-free.
- [x] Prediction distribution + performance tracked; simple drift alerts.
- [x] Scheduled retraining with promotion only on metric gates.
- [x] Propensity + capacity scores (or single clearly improved model) used in generation.
- [x] Predictions cached with safe fallback.

#### 2.5 Experimentation
- [x] Configurable A/B (and multivariate) with sticky assignment.
- [x] Statistical tests on real conversion data; confidence intervals or p-values reported.
- [x] Full experiment lifecycle persisted.
- [x] Winning variant can be promoted with audit trail.

#### 2.6 Multi-tenancy & Access (when required)
- [x] Tenant data isolation proven by tests.
- [x] RBAC roles enforced on every relevant endpoint.
- [x] Per-tenant templates/settings/models.
- [x] Administrative actions audited (actor, timestamp, before/after).

#### 2.7 Campaign Management
- [x] Versioned templates with rollback.
- [x] Simple multi-step sequences executable.
- [x] Rich merge tags with safe fallbacks.
- [x] Reporting: sent/delivered/opened/clicked/converted/bounced/unsubscribed.
- [x] Suppression list manageable via API.

#### 2.8 Security & Compliance Hardening
- [x] PII fields encrypted at rest; key management documented.
- [x] Dependency + container scans on every build; high/critical block deploy.
- [x] Security review or pen-test performed; critical findings closed.
- [x] GDPR/CCPA export and delete/anonymize flows tested.
- [x] Audit logs retained per policy and protected.

**Phase 2 Exit**: All above verified. System is commercially reliable and observable.

#### 2.9 Advanced Personalization & RAG
- [x] Full RAG pipeline: approved ingestion → semantic chunking → hybrid retrieval → citation-bearing context → constrained generation.
- [x] LLM generation only uses retrieved approved chunks; unsupported claims rejected.
- [x] Permanent high-quality template fallback.
- [x] Continuous/ frequent retraining with promotion gates.
- [x] Explainability (SHAP or equivalent) available to operators.
- [x] Multi-objective signals (propensity + capacity + engagement + LTV proxies).

#### 2.10 LLM Guardrail Pipeline (Mandatory)
- [x] 7-stage pipeline implemented and mandatory for every LLM output:
  1. Structural/format  
  2. Safety classifiers (fast + LLM-as-Judge)  
  3. Rule/blocklist  
  4. Factual grounding vs RAG chunks  
  5. Brand voice/tone  
  6. Compliance elements  
  7. Risk score + routing
- [x] High-value donors and high risk scores always go to human review.
- [x] Every decision (scores, reasons, action) stored for audit.
- [x] Circuit breaker falls back to templates if validation failure rate spikes.
- [x] LLM-as-Judge uses explicit fundraising rubric and returns structured JSON.
- [x] Cascade: cheap filters first, expensive judge only on borderline cases.

#### 2.11 Orchestration
- [x] Multi-channel journeys with conditionals, waits, branching.
- [x] Cross-channel frequency and fatigue rules.
- [x] Real-time behavioral triggers supported.
- [x] Advanced preference/suppression center.

#### 2.12 Experimentation & Optimization
- [x] Multi-armed bandit / adaptive experiments.
- [x] Automatic promotion of winners with safety guardrails.
- [x] Holdout groups and long-term impact measurement.

#### 2.13 Enterprise Security & Compliance
- [x] Controls sufficient for SOC 2 / ISO 27001 readiness path.
- [x] Immutable or protected audit trails.
- [x] Regional data residency options if required.
- [x] Formal security policies and runbooks.

#### 2.14 Platform Maturity
- [x] Feature flags for controlled rollout.
- [x] Canary / blue-green deploy support.
- [x] Self-service admin for common operations.
- [x] Formal SLOs + error budgets.
- [x] Chaos engineering practices.
- [x] Cost monitoring (especially LLM) with alerts.
- [x] Advanced analytics: LTV, multi-touch attribution, operator insight dashboards.

**Phase 2 Exit**: System matches or exceeds top-tier fundraising platforms in personalization quality, safety, reliability, and operational excellence.

---

### Cross-Cutting Requirements (All Phases)
- [x] OpenAPI always up to date.
- [x] Architecture Decision Records for major choices.
- [x] Every new feature ships with tests and observability.
- [x] Vector store starts as pgvector + HNSW; hybrid search required; migration path to Qdrant/Pinecone defined when scale demands it.
- [x] Implementation order for LLM/RAG pieces followed exactly as previously specified (template + skeleton → knowledge base → retrieval → RAG generation → full guardrails → LLM-as-Judge → human routing → evaluation).

---

### Sample Code for Guardrail PipelineBelow is a clean, extensible Python skeleton (FastAPI / Pydantic style) that implements the 7-stage pipeline. It is deliberately modular so you can swap real classifiers, LLM judges, and RAG checkers later.python

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import re
import logging

logger = logging.getLogger(__name__)

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationResult(BaseModel):
    stage: str
    passed: bool
    score: float = 0.0          # 0 = safe, 100 = maximum risk
    reasons: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

class AppealCandidate(BaseModel):
    subject: str
    body: str
    cta: str
    tone: str
    donor_id: int
    capacity: float
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    raw_llm_output: Optional[str] = None

class GuardrailDecision(BaseModel):
    approved: bool
    risk_score: float
    risk_level: RiskLevel
    action: str                 # "auto_approve" | "human_review" | "reject" | "regenerate"
    stage_results: list[ValidationResult]
    final_reasons: list[str] = Field(default_factory=list)
# ---------------------------------------------------------------------------
# Individual stage functions (replace with real implementations)
# ---------------------------------------------------------------------------

def stage_structural(candidate: AppealCandidate) -> ValidationResult:
    reasons = []
    if not candidate.subject or len(candidate.subject) > 78:
        reasons.append("Subject missing or too long")
    if not candidate.body or len(candidate.body) < 50:
        reasons.append("Body too short")
    if not candidate.cta:
        reasons.append("Missing CTA")
    passed = len(reasons) == 0
    return ValidationResult(
        stage="structural",
        passed=passed,
        score=0.0 if passed else 80.0,
        reasons=reasons,
    )
FORBIDDEN_PATTERNS = [
    r"guaranteed?\s+(results?|impact|outcome)",
    r"act\s+now\s+or\s+.+\s+will\s+die",
    r"100%\s+of\s+your\s+donation",
    # add more...
]

def stage_rule_based(candidate: AppealCandidate) -> ValidationResult:
    text = f"{candidate.subject} {candidate.body}".lower()
    reasons = []
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, text, re.I):
            reasons.append(f"Forbidden pattern: {pat}")
    score = min(100.0, len(reasons) * 40.0)
    return ValidationResult(
        stage="rule_based",
        passed=len(reasons) == 0,
        score=score,
        reasons=reasons,
    )
def stage_safety_classifier(candidate: AppealCandidate) -> ValidationResult:
    """
    Placeholder for fast classifier or LLM-as-Judge.
    Replace with real call.
    """
    # Example: call a small toxicity model or LLM judge here
    toxicity_score = 0.05          # 0–1
    manipulation_score = 0.12
    combined = max(toxicity_score, manipulation_score) * 100
    reasons = []
    if toxicity_score > 0.3:
        reasons.append("Elevated toxicity")
    if manipulation_score > 0.25:
        reasons.append("Potential manipulative language")
    return ValidationResult(
        stage="safety_classifier",
        passed=combined < 30,
        score=combined,
        reasons=reasons,
        details={"toxicity": toxicity_score, "manipulation": manipulation_score},
    )
def stage_grounding(candidate: AppealCandidate, retrieved_chunks: list[str]) -> ValidationResult:
    """
    Simple claim check – replace with proper entailment / LLM judge.
    """
    if not retrieved_chunks:
        return ValidationResult(
            stage="grounding",
            passed=False,
            score=70.0,
            reasons=["No retrieved context provided"],
        )
    # In real system: extract claims and verify against chunks
    return ValidationResult(
        stage="grounding",
        passed=True,
        score=10.0,
        reasons=[],
        details={"chunks_used": len(retrieved_chunks)},
    )
def stage_brand_voice(candidate: AppealCandidate) -> ValidationResult:
    # Placeholder – embed and compare to approved examples
    return ValidationResult(stage="brand_voice", passed=True, score=5.0)
def stage_compliance(candidate: AppealCandidate) -> ValidationResult:
    reasons = []
    body_lower = candidate.body.lower()
    if "unsubscribe" not in body_lower and "opt out" not in body_lower:
        reasons.append("Missing unsubscribe language")
    # Physical address check would live here too
    score = 40.0 if reasons else 0.0
    return ValidationResult(
        stage="compliance",
        passed=len(reasons) == 0,
        score=score,
        reasons=reasons,
    )
# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class GuardrailPipeline:
    def __init__(self, high_value_threshold: float = 5000.0):
        self.high_value_threshold = high_value_threshold

    def validate(
        self,
        candidate: AppealCandidate,
        retrieved_chunks: list[str] | None = None,
    ) -> GuardrailDecision:
        retrieved_chunks = retrieved_chunks or []
        results: list[ValidationResult] = []

        # Stage order matters – cheap checks first
        results.append(stage_structural(candidate))
        if not results[-1].passed:
            return self._decide(results, candidate, short_circuit=True)

        results.append(stage_rule_based(candidate))
        results.append(stage_safety_classifier(candidate))
        results.append(stage_grounding(candidate, retrieved_chunks))
        results.append(stage_brand_voice(candidate))
        results.append(stage_compliance(candidate))

        return self._decide(results, candidate)

    def _decide(
        self,
        results: list[ValidationResult],
        candidate: AppealCandidate,
        short_circuit: bool = False,
    ) -> GuardrailDecision:
        total_score = max(r.score for r in results) if results else 0.0
        # Weighted average is also common; max is conservative
        all_reasons = [r for res in results for r in res.reasons]

        if short_circuit or total_score >= 70:
            action = "reject"
            level = RiskLevel.CRITICAL if total_score >= 90 else RiskLevel.HIGH
            approved = False
        elif total_score >= 35 or candidate.capacity >= self.high_value_threshold:
            action = "human_review"
            level = RiskLevel.MEDIUM
            approved = False
        else:
            action = "auto_approve"
            level = RiskLevel.LOW
            approved = True

        decision = GuardrailDecision(
            approved=approved,
            risk_score=total_score,
            risk_level=level,
            action=action,
            stage_results=results,
            final_reasons=all_reasons,
        )
        logger.info(
            "Guardrail decision",
            extra={
                "donor_id": candidate.donor_id,
                "action": action,
                "risk_score": total_score,
                "reasons": all_reasons,
            },
        )
        return decision
# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = GuardrailPipeline(high_value_threshold=5000.0)

    candidate = AppealCandidate(
        subject="Your support can change lives this month",
        body="Dear friend, thanks to donors like you we reached 12,000 families last year. Will you help us continue this work?",
        cta="Donate now",
        tone="inspiring",
        donor_id=42,
        capacity=250.0,
        retrieved_chunk_ids=["IR-2025-14", "CS-88"],
    )

    decision = pipeline.validate(candidate, retrieved_chunks=["...approved impact text..."])
    print(decision.model_dump_json(indent=2))

How to extend. Replace the placeholder stages with real model calls (toxicity model, LLM-as-Judge, entailment model).
Make stages async and run independent ones concurrently.
Persist every GuardrailDecision + original candidate for audit and later training data.
Add a regeneration path that feeds failure reasons back into the LLM prompt.