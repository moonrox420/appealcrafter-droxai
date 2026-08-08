# AppealCrafter

**AI-powered fundraising appeal generation platform** — personalize donor outreach, enforce safety guardrails, and measure campaign performance with enterprise-grade reliability.

Built by **Dustin Hill / DroxAI**. Proprietary software — see [LICENSE](LICENSE).

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [RAG & Guardrails](#rag--guardrails)
- [ML & Experimentation](#ml--experimentation)
- [Observability](#observability)
- [Testing](#testing)
- [Deployment](#deployment)
- [Security & Compliance](#security--compliance)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

AppealCrafter is a production-grade platform that helps nonprofits generate personalized fundraising appeals at scale. It combines:

- **LLM-powered appeal generation** grounded in approved knowledge documents (RAG)
- **A mandatory 7-stage guardrail pipeline** ensuring every email is safe, compliant, and on-brand
- **ML-driven donor scoring** (propensity, capacity, engagement) for smarter targeting
- **A/B experimentation** to continuously improve conversion
- **Full delivery lifecycle tracking** with provider webhooks
- **Enterprise observability** — Prometheus, Grafana, OpenTelemetry, structured JSON logs

---

## Features

### Core Capabilities
- ✅ Donor ingestion with per-record validation
- ✅ Personalized appeal generation (LLM + RAG or template fallback)
- ✅ Real email delivery via SendGrid, Postmark, or AWS SES
- ✅ Delivery lifecycle tracking: queued → sent → delivered → opened → clicked → converted
- ✅ Provider webhooks with HMAC signature verification
- ✅ Suppression list + frequency capping
- ✅ Versioned templates with rollback and rich merge tags

### Phase 2 (Elite-Production-Grade)
- ✅ **7-Stage Guardrail Pipeline** — structural, safety (fast + LLM-as-Judge), rule/blocklist, factual grounding, brand voice, compliance, risk scoring
- ✅ **Full RAG Pipeline** — approved ingestion → semantic chunking → hybrid pgvector retrieval → citation-bearing context → constrained generation
- ✅ **ML Scoring** — RFM features, propensity/capacity/engagement scores, versioned model registry, drift detection, metric-gated promotion
- ✅ **A/B Experimentation** — sticky assignment, statistical z-tests, confidence intervals, promotion with audit trail
- ✅ **Multi-step Journeys** — conditional branching, delays, multi-channel
- ✅ **Feature Flags** — percentage-based rollout for controlled deploys
- ✅ **GDPR/CCPA Compliance** — export, anonymize, delete flows
- ✅ **PII Encryption** — AES-256-GCM field-level encryption at rest
- ✅ **Audit Logs** — immutable trail of administrative actions
- ✅ **Rate Limiting** — 429 + Retry-After
- ✅ **Async Bulk Jobs** — immediate job ID return
- ✅ **Redis Caching** — donor profiles, predictions, templates with hit-rate metrics
- ✅ **Automated Backups** — daily Postgres dumps to S3
- ✅ **Multi-tenancy** — tenant isolation with per-tenant templates/settings

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI (API)                        │
│  /donors  /generate-appeal  /campaigns  /templates          │
│  /knowledge  /experiments  /journeys  /feature-flags        │
│  /suppression  /reporting  /jobs  /compliance  /ml          │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────┐        ┌──────────────────────────┐
│   PostgreSQL + pgvector│        │        Redis 7           │
│  • Relational data    │        │  • Celery broker         │
│  • Vector embeddings  │        │  • Result backend        │
│  • HNSW hybrid search │        │  • Cache                 │
└──────────────┬────────┘        └────────────┬─────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Celery Workers + Beat                     │
│  • send_campaign_task      • retrain_model_task             │
│  • auto_send_appeals_task  • backup_database_task           │
│  • run_async_job_task      • monitor_queue_depth_task       │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│              Email Providers (SendGrid/Postmark/SES)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI, Pydantic v2, Pydantic Settings |
| **Database** | PostgreSQL 16 + pgvector |
| **ORM** | SQLAlchemy 2.0, Alembic migrations |
| **Async** | Celery 5, Redis 7 |
| **Auth** | JWT (access ≤ 60 min + refresh), bcrypt |
| **Email** | SendGrid, Postmark, AWS SES |
| **ML** | scikit-learn, sentence-transformers |
| **LLM** | OpenAI-compatible API (configurable) |
| **Observability** | Prometheus, Grafana, OpenTelemetry, Sentry |
| **Security** | AES-256-GCM, HMAC-SHA256, rate limiting |
| **Testing** | pytest, pytest-asyncio, pytest-cov |

---

## Getting Started

### Prerequisites
- Docker 24+ with Docker Compose v2
- Python 3.12+ (for local development)
- An email provider account (SendGrid/Postmark/SES) or use `log_only` in development

### Quick Start (Docker)

```bash
# 1. Clone the repository
git clone https://github.com/moonrox420/appealcrafter-droxai.git
cd appealcrafter-droxai

# 2. Create your environment file
cp .env.example .env   # then fill in required values

# 3. Build and start all services
docker compose up -d --build

# 4. Run database migrations
docker compose exec api alembic upgrade head

# 5. Verify health
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables (see Configuration below)

# 3. Run migrations
alembic upgrade head

# 4. Start the API
uvicorn app.main:app --reload

# 5. Start Celery workers (in separate terminals)
celery -A app.workers.celery_app worker --loglevel=info
celery -A app.workers.celery_app beat --loglevel=info
```

---

## Configuration

All configuration is loaded from environment variables (or a `.env` file). The app **refuses to start** if required secrets are missing.

### Required Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_HOST` | PostgreSQL host |
| `DATABASE_NAME` | Database name |
| `DATABASE_USER` | Database user |
| `DATABASE_PASSWORD` | Database password |
| `SECURITY_JWT_SECRET` | JWT signing secret (≥ 32 chars) |
| `EMAIL_FROM_ADDRESS` | Verified sender email |
| `EMAIL_PHYSICAL_ADDRESS` | Mailing address for CAN-SPAM |
| `EMAIL_WEBHOOK_SECRET` | Webhook signature secret (≥ 16 chars) |
| `REDIS_URL` | Redis connection URL |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECURITY_ENCRYPTION_KEY` | AES-256 key for PII encryption | disabled |
| `LLM_API_BASE_URL` | OpenAI-compatible API base URL | disabled |
| `LLM_API_KEY` | LLM API key | disabled |
| `LLM_RAG_ENABLED` | Enable RAG generation | `false` |
| `LLM_GUARDRAILS_ENABLED` | Enable guardrail pipeline | `true` |
| `OBSERVABILITY_OTLP_ENDPOINT` | OpenTelemetry exporter endpoint | disabled |
| `DATABASE_BACKUP_BUCKET` | S3 bucket for backups | disabled |
| `EMAIL_PROVIDER` | `sendgrid` / `postmark` / `ses` / `log_only` | `log_only` |
| `SECURITY_RATE_LIMIT_PER_MINUTE` | API rate limit | `120` |

---

## API Reference

Interactive docs are available at `http://localhost:8000/docs` (Swagger UI) and `/redoc`.

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/login` | Login, returns access + refresh tokens |
| POST | `/auth/refresh` | Exchange refresh token for new pair |
| POST | `/auth/users` | Create user (admin) |

### Donors
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/donors/ingest-donors` | Bulk ingest with per-record validation |
| GET | `/donors` | List active donors |
| DELETE | `/donors/{id}` | Soft-delete a donor |

### Appeals & Campaigns
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate-appeal` | Generate appeals + queue sends |
| POST | `/campaigns` | Create campaign |
| GET | `/campaigns` | List campaigns |

### Templates
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/templates` | Create versioned template |
| GET | `/templates` | List templates |
| POST | `/templates/{id}/versions` | Create new version |
| POST | `/templates/{id}/rollback/{version}` | Roll back to version |
| GET | `/templates/{id}/versions` | List versions |

### Knowledge & RAG
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/knowledge` | Create knowledge document |
| GET | `/knowledge` | List documents |
| POST | `/knowledge/{id}/ingest` | Chunk + embed for RAG |
| GET | `/knowledge/{id}` | Get document with chunks |

### Experiments
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/experiments` | Create A/B experiment |
| GET | `/experiments` | List experiments |
| POST | `/experiments/{id}/start` | Start experiment |
| GET | `/experiments/{id}/results` | Statistical results |
| POST | `/experiments/{id}/promote/{variant}` | Promote winner |

### Journeys
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/journeys` | Create multi-step journey |
| GET | `/journeys` | List journeys |

### Feature Flags
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/feature-flags` | Create flag (admin) |
| GET | `/feature-flags/{name}` | Evaluate flag |
| GET | `/feature-flags` | List flags |

### Suppression
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/suppression` | Add suppression entry |
| GET | `/suppression` | List entries |
| DELETE | `/suppression/{email}` | Remove entry |

### Reporting
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/reporting/campaigns/{id}` | Full campaign report |
| GET | `/reporting/deliveries` | Time-bucketed analytics |

### Jobs
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/jobs/{job_type}` | Create async job (returns ID) |
| GET | `/jobs/{id}` | Get job status |
| GET | `/jobs` | List jobs |

### Compliance (GDPR/CCPA)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/compliance/donors/{id}/export` | Export all donor data |
| POST | `/compliance/donors/{id}/anonymize` | Anonymize donor data |
| DELETE | `/compliance/donors/{id}` | Delete donor data |

### ML
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ml/train/{model}/{version}` | Train model (admin) |
| GET | `/ml/models/{name}/drift` | Check prediction drift |

### Webhooks
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/webhooks/email-events` | Provider delivery events |

### Health & Metrics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness (DB + broker) |
| GET | `/metrics` | Prometheus metrics |

---

## RAG & Guardrails

### RAG Pipeline
1. **Approved ingestion** — only `is_approved=True` documents are chunked and embedded
2. **Semantic chunking** — 200-word chunks with 50-word overlap
3. **Hybrid retrieval** — pgvector cosine similarity + keyword search merged
4. **Citation-bearing context** — retrieved chunks include source metadata
5. **Constrained generation** — LLM instructed to use ONLY approved chunks; unsupported claims rejected
6. **Template fallback** — permanent high-quality fallback when RAG is unavailable

### 7-Stage Guardrail Pipeline
Every LLM output passes through all 7 stages:

| Stage | Purpose |
|-------|---------|
| 1. Structural | Subject ≤ 78 chars, body ≥ 50 chars, CTA present |
| 2. Safety | Fast toxicity/manipulation classifier + LLM-as-Judge on borderline |
| 3. Rule/Blocklist | Forbidden patterns (guarantees, pressure tactics, etc.) |
| 4. Grounding | Claims verified against retrieved RAG chunks |
| 5. Brand Voice | Tone alignment, word count limits |
| 6. Compliance | Unsubscribe language + physical address present |
| 7. Risk Scoring | Aggregate risk + high-value donor routing |

**Routing rules:**
- Score ≥ 70 → **Reject**
- Score ≥ 35 OR capacity ≥ $5,000 → **Human Review**
- Otherwise → **Auto-approve**

**Circuit breaker:** If validation failure rate exceeds threshold, the pipeline falls back to templates automatically.

---

## ML & Experimentation

### Donor Scoring
- **RFM features**: recency (days since last donation), frequency (donation count), monetary (total donated)
- **Engagement signals**: open rate, click rate from delivery history
- **Propensity score**: 0–1 likelihood of donating again
- **Capacity score**: estimated donation capacity
- **Predictions cached** with safe fallback

### Model Registry
- Versioned models with train/validate/test split (70/15/15)
- Promotion only when AUC ≥ 0.65 and precision ≥ 0.60
- Scheduled weekly retraining via Celery beat
- Drift detection on prediction distributions

### A/B Experiments
- Sticky assignment via deterministic hashing
- Statistical z-test with p-values and 95% confidence intervals
- Full lifecycle: draft → running → completed → promoted
- Promotion with audit trail

---

## Observability

### Prometheus Metrics
- HTTP request count, latency, error rate
- Celery queue depth
- Delivery rates by status
- Prediction latency
- Guardrail decisions by action
- Model promotions
- LLM cost tracking

### Grafana
- Pre-configured in docker-compose at `http://localhost:3000`
- Dashboards for API, workers, campaigns, ML

### OpenTelemetry
- Set `OBSERVABILITY_OTLP_ENDPOINT` to enable distributed tracing
- Traces cover request → generation → send

### Structured Logging
- All logs are JSON with `trace_id` correlation
- Written to stderr (never stdout)
- No PII or secrets in logs

---

## Testing

```bash
# Run all tests (requires PostgreSQL with test role)
python -m pytest

# Run with coverage
python -m pytest --cov=app --cov-report=term-missing

# Run Phase 2 tests only
python -m pytest tests/test_phase2.py -v
```

### Test Coverage
- **Phase 1**: integration flow (ingest → generate → queue → status), security, suppression
- **Phase 2**: guardrails (approve/reject/human-review/persist), RAG chunking, experiments (sticky assignment, stats), compliance (export/anonymize/delete), ML (predictions, RFM), suppression API, template versioning, rate limiting

---

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full deployment guide including:

- Docker Compose deployment
- Environment configuration
- Scheduled task reference
- Operations runbook (restart workers, view DLQ, rotate secrets, restore backups)
- Security hardening
- GDPR/CCPA compliance procedures

---

## Security & Compliance

- **JWT auth** — access ≤ 60 min, refresh ≤ 7 days, restricted algorithms
- **PII encryption** — AES-256-GCM field-level encryption at rest
- **Rate limiting** — 429 + Retry-After
- **Security headers** — HSTS, X-Content-Type-Options, X-Frame-Options, CSP
- **Webhook verification** — HMAC-SHA256 signatures
- **Audit logs** — actor, timestamp, before/after state for admin actions
- **GDPR/CCPA** — export, anonymize, delete flows
- **CAN-SPAM** — unsubscribe + physical address on every email
- **Suppression** — unsubscribed addresses never receive mail

---

## Project Structure

```
appealcrafter-droxai/
├── alembic/                  # Database migrations
│   └── versions/
│       ├── 0001_initial_schema.py
│       └── 0002_phase2_schema.py
├── app/
│   ├── api/                  # API routers
│   │   ├── appeals.py        # Appeal generation
│   │   ├── auth.py           # Authentication
│   │   ├── campaigns.py      # Campaign management
│   │   ├── donors.py         # Donor management
│   │   ├── health.py         # Health checks
│   │   ├── metrics.py        # Basic metrics
│   │   ├── operations.py     # Suppression, reporting, jobs, compliance, ML
│   │   ├── platform.py       # Knowledge, templates, experiments, journeys, flags
│   │   └── webhooks.py       # Provider webhooks
│   ├── core/                 # Core infrastructure
│   │   ├── cache.py          # Redis caching
│   │   ├── config.py         # Pydantic settings
│   │   ├── encryption.py     # PII encryption
│   │   ├── logging.py        # Structured JSON logging
│   │   ├── metrics.py        # Prometheus metrics
│   │   ├── rate_limit.py     # Rate limiting
│   │   └── security.py       # JWT + RBAC
│   ├── db/                   # Database session
│   ├── models/               # SQLAlchemy ORM models
│   ├── schemas/              # Pydantic schemas
│   ├── services/             # Business logic
│   │   ├── appeal.py         # Appeal generation
│   │   ├── async_jobs.py     # Async job tracking
│   │   ├── audit.py          # Audit logging
│   │   ├── compliance.py     # GDPR/CCPA
│   │   ├── delivery.py       # Delivery lifecycle
│   │   ├── donor.py          # Donor management
│   │   ├── email_provider.py # ESP abstraction
│   │   ├── experiments.py    # A/B testing
│   │   ├── feature_flags.py  # Feature flags
│   │   ├── guardrails.py     # 7-stage guardrail pipeline
│   │   ├── journeys.py       # Multi-step journeys
│   │   ├── llm_generation.py # Constrained LLM generation
│   │   ├── ml.py             # RFM + predictions
│   │   ├── rag.py            # RAG pipeline
│   │   ├── reporting.py      # Campaign reporting
│   │   ├── suppression.py    # Suppression + frequency cap
│   │   └── template.py       # Versioned templates
│   └── workers/              # Celery tasks
│       ├── celery_app.py     # Celery config + beat schedule
│       └── tasks.py          # Task definitions
├── tests/                    # Test suite
│   ├── conftest.py
│   ├── test_integration.py
│   ├── test_phase2.py
│   ├── test_security.py
│   └── test_suppression.py
├── docs/
│   └── DEPLOYMENT.md         # Deployment guide
├── docker-compose.yml        # All services
├── Dockerfile
├── prometheus.yml             # Prometheus config
├── requirements.txt
├── LICENSE                    # Proprietary license
└── README.md
```

---

## License

**Proprietary Software** — Copyright (c) 2026 **Dustin Hill, DroxAI**. All Rights Reserved.

This software is the exclusive property of Dustin Hill and DroxAI. It may not be modified, distributed, or used without verified written permission from the owner. See [LICENSE](LICENSE) for full terms.