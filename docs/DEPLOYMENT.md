# AppealCrafter Deployment Guide

## Architecture Overview

AppealCrafter is a FastAPI-based fundraising appeal platform with:

- **PostgreSQL 16 + pgvector** for relational data and vector embeddings
- **Redis 7** for Celery broker, result backend, and caching
- **Celery workers + beat** for async processing and scheduled tasks
- **Prometheus + Grafana** for metrics and dashboards
- **OpenTelemetry** for distributed tracing
- **SendGrid / Postmark / SES** for email delivery

## Prerequisites

- Docker 24+ with Docker Compose v2
- Environment file (`.env`) with all required secrets
- AWS CLI configured for S3 backups (optional)

## Environment Variables

### Required
```
DATABASE_HOST=postgres
DATABASE_NAME=appealcrafter
DATABASE_USER=appealcrafter
DATABASE_PASSWORD=<strong-password>
SECURITY_JWT_SECRET=<32+ char random string>
EMAIL_FROM_ADDRESS=no-reply@example.org
EMAIL_PHYSICAL_ADDRESS=<your mailing address>
EMAIL_WEBHOOK_SECRET=<16+ char random string>
REDIS_URL=redis://redis:6379/0
```

### Optional
```
SECURITY_ENCRYPTION_KEY=<32+ char key for PII encryption>
LLM_API_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=<your LLM API key>
LLM_RAG_ENABLED=true
OBSERVABILITY_OTLP_ENDPOINT=http://otel-collector:4318
DATABASE_BACKUP_BUCKET=<s3-bucket-name>
GRAFANA_ADMIN_PASSWORD=<admin-password>
```

## Deployment

### 1. Build and start

```bash
docker compose up -d --build
```

### 2. Run migrations

```bash
docker compose exec api alembic upgrade head
```

### 3. Verify health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### 4. Access services

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

## Scheduled Tasks (Celery Beat)

| Task | Schedule | Purpose |
|------|----------|---------|
| auto-send-appeals-daily | Daily 09:00 UTC | Generate and queue appeals |
| retrain-model-weekly | Sunday 02:00 UTC | Retrain propensity model with promotion gates |
| backup-database-daily | Daily 01:30 UTC | Postgres backup to S3 |
| monitor-queue-depth | Every 5 min | Expose queue depth to Prometheus |

## Operations

### Restart workers
```bash
docker compose restart worker beat
```

### View dead-letter queue
```bash
docker compose exec redis redis-cli LLEN celery@dead
```

### Rotate secrets
1. Update `.env` with new values
2. `docker compose up -d --force-recreate api worker beat`

### Restore from backup
```bash
# Download backup from S3
aws s3 cp s3://<bucket>/postgres-backups/<file>.dump /tmp/restore.dump
# Restore
docker compose exec -T postgres pg_restore -U appealcrafter -d appealcrafter /tmp/restore.dump
```

## Security

- All secrets loaded from environment only; app refuses to start on missing secrets
- JWT access tokens ≤ 60 min, refresh ≤ 7 days
- PII fields encrypted at rest with AES-256-GCM when `SECURITY_ENCRYPTION_KEY` is set
- Rate limiting returns 429 with Retry-After
- Security headers: HSTS, X-Content-Type-Options, X-Frame-Options, CSP
- Webhook signatures verified with HMAC-SHA256
- Audit logs record actor, timestamp, before/after state for admin actions

## Observability

### Prometheus metrics
- HTTP request count, latency, error rate
- Celery queue depth
- Delivery rates by status
- Prediction latency
- Guardrail decisions by action
- Model promotions
- LLM cost tracking

### Grafana dashboards
- API: request rate, latency, error rate
- Workers: queue depth, task throughput
- Campaigns: delivery lifecycle metrics
- ML: prediction distribution, drift alerts

### OpenTelemetry
- Configure `OBSERVABILITY_OTLP_ENDPOINT` to enable distributed tracing
- Traces cover request → generation → send

## GDPR/CCPA Compliance

- Export: `GET /compliance/donors/{id}/export`
- Anonymize: `POST /compliance/donors/{id}/anonymize`
- Delete: `DELETE /compliance/donors/{id}`

## Runbook

### High error rate
1. Check Prometheus for error rate spike
2. Check Sentry for exception details
3. Check worker logs for provider failures
4. Verify email provider credentials

### Queue backlog
1. Check queue depth metric
2. Scale workers: `docker compose up -d --scale worker=8`
3. Check for stuck tasks in dead-letter queue

### Bounce spike
1. Check delivery metrics for bounce rate
2. Review suppression list for affected domains
3. Contact email provider if needed

### Model latency
1. Check prediction latency metric
2. Verify LLM API endpoint health
3. Consider reducing batch size or model size