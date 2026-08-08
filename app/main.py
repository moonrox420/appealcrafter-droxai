"""AppealCrafter FastAPI application entry point."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import appeals, auth, campaigns, donors, health, metrics, webhooks
from app.api.operations import (
    compliance_router,
    jobs_router,
    ml_router,
    reporting_router,
    suppression_router,
)
from app.api.platform import (
    experiments_router,
    feature_flags_router,
    journeys_router,
    knowledge_router,
    templates_router,
)
from app.core.config import get_settings
from app.core.logging import configure_logging, set_trace_id
from app.core.metrics import prometheus_metrics_endpoint, prometheus_middleware

settings = get_settings()
configure_logging(settings.log_level)

if settings.sentry_dsn:
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment.value,
        traces_sample_rate=0.1,
    )

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.observability.prometheus_enabled:
    app.middleware("http")(prometheus_middleware)

if settings.observability.otlp_endpoint:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": settings.observability.otlp_service_name})
    tracer_provider = TracerProvider(resource=resource)
    otlp_exporter = OTLPSpanExporter(endpoint=settings.observability.otlp_endpoint)
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(tracer_provider)
    FastAPIInstrumentor.instrument_app(app)


@app.middleware("http")
async def add_security_headers_and_trace_id(request: Request, call_next):
    """Add security headers and trace_id correlation to every request."""
    trace_id = str(uuid.uuid4())
    set_trace_id(trace_id)
    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Trace-Id"] = trace_id
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    logger.info(
        "HTTP request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic 500 response without leaking internals."""
    logger.error(
        "Unhandled exception",
        extra={"path": request.url.path, "error": str(exc)},
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


app.include_router(auth.router)
app.include_router(donors.router)
app.include_router(appeals.router)
app.include_router(campaigns.router)
app.include_router(webhooks.router)
app.include_router(health.router)
app.include_router(metrics.router)

app.include_router(knowledge_router)
app.include_router(templates_router)
app.include_router(experiments_router)
app.include_router(journeys_router)
app.include_router(feature_flags_router)
app.include_router(suppression_router)
app.include_router(reporting_router)
app.include_router(jobs_router)
app.include_router(compliance_router)
app.include_router(ml_router)

app.add_api_route("/metrics", prometheus_metrics_endpoint, methods=["GET"])