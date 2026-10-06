"""Prometheus metrics collection for the AppealCrafter API.

Exposes request latency, error rates, queue depth, delivery rates, and
prediction latency metrics for Grafana dashboards and alerting.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REQUEST_COUNT = Counter(
    "appealcrafter_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "appealcrafter_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

ERROR_COUNT = Counter(
    "appealcrafter_http_errors_total",
    "Total HTTP 5xx errors",
    ["method", "path"],
)

QUEUE_DEPTH = Gauge(
    "appealcrafter_celery_queue_depth",
    "Current Celery queue depth",
    ["queue"],
)

DELIVERY_RATE = Counter(
    "appealcrafter_deliveries_total",
    "Total deliveries by status",
    ["status"],
)

PREDICTION_LATENCY = Histogram(
    "appealcrafter_prediction_duration_seconds",
    "ML prediction latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
)

GUARDRAIL_DECISIONS = Counter(
    "appealcrafter_guardrail_decisions_total",
    "Guardrail decisions by action",
    ["action"],
)

MODEL_PROMOTIONS = Counter(
    "appealcrafter_model_promotions_total",
    "Total model promotions",
    ["model_name"],
)

EXPERIMENT_EVENTS = Counter(
    "appealcrafter_experiment_events_total",
    "Experiment events by type",
    ["event_type"],
)

LLM_COST_DOLLARS = Counter(
    "appealcrafter_llm_cost_dollars_total",
    "Cumulative LLM API cost in dollars",
    ["model"],
)


async def prometheus_metrics_endpoint() -> Response:
    """Return Prometheus metrics in text format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def prometheus_middleware(request: Request, call_next: Callable) -> Response:
    """Instrument every HTTP request with Prometheus metrics."""
    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_seconds = time.perf_counter() - start_time
    path = request.url.path
    method = request.method
    status_code = str(response.status_code)
    REQUEST_COUNT.labels(method=method, path=path, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed_seconds)
    if response.status_code >= 500:
        ERROR_COUNT.labels(method=method, path=path).inc()
    return response
