"""Structured JSON logging with trace_id correlation.

Every log record is emitted as valid JSON with a trace_id field for
end-to-end request and task tracing. No output is written to stdout.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from app.core.config import LogLevel

trace_id_context: ContextVar[str] = ContextVar("trace_id", default="")


def set_trace_id(trace_id: str) -> None:
    """Set the trace_id for the current async context."""
    trace_id_context.set(trace_id)


def get_trace_id() -> str:
    """Return the current trace_id or generate a new one."""
    current = trace_id_context.get()
    if not current:
        current = str(uuid.uuid4())
        trace_id_context.set(current)
    return current


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the log record to a JSON string."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_trace_id(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: LogLevel) -> None:
    """Configure the root logger with JSON formatting."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level.value)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(stream_handler)