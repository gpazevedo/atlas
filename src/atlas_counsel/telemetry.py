"""OpenTelemetry setup.

Silent no-op when OTEL is not installed or OTLP endpoint is not configured.
Otherwise sets up a BatchSpanProcessor exporting to OTLP and returns a tracer
for the application.

Env vars:
  OTEL_EXPORTER_OTLP_ENDPOINT  — required; e.g. http://localhost:4318
  OTEL_SERVICE_NAME            — defaults to "atlas-counsel"
"""

from __future__ import annotations

import logging
import os
from functools import cache
from typing import Any

logger = logging.getLogger(__name__)

_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _OTEL_AVAILABLE = True
except ImportError:
    pass


def _endpoint() -> str | None:
    return os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")


def _service_name() -> str:
    return os.environ.get("OTEL_SERVICE_NAME", "atlas-counsel")


@cache
def _provider() -> trace.TracerProvider | None:
    if not _OTEL_AVAILABLE:
        logger.debug("OTEL SDK not installed; tracing disabled")
        return None
    endpoint = _endpoint()
    if not endpoint:
        logger.debug("OTEL_EXPORTER_OTLP_ENDPOINT not set; tracing disabled")
        return None
    resource = Resource(attributes={SERVICE_NAME: _service_name()})
    exporter = OTLPSpanExporter(endpoint=endpoint)
    processor = BatchSpanProcessor(exporter)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    logger.info("OTEL tracing enabled: service=%s endpoint=%s",
                _service_name(), endpoint)
    return provider


def get_tracer(name: str = "atlas-counsel") -> Any:
    """Return a tracer. Returns a no-op if OTEL is not configured."""
    provider = _provider()
    if provider is None:
        return trace.NoOpTracer() if _OTEL_AVAILABLE else _NoOpTracer()
    return trace.get_tracer(name)


def instrument_fastapi(app) -> None:
    """Apply FastAPI auto-instrumentation. No-op if OTEL not configured."""
    if _provider() is None:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        logger.warning("opentelemetry-instrumentation-fastapi not installed")


def shutdown() -> None:
    """Flush pending spans. Call at process exit."""
    provider = _provider()
    if provider is not None:
        provider.shutdown()


class _NoOpTracer:
    """Minimal no-op for when the OTEL SDK isn't installed at all."""
    def start_as_current_span(self, *args, **kwargs):
        return _NoOpSpan()

    def start_span(self, *args, **kwargs):
        return _NoOpSpan()


class _NoOpSpan:
    """No-op span context manager."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass

    def record_exception(self, *args, **kwargs):
        pass
