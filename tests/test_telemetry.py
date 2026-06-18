"""Tests for the OpenTelemetry setup module.

No real OTLP endpoint is required — the module is a silent no-op when
the endpoint is not configured. We test that the no-op path works and
that the module is importable.
"""

import importlib
import os

import pytest


class TestTelemetryNoOp:
    """All tests run without OTEL_EXPORTER_OTLP_ENDPOINT — covers the no-op path."""

    def test_import(self):
        """telemetry module is importable."""
        from atlas_counsel import telemetry
        assert hasattr(telemetry, "get_tracer")

    def test_get_tracer_returns_noop(self):
        """Tracer is a no-op when no endpoint is configured."""
        from atlas_counsel.telemetry import get_tracer
        tracer = get_tracer()
        # The tracer should exist and have start_as_current_span
        assert hasattr(tracer, "start_as_current_span")

    def test_tracer_span_context_manager(self):
        """No-op span works as a context manager without crashing."""
        from atlas_counsel.telemetry import get_tracer
        tracer = get_tracer()
        with tracer.start_as_current_span("test-span"):
            pass

    def test_noop_span_set_attribute(self):
        """No-op span accepts set_attribute."""
        from atlas_counsel.telemetry import get_tracer
        tracer = get_tracer()
        with tracer.start_as_current_span("test-span") as span:
            span.set_attribute("key", "value")

    def test_instrument_fastapi_noop(self):
        """instrument_fastapi is safe to call when OTEL is off."""
        from atlas_counsel.telemetry import instrument_fastapi
        # Should not raise — just returns silently
        instrument_fastapi(None)  # object is fine since it returns early

    def test_shutdown_noop(self):
        """shutdown is safe to call when OTEL is off."""
        from atlas_counsel.telemetry import shutdown
        shutdown()

    def test_tenants_uses_telemetry(self):
        """TenantRegistry._create uses get_tracer without crashing."""
        from atlas_counsel.service.tenants import TenantRegistry
        registry = TenantRegistry()
        tenant = registry.get("acme")
        assert tenant.tenant_id == "acme"

    def test_core_uses_telemetry(self):
        """CounselService ask/resume use OTEL spans without crashing."""
        from atlas_counsel.service.tenants import TenantRegistry
        from atlas_counsel.service.core import CounselService
        registry = TenantRegistry()
        svc = CounselService()
        # Replace the registry so we share the same TenantRegistry
        svc._registry = registry
        result = svc.ask("test question", tenant_id="oteltest")
        assert result.status is not None
