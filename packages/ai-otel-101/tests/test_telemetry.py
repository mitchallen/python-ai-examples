"""The provider wiring, checked without touching the global providers."""

from __future__ import annotations

from ai_otel_101.telemetry import configure_telemetry


def test_configure_telemetry_returns_usable_providers():
    telemetry = configure_telemetry("test-service", set_global=False)
    try:
        assert telemetry.tracer() is not None
        assert telemetry.meter() is not None
    finally:
        telemetry.shutdown()


def test_resource_carries_the_service_name():
    telemetry = configure_telemetry("test-service", set_global=False)
    try:
        resource = telemetry.tracer_provider.resource
        assert resource.attributes["service.name"] == "test-service"
    finally:
        telemetry.shutdown()
