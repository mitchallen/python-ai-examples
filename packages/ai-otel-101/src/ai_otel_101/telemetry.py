"""Provider setup: where the spans and metrics actually go.

Real services configure this once at startup. The default here is the console
exporter so the example prints its own telemetry with no collector running;
set ``OTEL_EXPORTER_OTLP_ENDPOINT`` (and install the ``otlp`` extra) to ship it
to a real backend instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

SERVICE_NAME = "service.name"


@dataclass
class Telemetry:
    """The two providers, kept so the demo can flush them before exiting."""

    tracer_provider: TracerProvider
    meter_provider: MeterProvider

    def tracer(self, name: str = "ai-otel-101") -> trace.Tracer:
        return self.tracer_provider.get_tracer(name)

    def meter(self, name: str = "ai-otel-101") -> metrics.Meter:
        return self.meter_provider.get_meter(name)

    def shutdown(self) -> None:
        """Flush both pipelines. Batch exporters drop data without this."""
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()


def configure_telemetry(
    service_name: str = "ai-otel-101",
    *,
    set_global: bool = True,
    export_interval_ms: int = 5_000,
) -> Telemetry:
    """Build tracer and meter providers wired to console (or OTLP) exporters."""
    resource = Resource.create({SERVICE_NAME: service_name})
    span_exporter, metric_exporter = _exporters()

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                metric_exporter, export_interval_millis=export_interval_ms
            )
        ],
    )

    if set_global:
        # Global providers can only be set once per process, which is why the
        # instrumentation takes explicit tracer/meter arguments -- the tests
        # build their own providers and never touch the globals.
        trace.set_tracer_provider(tracer_provider)
        metrics.set_meter_provider(meter_provider)

    return Telemetry(tracer_provider=tracer_provider, meter_provider=meter_provider)


def _exporters():
    """OTLP when an endpoint is configured and the extra is installed, else console."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return ConsoleSpanExporter(), ConsoleMetricExporter()

    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError:  # pragma: no cover - depends on the optional extra
        print(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but the OTLP exporter is not "
            "installed; falling back to the console. Install the extra with "
            "`uv add --package ai-otel-101 'ai-otel-101[otlp]'`."
        )
        return ConsoleSpanExporter(), ConsoleMetricExporter()

    return OTLPSpanExporter(), OTLPMetricExporter()
