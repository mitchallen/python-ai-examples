"""Everything the example needs, in one file you can copy into your own project.

The contrast with ai-otel-101: there, a wrapper object owns the OpenAI call.
Here the call stays in *your* code and telemetry wraps around it as a context
manager::

    with telemetry.chat(model="gpt-4o-mini") as observed:
        response = client.chat.completions.create(model=..., messages=...)
        observed.record(response)

That shape is what you want when the call site is already doing something
interesting -- retries, fallbacks between providers, streaming -- and you would
rather not route all of it through a wrapper's signature.

Nothing here imports from another package in this repo; the only dependencies
are ``openai`` and the OpenTelemetry API/SDK.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlparse

from openai import OpenAI
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind

INSTRUMENTATION_NAME = "ai-otel-102"
INSTRUMENTATION_VERSION = "0.1.0"

DEFAULT_MODEL = "gpt-4o-mini"
PIRATE_SYSTEM_PROMPT = (
    "You are a pirate. Answer every message in salty pirate speak, "
    "and keep it to a sentence or two."
)

# --- OpenTelemetry GenAI semantic conventions -------------------------------
# Written out rather than imported: these names are the contract with whatever
# backend reads the data, and an OTel backend can only chart "tokens by model"
# because everyone spells them the same way.
OPERATION_NAME = "gen_ai.operation.name"
OPERATION_CHAT = "chat"
SYSTEM = "gen_ai.system"  # renamed to gen_ai.provider.name; both are emitted
PROVIDER_NAME = "gen_ai.provider.name"
PROVIDER_OPENAI = "openai"
PROVIDER_OLLAMA = "ollama"
PROVIDER_AZURE_OPENAI = "azure.ai.openai"
REQUEST_MODEL = "gen_ai.request.model"
REQUEST_TEMPERATURE = "gen_ai.request.temperature"
REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
RESPONSE_ID = "gen_ai.response.id"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ERROR_TYPE = "error.type"
METRIC_TOKEN_USAGE = "gen_ai.client.token.usage"
METRIC_OPERATION_DURATION = "gen_ai.client.operation.duration"
TOKEN_TYPE = "gen_ai.token.type"
TOKEN_TYPE_INPUT = "input"
TOKEN_TYPE_OUTPUT = "output"


# --- the chat bits ----------------------------------------------------------


_KNOWN_PROVIDER_HOSTS = {
    "api.openai.com": PROVIDER_OPENAI,
    "localhost:11434": PROVIDER_OLLAMA,
    "127.0.0.1:11434": PROVIDER_OLLAMA,
    "[::1]:11434": PROVIDER_OLLAMA,
}


def provider_from_base_url(base_url: Any) -> str:
    """Name the provider from the endpoint the client actually talks to.

    The OpenAI SDK follows ``base_url``/``OPENAI_BASE_URL``, so a hardcoded
    ``"openai"`` is wrong the moment the call goes to Ollama or Azure -- and
    wrong in the attribute a cost dashboard groups by.
    """
    if not base_url:
        return PROVIDER_OPENAI

    host = (urlparse(str(base_url)).netloc or str(base_url)).lower()
    if host in _KNOWN_PROVIDER_HOSTS:
        return _KNOWN_PROVIDER_HOSTS[host]

    hostname = host.rsplit(":", 1)[0] if not host.endswith("]") else host
    if "ollama" in hostname:
        return PROVIDER_OLLAMA
    if hostname.endswith(".openai.azure.com"):
        return PROVIDER_AZURE_OPENAI
    if hostname.endswith(".openai.com"):
        return PROVIDER_OPENAI
    return host


def provider_for(client: Any) -> str:
    """The provider a constructed client points at."""
    return provider_from_base_url(getattr(client, "base_url", None))


def create_client() -> OpenAI:
    """Construct the OpenAI client; the key comes from ``OPENAI_API_KEY``."""
    return OpenAI()


def resolve_model(model: str | None = None) -> str:
    return model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def build_conversation(user_message: str = "Hello") -> list[dict[str, str]]:
    """The hard-coded conversation: pirate system prompt plus one user turn."""
    return [
        {"role": "system", "content": PIRATE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute or key lookup, so stubs and real SDK objects both work."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# --- the telemetry bits -----------------------------------------------------


class Observation:
    """Handed to the ``with`` body; call :meth:`record` with the response.

    If you never call it -- an early return, a cached hit, a failure -- the
    span still closes cleanly, just without usage data. Instrumentation that
    forces you to restructure your code does not survive contact with a real
    call site.
    """

    def __init__(self, span: trace.Span, attributes: dict[str, Any]) -> None:
        self.span = span
        self.attributes = attributes
        self.response: Any = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def record(self, response: Any) -> Any:
        """Copy response id, model, finish reasons and token counts onto the span."""
        self.response = response

        for name, value in (
            (RESPONSE_ID, _get(response, "id")),
            (RESPONSE_MODEL, _get(response, "model")),
        ):
            if value is not None:
                self.span.set_attribute(name, value)

        finish_reasons = [
            reason
            for reason in (
                _get(choice, "finish_reason") for choice in _get(response, "choices", [])
            )
            if reason is not None
        ]
        if finish_reasons:
            self.span.set_attribute(RESPONSE_FINISH_REASONS, finish_reasons)

        usage = _get(response, "usage")
        if usage is not None:
            # OpenAI says prompt/completion; the conventions say input/output.
            self.input_tokens = _get(usage, "prompt_tokens")
            self.output_tokens = _get(usage, "completion_tokens")
            if self.input_tokens is not None:
                self.span.set_attribute(USAGE_INPUT_TOKENS, self.input_tokens)
            if self.output_tokens is not None:
                self.span.set_attribute(USAGE_OUTPUT_TOKENS, self.output_tokens)

        return response

    def text(self) -> str:
        """Convenience for the demo: the first choice's content."""
        choices = _get(self.response, "choices", []) if self.response else []
        if not choices:
            return ""
        return _get(_get(choices[0], "message"), "content") or ""


class ChatTelemetry:
    """Owns the tracer and the two instruments. Build one, keep it around.

    Instruments are created once here rather than per call, which is the whole
    reason this is a class and not a bare function.
    """

    def __init__(
        self,
        *,
        tracer: trace.Tracer | None = None,
        meter: metrics.Meter | None = None,
        provider: str | None = None,
    ) -> None:
        self._provider = provider or PROVIDER_OPENAI
        self._tracer = tracer or trace.get_tracer(
            INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION
        )
        meter = meter or metrics.get_meter(
            INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION
        )
        self._token_usage = meter.create_histogram(
            name=METRIC_TOKEN_USAGE,
            unit="{token}",
            description="Number of input and output tokens used.",
        )
        self._duration = meter.create_histogram(
            name=METRIC_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration.",
        )

    @contextmanager
    def chat(
        self,
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: str | None = None,
    ) -> Iterator[Observation]:
        """Open a span around a chat call you make yourself.

        ``provider`` overrides the instance default per call, which is what a
        call site that falls back from one provider to another needs.
        """
        provider = provider or self._provider
        attributes: dict[str, Any] = {
            OPERATION_NAME: OPERATION_CHAT,
            SYSTEM: provider,
            PROVIDER_NAME: provider,
            REQUEST_MODEL: model,
        }
        if temperature is not None:
            attributes[REQUEST_TEMPERATURE] = temperature
        if max_tokens is not None:
            attributes[REQUEST_MAX_TOKENS] = max_tokens

        started = time.perf_counter()
        with self._tracer.start_as_current_span(
            f"{OPERATION_CHAT} {model}",
            kind=SpanKind.CLIENT,
            attributes=attributes,
        ) as span:
            observation = Observation(span, attributes)
            try:
                yield observation
            except Exception as exc:
                # The span context manager records the exception and sets
                # ERROR status; add the low-cardinality error.type that the
                # latency metric is keyed on, then let it propagate.
                error_type = type(exc).__qualname__
                span.set_attribute(ERROR_TYPE, error_type)
                self._duration.record(
                    time.perf_counter() - started,
                    {**attributes, ERROR_TYPE: error_type},
                )
                raise
            self._record_metrics(observation, time.perf_counter() - started)

    def _record_metrics(self, observation: Observation, elapsed: float) -> None:
        attributes = dict(observation.attributes)
        response_model = _get(observation.response, "model")
        if response_model is not None:
            # Billing follows the model that answered, not the one requested.
            attributes[RESPONSE_MODEL] = response_model

        for count, token_type in (
            (observation.input_tokens, TOKEN_TYPE_INPUT),
            (observation.output_tokens, TOKEN_TYPE_OUTPUT),
        ):
            if count is not None:
                self._token_usage.record(count, {**attributes, TOKEN_TYPE: token_type})

        self._duration.record(elapsed, attributes)


# --- provider setup ---------------------------------------------------------


@dataclass
class Telemetry:
    """The providers, kept so the demo can flush them before exiting."""

    tracer_provider: TracerProvider
    meter_provider: MeterProvider

    def tracer(self, name: str = INSTRUMENTATION_NAME) -> trace.Tracer:
        return self.tracer_provider.get_tracer(name)

    def meter(self, name: str = INSTRUMENTATION_NAME) -> metrics.Meter:
        return self.meter_provider.get_meter(name)

    def chat_telemetry(self, provider: str | None = None) -> ChatTelemetry:
        return ChatTelemetry(
            tracer=self.tracer(), meter=self.meter(), provider=provider
        )

    def shutdown(self) -> None:
        """Flush both pipelines. Batch exporters drop data without this."""
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()


def configure_telemetry(
    service_name: str = INSTRUMENTATION_NAME,
    *,
    set_global: bool = True,
    export_interval_ms: int = 5_000,
) -> Telemetry:
    """Wire up tracer and meter providers against console (or OTLP) exporters."""
    resource = Resource.create({"service.name": service_name})
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
        # Globals can only be set once per process, which is why ChatTelemetry
        # takes explicit tracer/meter arguments -- the tests build their own.
        trace.set_tracer_provider(tracer_provider)
        metrics.set_meter_provider(meter_provider)

    return Telemetry(tracer_provider=tracer_provider, meter_provider=meter_provider)


def _exporters() -> tuple[Any, Any]:
    """OTLP when an endpoint is set and the extra is installed, else console."""
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
            "installed; falling back to the console."
        )
        return ConsoleSpanExporter(), ConsoleMetricExporter()

    return OTLPSpanExporter(), OTLPMetricExporter()


def ask_pirate(
    user_message: str = "Hello",
    *,
    client: Any = None,
    telemetry: ChatTelemetry | None = None,
    model: str | None = None,
    messages: Sequence[Mapping[str, str]] | None = None,
) -> str:
    """The whole example in one call: instrumented, hard-coded, self-contained."""
    client = client or create_client()
    telemetry = telemetry or ChatTelemetry(provider=provider_for(client))
    model = resolve_model(model)
    conversation = list(messages) if messages else build_conversation(user_message)

    with telemetry.chat(model) as observed:
        observed.record(
            client.chat.completions.create(model=model, messages=conversation)
        )
        return observed.text()
