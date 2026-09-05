"""A thin OpenTelemetry wrapper around ``client.chat.completions.create``.

The wrapper does four things per call, which is the whole job of GenAI
instrumentation:

1. Opens a CLIENT span named ``chat <model>``.
2. Tags it with request attributes before the call and response attributes
   (id, model, finish reasons, **token usage**) after it.
3. Records two metrics -- token usage split input/output, and wall-clock
   latency -- so you get dashboards without reading a single trace.
4. Marks the span and the latency metric with ``error.type`` when the call
   raises, then re-raises. Instrumentation never swallows a failure.

Nothing here is OpenAI-specific beyond the response field names; the same
shape works for any chat provider.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind

from . import semconv as sc

INSTRUMENTATION_NAME = "ai-otel-101"
INSTRUMENTATION_VERSION = "0.1.0"


# Hosts whose provider name the conventions actually define. Anything else
# OpenAI-compatible -- a gateway, a proxy, vLLM -- is named by its host, which
# beats reporting "openai" for a request OpenAI never saw.
_KNOWN_PROVIDER_HOSTS = {
    "api.openai.com": sc.PROVIDER_OPENAI,
    "localhost:11434": sc.PROVIDER_OLLAMA,
    "127.0.0.1:11434": sc.PROVIDER_OLLAMA,
    "[::1]:11434": sc.PROVIDER_OLLAMA,
}


def provider_from_base_url(base_url: Any) -> str:
    """Name the provider from the endpoint the client actually talks to.

    The OpenAI SDK points at whatever ``base_url``/``OPENAI_BASE_URL`` says, so
    a hardcoded ``"openai"`` is wrong the moment someone runs against Ollama or
    an Azure deployment -- and wrong in the one attribute a cost dashboard
    groups by.
    """
    if not base_url:
        # No override: the SDK's own default endpoint.
        return sc.PROVIDER_OPENAI

    host = (urlparse(str(base_url)).netloc or str(base_url)).lower()
    if host in _KNOWN_PROVIDER_HOSTS:
        return _KNOWN_PROVIDER_HOSTS[host]

    hostname = host.rsplit(":", 1)[0] if not host.endswith("]") else host
    if "ollama" in hostname:
        return sc.PROVIDER_OLLAMA
    if hostname.endswith(".openai.azure.com"):
        return sc.PROVIDER_AZURE_OPENAI
    if hostname.endswith(".openai.com"):
        return sc.PROVIDER_OPENAI
    # Unknown but real: report where the tokens went.
    return host


def _capture_content_default() -> bool:
    """Message content is opt-in via env var; it can hold user PII."""
    return os.environ.get(sc.CAPTURE_CONTENT_ENV, "").strip().lower() in {
        "true",
        "1",
        "yes",
    }


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute or key lookup, so stubs and real SDK objects both work."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


class InstrumentedChat:
    """Wraps an OpenAI client and emits telemetry for every completion.

    Pass your own ``tracer``/``meter`` (tests do exactly that) or leave them
    out to pick up whatever global providers the process configured.
    """

    def __init__(
        self,
        client: Any,
        *,
        tracer: trace.Tracer | None = None,
        meter: metrics.Meter | None = None,
        capture_content: bool | None = None,
        provider: str | None = None,
    ) -> None:
        self._client = client
        # Derived from the client unless the caller knows better.
        self._provider = provider or provider_from_base_url(
            getattr(client, "base_url", None)
        )
        self._tracer = tracer or trace.get_tracer(
            INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION
        )
        meter = meter or metrics.get_meter(
            INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION
        )
        self._token_usage = meter.create_histogram(
            name=sc.METRIC_TOKEN_USAGE,
            unit="{token}",
            description="Number of input and output tokens used.",
        )
        self._duration = meter.create_histogram(
            name=sc.METRIC_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration.",
        )
        self._capture_content = (
            _capture_content_default() if capture_content is None else capture_content
        )

    # -- public API ---------------------------------------------------------

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Call chat completions inside a span, recording usage metrics."""
        attributes = self._request_attributes(model, kwargs)
        started = time.perf_counter()

        # Span name is "<operation> <model>" per the GenAI conventions, which
        # is what makes traces from different providers line up.
        with self._tracer.start_as_current_span(
            f"{sc.OPERATION_CHAT} {model}",
            kind=SpanKind.CLIENT,
            attributes=attributes,
        ) as span:
            if self._capture_content:
                self._record_prompt(span, messages)
            try:
                response = self._client.chat.completions.create(
                    model=model, messages=list(messages), **kwargs
                )
            except Exception as exc:
                # The span context manager already records the exception and
                # sets ERROR status; we add the low-cardinality error.type
                # that the metric is also keyed on.
                error_type = type(exc).__qualname__
                span.set_attribute(sc.ERROR_TYPE, error_type)
                self._duration.record(
                    time.perf_counter() - started,
                    {**attributes, sc.ERROR_TYPE: error_type},
                )
                raise

            elapsed = time.perf_counter() - started
            self._record_response(span, response)
            self._record_metrics(response, attributes, elapsed)
            return response

    # -- attribute plumbing -------------------------------------------------

    def _request_attributes(
        self, model: str, kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            sc.OPERATION_NAME: sc.OPERATION_CHAT,
            sc.SYSTEM: self._provider,
            sc.PROVIDER_NAME: self._provider,
            sc.REQUEST_MODEL: model,
        }
        # Only sampling knobs that were actually set; absent is not the same
        # as a default.
        if "temperature" in kwargs:
            attributes[sc.REQUEST_TEMPERATURE] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            attributes[sc.REQUEST_MAX_TOKENS] = kwargs["max_tokens"]
        return attributes

    def _record_response(self, span: trace.Span, response: Any) -> None:
        if not span.is_recording():
            return

        for name, value in (
            (sc.RESPONSE_ID, _get(response, "id")),
            (sc.RESPONSE_MODEL, _get(response, "model")),
        ):
            if value is not None:
                span.set_attribute(name, value)

        finish_reasons = [
            reason
            for reason in (
                _get(choice, "finish_reason") for choice in _get(response, "choices", [])
            )
            if reason is not None
        ]
        if finish_reasons:
            span.set_attribute(sc.RESPONSE_FINISH_REASONS, finish_reasons)

        input_tokens, output_tokens = self._token_counts(response)
        if input_tokens is not None:
            span.set_attribute(sc.USAGE_INPUT_TOKENS, input_tokens)
        if output_tokens is not None:
            span.set_attribute(sc.USAGE_OUTPUT_TOKENS, output_tokens)

        if self._capture_content:
            for choice in _get(response, "choices", []):
                message = _get(choice, "message")
                span.add_event(
                    sc.EVENT_CHOICE,
                    {
                        "finish_reason": _get(choice, "finish_reason") or "",
                        "content": _get(message, "content") or "",
                    },
                )

    def _record_prompt(
        self, span: trace.Span, messages: Iterable[Mapping[str, str]]
    ) -> None:
        for message in messages:
            role = message.get("role", "user")
            event = {
                "system": sc.EVENT_SYSTEM_MESSAGE,
                "user": sc.EVENT_USER_MESSAGE,
            }.get(role, sc.EVENT_USER_MESSAGE)
            span.add_event(event, {"content": message.get("content", "")})

    @staticmethod
    def _token_counts(response: Any) -> tuple[int | None, int | None]:
        """OpenAI reports prompt/completion tokens; semconv calls them input/output."""
        usage = _get(response, "usage")
        if usage is None:
            return None, None
        return _get(usage, "prompt_tokens"), _get(usage, "completion_tokens")

    def _record_metrics(
        self, response: Any, attributes: Mapping[str, Any], elapsed: float
    ) -> None:
        metric_attributes = dict(attributes)
        response_model = _get(response, "model")
        if response_model is not None:
            # The model that answered can differ from the one requested
            # ("gpt-4o-mini" -> "gpt-4o-mini-2024-07-18"), and billing follows
            # the former.
            metric_attributes[sc.RESPONSE_MODEL] = response_model

        input_tokens, output_tokens = self._token_counts(response)
        if input_tokens is not None:
            self._token_usage.record(
                input_tokens,
                {**metric_attributes, sc.TOKEN_TYPE: sc.TOKEN_TYPE_INPUT},
            )
        if output_tokens is not None:
            self._token_usage.record(
                output_tokens,
                {**metric_attributes, sc.TOKEN_TYPE: sc.TOKEN_TYPE_OUTPUT},
            )

        self._duration.record(elapsed, metric_attributes)
