"""Instrumented streaming chat completions.

Two things change when a response streams:

1. **Usage stops arriving for free.** A streamed response carries no ``usage``
   block unless you ask for ``stream_options={"include_usage": True}``, and the
   chunk that then carries it has an *empty* ``choices`` list. Miss either
   detail and your token telemetry is quietly zero for every streamed call.
2. **Total duration stops meaning much.** It grows with the length of the
   answer, so a slow start and a long answer look the same. Time to first
   token is the number a user actually feels, so it gets its own histogram.

The span stays open for the whole stream rather than just the initial request,
which is what makes both of those measurable.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

from ai_otel_101 import semconv as sc
from ai_otel_101.instrumented import provider_from_base_url
from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind

INSTRUMENTATION_NAME = "ai-otel-103"
INSTRUMENTATION_VERSION = "0.1.0"

# Not semantic conventions. The spec defines gen_ai.server.time_to_first_token
# for the server side and blesses no client-side equivalent, and the app.*
# attributes are this example's own -- named so the difference is visible.
METRIC_TIME_TO_FIRST_TOKEN = "gen_ai.client.time_to_first_token"
STREAM_CHUNKS = "app.stream.chunks"
STREAM_COMPLETED = "app.stream.completed"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


class StreamedResponse:
    """Iterate it for text deltas; read the totals afterwards.

    Iteration is what advances the stream, so ``text``, the token counts and
    ``ttft`` are only complete once iteration finishes.
    """

    def __init__(self, raw: Any, span: trace.Span, started: float) -> None:
        self._raw = raw
        self._span = span
        self._started = started
        self.text = ""
        self.chunks = 0
        self.ttft: float | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.finish_reasons: list[str] = []
        self.completed = False

    def __iter__(self) -> Iterator[str]:
        for chunk in self._raw:
            self.chunks += 1

            # The usage chunk arrives last with choices == [], which is exactly
            # where chunk.choices[0] blows up in naive code.
            usage = _get(chunk, "usage")
            if usage is not None:
                self.input_tokens = _get(usage, "prompt_tokens")
                self.output_tokens = _get(usage, "completion_tokens")

            for choice in _get(chunk, "choices", []) or []:
                reason = _get(choice, "finish_reason")
                if reason is not None:
                    self.finish_reasons.append(reason)

                content = _get(_get(choice, "delta"), "content")
                if not content:
                    continue
                if self.ttft is None:
                    # First visible token: the latency a user perceives.
                    self.ttft = time.perf_counter() - self._started
                self.text += content
                yield content

        self.completed = True

    def close(self) -> None:
        """Release the HTTP response if iteration stopped early."""
        closer = getattr(self._raw, "close", None)
        if callable(closer):
            closer()


class StreamedChat:
    """Wraps a client and instruments streamed completions."""

    def __init__(
        self,
        client: Any,
        *,
        tracer: trace.Tracer | None = None,
        meter: metrics.Meter | None = None,
        provider: str | None = None,
        include_usage: bool = True,
    ) -> None:
        self._client = client
        self._provider = provider or provider_from_base_url(
            getattr(client, "base_url", None)
        )
        # Off only to demonstrate what breaks; leave it on in real code.
        self._include_usage = include_usage
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
        self._ttft = meter.create_histogram(
            name=METRIC_TIME_TO_FIRST_TOKEN,
            unit="s",
            description="Time from request to the first streamed token.",
        )

    @contextmanager
    def stream(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        **kwargs: Any,
    ) -> Iterator[StreamedResponse]:
        """Open a span that lives for the whole stream, not just the request."""
        attributes: dict[str, Any] = {
            sc.OPERATION_NAME: sc.OPERATION_CHAT,
            sc.SYSTEM: self._provider,
            sc.PROVIDER_NAME: self._provider,
            sc.REQUEST_MODEL: model,
        }
        if "temperature" in kwargs:
            attributes[sc.REQUEST_TEMPERATURE] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            attributes[sc.REQUEST_MAX_TOKENS] = kwargs["max_tokens"]

        if self._include_usage:
            # Without this the stream carries no usage at all, and every
            # streamed call reports zero tokens.
            kwargs.setdefault("stream_options", {"include_usage": True})

        started = time.perf_counter()
        with self._tracer.start_as_current_span(
            f"{sc.OPERATION_CHAT} {model}",
            kind=SpanKind.CLIENT,
            attributes=attributes,
        ) as span:
            try:
                raw = self._client.chat.completions.create(
                    model=model, messages=list(messages), stream=True, **kwargs
                )
            except Exception as exc:
                self._record_failure(span, attributes, exc, started)
                raise

            response = StreamedResponse(raw, span, started)
            try:
                yield response
            except Exception as exc:
                self._record_failure(span, attributes, exc, started)
                raise
            finally:
                if not response.completed:
                    # Abandoned mid-stream: close the socket rather than leak it.
                    response.close()

            self._record_success(span, response, attributes, time.perf_counter() - started)

    def _record_failure(
        self,
        span: trace.Span,
        attributes: Mapping[str, Any],
        exc: Exception,
        started: float,
    ) -> None:
        error_type = type(exc).__qualname__
        span.set_attribute(sc.ERROR_TYPE, error_type)
        self._duration.record(
            time.perf_counter() - started, {**attributes, sc.ERROR_TYPE: error_type}
        )

    def _record_success(
        self,
        span: trace.Span,
        response: StreamedResponse,
        attributes: Mapping[str, Any],
        elapsed: float,
    ) -> None:
        span.set_attribute(STREAM_CHUNKS, response.chunks)
        span.set_attribute(STREAM_COMPLETED, response.completed)
        if response.finish_reasons:
            span.set_attribute(sc.RESPONSE_FINISH_REASONS, response.finish_reasons)
        if response.input_tokens is not None:
            span.set_attribute(sc.USAGE_INPUT_TOKENS, response.input_tokens)
        if response.output_tokens is not None:
            span.set_attribute(sc.USAGE_OUTPUT_TOKENS, response.output_tokens)

        for count, token_type in (
            (response.input_tokens, sc.TOKEN_TYPE_INPUT),
            (response.output_tokens, sc.TOKEN_TYPE_OUTPUT),
        ):
            if count is not None:
                self._token_usage.record(
                    count, {**attributes, sc.TOKEN_TYPE: token_type}
                )

        if response.ttft is not None:
            self._ttft.record(response.ttft, dict(attributes))
        self._duration.record(elapsed, dict(attributes))
