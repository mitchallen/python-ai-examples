"""Async streaming, where two things that were easy become traps.

**Cancellation.** ``asyncio.CancelledError`` inherits from ``BaseException``,
not ``Exception``. Instrumentation written as ``except Exception`` therefore
never sees a cancelled request -- and cancellation is the *normal* ending for a
streaming chat when the user navigates away. Handled wrong, those requests
vanish from telemetry entirely, taking their tokens with them; handled as an
error, they poison your error rate with something nobody needs to fix. This
records them as cancelled, leaves the span status alone, and always re-raises,
because swallowing cancellation breaks task shutdown.

**Concurrency.** Spans live in a ``contextvars`` context, and every asyncio task
gets its own copy at creation. Start the span *inside* the coroutine that does
the work -- as this does -- and N concurrent streams produce N sibling spans
under whatever span was current when the tasks were spawned. Start one span
around a ``gather`` of many calls and you get a single span whose duration means
nothing.

Attribute and metric names are shared with :mod:`ai_otel_103`, so the sync and
async paths land in the same dashboards.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping, Sequence

from ai_otel_101 import semconv as sc  # noqa: I001
from ai_otel_101.instrumented import provider_from_base_url
from ai_otel_103.streaming import (
    METRIC_TIME_TO_FIRST_TOKEN,
    STREAM_CHUNKS,
    STREAM_COMPLETED,
)
from openai import AsyncOpenAI
from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind

INSTRUMENTATION_NAME = "ai-otel-104"
INSTRUMENTATION_VERSION = "0.1.0"

# Local, like the app.* attributes in ai-otel-103: a cancelled stream is not a
# failure, so it needs a name of its own rather than error.type.
STREAM_CANCELLED = "app.stream.cancelled"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


class AsyncStreamedResponse:
    """``async for`` it for text deltas; read the totals afterwards."""

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

    async def __aiter__(self) -> AsyncIterator[str]:
        async for chunk in self._raw:
            self.chunks += 1

            # Last chunk: usage, with an empty choices list.
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
                    self.ttft = time.perf_counter() - self._started
                self.text += content
                yield content

        self.completed = True

    async def aclose(self) -> None:
        """Release the response if iteration stopped early or was cancelled."""
        closer = getattr(self._raw, "aclose", None) or getattr(self._raw, "close", None)
        if closer is None:
            return
        result = closer()
        if inspect.isawaitable(result):
            await result


class AsyncStreamedChat:
    """Instruments streamed completions from an ``AsyncOpenAI`` client."""

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

    @asynccontextmanager
    async def stream(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[AsyncStreamedResponse]:
        """Span the whole stream, from inside whichever task is running it."""
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
            kwargs.setdefault("stream_options", {"include_usage": True})

        started = time.perf_counter()
        with self._tracer.start_as_current_span(
            f"{sc.OPERATION_CHAT} {model}",
            kind=SpanKind.CLIENT,
            attributes=attributes,
        ) as span:
            try:
                raw = await self._client.chat.completions.create(
                    model=model, messages=list(messages), stream=True, **kwargs
                )
            except asyncio.CancelledError:
                self._record_cancelled(span, attributes, started)
                raise
            except Exception as exc:
                self._record_failure(span, attributes, exc, started)
                raise

            response = AsyncStreamedResponse(raw, span, started)
            try:
                yield response
            except asyncio.CancelledError:
                # BaseException, so a bare `except Exception` above would miss
                # this. Never swallowed: re-raising is what lets the task die.
                self._record_cancelled(span, attributes, started, response)
                raise
            except Exception as exc:
                self._record_failure(span, attributes, exc, started)
                raise
            else:
                self._record_success(
                    span, response, attributes, time.perf_counter() - started
                )
            finally:
                if not response.completed:
                    await response.aclose()

    # -- recording ----------------------------------------------------------

    def _record_cancelled(
        self,
        span: trace.Span,
        attributes: Mapping[str, Any],
        started: float,
        response: AsyncStreamedResponse | None = None,
    ) -> None:
        # Deliberately not an error: a user who closed the tab is nothing to
        # fix, and marking it ERROR would poison the error rate. OpenTelemetry
        # agrees by accident -- its span manager only reacts to Exception, so
        # the status stays unset here.
        span.set_attribute(STREAM_CANCELLED, True)
        span.set_attribute(STREAM_COMPLETED, False)
        if response is not None:
            span.set_attribute(STREAM_CHUNKS, response.chunks)
            self._record_partial_tokens(response, attributes)
        self._duration.record(
            time.perf_counter() - started, {**attributes, STREAM_CANCELLED: True}
        )

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

    def _record_partial_tokens(
        self, response: AsyncStreamedResponse, attributes: Mapping[str, Any]
    ) -> None:
        """Bill what was actually generated, even for a stream nobody read to the end."""
        for count, token_type in (
            (response.input_tokens, sc.TOKEN_TYPE_INPUT),
            (response.output_tokens, sc.TOKEN_TYPE_OUTPUT),
        ):
            if count is not None:
                self._token_usage.record(
                    count, {**attributes, sc.TOKEN_TYPE: token_type}
                )

    def _record_success(
        self,
        span: trace.Span,
        response: AsyncStreamedResponse,
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

        self._record_partial_tokens(response, attributes)
        if response.ttft is not None:
            self._ttft.record(response.ttft, dict(attributes))
        self._duration.record(elapsed, dict(attributes))


def create_async_client() -> AsyncOpenAI:
    """The async client; same env vars as the sync one (key, base URL)."""
    return AsyncOpenAI()


async def stream_many(
    streamed: AsyncStreamedChat,
    conversations: Sequence[Sequence[Mapping[str, str]]],
    *,
    model: str,
) -> list[AsyncStreamedResponse]:
    """Run several streams concurrently, one span each.

    The span is opened inside :meth:`AsyncStreamedChat.stream`, which runs in
    the task, so each concurrent request gets its own span parented to whatever
    was current when the tasks were created -- rather than one span around the
    whole gather, whose duration would only tell you how long the slowest call
    took.
    """

    async def one(conversation: Sequence[Mapping[str, str]]) -> AsyncStreamedResponse:
        async with streamed.stream(conversation, model=model) as stream:
            async for _ in stream:
                pass
            return stream

    return list(await asyncio.gather(*(one(c) for c in conversations)))
