"""A chat wrapper that retries visibly.

The SDK retries 429s and 5xx on its own -- ``OpenAI().max_retries`` is 2 -- from
inside the call you wrapped a span around. The retries are real, the waiting is
real, and none of it reaches your telemetry: one span, no error, and latency
with no explanation.

So this turns that off and does it here, where every attempt gets a span and
every wait gets counted.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ai_otel_101 import semconv as sc
from ai_otel_101.instrumented import provider_from_base_url
from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind

from .retry import RetryPolicy, headers_of, is_retryable, retry_after, status_of

INSTRUMENTATION_NAME = "ai-otel-108"
INSTRUMENTATION_VERSION = "0.1.0"

# Standard HTTP semconv.
HTTP_STATUS = "http.response.status_code"

# Local: retry accounting has no semconv names.
RETRY_ATTEMPT = "app.retry.attempt"
RETRY_ATTEMPTS = "app.retry.attempts"
RETRY_SLEPT = "app.retry.slept"
RETRY_AFTER = "app.retry.after_seconds"
RETRY_EXHAUSTED = "app.retry.exhausted"
METRIC_ATTEMPTS = "app.retry.attempts"
METRIC_SLEPT = "app.retry.slept"

# Quota, read from the response headers of a *successful* call.
RATELIMIT_ATTRIBUTES = {
    "x-ratelimit-remaining-requests": "app.ratelimit.remaining_requests",
    "x-ratelimit-remaining-tokens": "app.ratelimit.remaining_tokens",
    "x-ratelimit-limit-requests": "app.ratelimit.limit_requests",
    "x-ratelimit-limit-tokens": "app.ratelimit.limit_tokens",
}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class Attempt:
    """One try, for the caller who wants the story rather than the span."""

    number: int
    error: str | None = None
    status: int | None = None
    slept: float = 0.0


@dataclass
class RetryOutcome:
    """The result plus what it took to get it."""

    response: Any = None
    attempts: list[Attempt] = field(default_factory=list)
    slept: float = 0.0
    quota: dict[str, Any] = field(default_factory=dict)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def retried(self) -> bool:
        return self.attempt_count > 1


class RetryingChat:
    """Chat completions with visible retries."""

    def __init__(
        self,
        client: Any,
        *,
        tracer: trace.Tracer | None = None,
        meter: metrics.Meter | None = None,
        policy: RetryPolicy | None = None,
        provider: str | None = None,
        sleeper: Any = time.sleep,
        rng: Any = random,
    ) -> None:
        self._client = client
        self._policy = policy or RetryPolicy()
        # Injected so tests assert the backoff schedule without waiting for it.
        self._sleep = sleeper
        self._rng = rng
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
            name=sc.METRIC_TOKEN_USAGE, unit="{token}",
            description="Number of input and output tokens used.",
        )
        self._duration = meter.create_histogram(
            name=sc.METRIC_OPERATION_DURATION, unit="s",
            description="GenAI operation duration, including retries.",
        )
        self._attempts_metric = meter.create_histogram(
            name=METRIC_ATTEMPTS, unit="{attempt}",
            description="Attempts made per logical call.",
        )
        self._slept_metric = meter.create_histogram(
            name=METRIC_SLEPT, unit="s",
            description="Time spent waiting between attempts.",
        )

    def complete(
        self, messages: Sequence[Mapping[str, str]], *, model: str, **kwargs: Any
    ) -> RetryOutcome:
        """Call the model, retrying what is worth retrying."""
        attributes: dict[str, Any] = {
            sc.OPERATION_NAME: sc.OPERATION_CHAT,
            sc.SYSTEM: self._provider,
            sc.PROVIDER_NAME: self._provider,
            sc.REQUEST_MODEL: model,
        }
        outcome = RetryOutcome()
        started = time.perf_counter()

        # The logical call. Its children are tries of this, not separate
        # operations, which is why they are named "attempt N".
        with self._tracer.start_as_current_span(
            f"{sc.OPERATION_CHAT} {model}",
            kind=SpanKind.INTERNAL,
            attributes=attributes,
        ) as span:
            last_error: Exception | None = None

            for number in range(1, self._policy.max_attempts + 1):
                attempt = Attempt(number=number)
                outcome.attempts.append(attempt)

                try:
                    response = self._attempt(number, attributes, outcome, model, messages, kwargs)
                except Exception as error:  # noqa: BLE001 - re-raised below
                    last_error = error
                    attempt.error = type(error).__qualname__
                    attempt.status = status_of(error)

                    if not is_retryable(error) or number == self._policy.max_attempts:
                        break

                    delay = self._policy.delay_for(number, error, rng=self._rng)
                    attempt.slept = delay
                    outcome.slept += delay
                    if delay:
                        self._sleep(delay)
                    continue

                outcome.response = response
                self._finish_success(span, outcome, attributes, started)
                return outcome

            self._finish_failure(span, outcome, attributes, started, last_error)
            assert last_error is not None
            raise last_error

    # -- one attempt --------------------------------------------------------

    def _attempt(
        self,
        number: int,
        attributes: Mapping[str, Any],
        outcome: RetryOutcome,
        model: str,
        messages: Sequence[Mapping[str, str]],
        kwargs: Mapping[str, Any],
    ) -> Any:
        with self._tracer.start_as_current_span(
            f"attempt {number}",
            kind=SpanKind.CLIENT,
            attributes={**attributes, RETRY_ATTEMPT: number},
        ) as span:
            try:
                response, headers = self._invoke(model, messages, kwargs)
            except Exception as error:
                span.set_attribute(sc.ERROR_TYPE, type(error).__qualname__)
                status = status_of(error)
                if status is not None:
                    span.set_attribute(HTTP_STATUS, status)
                asked = retry_after(error)
                if asked is not None:
                    span.set_attribute(RETRY_AFTER, asked)
                raise

            outcome.quota = self._record_quota(span, headers)
            usage = _get(response, "usage")
            if usage is not None:
                for name, value in (
                    (sc.USAGE_INPUT_TOKENS, _get(usage, "prompt_tokens")),
                    (sc.USAGE_OUTPUT_TOKENS, _get(usage, "completion_tokens")),
                ):
                    if value is not None:
                        span.set_attribute(name, value)
            return response

    def _invoke(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        kwargs: Mapping[str, Any],
    ) -> tuple[Any, dict[str, str]]:
        """Prefer the raw response, because the quota lives in its headers."""
        completions = self._client.chat.completions
        raw_api = getattr(completions, "with_raw_response", None)
        if raw_api is not None:
            raw = raw_api.create(model=model, messages=list(messages), **kwargs)
            return raw.parse(), headers_of(raw)

        response = completions.create(model=model, messages=list(messages), **kwargs)
        return response, headers_of(response)

    # -- recording ----------------------------------------------------------

    @staticmethod
    def _record_quota(span: trace.Span, headers: Mapping[str, str]) -> dict[str, Any]:
        """Remaining quota, recorded while calls still succeed.

        This is the early warning: by the time a 429 arrives the limit has
        already been hit.
        """
        quota: dict[str, Any] = {}
        for header, attribute in RATELIMIT_ATTRIBUTES.items():
            raw = headers.get(header)
            if raw is None:
                continue
            value: Any = int(raw) if raw.isdigit() else raw
            quota[attribute] = value
            span.set_attribute(attribute, value)
        return quota

    def _finish_success(
        self,
        span: trace.Span,
        outcome: RetryOutcome,
        attributes: Mapping[str, Any],
        started: float,
    ) -> None:
        span.set_attribute(RETRY_ATTEMPTS, outcome.attempt_count)
        span.set_attribute(RETRY_SLEPT, outcome.slept)
        for attribute, value in outcome.quota.items():
            span.set_attribute(attribute, value)

        usage = _get(outcome.response, "usage")
        if usage is not None:
            for token_type, value in (
                (sc.TOKEN_TYPE_INPUT, _get(usage, "prompt_tokens")),
                (sc.TOKEN_TYPE_OUTPUT, _get(usage, "completion_tokens")),
            ):
                if value is not None:
                    span.set_attribute(
                        sc.USAGE_INPUT_TOKENS
                        if token_type == sc.TOKEN_TYPE_INPUT
                        else sc.USAGE_OUTPUT_TOKENS,
                        value,
                    )
                    self._token_usage.record(
                        value, {**attributes, sc.TOKEN_TYPE: token_type}
                    )

        self._record_shared(outcome, attributes, started)

    def _finish_failure(
        self,
        span: trace.Span,
        outcome: RetryOutcome,
        attributes: Mapping[str, Any],
        started: float,
        error: Exception | None,
    ) -> None:
        span.set_attribute(RETRY_ATTEMPTS, outcome.attempt_count)
        span.set_attribute(RETRY_SLEPT, outcome.slept)
        if error is not None:
            span.set_attribute(sc.ERROR_TYPE, type(error).__qualname__)
            status = status_of(error)
            if status is not None:
                span.set_attribute(HTTP_STATUS, status)
            # Distinguishes "gave up after N" from "failed once, unretryable".
            span.set_attribute(RETRY_EXHAUSTED, is_retryable(error))
        self._record_shared(outcome, attributes, started, error=error)

    def _record_shared(
        self,
        outcome: RetryOutcome,
        attributes: Mapping[str, Any],
        started: float,
        error: Exception | None = None,
    ) -> None:
        metric_attributes = dict(attributes)
        if error is not None:
            metric_attributes[sc.ERROR_TYPE] = type(error).__qualname__

        self._attempts_metric.record(outcome.attempt_count, metric_attributes)
        # Separate from duration on purpose: time asleep is quota, time in
        # flight is the model. They need different fixes.
        self._slept_metric.record(outcome.slept, metric_attributes)
        self._duration.record(time.perf_counter() - started, metric_attributes)
