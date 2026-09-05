"""Retry telemetry: one span per attempt, and what the spans say."""

from __future__ import annotations

from typing import Any

import pytest
from ai_otel_101 import semconv as sc
from openai import RateLimitError

from ai_otel_108 import (
    HTTP_STATUS,
    METRIC_ATTEMPTS,
    METRIC_SLEPT,
    RETRY_AFTER,
    RETRY_ATTEMPT,
    RETRY_ATTEMPTS,
    RETRY_EXHAUSTED,
    RETRY_SLEPT,
    RetryingChat,
    RetryPolicy,
)

MESSAGES = [{"role": "user", "content": "Hello"}]


def points(metric_reader: Any, name: str) -> list[Any]:
    data = metric_reader.get_metrics_data()
    return [
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def chat(client, tracer, meter, sleeper, no_jitter, **kwargs) -> RetryingChat:
    kwargs.setdefault("policy", RetryPolicy(max_attempts=3, base_delay=0.5, jitter=0.0))
    return RetryingChat(
        client, tracer=tracer, meter=meter, sleeper=sleeper, rng=no_jitter, **kwargs
    )


def attempt_spans(spans):
    return [s for s in spans.get_finished_spans() if s.name.startswith("attempt")]


def parent_span(spans):
    (parent,) = [s for s in spans.get_finished_spans() if s.name.startswith("chat")]
    return parent


# -- the happy path ----------------------------------------------------------


def test_a_successful_call_is_one_attempt(
    tracer, meter, spans, sleeper, no_jitter, make_client, ok
):
    outcome = chat(make_client(ok()), tracer, meter, sleeper, no_jitter).complete(
        MESSAGES, model="gpt-4o-mini"
    )

    assert outcome.attempt_count == 1
    assert outcome.retried is False
    assert sleeper.waits == []
    assert len(attempt_spans(spans)) == 1


def test_attempts_are_children_of_the_logical_call(
    tracer, meter, spans, sleeper, no_jitter, make_client, ok, errors
):
    client = make_client(errors.http(429), errors.http(429), ok())

    chat(client, tracer, meter, sleeper, no_jitter).complete(MESSAGES, model="m")

    parent = parent_span(spans)
    attempts = attempt_spans(spans)
    assert [s.name for s in attempts] == ["attempt 1", "attempt 2", "attempt 3"]
    assert {s.parent.span_id for s in attempts} == {parent.get_span_context().span_id}


# -- retrying ----------------------------------------------------------------


def test_a_rate_limit_is_retried_until_it_succeeds(
    tracer, meter, spans, sleeper, no_jitter, make_client, ok, errors
):
    client = make_client(errors.http(429), errors.http(429), ok())

    outcome = chat(client, tracer, meter, sleeper, no_jitter).complete(
        MESSAGES, model="m"
    )

    assert outcome.attempt_count == 3
    assert outcome.response.choices[0].message.content == "Ahoy!"
    assert [a.error for a in outcome.attempts] == [
        "RateLimitError",
        "RateLimitError",
        None,
    ]


def test_each_failed_attempt_records_its_status(
    tracer, meter, spans, sleeper, no_jitter, make_client, ok, errors
):
    client = make_client(errors.http(429), ok())

    chat(client, tracer, meter, sleeper, no_jitter).complete(MESSAGES, model="m")

    first, second = attempt_spans(spans)
    assert first.attributes[sc.ERROR_TYPE] == "RateLimitError"
    assert first.attributes[HTTP_STATUS] == 429
    assert first.attributes[RETRY_ATTEMPT] == 1
    assert sc.ERROR_TYPE not in second.attributes


def test_backoff_doubles_between_attempts(
    tracer, meter, sleeper, no_jitter, make_client, ok, errors
):
    client = make_client(errors.http(500), errors.http(500), ok())

    chat(client, tracer, meter, sleeper, no_jitter).complete(MESSAGES, model="m")

    assert sleeper.waits == [0.5, 1.0]


def test_retry_after_overrides_the_backoff(
    tracer, meter, spans, sleeper, no_jitter, make_client, ok, errors
):
    client = make_client(errors.http(429, headers={"retry-after": "2"}), ok())

    chat(client, tracer, meter, sleeper, no_jitter).complete(MESSAGES, model="m")

    assert sleeper.waits == [2.0]
    assert attempt_spans(spans)[0].attributes[RETRY_AFTER] == 2.0


def test_connection_failures_are_retried(
    tracer, meter, sleeper, no_jitter, make_client, ok, errors
):
    client = make_client(errors.connection(), ok())

    outcome = chat(client, tracer, meter, sleeper, no_jitter).complete(
        MESSAGES, model="m"
    )

    assert outcome.attempt_count == 2


# -- giving up ---------------------------------------------------------------


def test_an_unretryable_error_fails_immediately(
    tracer, meter, spans, sleeper, no_jitter, make_client, errors
):
    client = make_client(errors.http(401))

    with pytest.raises(Exception) as caught:
        chat(client, tracer, meter, sleeper, no_jitter).complete(MESSAGES, model="m")

    assert caught.value.status_code == 401
    assert len(client.calls) == 1  # no pointless second try
    assert sleeper.waits == []
    parent = parent_span(spans)
    assert parent.attributes[RETRY_ATTEMPTS] == 1
    # Not exhausted -- it was never retryable.
    assert parent.attributes[RETRY_EXHAUSTED] is False


def test_exhausting_the_attempts_raises_the_last_error(
    tracer, meter, spans, sleeper, no_jitter, make_client, errors
):
    client = make_client(errors.http(429))

    with pytest.raises(RateLimitError):
        chat(client, tracer, meter, sleeper, no_jitter).complete(MESSAGES, model="m")

    assert len(client.calls) == 3
    parent = parent_span(spans)
    assert parent.attributes[RETRY_ATTEMPTS] == 3
    assert parent.attributes[RETRY_EXHAUSTED] is True
    assert parent.attributes[sc.ERROR_TYPE] == "RateLimitError"


def test_the_last_wait_is_not_taken_after_the_last_attempt(
    tracer, meter, sleeper, no_jitter, make_client, errors
):
    # Sleeping after the final failure just delays the exception.
    client = make_client(errors.http(429))

    with pytest.raises(RateLimitError):
        chat(client, tracer, meter, sleeper, no_jitter).complete(MESSAGES, model="m")

    assert len(sleeper.waits) == 2  # between three attempts, not after them


# -- what the parent span says ----------------------------------------------


def test_the_parent_records_attempts_and_time_asleep(
    tracer, meter, spans, sleeper, no_jitter, make_client, ok, errors
):
    client = make_client(errors.http(429), errors.http(429), ok())

    chat(client, tracer, meter, sleeper, no_jitter).complete(MESSAGES, model="m")

    parent = parent_span(spans)
    assert parent.attributes[RETRY_ATTEMPTS] == 3
    # Time asleep is quota; time in flight is the model. Different fixes.
    assert parent.attributes[RETRY_SLEPT] == 1.5
    assert parent.attributes[sc.USAGE_INPUT_TOKENS] == 48


def test_quota_headers_are_recorded_on_success(
    tracer, meter, spans, sleeper, no_jitter, make_client, ok
):
    # The early warning: read the remaining quota while calls still work.
    client = make_client(
        ok(),
        headers={
            "x-ratelimit-remaining-requests": "59",
            "x-ratelimit-remaining-tokens": "149",
            "x-ratelimit-limit-requests": "60",
        },
    )

    outcome = chat(client, tracer, meter, sleeper, no_jitter).complete(
        MESSAGES, model="m"
    )

    parent = parent_span(spans)
    assert parent.attributes["app.ratelimit.remaining_requests"] == 59
    assert parent.attributes["app.ratelimit.remaining_tokens"] == 149
    assert outcome.quota["app.ratelimit.limit_requests"] == 60


def test_a_client_without_raw_responses_still_works(
    tracer, meter, sleeper, no_jitter, make_client, ok
):
    client = make_client(ok(), raw=False)

    outcome = chat(client, tracer, meter, sleeper, no_jitter).complete(
        MESSAGES, model="m"
    )

    assert outcome.attempt_count == 1
    assert outcome.quota == {}


# -- metrics -----------------------------------------------------------------


def test_attempts_and_sleep_are_separate_metrics(
    tracer, meter, metric_reader, sleeper, no_jitter, make_client, ok, errors
):
    client = make_client(errors.http(429), ok())

    chat(client, tracer, meter, sleeper, no_jitter).complete(MESSAGES, model="m")

    (attempts,) = points(metric_reader, METRIC_ATTEMPTS)
    (slept,) = points(metric_reader, METRIC_SLEPT)
    assert attempts.sum == 2
    assert slept.sum == 0.5


def test_a_failed_call_still_records_its_attempts(
    tracer, meter, metric_reader, sleeper, no_jitter, make_client, errors
):
    client = make_client(errors.http(429))

    with pytest.raises(RateLimitError):
        chat(client, tracer, meter, sleeper, no_jitter).complete(MESSAGES, model="m")

    (attempts,) = points(metric_reader, METRIC_ATTEMPTS)
    assert attempts.sum == 3
    assert attempts.attributes[sc.ERROR_TYPE] == "RateLimitError"


def test_tokens_are_recorded_once_not_per_attempt(
    tracer, meter, metric_reader, sleeper, no_jitter, make_client, ok, errors
):
    client = make_client(errors.http(429), errors.http(429), ok())

    chat(client, tracer, meter, sleeper, no_jitter).complete(MESSAGES, model="m")

    by_type = {
        p.attributes[sc.TOKEN_TYPE]: p.sum
        for p in points(metric_reader, sc.METRIC_TOKEN_USAGE)
    }
    # Failed attempts produced no tokens; counting them thrice would inflate cost.
    assert by_type == {"input": 48, "output": 12}
