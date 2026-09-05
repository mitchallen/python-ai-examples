"""What to retry, and how long to wait -- the decisions, without the telemetry."""

from __future__ import annotations

import pytest

from ai_otel_108 import RetryPolicy, is_retryable, retry_after, status_of


# -- classification ----------------------------------------------------------


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_transient_failures_are_retryable(status, errors):
    assert is_retryable(errors.http(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_broken_request_is_not_retryable(status, errors):
    # Retrying a 401 cannot succeed: the key will not become valid. It only
    # multiplies the latency of a failure and delays the error the caller needs.
    assert is_retryable(errors.http(status)) is False


def test_connection_and_timeout_failures_are_retryable(errors):
    assert is_retryable(errors.connection()) is True
    assert is_retryable(errors.timeout()) is True


def test_an_unknown_exception_is_not_retryable():
    assert is_retryable(ValueError("something else")) is False


def test_status_is_read_off_the_response(errors):
    assert status_of(errors.http(429)) == 429
    assert status_of(ValueError()) is None


# -- retry-after -------------------------------------------------------------


def test_retry_after_header_is_read(errors):
    assert retry_after(errors.http(429, headers={"retry-after": "3"})) == 3.0


def test_retry_after_ms_is_converted_to_seconds(errors):
    assert retry_after(errors.http(429, headers={"retry-after-ms": "1500"})) == 1.5


def test_a_reset_header_is_used_when_retry_after_is_absent(errors):
    error = errors.http(429, headers={"x-ratelimit-reset-requests": "6s"})

    assert retry_after(error) == 6.0


def test_no_header_means_no_instruction(errors):
    assert retry_after(errors.http(429)) is None


def test_a_nonsense_header_is_ignored_rather_than_fatal(errors):
    assert retry_after(errors.http(429, headers={"retry-after": "soon"})) is None


# -- backoff -----------------------------------------------------------------


def test_delay_doubles_per_attempt(no_jitter):
    policy = RetryPolicy(base_delay=0.5, jitter=0.0)

    delays = [policy.delay_for(n, rng=no_jitter) for n in (1, 2, 3, 4)]

    assert delays == [0.5, 1.0, 2.0, 4.0]


def test_delay_is_capped(no_jitter):
    policy = RetryPolicy(base_delay=1.0, max_delay=4.0, jitter=0.0)

    assert policy.delay_for(10, rng=no_jitter) == 4.0


def test_the_server_wins_over_computed_backoff(errors, no_jitter):
    # The server knows when its window resets; backoff is a guess.
    policy = RetryPolicy(base_delay=0.5, jitter=0.0)
    error = errors.http(429, headers={"retry-after": "7"})

    assert policy.delay_for(1, error, rng=no_jitter) == 7.0


def test_jitter_moves_the_delay_without_going_negative():
    policy = RetryPolicy(base_delay=1.0, jitter=0.5)

    delays = {policy.delay_for(1) for _ in range(50)}

    assert len(delays) > 1  # actually jittered
    assert all(0.5 <= d <= 1.5 for d in delays)


def test_a_negative_retry_after_is_clamped(errors, no_jitter):
    policy = RetryPolicy()
    error = errors.http(429, headers={"retry-after": "-5"})

    assert policy.delay_for(1, error, rng=no_jitter) == 0.0
