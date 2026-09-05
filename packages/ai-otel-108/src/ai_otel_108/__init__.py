"""ai-otel-108: retries and rate limits, made visible."""

from .retry import RETRYABLE_STATUS, RetryPolicy, is_retryable, retry_after, status_of
from .retrying import (
    HTTP_STATUS,
    METRIC_ATTEMPTS,
    METRIC_SLEPT,
    RETRY_AFTER,
    RETRY_ATTEMPT,
    RETRY_ATTEMPTS,
    RETRY_EXHAUSTED,
    RETRY_SLEPT,
    Attempt,
    RetryingChat,
    RetryOutcome,
)

__all__ = [
    "Attempt",
    "HTTP_STATUS",
    "METRIC_ATTEMPTS",
    "METRIC_SLEPT",
    "RETRYABLE_STATUS",
    "RETRY_AFTER",
    "RETRY_ATTEMPT",
    "RETRY_ATTEMPTS",
    "RETRY_EXHAUSTED",
    "RETRY_SLEPT",
    "RetryOutcome",
    "RetryPolicy",
    "RetryingChat",
    "is_retryable",
    "retry_after",
    "status_of",
]
