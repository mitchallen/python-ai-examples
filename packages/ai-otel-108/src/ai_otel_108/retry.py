"""Retry policy: what to retry, how long to wait, and how to say so.

Kept separate from the instrumentation because the decisions here -- which
failures are worth repeating, and how long to wait -- are the ones worth
testing, and they have nothing to do with OpenTelemetry.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# Transient by nature: the same request may well work in a moment.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def status_of(error: Exception) -> int | None:
    """The HTTP status behind an exception, if it has one."""
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def headers_of(error_or_response: Any) -> dict[str, str]:
    """Response headers off an exception or a response, lower-cased."""
    response = getattr(error_or_response, "response", error_or_response)
    headers = getattr(response, "headers", None)
    if not headers:
        return {}
    try:
        return {str(key).lower(): str(value) for key, value in dict(headers).items()}
    except (TypeError, ValueError):  # pragma: no cover - exotic header objects
        return {}


def retry_after(error: Exception) -> float | None:
    """Seconds the server asked us to wait, if it said.

    The server knows when its window resets and our backoff is a guess, so this
    wins whenever it is present.
    """
    headers = headers_of(error)
    for name in ("retry-after", "x-ratelimit-reset-requests", "retry-after-ms"):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw.rstrip("ms") if name.endswith("-ms") else raw.rstrip("s"))
        except ValueError:
            continue
        return value / 1000 if name.endswith("-ms") else value
    return None


def is_retryable(error: Exception) -> bool:
    """True for failures a second attempt could plausibly fix.

    A 401 will not become authorised and a 400 will not become well-formed, so
    retrying either only multiplies the latency of a failure and delays the
    error the caller has to see.
    """
    name = type(error).__name__
    if name in {"APIConnectionError", "APITimeoutError", "ConnectionError", "TimeoutError"}:
        return True
    status = status_of(error)
    return status in RETRYABLE_STATUS if status is not None else False


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter, capped."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 0.1

    def delay_for(
        self, attempt: int, error: Exception | None = None, *, rng: Any = random
    ) -> float:
        """How long to wait before `attempt` + 1, in seconds."""
        if error is not None:
            asked = retry_after(error)
            if asked is not None:
                return max(0.0, asked)

        # 0.5, 1, 2, 4 ... capped.
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        if not self.jitter:
            return delay
        # Jitter keeps a fleet of clients from waking up together and
        # re-creating the burst that caused the 429.
        spread = delay * self.jitter
        return max(0.0, delay + rng.uniform(-spread, spread))
