"""Retry demo: ``make run PKG=ai-otel-108``.

Runs a normal call, then the same call through a shim that raises 429 twice
before letting it through -- provoking a genuine rate limit is neither cheap nor
polite, and every code path exercised here is the one a real 429 takes.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import httpx
from ai_otel_101.telemetry import configure_telemetry
from ai_python_101.chat import build_conversation, create_client, resolve_model
from openai import OpenAI, RateLimitError

from .retry import RetryPolicy
from .retrying import RetryingChat


def rate_limit_error(retry_after: float = 1.0) -> RateLimitError:
    """A real openai.RateLimitError, so the demo classifies the real type."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": str(retry_after), "x-ratelimit-remaining-requests": "0"},
        json={"error": {"message": "Rate limit reached", "type": "rate_limit_error"}},
    )
    return RateLimitError("Rate limit reached", response=response, body=None)


class FlakyCompletions:
    """Fails `failures` times with 429, then delegates to the real client."""

    def __init__(self, real: Any, failures: int) -> None:
        self._real = real
        self._remaining = failures
        self.attempts = 0

    def create(self, **kwargs: Any) -> Any:
        self.attempts += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise rate_limit_error()
        return self._real.create(**kwargs)


class FlakyClient:
    """The real client with a rate-limiting shim in front of completions."""

    def __init__(self, real: OpenAI, failures: int = 2) -> None:
        self.base_url = real.base_url
        self.completions = FlakyCompletions(real.chat.completions, failures)
        self.chat = type("Chat", (), {"completions": self.completions})()


def describe(label: str, outcome: Any) -> None:
    print(f"\n{label}")
    print(f"  attempts: {outcome.attempt_count}   slept: {outcome.slept:.2f}s")
    for attempt in outcome.attempts:
        if attempt.error:
            print(
                f"    attempt {attempt.number}: {attempt.error} "
                f"(status {attempt.status}) -> waited {attempt.slept:.2f}s"
            )
        else:
            print(f"    attempt {attempt.number}: ok")
    if outcome.quota:
        print(f"  quota: {outcome.quota}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    question = argv[0] if argv else "Hello"

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Either:\n"
            "    make run PKG=ai-otel-108          # falls back to a local Ollama model\n"
            "    export OPENAI_API_KEY=sk-...      # then: make run-openai PKG=ai-otel-108",
            file=sys.stderr,
        )
        return 1

    telemetry = configure_telemetry("ai-otel-108")
    model = resolve_model()
    messages = build_conversation(question)

    # max_retries=0: the SDK's own retries would happen inside a single span
    # and never reach the telemetry. We want to see them.
    client = create_client().with_options(max_retries=0)
    policy = RetryPolicy(max_attempts=3, base_delay=0.5)

    try:
        print(f"model: {model}   sdk max_retries: {client.max_retries} (ours: "
              f"{policy.max_attempts} attempts)\n")

        chat = RetryingChat(client, tracer=telemetry.tracer(), meter=telemetry.meter(),
                            policy=policy)
        outcome = chat.complete(messages, model=model)
        print(f"Pirate: {outcome.response.choices[0].message.content}")
        describe("normal call:", outcome)

        # Now the same call, rate limited twice.
        flaky = RetryingChat(
            FlakyClient(client, failures=2),
            tracer=telemetry.tracer(),
            meter=telemetry.meter(),
            policy=policy,
        )
        outcome = flaky.complete(messages, model=model)
        print(f"\nPirate: {outcome.response.choices[0].message.content}")
        describe("rate-limited call (429 x2, simulated):", outcome)
        print(
            "\nThe SDK would have retried these itself, inside one span, and the "
            "429s\nwould never have reached your telemetry -- just unexplained "
            "latency."
        )
        print("\n--- telemetry ---")
    finally:
        telemetry.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
