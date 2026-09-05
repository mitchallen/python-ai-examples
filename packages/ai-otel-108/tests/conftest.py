"""Real openai error types, a scripted client, and a sleeper that never sleeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def http_error(status: int, *, headers: dict[str, str] | None = None) -> APIStatusError:
    """The real exception class the SDK would raise for this status."""
    response = httpx.Response(status, request=REQUEST, headers=headers or {}, json={})
    kind = {
        400: BadRequestError,
        401: AuthenticationError,
        404: NotFoundError,
        429: RateLimitError,
        500: InternalServerError,
    }.get(status, APIStatusError)
    return kind(f"status {status}", response=response, body=None)


def connection_error() -> APIConnectionError:
    return APIConnectionError(request=REQUEST)


def timeout_error() -> APITimeoutError:
    return APITimeoutError(request=REQUEST)


@dataclass
class FakeUsage:
    prompt_tokens: int = 48
    completion_tokens: int = 12
    total_tokens: int = 60


@dataclass
class FakeMessage:
    content: str = "Ahoy!"


@dataclass
class FakeChoice:
    message: FakeMessage = field(default_factory=FakeMessage)
    finish_reason: str = "stop"


@dataclass
class FakeResponse:
    choices: list[FakeChoice] = field(default_factory=lambda: [FakeChoice()])
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "gpt-4o-mini"


class FakeRawResponse:
    """What ``with_raw_response`` returns: headers plus a parse()."""

    def __init__(self, response: Any, headers: dict[str, str]) -> None:
        self._response = response
        self.headers = headers

    def parse(self) -> Any:
        return self._response


@dataclass
class FakeCompletions:
    script: list[Any] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def _next(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.script) - 1)
        item = self.script[index]
        if isinstance(item, Exception):
            raise item
        return item

    def create(self, **kwargs: Any) -> Any:
        return self._next(**kwargs)


class RawResponseAPI:
    def __init__(self, completions: FakeCompletions) -> None:
        self._completions = completions

    def create(self, **kwargs: Any) -> FakeRawResponse:
        response = self._completions._next(**kwargs)
        return FakeRawResponse(response, self._completions.headers)


class FakeClient:
    """Replays a script of responses and exceptions, in order."""

    def __init__(
        self,
        *script: Any,
        headers: dict[str, str] | None = None,
        raw: bool = True,
    ) -> None:
        self.completions = FakeCompletions(
            script=list(script) or [FakeResponse()], headers=headers or {}
        )
        if raw:
            self.completions.with_raw_response = RawResponseAPI(self.completions)
        self.chat = type("Chat", (), {"completions": self.completions})()

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls


class FakeSleeper:
    """Records what it was asked to wait, and waits for none of it."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.waits)


class NoJitter:
    """A deterministic stand-in for `random`."""

    @staticmethod
    def uniform(low: float, high: float) -> float:
        return 0.0


@pytest.fixture
def spans() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(spans: InMemorySpanExporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    return provider.get_tracer("test")


@pytest.fixture
def metric_reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture
def meter(metric_reader: InMemoryMetricReader):
    return MeterProvider(metric_readers=[metric_reader]).get_meter("test")


@pytest.fixture
def sleeper() -> FakeSleeper:
    return FakeSleeper()


@pytest.fixture
def make_client():
    return FakeClient


@pytest.fixture
def errors():
    return type(
        "Errors",
        (),
        {
            "http": staticmethod(http_error),
            "connection": staticmethod(connection_error),
            "timeout": staticmethod(timeout_error),
        },
    )


@pytest.fixture
def ok():
    return FakeResponse


@pytest.fixture
def no_jitter():
    return NoJitter
