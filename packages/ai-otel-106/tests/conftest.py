"""Stub responses covering every structured-output outcome."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel


class Ship(BaseModel):
    name: str
    crew: int
    cannons: int


@dataclass
class FakeUsage:
    prompt_tokens: int = 32
    completion_tokens: int = 31
    total_tokens: int = 63


@dataclass
class FakeMessage:
    content: str | None = None
    refusal: str | None = None
    role: str = "assistant"


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage | None = field(default_factory=FakeUsage)
    id: str = "chatcmpl-123"
    model: str = "gpt-4o-mini"


def returns(content: str, *, finish_reason: str = "stop") -> FakeResponse:
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(content=content), finish_reason=finish_reason)]
    )


def refuses(reason: str = "I can't help with that.") -> FakeResponse:
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(content=None, refusal=reason))]
    )


@dataclass
class FakeCompletions:
    response: Any = None
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: Any = None, *, error: Exception | None = None) -> None:
        self.completions = FakeCompletions(
            response=response if response is not None else returns('{"name":"Black Pearl","crew":40,"cannons":32}'),
            error=error,
        )
        self.chat = type("Chat", (), {"completions": self.completions})()

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls


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
def ship():
    return Ship


@pytest.fixture
def make_client():
    return FakeClient


@pytest.fixture
def responses():
    return type("Responses", (), {"returns": staticmethod(returns), "refuses": staticmethod(refuses)})


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()
