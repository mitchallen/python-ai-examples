"""Stub OpenAI client plus in-memory OTel pipelines.

Everything is in-process: no network, no API key, no collector. The tests read
spans and metrics straight out of memory and assert on the attribute names,
because those names are the contract with whatever backend consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@dataclass
class FakeUsage:
    prompt_tokens: int = 11
    completion_tokens: int = 7
    total_tokens: int = 18


@dataclass
class FakeMessage:
    content: str = "Ahoy, matey!"
    role: str = "assistant"


@dataclass
class FakeChoice:
    message: FakeMessage = field(default_factory=FakeMessage)
    finish_reason: str = "stop"


@dataclass
class FakeResponse:
    id: str = "chatcmpl-123"
    model: str = "gpt-4o-mini-2024-07-18"
    choices: list[FakeChoice] = field(default_factory=lambda: [FakeChoice()])
    usage: FakeUsage | None = field(default_factory=FakeUsage)


@dataclass
class FakeCompletions:
    response: Any = field(default_factory=FakeResponse)
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.completions = FakeCompletions(
            response=response if response is not None else FakeResponse(),
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
    # A local provider, never the global one, so tests stay independent.
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
def client() -> FakeClient:
    return FakeClient()
