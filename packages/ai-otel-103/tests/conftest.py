"""Fake streaming client: chunk sequences shaped like the real wire format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@dataclass
class FakeDelta:
    content: str | None = None


@dataclass
class FakeChoice:
    delta: FakeDelta = field(default_factory=FakeDelta)
    finish_reason: str | None = None


@dataclass
class FakeUsage:
    prompt_tokens: int = 35
    completion_tokens: int = 13
    total_tokens: int = 48


@dataclass
class FakeChunk:
    choices: list[FakeChoice] = field(default_factory=list)
    usage: FakeUsage | None = None


class FakeStream:
    """Iterable of chunks that records whether anyone closed it."""

    def __init__(self, chunks: Iterable[FakeChunk], error: Exception | None = None):
        self._chunks = list(chunks)
        self._error = error
        self.closed = False

    def __iter__(self):
        for index, chunk in enumerate(self._chunks):
            if self._error is not None and index == len(self._chunks) // 2:
                raise self._error
            yield chunk

    def close(self) -> None:
        self.closed = True


def text_chunks(*words: str) -> list[FakeChunk]:
    return [FakeChunk(choices=[FakeChoice(delta=FakeDelta(content=w))]) for w in words]


def finish_chunk(reason: str = "stop") -> FakeChunk:
    return FakeChunk(choices=[FakeChoice(delta=FakeDelta(), finish_reason=reason)])


def usage_chunk(**kwargs: Any) -> FakeChunk:
    # The real thing: usage arrives with an EMPTY choices list.
    return FakeChunk(choices=[], usage=FakeUsage(**kwargs))


def default_chunks() -> list[FakeChunk]:
    return [*text_chunks("Ahoy", ", ", "matey!"), finish_chunk(), usage_chunk()]


@dataclass
class FakeCompletions:
    stream: FakeStream | None = None
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.stream if self.stream is not None else FakeStream(default_chunks())


class FakeClient:
    def __init__(
        self,
        chunks: Iterable[FakeChunk] | None = None,
        *,
        error: Exception | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        stream = (
            FakeStream(chunks if chunks is not None else default_chunks(), stream_error)
            if error is None
            else None
        )
        self.completions = FakeCompletions(stream=stream, error=error)
        self.chat = type("Chat", (), {"completions": self.completions})()
        self.stream = stream

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
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def make_client():
    return FakeClient


@pytest.fixture
def chunks():
    """Builders for custom chunk sequences."""
    return type(
        "Chunks",
        (),
        {
            "text": staticmethod(text_chunks),
            "finish": staticmethod(finish_chunk),
            "usage": staticmethod(usage_chunk),
            "default": staticmethod(default_chunks),
        },
    )
