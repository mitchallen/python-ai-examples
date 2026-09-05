"""Async fake stream: chunks with awaits between them, so cancellation is real."""

from __future__ import annotations

import asyncio
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


def text_chunks(*words: str) -> list[FakeChunk]:
    return [FakeChunk(choices=[FakeChoice(delta=FakeDelta(content=w))]) for w in words]


def finish_chunk(reason: str | None = "stop") -> FakeChunk:
    return FakeChunk(choices=[FakeChoice(delta=FakeDelta(), finish_reason=reason)])


def usage_chunk(**kwargs: Any) -> FakeChunk:
    return FakeChunk(choices=[], usage=FakeUsage(**kwargs))


def default_chunks() -> list[FakeChunk]:
    return [*text_chunks("Ahoy", ", ", "matey!"), finish_chunk(), usage_chunk()]


class FakeAsyncStream:
    """Async-iterable chunks, with a real await between each one.

    The await matters: it gives the event loop a chance to deliver a
    cancellation, so the tests exercise the genuine article rather than a
    hand-raised CancelledError.
    """

    def __init__(
        self,
        chunks: Iterable[FakeChunk],
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self._chunks = list(chunks)
        self._error = error
        self._delay = delay
        self.closed = False

    async def __aiter__(self):
        for index, chunk in enumerate(self._chunks):
            await asyncio.sleep(self._delay)
            if self._error is not None and index == len(self._chunks) // 2:
                raise self._error
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class FakeAsyncCompletions:
    stream: FakeAsyncStream | None = None
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    streams: list[FakeAsyncStream] = field(default_factory=list)
    chunks: list[FakeChunk] = field(default_factory=default_chunks)
    delay: float = 0.0

    async def create(self, **kwargs: Any) -> FakeAsyncStream:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        # A fresh stream per call, so concurrent requests do not share one.
        stream = self.stream or FakeAsyncStream(self.chunks, delay=self.delay)
        self.streams.append(stream)
        return stream


class FakeAsyncClient:
    def __init__(
        self,
        chunks: Iterable[FakeChunk] | None = None,
        *,
        error: Exception | None = None,
        stream_error: Exception | None = None,
        delay: float = 0.0,
        shared_stream: bool = True,
    ) -> None:
        resolved = list(chunks) if chunks is not None else default_chunks()
        stream = (
            FakeAsyncStream(resolved, stream_error, delay)
            if error is None and shared_stream
            else None
        )
        self.completions = FakeAsyncCompletions(
            stream=stream, error=error, chunks=resolved, delay=delay
        )
        self.chat = type("Chat", (), {"completions": self.completions})()
        self.stream = stream

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls

    @property
    def streams(self) -> list[FakeAsyncStream]:
        return self.completions.streams


@pytest.fixture
def spans() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(spans: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    return provider


@pytest.fixture
def tracer(tracer_provider: TracerProvider):
    return tracer_provider.get_tracer("test")


@pytest.fixture
def metric_reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture
def meter(metric_reader: InMemoryMetricReader):
    return MeterProvider(metric_readers=[metric_reader]).get_meter("test")


@pytest.fixture
def client() -> FakeAsyncClient:
    return FakeAsyncClient()


@pytest.fixture
def make_client():
    return FakeAsyncClient


@pytest.fixture
def chunks():
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
