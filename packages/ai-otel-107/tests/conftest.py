"""A stub embeddings endpoint with deterministic vectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@dataclass
class FakeEmbedding:
    embedding: list[float]
    index: int = 0
    object: str = "embedding"


@dataclass
class FakeUsage:
    """Note what is absent: embeddings have no completion_tokens."""

    prompt_tokens: int = 7
    total_tokens: int = 7


@dataclass
class FakeEmbeddingResponse:
    data: list[FakeEmbedding]
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "text-embedding-3-small"


def vector_for(text: str, dimensions: int = 4) -> list[float]:
    """A stable pseudo-vector: same text always gives the same numbers."""
    return [float((sum(bytearray(text.encode())) + i * 31) % 97) for i in range(dimensions)]


@dataclass
class FakeEmbeddings:
    calls: list[dict[str, Any]] = field(default_factory=list)
    error: Exception | None = None
    tokens_per_input: int = 7
    dimensions: int = 4

    def create(self, **kwargs: Any) -> FakeEmbeddingResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        inputs = kwargs["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        return FakeEmbeddingResponse(
            data=[
                FakeEmbedding(embedding=vector_for(text, self.dimensions), index=i)
                for i, text in enumerate(inputs)
            ],
            usage=FakeUsage(
                prompt_tokens=self.tokens_per_input * len(inputs),
                total_tokens=self.tokens_per_input * len(inputs),
            ),
        )

    @property
    def inputs_sent(self) -> list[Sequence[str]]:
        return [call["input"] for call in self.calls]


class FakeClient:
    def __init__(self, *, error: Exception | None = None, dimensions: int = 4) -> None:
        self.embeddings = FakeEmbeddings(error=error, dimensions=dimensions)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.embeddings.calls


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
def vectors():
    return vector_for
