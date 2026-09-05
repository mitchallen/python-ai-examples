"""A scripted client: hand it the responses the model would give, in order."""

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
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None = None
    role: str = "assistant"
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 20
    total_tokens: int = 120


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage | None = field(default_factory=FakeUsage)
    id: str = "chatcmpl-123"
    model: str = "gpt-4o-mini"


def says(text: str, **usage: int) -> FakeResponse:
    """A plain answer, no tools."""
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(content=text))],
        usage=FakeUsage(**usage) if usage else FakeUsage(),
    )


def wants(*calls: tuple[str, str], **usage: int) -> FakeResponse:
    """A tool request: (tool_name, json_arguments) pairs."""
    tool_calls = [
        FakeToolCall(id=f"call_{index}", function=FakeFunction(name=name, arguments=args))
        for index, (name, args) in enumerate(calls)
    ]
    return FakeResponse(
        choices=[
            FakeChoice(
                message=FakeMessage(content=None, tool_calls=tool_calls),
                finish_reason="tool_calls",
            )
        ],
        usage=FakeUsage(**usage) if usage else FakeUsage(),
    )


@dataclass
class FakeCompletions:
    script: list[FakeResponse]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.script) - 1)
        return self.script[index]


class FakeClient:
    """Replays a script of responses, recording every request it was sent."""

    def __init__(self, *script: FakeResponse) -> None:
        self.completions = FakeCompletions(script=list(script) or [says("Ahoy!")])
        self.chat = type("Chat", (), {"completions": self.completions})()

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls

    def messages_sent(self, round_index: int) -> list[dict[str, Any]]:
        return self.calls[round_index]["messages"]


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
def script():
    """Builders for model turns."""
    return type("Script", (), {"says": staticmethod(says), "wants": staticmethod(wants)})


@pytest.fixture
def make_client():
    return FakeClient


@pytest.fixture
def registry():
    from ai_otel_105.tools import default_registry

    return default_registry()
