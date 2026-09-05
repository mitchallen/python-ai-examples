"""Scripted replies, so token accounting and trimming are exact."""

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
    prompt_tokens: int = 40
    completion_tokens: int = 10
    total_tokens: int = 50


@dataclass
class FakeMessage:
    content: str = "Ahoy!"


@dataclass
class FakeChoice:
    message: FakeMessage = field(default_factory=FakeMessage)
    finish_reason: str = "stop"


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "gpt-4o-mini"


@dataclass
class FakeCompletions:
    replies: list[str] = field(default_factory=lambda: ["Ahoy!"])
    calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 40

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        # prompt_tokens tracks the message count, so tests can tell a trimmed
        # request from a full one by its cost.
        sent = len(kwargs.get("messages", []))
        return FakeResponse(
            choices=[FakeChoice(message=FakeMessage(content=self.replies[index]))],
            usage=FakeUsage(prompt_tokens=self.prompt_tokens * sent, completion_tokens=10),
        )


class FakeClient:
    def __init__(self, *replies: str) -> None:
        self.completions = FakeCompletions(replies=list(replies) or ["Ahoy!"])
        self.chat = type("Chat", (), {"completions": self.completions})()

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls

    def messages_sent(self, index: int) -> list[dict[str, Any]]:
        return self.calls[index]["messages"]


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


def turn(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def history(exchanges: int, *, system: bool = True) -> list[dict[str, str]]:
    """A conversation with `exchanges` complete user/assistant pairs."""
    messages = [turn("system", "You are a pirate.")] if system else []
    for index in range(exchanges):
        messages.append(turn("user", f"question {index}"))
        messages.append(turn("assistant", f"answer {index}"))
    return messages


@pytest.fixture
def conversation_history():
    return history
