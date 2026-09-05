"""Tests for the pirate demo.

The OpenAI client is replaced with a stub, so nothing here touches the network
or needs an API key -- the tests assert on the *request* we build and on how the
reply is threaded back into the history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ai_python_101 import (
    DEFAULT_MODEL,
    PIRATE_SYSTEM_PROMPT,
    Conversation,
    ask_pirate,
    build_conversation,
)
from ai_python_101.chat import resolve_model


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]


@dataclass
class FakeCompletions:
    replies: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return FakeResponse(choices=[FakeChoice(message=FakeMessage(content=reply))])


class FakeClient:
    """Stands in for ``OpenAI()``; only the bits the demo touches exist."""

    def __init__(self, *replies: str) -> None:
        self.completions = FakeCompletions(replies=list(replies) or ["Ahoy!"])
        self.chat = type("Chat", (), {"completions": self.completions})()

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls


def test_build_conversation_is_system_then_user():
    messages = build_conversation("Hello")

    assert messages == [
        {"role": "system", "content": PIRATE_SYSTEM_PROMPT},
        {"role": "user", "content": "Hello"},
    ]


def test_ask_pirate_returns_reply_text():
    client = FakeClient("Ahoy, matey!")

    assert ask_pirate("Hello", client=client) == "Ahoy, matey!"


def test_ask_pirate_sends_the_hard_coded_conversation():
    client = FakeClient("Ahoy, matey!")

    ask_pirate("Hello", client=client)

    (call,) = client.calls
    assert call["model"] == DEFAULT_MODEL
    assert call["messages"] == build_conversation("Hello")


def test_ask_pirate_honors_an_explicit_model():
    client = FakeClient("Ahoy!")

    ask_pirate("Hello", client=client, model="gpt-4o")

    assert client.calls[0]["model"] == "gpt-4o"


def test_missing_content_becomes_empty_string():
    client = FakeClient("")
    client.completions.replies = [None]

    assert ask_pirate("Hello", client=client) == ""


def test_resolve_model_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert resolve_model() == DEFAULT_MODEL

    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    assert resolve_model() == "gpt-4o"
    assert resolve_model("gpt-4.1") == "gpt-4.1"


def test_conversation_starts_with_only_the_system_prompt():
    chat = Conversation(client=FakeClient())

    assert chat.messages == [{"role": "system", "content": PIRATE_SYSTEM_PROMPT}]


def test_conversation_accumulates_history():
    client = FakeClient("Ahoy, matey!", "Stormy, arr.")
    chat = Conversation(client=client)

    assert chat.ask("Hello") == "Ahoy, matey!"
    assert chat.ask("What's the weather like at sea today?") == "Stormy, arr."

    assert [m["role"] for m in chat.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_conversation_resends_prior_turns():
    client = FakeClient("Ahoy, matey!", "Stormy, arr.")
    chat = Conversation(client=client)

    chat.ask("Hello")
    chat.ask("What's the weather like at sea today?")

    first, second = client.calls
    assert len(first["messages"]) == 2
    # The second request replays system + first exchange + the new user turn.
    assert len(second["messages"]) == 4
    assert second["messages"][2] == {"role": "assistant", "content": "Ahoy, matey!"}


def test_transcript_renders_each_turn():
    chat = Conversation(system_prompt="You are a pirate.", client=FakeClient("Ahoy!"))
    chat.ask("Hello")

    assert chat.transcript() == (
        "System: You are a pirate.\nUser: Hello\nAssistant: Ahoy!"
    )
