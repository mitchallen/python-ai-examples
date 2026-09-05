"""A hard-coded pirate conversation, driven through the OpenAI Python client.

Two levels of the same idea:

* :func:`ask_pirate` -- one system prompt, one user turn, one reply.
* :class:`Conversation` -- the same call, but keeping the message history so
  follow-up turns carry context.

Every entry point takes an optional ``client``, which is what makes the whole
thing testable without a network call or an API key.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"

PIRATE_SYSTEM_PROMPT = (
    "You are a pirate. Answer every message in salty pirate speak, "
    "and keep it to a sentence or two."
)

Message = dict[str, str]


def create_client() -> OpenAI:
    """Construct the OpenAI client.

    The constructor reads ``OPENAI_API_KEY`` from the environment; pass
    ``api_key=...`` instead if you would rather be explicit.
    """
    return OpenAI()


def resolve_model(model: str | None = None) -> str:
    """Pick the model: explicit argument, then ``OPENAI_MODEL``, then the default."""
    return model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def build_conversation(user_message: str = "Hello") -> list[Message]:
    """The hard-coded conversation: a pirate system prompt plus one user turn."""
    return [
        {"role": "system", "content": PIRATE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _reply_text(response: Any) -> str:
    """Pull the assistant text out of a chat completion response."""
    return response.choices[0].message.content or ""


def ask_pirate(
    user_message: str = "Hello",
    *,
    client: OpenAI | None = None,
    model: str | None = None,
) -> str:
    """Send a single-turn conversation and return the pirate's reply."""
    client = client or create_client()
    response = client.chat.completions.create(
        model=resolve_model(model),
        messages=build_conversation(user_message),
    )
    return _reply_text(response)


class Conversation:
    """A chat that remembers its own history.

    ``messages`` starts with the system prompt and grows by two entries per
    :meth:`ask` -- the user turn and the assistant's reply -- so the model sees
    everything said so far.
    """

    def __init__(
        self,
        system_prompt: str = PIRATE_SYSTEM_PROMPT,
        *,
        client: OpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self.client = client or create_client()
        self.model = resolve_model(model)
        self.messages: list[Message] = [{"role": "system", "content": system_prompt}]

    def ask(self, user_message: str) -> str:
        """Add a user turn, call the model, record and return the reply."""
        self.messages.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            model=self.model,
            # A copy: the request is a snapshot of the history at send time,
            # not a view that keeps changing as the conversation grows.
            messages=list(self.messages),
        )
        reply = _reply_text(response)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def transcript(self) -> str:
        """The conversation so far, one ``Role: text`` line per turn."""
        return "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in self.messages
        )
