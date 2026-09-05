"""Run the demo: ``uv run --package ai-python-101 ai-python-101`` (or ``make run``)."""

from __future__ import annotations

import os
import sys

from .chat import Conversation, create_client, resolve_model

DEMO_TURNS = [
    "Hello",
    "What's the weather like at sea today?",
]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    turns = argv or DEMO_TURNS

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Either:\n"
            "    make run PKG=ai-python-101          # falls back to a local Ollama model\n"
            "    export OPENAI_API_KEY=sk-...  # then: make run-openai PKG=ai-python-101",
            file=sys.stderr,
        )
        return 1

    chat = Conversation(client=create_client())
    print(f"model: {resolve_model()}\n")

    for turn in turns:
        print(f"User: {turn}")
        print(f"Pirate: {chat.ask(turn)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
