"""Run the instrumented demo: ``make run PKG=ai-otel-101``.

Reuses the pirate conversation from ai-python-101 (a workspace sibling) so the
only new thing on screen is the telemetry.
"""

from __future__ import annotations

import os
import sys

from ai_python_101.chat import build_conversation, create_client, resolve_model

from .instrumented import InstrumentedChat
from .telemetry import configure_telemetry


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    user_message = argv[0] if argv else "Hello"

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Either:\n"
            "    make run PKG=ai-otel-101          # falls back to a local Ollama model\n"
            "    export OPENAI_API_KEY=sk-...  # then: make run-openai PKG=ai-otel-101",
            file=sys.stderr,
        )
        return 1

    telemetry = configure_telemetry()
    chat = InstrumentedChat(
        create_client(), tracer=telemetry.tracer(), meter=telemetry.meter()
    )
    model = resolve_model()

    try:
        response = chat.complete(build_conversation(user_message), model=model)
        print(f"User: {user_message}")
        print(f"Pirate: {response.choices[0].message.content}\n")
        usage = response.usage
        print(
            f"tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out "
            f"({usage.total_tokens} total)\n"
        )
        print("--- telemetry ---")
    finally:
        # Flush before exit, or the batch exporters take the data with them.
        telemetry.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
