"""Run the demo: ``make run PKG=ai-otel-102``."""

from __future__ import annotations

import os
import sys

from .observe import (
    build_conversation,
    configure_telemetry,
    create_client,
    provider_for,
    resolve_model,
)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    user_message = argv[0] if argv else "Hello"

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Export a key and try again:\n"
            "    export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        return 1

    telemetry = configure_telemetry()
    client = create_client()
    # Reports "ollama" when OPENAI_BASE_URL points at a local model.
    chat = telemetry.chat_telemetry(provider=provider_for(client))
    model = resolve_model()

    try:
        # The SDK call stays right here in the demo, which is the point of the
        # context-manager shape.
        with chat.chat(model) as observed:
            observed.record(
                client.chat.completions.create(
                    model=model, messages=build_conversation(user_message)
                )
            )
            print(f"User: {user_message}")
            print(f"Pirate: {observed.text()}\n")
            print(
                f"tokens: {observed.input_tokens} in / "
                f"{observed.output_tokens} out\n"
            )
        print("--- telemetry ---")
    finally:
        # Flush before exit, or the batch exporters take the data with them.
        telemetry.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
