"""Stream a pirate reply to the terminal: ``make run-ollama PKG=ai-otel-103``.

Pass ``--no-usage`` to watch the token counts disappear, which is what happens
to any streamed call that forgets ``stream_options``.
"""

from __future__ import annotations

import os
import sys

from ai_otel_101.telemetry import configure_telemetry
from ai_python_101.chat import build_conversation, create_client, resolve_model

from .streaming import StreamedChat


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    include_usage = "--no-usage" not in argv
    prompts = [arg for arg in argv if not arg.startswith("--")]
    user_message = prompts[0] if prompts else "Tell me about the sea."

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Either:\n"
            "    make run PKG=ai-otel-103          # falls back to a local Ollama model\n"
            "    export OPENAI_API_KEY=sk-...  # then: make run-openai PKG=ai-otel-103",
            file=sys.stderr,
        )
        return 1

    telemetry = configure_telemetry("ai-otel-103")
    streamed = StreamedChat(
        create_client(),
        tracer=telemetry.tracer(),
        meter=telemetry.meter(),
        include_usage=include_usage,
    )
    model = resolve_model()

    try:
        print(f"User: {user_message}")
        print("Pirate: ", end="", flush=True)

        with streamed.stream(build_conversation(user_message), model=model) as stream:
            for delta in stream:
                print(delta, end="", flush=True)

        print("\n")
        ttft = f"{stream.ttft * 1000:.0f} ms" if stream.ttft else "n/a"
        print(f"time to first token: {ttft}   chunks: {stream.chunks}")
        if stream.output_tokens is None:
            print(
                "tokens: unavailable -- this is what --no-usage costs you; the "
                "provider sends no usage unless you ask for it."
            )
        else:
            print(f"tokens: {stream.input_tokens} in / {stream.output_tokens} out")
        print("\n--- telemetry ---")
    finally:
        telemetry.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
