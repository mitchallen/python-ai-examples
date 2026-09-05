"""Async demo: ``make run-ollama PKG=ai-otel-104``.

Streams one reply to the terminal, then fires three concurrent requests to show
that each gets its own span and its own time-to-first-token.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from ai_otel_101.telemetry import configure_telemetry
from ai_python_101.chat import build_conversation, resolve_model

from .async_streaming import AsyncStreamedChat, create_async_client, stream_many

CONCURRENT_PROMPTS = [
    "Name your ship.",
    "What's for supper?",
    "Where be the treasure?",
]


async def run(user_message: str) -> None:
    telemetry = configure_telemetry("ai-otel-104")
    client = create_async_client()
    streamed = AsyncStreamedChat(
        client, tracer=telemetry.tracer(), meter=telemetry.meter()
    )
    model = resolve_model()

    try:
        print(f"User: {user_message}")
        print("Pirate: ", end="", flush=True)
        async with streamed.stream(
            build_conversation(user_message), model=model
        ) as stream:
            async for delta in stream:
                print(delta, end="", flush=True)
        ttft = f"{stream.ttft * 1000:.0f} ms" if stream.ttft else "n/a"
        print(f"\n\ntime to first token: {ttft}   tokens: "
              f"{stream.input_tokens} in / {stream.output_tokens} out\n")

        print(f"--- {len(CONCURRENT_PROMPTS)} requests concurrently ---")
        started = time.perf_counter()
        streams = await stream_many(
            streamed,
            [build_conversation(prompt) for prompt in CONCURRENT_PROMPTS],
            model=model,
        )
        elapsed = time.perf_counter() - started

        for prompt, result in zip(CONCURRENT_PROMPTS, streams):
            first = f"{result.ttft * 1000:.0f} ms" if result.ttft else "n/a"
            print(
                f"  {prompt:<24} ttft {first:>8}   "
                f"{result.output_tokens} out   {result.text[:44]!r}"
            )
        slowest = max((s.ttft or 0) for s in streams)
        print(
            f"\nwall clock {elapsed:.2f}s for {len(streams)} streams "
            f"(slowest first token {slowest * 1000:.0f} ms)"
        )
        print("\n--- telemetry ---")
    finally:
        await client.close()
        telemetry.shutdown()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    user_message = argv[0] if argv else "Hello"

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Either:\n"
            "    make run PKG=ai-otel-104          # falls back to a local Ollama model\n"
            "    export OPENAI_API_KEY=sk-...  # then: make run-openai PKG=ai-otel-104",
            file=sys.stderr,
        )
        return 1

    asyncio.run(run(user_message))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
