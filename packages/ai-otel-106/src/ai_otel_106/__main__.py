"""Structured output demo: ``make run PKG=ai-otel-106`` (``ARGS=--loose`` to break it)."""

from __future__ import annotations

import os
import sys

from ai_otel_101.telemetry import configure_telemetry
from ai_python_101.chat import create_client, resolve_model
from pydantic import BaseModel, Field

from .structured import StructuredChat


class Ship(BaseModel):
    """The shape the caller actually needs."""

    name: str = Field(description="The ship's name.")
    crew: int = Field(description="How many souls aboard.")
    cannons: int = Field(description="Number of cannons.")
    notorious_for: str = Field(description="What she is known for, in a few words.")


QUESTION = "Describe the pirate ship Black Pearl."


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    strict = "--loose" not in argv
    prompts = [arg for arg in argv if not arg.startswith("--")]
    question = prompts[0] if prompts else QUESTION

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Either:\n"
            "    make run PKG=ai-otel-106          # falls back to a local Ollama model\n"
            "    export OPENAI_API_KEY=sk-...      # then: make run-openai PKG=ai-otel-106",
            file=sys.stderr,
        )
        return 1

    telemetry = configure_telemetry("ai-otel-106")
    structured = StructuredChat(
        create_client(), tracer=telemetry.tracer(), meter=telemetry.meter()
    )

    try:
        mode = "strict json_schema" if strict else "plain json_object (--loose)"
        print(f"User: {question}")
        print(f"Mode: {mode}\n")

        result = structured.parse(
            [{"role": "user", "content": question}],
            model=resolve_model(),
            schema=Ship,
            strict=strict,
        )

        print(f"outcome: {result.outcome}")
        if result.ok:
            ship = result.parsed
            print(f"  name:          {ship.name}")
            print(f"  crew:          {ship.crew}")
            print(f"  cannons:       {ship.cannons}")
            print(f"  notorious for: {ship.notorious_for}")
        else:
            print(f"  raw:   {(result.raw or result.refusal or '')[:160]}")
            print(f"  error: {result.error}")
            if not strict:
                print(
                    "\n  Valid JSON is not the success condition. This is what "
                    "json_object mode\n  buys you, and why the schema is checked "
                    "on arrival rather than assumed."
                )
        print(f"\ntokens: {result.input_tokens} in / {result.output_tokens} out")
        print("\n--- telemetry ---")
    finally:
        telemetry.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
