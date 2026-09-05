"""Tool-calling demo: ``make run PKG=ai-otel-105``."""

from __future__ import annotations

import os
import sys

from ai_otel_101.telemetry import configure_telemetry
from ai_python_101.chat import create_client, resolve_model

from .agent import ToolCallingChat
from .tools import default_registry

SYSTEM_PROMPT = (
    "You are a pirate quartermaster. Use the tools you are given to answer "
    "questions about the weather and the ship's stores. Answer in pirate speak, "
    "in a sentence or two."
)
DEFAULT_QUESTION = "What's the weather at Tortuga, and how much rum is left?"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    question = argv[0] if argv else DEFAULT_QUESTION

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Either:\n"
            "    make run PKG=ai-otel-105          # falls back to a local Ollama model\n"
            "    export OPENAI_API_KEY=sk-...      # then: make run-openai PKG=ai-otel-105",
            file=sys.stderr,
        )
        return 1

    telemetry = configure_telemetry("ai-otel-105")
    registry = default_registry()
    agent = ToolCallingChat(
        create_client(),
        registry,
        tracer=telemetry.tracer(),
        meter=telemetry.meter(),
    )

    try:
        print(f"User: {question}\n")
        turn = agent.run(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            model=resolve_model(),
        )

        print(f"Quartermaster: {turn.text}\n")
        print(f"rounds:     {turn.rounds} model call(s) for one question")
        print(f"tools:      {', '.join(turn.tool_calls) or 'none'}")
        print(
            f"tokens:     {turn.input_tokens} in / {turn.output_tokens} out "
            f"({turn.total_tokens} total across the turn)"
        )
        for number, (prompt, completion) in enumerate(turn.usage_by_round, start=1):
            print(f"  round {number}:  {prompt} in / {completion} out")
        if turn.truncated:
            print("truncated:  hit max_rounds with the model still asking for tools")
        print("\n--- telemetry ---")
    finally:
        telemetry.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
