"""Conversation memory demo: ``make run PKG=ai-otel-109``.

Runs the same six questions under two strategies and prints the cost curve.
``--summarize`` adds a third run that compacts old turns.
"""

from __future__ import annotations

import os
import sys

from ai_otel_101.telemetry import configure_telemetry
from ai_python_101.chat import create_client, resolve_model

from .conversation import TracedConversation
from .memory import KeepAll, SlidingWindow, Strategy, Summarizing

SYSTEM_PROMPT = (
    "You are a pirate quartermaster. Answer in one short sentence of pirate speak."
)
QUESTIONS = [
    "What's your name?",
    "How many souls are in the crew?",
    "What's in the hold?",
    "Where are we sailing?",
    "How's the weather looking?",
    "Remind me: what did I ask you first?",
]


def run_conversation(client, model, strategy: Strategy, telemetry) -> TracedConversation:
    conversation = TracedConversation(
        client,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        strategy=strategy,
        tracer=telemetry.tracer(),
        meter=telemetry.meter(),
    )
    for question in QUESTIONS:
        conversation.ask(question)
    return conversation


def report(label: str, conversation: TracedConversation) -> None:
    print(f"\n{label}  (session {conversation.conversation_id})")
    print("  turn  msgs  dropped   in    out   estimate")
    for result in conversation.stats.per_turn:
        error = result.estimate_error
        drift = f"{error:+.1%}" if error is not None else "n/a"
        flag = "  <- compacted" if result.compacted else ""
        print(
            f"   {result.turn:>2}   {result.messages_sent:>3}   {result.messages_dropped:>5}"
            f"  {result.input_tokens:>5}  {result.output_tokens:>4}   {drift:>7}{flag}"
        )
    stats = conversation.stats
    print(
        f"  total: {stats.input_tokens} in / {stats.output_tokens} out "
        f"({stats.total_tokens} tokens, {stats.compactions} compactions)"
    )
    # The last answer shows whether trimming cost the model its memory.
    print(f"  last answer: {conversation.stats.per_turn[-1].text[:70]!r}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with_summary = "--summarize" in argv

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Either:\n"
            "    make run PKG=ai-otel-109          # falls back to a local Ollama model\n"
            "    export OPENAI_API_KEY=sk-...      # then: make run-openai PKG=ai-otel-109",
            file=sys.stderr,
        )
        return 1

    telemetry = configure_telemetry("ai-otel-109")
    client = create_client()
    model = resolve_model()

    try:
        print(f"model: {model}   {len(QUESTIONS)} questions, same order each time")

        keep_all = run_conversation(client, model, KeepAll(), telemetry)
        report("keep-all:", keep_all)

        windowed = run_conversation(client, model, SlidingWindow(max_turns=2), telemetry)
        report("sliding window (2 turns):", windowed)

        baseline = keep_all.stats.input_tokens
        trimmed = windowed.stats.input_tokens
        if baseline:
            print(
                f"\nsliding window sent {baseline - trimmed} fewer input tokens "
                f"({(trimmed - baseline) / baseline:+.0%}) -- and the last answer "
                "shows what it forgot."
            )

        if with_summary:
            summarizing = run_conversation(
                client, model, Summarizing(keep_turns=2, trigger_after=4), telemetry
            )
            report("summarizing:", summarizing)
            print(
                "\nCompaction is not free: its own model call is in the totals "
                "above,\nand in the trace as a `compaction` span inside the turn "
                "that triggered it."
            )

        print("\n--- telemetry ---")
    finally:
        telemetry.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
