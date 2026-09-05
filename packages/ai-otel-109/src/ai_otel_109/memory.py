"""Memory strategies: what to send back, and what to leave behind.

A chat model remembers nothing, so every turn resends the history and the cost
of a conversation grows with the square of its length. These are the usual ways
to bend that curve, behind one interface so the telemetry can say which one was
in force and what it discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

Message = dict[str, Any]

#: Characters per token. A heuristic, not a tokeniser: trimming happens before
#: the call, when the real count does not exist yet. Anything derived from it
#: is named "estimated" for that reason.
CHARS_PER_TOKEN = 4


def estimate_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Roughly how many tokens these messages will cost."""
    characters = sum(len(str(message.get("content") or "")) for message in messages)
    return characters // CHARS_PER_TOKEN


@dataclass
class MemoryDecision:
    """What the strategy chose to send, and what it dropped to get there."""

    messages: list[Message] = field(default_factory=list)
    dropped: int = 0
    summarized: int = 0
    estimated_tokens: int = 0


def _split_system(history: Sequence[Message]) -> tuple[list[Message], list[Message]]:
    """System messages are instructions, not history -- they are never trimmed."""
    system = [m for m in history if m.get("role") == "system"]
    rest = [m for m in history if m.get("role") != "system"]
    return system, rest


def _split_pending(rest: Sequence[Message]) -> tuple[list[Message], list[Message]]:
    """Hold aside the question being asked right now.

    Trimming counts backwards from the end, and the newest message is usually
    the unanswered question. Counting it as history splits an exchange in half
    and can leave the window opening on an assistant reply to a question the
    model can no longer see.
    """
    if rest and rest[-1].get("role") == "user":
        return list(rest[:-1]), [rest[-1]]
    return list(rest), []


def _drop_leading_answer(messages: Sequence[Message]) -> list[Message]:
    """An assistant message with no question in front of it is noise."""
    kept = list(messages)
    while kept and kept[0].get("role") == "assistant":
        kept.pop(0)
    return kept


class Strategy:
    """Base: decide which messages go into the next request."""

    name = "base"

    def prepare(self, history: Sequence[Message]) -> MemoryDecision:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _decide(messages: list[Message], dropped: int = 0, summarized: int = 0) -> MemoryDecision:
        return MemoryDecision(
            messages=messages,
            dropped=dropped,
            summarized=summarized,
            estimated_tokens=estimate_tokens(messages),
        )


class KeepAll(Strategy):
    """Send everything. Simple, correct, and quadratic."""

    name = "keep_all"

    def prepare(self, history: Sequence[Message]) -> MemoryDecision:
        return self._decide(list(history))


class SlidingWindow(Strategy):
    """Keep the system prompt and the last `max_turns` exchanges.

    Cheap and predictable, and it forgets without telling anyone -- which is
    exactly why the count of dropped messages belongs on the span.
    """

    name = "sliding_window"

    def __init__(self, max_turns: int = 3) -> None:
        self.max_turns = max_turns

    def prepare(self, history: Sequence[Message]) -> MemoryDecision:
        system, rest = _split_system(history)
        answered, pending = _split_pending(rest)
        keep = self.max_turns * 2  # a turn is a user message plus a reply

        kept = _drop_leading_answer(answered[-keep:] if keep else [])
        messages = system + kept + pending
        return self._decide(messages, dropped=len(rest) - len(kept) - len(pending))


class TokenBudget(Strategy):
    """Keep as much recent history as fits a token budget.

    Trims by size rather than by count, which is what actually matters when
    turns vary from "yes" to a pasted stack trace.
    """

    name = "token_budget"

    def __init__(self, max_tokens: int = 400) -> None:
        self.max_tokens = max_tokens

    def prepare(self, history: Sequence[Message]) -> MemoryDecision:
        system, rest = _split_system(history)
        kept: list[Message] = []

        # Newest first: the most recent turns are the ones worth keeping.
        for message in reversed(rest):
            candidate = [message, *kept]
            if estimate_tokens(system + candidate) > self.max_tokens and kept:
                break
            kept = candidate

        # The oldest kept message may be an answer whose question was trimmed.
        if len(kept) > 1:
            kept = _drop_leading_answer(kept)
        return self._decide(system + kept, dropped=len(rest) - len(kept))


class Summarizing(Strategy):
    """Compress old turns into a summary, keeping the recent ones verbatim.

    The summary is produced by a model call, so this strategy is the only one
    here that costs money to run. That call is instrumented separately -- see
    the compaction span -- because a strategy that quietly doubles your request
    count should not be invisible.
    """

    name = "summarizing"

    def __init__(
        self,
        summarizer: Callable[[Sequence[Message]], str] | None = None,
        *,
        keep_turns: int = 2,
        trigger_after: int = 6,
        recompact_after: int = 4,
    ) -> None:
        # Optional so a TracedConversation can wire in its own instrumented
        # summariser; supply one directly to compact without a model call.
        self.summarizer = summarizer
        self.keep_turns = keep_turns
        self.trigger_after = trigger_after
        # Re-summarising on every turn costs a model call per turn and saves
        # almost nothing, which measured *worse* than keeping everything. Wait
        # until this many messages have aged out since the last compaction.
        self.recompact_after = recompact_after
        self.summary: str | None = None
        self.summarized_count = 0
        self.compactions = 0

    def prepare(self, history: Sequence[Message]) -> MemoryDecision:
        system, rest = _split_system(history)
        keep = self.keep_turns * 2

        if len(rest) <= max(self.trigger_after, keep):
            return self._decide(system + list(rest))

        older, recent = rest[:-keep], rest[-keep:]
        # Messages that have aged out but are not in the summary yet. They ride
        # along verbatim until the next compaction, so nothing is ever dropped
        # without first being summarised.
        pending = older[self.summarized_count :]
        summarized = 0

        if len(pending) >= self.recompact_after:
            self.summary = self.summarizer(older)
            self.summarized_count = len(older)
            summarized = len(older)
            pending = []
            self.compactions += 1

        prefix = [self._summary_message()] if self.summary else []
        return self._decide(
            system + prefix + list(pending) + list(recent), summarized=summarized
        )

    def _summary_message(self) -> Message:
        # A system message: it is context about the conversation, not a turn in
        # it, and labelling it keeps the model from answering it.
        return {
            "role": "system",
            "content": f"Summary of the conversation so far: {self.summary}",
        }
