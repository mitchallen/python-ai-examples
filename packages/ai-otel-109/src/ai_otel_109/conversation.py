"""A conversation that records what its memory costs.

One span per turn, carrying the conversation id so a backend can reassemble the
session, with the model call (and any compaction call) nested inside it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ai_otel_101 import semconv as sc
from ai_otel_101.instrumented import InstrumentedChat
from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind

from .memory import KeepAll, Message, Strategy, Summarizing, estimate_tokens

INSTRUMENTATION_NAME = "ai-otel-109"
INSTRUMENTATION_VERSION = "0.1.0"

# Semconv: correlates every span of one conversation.
CONVERSATION_ID = "gen_ai.conversation.id"

# Local: how the history was managed for this turn.
TURN = "app.conversation.turn"
MEMORY_STRATEGY = "app.memory.strategy"
MESSAGES_SENT = "app.memory.messages_sent"
MESSAGES_DROPPED = "app.memory.messages_dropped"
MESSAGES_SUMMARIZED = "app.memory.messages_summarized"
ESTIMATED_TOKENS = "app.memory.estimated_tokens"
COMPACTION = "app.memory.compaction"
METRIC_MESSAGES = "app.memory.messages_sent"
METRIC_DROPPED = "app.memory.messages_dropped"
METRIC_COMPACTIONS = "app.memory.compactions"

SUMMARY_PROMPT = (
    "Summarise the following conversation in two sentences, keeping names, "
    "numbers and decisions. Reply with the summary only."
)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class TurnResult:
    """One exchange, and what it cost."""

    turn: int
    text: str
    messages_sent: int = 0
    messages_dropped: int = 0
    estimated_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compacted: bool = False

    @property
    def estimate_error(self) -> float | None:
        """How far the pre-flight estimate was from the provider's count."""
        if not self.input_tokens:
            return None
        return (self.estimated_tokens - self.input_tokens) / self.input_tokens


@dataclass
class ConversationStats:
    """Running totals -- the numbers the per-call view never shows."""

    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compactions: int = 0
    per_turn: list[TurnResult] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TracedConversation:
    """Multi-turn chat with a pluggable memory strategy."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        system_prompt: str | None = None,
        strategy: Strategy | None = None,
        tracer: trace.Tracer | None = None,
        meter: metrics.Meter | None = None,
        conversation_id: str | None = None,
        chat: InstrumentedChat | None = None,
    ) -> None:
        self._model = model
        self._strategy = strategy or KeepAll()
        self.conversation_id = conversation_id or uuid.uuid4().hex[:12]
        self.history: list[Message] = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        self.stats = ConversationStats()

        self._tracer = tracer or trace.get_tracer(
            INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION
        )
        meter = meter or metrics.get_meter(
            INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION
        )
        # Model calls reuse the 101 wrapper, so each turn emits an ordinary
        # chat span with its token counts.
        self._chat = chat or InstrumentedChat(client, tracer=self._tracer, meter=meter)
        self._messages_metric = meter.create_histogram(
            name=METRIC_MESSAGES, unit="{message}",
            description="Messages sent to the model per turn.",
        )
        self._dropped_metric = meter.create_counter(
            name=METRIC_DROPPED, unit="{message}",
            description="Messages the memory strategy left out.",
        )
        self._compactions_metric = meter.create_counter(
            name=METRIC_COMPACTIONS, unit="{compaction}",
            description="History compactions performed.",
        )

        if isinstance(self._strategy, Summarizing) and self._strategy.summarizer is None:
            self._strategy.summarizer = self.summarize

    def ask(self, question: str, **kwargs: Any) -> TurnResult:
        """One exchange: trim the history, ask, remember the answer."""
        self.history.append({"role": "user", "content": question})
        turn_number = self.stats.turns + 1

        attributes = {
            CONVERSATION_ID: self.conversation_id,
            TURN: turn_number,
            MEMORY_STRATEGY: self._strategy.name,
            sc.REQUEST_MODEL: self._model,
        }

        with self._tracer.start_as_current_span(
            f"turn {turn_number}", kind=SpanKind.INTERNAL, attributes=attributes
        ) as span:
            compactions_before = getattr(self._strategy, "compactions", 0)
            decision = self._strategy.prepare(self.history)
            compacted = getattr(self._strategy, "compactions", 0) > compactions_before

            response = self._chat.complete(
                decision.messages, model=self._model, **kwargs
            )
            reply = _get(_get(_get(response, "choices", [None])[0], "message"), "content") or ""
            self.history.append({"role": "assistant", "content": reply})

            usage = _get(response, "usage")
            result = TurnResult(
                turn=turn_number,
                text=reply,
                messages_sent=len(decision.messages),
                messages_dropped=decision.dropped,
                estimated_tokens=decision.estimated_tokens,
                input_tokens=_get(usage, "prompt_tokens") or 0 if usage else 0,
                output_tokens=_get(usage, "completion_tokens") or 0 if usage else 0,
                compacted=compacted,
            )

            span.set_attribute(MESSAGES_SENT, result.messages_sent)
            span.set_attribute(MESSAGES_DROPPED, result.messages_dropped)
            span.set_attribute(ESTIMATED_TOKENS, result.estimated_tokens)
            if decision.summarized:
                span.set_attribute(MESSAGES_SUMMARIZED, decision.summarized)

            self._record(result, attributes)
            self._accumulate(result)
            return result

    def summarize(self, messages: Sequence[Message]) -> str:
        """Compact old turns with a model call of their own.

        Its span sits inside the turn that triggered it and is marked as
        compaction, so a strategy that doubles the request count cannot hide.
        """
        transcript = "\n".join(
            f"{m.get('role')}: {m.get('content')}" for m in messages
        )
        with self._tracer.start_as_current_span(
            "compaction", kind=SpanKind.INTERNAL,
            attributes={
                CONVERSATION_ID: self.conversation_id,
                COMPACTION: True,
                MESSAGES_SUMMARIZED: len(messages),
            },
        ):
            response = self._chat.complete(
                [
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                model=self._model,
            )
            usage = _get(response, "usage")
            if usage is not None:
                # Compaction tokens are real tokens: they belong in the total.
                self.stats.input_tokens += _get(usage, "prompt_tokens") or 0
                self.stats.output_tokens += _get(usage, "completion_tokens") or 0
            self.stats.compactions += 1
            self._compactions_metric.add(1, {CONVERSATION_ID: self.conversation_id})
            return _get(_get(_get(response, "choices", [None])[0], "message"), "content") or ""

    def estimated_history_tokens(self) -> int:
        return estimate_tokens(self.history)

    # -- recording ----------------------------------------------------------

    def _record(self, result: TurnResult, attributes: Mapping[str, Any]) -> None:
        metric_attributes = {
            MEMORY_STRATEGY: attributes[MEMORY_STRATEGY],
            sc.REQUEST_MODEL: attributes[sc.REQUEST_MODEL],
        }
        self._messages_metric.record(result.messages_sent, metric_attributes)
        if result.messages_dropped:
            self._dropped_metric.add(result.messages_dropped, metric_attributes)

    def _accumulate(self, result: TurnResult) -> None:
        self.stats.turns += 1
        self.stats.input_tokens += result.input_tokens
        self.stats.output_tokens += result.output_tokens
        self.stats.per_turn.append(result)
