"""The tool-calling loop, instrumented.

One user question becomes several model calls: the model asks for a function,
you run it, you send the result back, it answers. Each round resends everything
said so far plus the tool output, so input tokens climb every round without the
user typing anything. The trace makes that visible -- one ``invoke_agent`` span
over alternating ``chat`` and ``execute_tool`` children.

Model calls are delegated to :class:`ai_otel_101.InstrumentedChat`, so they emit
exactly what a plain completion emits and land in the same dashboards. This
module adds only what tool calling introduces.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ai_otel_101.instrumented import InstrumentedChat
from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind

from . import semconv as sc

INSTRUMENTATION_NAME = "ai-otel-105"
INSTRUMENTATION_VERSION = "0.1.0"

DEFAULT_AGENT_NAME = "quartermaster"
DEFAULT_MAX_ROUNDS = 4


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _message_to_dict(message: Any) -> dict[str, Any]:
    """The assistant turn, in the shape the next request needs.

    The SDK hands back a pydantic model; test stubs hand back something
    simpler. Both have to round-trip into the message list.
    """
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        return {k: v for k, v in dump(exclude_none=True).items() if v not in ([], {})}

    payload: dict[str, Any] = {"role": _get(message, "role", "assistant")}
    content = _get(message, "content")
    if content is not None:
        payload["content"] = content
    tool_calls = _get(message, "tool_calls") or []
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": _get(call, "id"),
                "type": "function",
                "function": {
                    "name": _get(_get(call, "function"), "name"),
                    "arguments": _get(_get(call, "function"), "arguments"),
                },
            }
            for call in tool_calls
        ]
    return payload


@dataclass
class Turn:
    """What one user question actually cost."""

    text: str
    rounds: int = 0
    tool_calls: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    # (input, output) per round. Kept separately because how a growing
    # conversation maps to billed tokens is provider-specific -- measure it,
    # do not assume it.
    usage_by_round: list[tuple[int, int]] = field(default_factory=list)
    truncated: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ToolCallingChat:
    """Runs the model/tool loop and traces the whole turn."""

    def __init__(
        self,
        client: Any,
        registry: Any,
        *,
        tracer: trace.Tracer | None = None,
        meter: metrics.Meter | None = None,
        agent_name: str = DEFAULT_AGENT_NAME,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        chat: InstrumentedChat | None = None,
    ) -> None:
        self._registry = registry
        self._agent_name = agent_name
        self._max_rounds = max_rounds
        self._tracer = tracer or trace.get_tracer(
            INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION
        )
        meter = meter or metrics.get_meter(
            INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION
        )
        # Model calls go through the 101 wrapper: same spans, same histograms.
        self._chat = chat or InstrumentedChat(client, tracer=self._tracer, meter=meter)
        self._duration = meter.create_histogram(
            name=sc.METRIC_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration.",
        )

    def run(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        **kwargs: Any,
    ) -> Turn:
        """Ask, run whatever tools the model asks for, ask again, until it stops."""
        conversation: list[dict[str, Any]] = [dict(m) for m in messages]
        turn = Turn(text="", messages=conversation)

        with self._tracer.start_as_current_span(
            f"{sc.OPERATION_INVOKE_AGENT} {self._agent_name}",
            kind=SpanKind.INTERNAL,
            attributes={
                sc.OPERATION_NAME: sc.OPERATION_INVOKE_AGENT,
                sc.AGENT_NAME: self._agent_name,
                sc.REQUEST_MODEL: model,
            },
        ) as span:
            for _ in range(self._max_rounds):
                response = self._chat.complete(
                    conversation,
                    model=model,
                    tools=self._registry.schemas(),
                    **kwargs,
                )
                turn.rounds += 1
                self._accumulate_usage(turn, response)

                message = _get(_get(response, "choices", [None])[0], "message")
                tool_calls = _get(message, "tool_calls") or []

                if not tool_calls:
                    turn.text = _get(message, "content") or ""
                    break

                # The assistant's request has to go back verbatim, or the tool
                # results have nothing to attach to.
                conversation.append(_message_to_dict(message))
                for call in tool_calls:
                    conversation.append(self._execute(call, turn))
            else:
                # Ran out of rounds with the model still asking for tools.
                turn.truncated = True
                span.set_attribute(sc.AGENT_TRUNCATED, True)

            span.set_attribute(sc.AGENT_ROUNDS, turn.rounds)
            span.set_attribute(sc.AGENT_TOOL_CALLS, len(turn.tool_calls))

        return turn

    # -- tools --------------------------------------------------------------

    def _execute(self, call: Any, turn: Turn) -> dict[str, Any]:
        """Run one requested tool inside its own span; never raise."""
        call_id = _get(call, "id") or ""
        function = _get(call, "function")
        name = _get(function, "name") or ""
        raw_arguments = _get(function, "arguments") or "{}"
        tool = self._registry.get(name)

        attributes: dict[str, Any] = {
            sc.OPERATION_NAME: sc.OPERATION_EXECUTE_TOOL,
            sc.TOOL_NAME: name,
            sc.TOOL_CALL_ID: call_id,
            sc.TOOL_TYPE: sc.TOOL_TYPE_FUNCTION,
        }
        if tool is not None:
            attributes[sc.TOOL_DESCRIPTION] = tool.description

        turn.tool_calls.append(name)
        started = time.perf_counter()

        with self._tracer.start_as_current_span(
            f"{sc.OPERATION_EXECUTE_TOOL} {name}",
            kind=SpanKind.INTERNAL,
            attributes=attributes,
        ) as span:
            content, error_type = self._invoke(tool, name, raw_arguments)
            if error_type is not None:
                # Recorded here, but handed back to the model as text: it can
                # often recover, and killing the turn would guarantee it can't.
                span.set_attribute(sc.ERROR_TYPE, error_type)
            self._duration.record(
                time.perf_counter() - started,
                {**attributes, **({sc.ERROR_TYPE: error_type} if error_type else {})},
            )

        return {"role": "tool", "tool_call_id": call_id, "content": content}

    def _invoke(
        self, tool: Any, name: str, raw_arguments: str
    ) -> tuple[str, str | None]:
        if tool is None:
            return f"error: no tool named {name!r}", "UnknownTool"

        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, ValueError) as exc:
            # Models do emit malformed arguments. Tell the model, don't crash.
            return f"error: arguments were not valid JSON ({exc})", "JSONDecodeError"

        try:
            return str(tool.run(**arguments)), None
        except TypeError as exc:
            return f"error: wrong arguments for {name} ({exc})", "TypeError"
        except Exception as exc:  # noqa: BLE001 - the model gets to see this
            return f"error: {name} failed ({exc})", type(exc).__qualname__

    # -- accounting ---------------------------------------------------------

    @staticmethod
    def _accumulate_usage(turn: Turn, response: Any) -> None:
        usage = _get(response, "usage")
        if usage is None:
            turn.usage_by_round.append((0, 0))
            return
        prompt = _get(usage, "prompt_tokens") or 0
        completion = _get(usage, "completion_tokens") or 0
        turn.input_tokens += prompt
        turn.output_tokens += completion
        turn.usage_by_round.append((prompt, completion))
