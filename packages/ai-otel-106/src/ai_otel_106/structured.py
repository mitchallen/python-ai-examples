"""Structured outputs, classified rather than assumed.

A completion that "succeeded" can still be unusable: valid JSON in the wrong
shape, a refusal, or an object cut in half by the token limit. Each of those is
a different thing to do about it, so each gets its own outcome on the span
instead of collapsing into success-or-exception.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai_otel_101 import semconv as sc
from ai_otel_101.instrumented import provider_from_base_url
from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind
from pydantic import BaseModel, ValidationError

from .schema import response_format

INSTRUMENTATION_NAME = "ai-otel-106"
INSTRUMENTATION_VERSION = "0.1.0"

# Semconv: the kind of output requested ("text", "json", "image", "speech").
OUTPUT_TYPE = "gen_ai.output.type"
OUTPUT_TYPE_JSON = "json"

# Local. The outcome taxonomy is this example's own; semconv has no attribute
# for "the JSON parsed but did not match the schema", which is the failure that
# actually reaches production.
OUTPUT_SCHEMA = "app.output.schema"
OUTPUT_OUTCOME = "app.output.outcome"
METRIC_OUTPUTS = "app.structured.outputs"

PARSED = "parsed"
REFUSED = "refused"
INVALID_JSON = "invalid_json"
SCHEMA_INVALID = "schema_invalid"
TRUNCATED = "truncated"

#: Outcomes that mean the caller got nothing usable. A refusal is not among
#: them: the model did its job, and burying refusals in the error rate hides a
#: prompt problem inside what looks like an outage.
FAILURES = frozenset({INVALID_JSON, SCHEMA_INVALID, TRUNCATED})


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class StructuredResult:
    """What came back, and whether it can be used."""

    outcome: str
    parsed: BaseModel | None = None
    raw: str | None = None
    refusal: str | None = None
    error: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == PARSED

    @property
    def failed(self) -> bool:
        """A refusal is not a failure -- see FAILURES."""
        return self.outcome in FAILURES


class StructuredChat:
    """Requests a schema, then checks that it actually got one."""

    def __init__(
        self,
        client: Any,
        *,
        tracer: trace.Tracer | None = None,
        meter: metrics.Meter | None = None,
        provider: str | None = None,
    ) -> None:
        self._client = client
        self._provider = provider or provider_from_base_url(
            getattr(client, "base_url", None)
        )
        self._tracer = tracer or trace.get_tracer(
            INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION
        )
        meter = meter or metrics.get_meter(
            INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION
        )
        self._token_usage = meter.create_histogram(
            name=sc.METRIC_TOKEN_USAGE,
            unit="{token}",
            description="Number of input and output tokens used.",
        )
        self._duration = meter.create_histogram(
            name=sc.METRIC_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration.",
        )
        # A counter, not a histogram: these are events with a category, and the
        # question is "what share came back unusable?"
        self._outcomes = meter.create_counter(
            name=METRIC_OUTPUTS,
            unit="{output}",
            description="Structured output results by outcome.",
        )

    def parse(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        schema: type[BaseModel],
        strict: bool = True,
        **kwargs: Any,
    ) -> StructuredResult:
        """Ask for `schema`, and report what actually arrived."""
        attributes: dict[str, Any] = {
            sc.OPERATION_NAME: sc.OPERATION_CHAT,
            sc.SYSTEM: self._provider,
            sc.PROVIDER_NAME: self._provider,
            sc.REQUEST_MODEL: model,
            OUTPUT_TYPE: OUTPUT_TYPE_JSON,
            OUTPUT_SCHEMA: schema.__name__,
        }
        started = time.perf_counter()

        with self._tracer.start_as_current_span(
            f"{sc.OPERATION_CHAT} {model}",
            kind=SpanKind.CLIENT,
            attributes=attributes,
        ) as span:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=list(messages),
                    response_format=response_format(schema, strict=strict),
                    **kwargs,
                )
            except Exception as exc:
                error_type = type(exc).__qualname__
                span.set_attribute(sc.ERROR_TYPE, error_type)
                self._duration.record(
                    time.perf_counter() - started,
                    {**attributes, sc.ERROR_TYPE: error_type},
                )
                raise

            result = self._classify(response, schema)
            self._record(span, result, attributes, time.perf_counter() - started)
            return result

    # -- classification -----------------------------------------------------

    def _classify(self, response: Any, schema: type[BaseModel]) -> StructuredResult:
        choice = (_get(response, "choices") or [None])[0]
        message = _get(choice, "message")
        finish_reason = _get(choice, "finish_reason")
        usage = _get(response, "usage")

        result = StructuredResult(
            outcome=PARSED,
            finish_reason=finish_reason,
            input_tokens=_get(usage, "prompt_tokens") if usage else None,
            output_tokens=_get(usage, "completion_tokens") if usage else None,
        )

        refusal = _get(message, "refusal")
        if refusal:
            # The model declined. Not an error: it worked as designed.
            result.outcome = REFUSED
            result.refusal = refusal
            return result

        content = _get(message, "content")
        result.raw = content

        if finish_reason == "length":
            # The object stops mid-key. Parsing it is pointless.
            result.outcome = TRUNCATED
            result.error = "response hit the token limit before the JSON closed"
            return result

        try:
            payload = json.loads(content or "")
        except (TypeError, ValueError) as exc:
            result.outcome = INVALID_JSON
            result.error = str(exc)
            return result

        try:
            result.parsed = schema.model_validate(payload)
        except ValidationError as exc:
            # Valid JSON, wrong shape. The failure a naive implementation calls
            # a success.
            result.outcome = SCHEMA_INVALID
            result.error = f"{exc.error_count()} schema violation(s): {exc.errors()[0]['msg']}"
            return result

        return result

    def _record(
        self,
        span: trace.Span,
        result: StructuredResult,
        attributes: Mapping[str, Any],
        elapsed: float,
    ) -> None:
        span.set_attribute(OUTPUT_OUTCOME, result.outcome)
        if result.finish_reason:
            span.set_attribute(sc.RESPONSE_FINISH_REASONS, [result.finish_reason])
        if result.input_tokens is not None:
            span.set_attribute(sc.USAGE_INPUT_TOKENS, result.input_tokens)
        if result.output_tokens is not None:
            span.set_attribute(sc.USAGE_OUTPUT_TOKENS, result.output_tokens)
        if result.failed:
            # error.type carries the outcome, so a backend that only knows
            # error.type still sees these; a refusal deliberately does not.
            span.set_attribute(sc.ERROR_TYPE, result.outcome)

        outcome_attributes = {**attributes, OUTPUT_OUTCOME: result.outcome}
        for count, token_type in (
            (result.input_tokens, sc.TOKEN_TYPE_INPUT),
            (result.output_tokens, sc.TOKEN_TYPE_OUTPUT),
        ):
            if count is not None:
                self._token_usage.record(
                    count, {**attributes, sc.TOKEN_TYPE: token_type}
                )

        self._outcomes.add(1, outcome_attributes)
        self._duration.record(elapsed, outcome_attributes)
