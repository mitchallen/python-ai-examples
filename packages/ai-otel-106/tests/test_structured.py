"""Every outcome a structured call can have, and how each is recorded."""

from __future__ import annotations

from typing import Any

import pytest
from ai_otel_101 import semconv as sc
from opentelemetry.trace import StatusCode

from ai_otel_106 import (
    INVALID_JSON,
    METRIC_OUTPUTS,
    OUTPUT_OUTCOME,
    OUTPUT_SCHEMA,
    OUTPUT_TYPE,
    PARSED,
    REFUSED,
    SCHEMA_INVALID,
    TRUNCATED,
    StructuredChat,
)

MESSAGES = [{"role": "user", "content": "Describe the Black Pearl."}]

# Exactly what llama3.2:3b returned in json_object mode: valid JSON, useless.
REAL_WORLD_JUNK = '{ "The Black Pearl: An Infamous Pirate Ship":\n\n\n 1.4 }'


def points(metric_reader: Any, name: str) -> list[Any]:
    data = metric_reader.get_metrics_data()
    return [
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def parse(client, ship, tracer, meter, **kwargs):
    return StructuredChat(client, tracer=tracer, meter=meter).parse(
        MESSAGES, model="gpt-4o-mini", schema=ship, **kwargs
    )


# -- the good case -----------------------------------------------------------


def test_a_matching_response_parses_into_the_model(client, ship, tracer, meter):
    result = parse(client, ship, tracer, meter)

    assert result.outcome == PARSED
    assert result.ok is True
    assert result.failed is False
    # A Ship, not a dict.
    assert isinstance(result.parsed, ship)
    assert (result.parsed.name, result.parsed.crew) == ("Black Pearl", 40)


def test_the_span_says_what_was_asked_for_and_what_arrived(
    client, ship, tracer, meter, spans
):
    parse(client, ship, tracer, meter)

    (span,) = spans.get_finished_spans()
    assert span.attributes[OUTPUT_TYPE] == "json"
    assert span.attributes[OUTPUT_SCHEMA] == "Ship"
    assert span.attributes[OUTPUT_OUTCOME] == PARSED
    assert span.attributes[sc.USAGE_INPUT_TOKENS] == 32
    assert sc.ERROR_TYPE not in span.attributes


def test_the_request_carries_a_strict_schema(client, ship, tracer, meter):
    parse(client, ship, tracer, meter)

    fmt = client.calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "Ship"
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False


def test_loose_mode_asks_only_for_json(client, ship, tracer, meter):
    parse(client, ship, tracer, meter, strict=False)

    assert client.calls[0]["response_format"] == {"type": "json_object"}


# -- the failures ------------------------------------------------------------


def test_valid_json_in_the_wrong_shape_is_not_a_success(
    ship, tracer, meter, spans, make_client, responses
):
    client = make_client(responses.returns(REAL_WORLD_JUNK))

    result = parse(client, ship, tracer, meter, strict=False)

    assert result.outcome == SCHEMA_INVALID
    assert result.ok is False
    assert result.failed is True
    assert result.parsed is None
    # The raw text is kept, because you cannot debug what you did not record.
    assert result.raw == REAL_WORLD_JUNK
    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.ERROR_TYPE] == SCHEMA_INVALID


def test_an_empty_object_is_schema_invalid(ship, tracer, meter, make_client, responses):
    client = make_client(responses.returns("{}"))

    result = parse(client, ship, tracer, meter, strict=False)

    assert result.outcome == SCHEMA_INVALID
    assert "Field required" in result.error


def test_unparseable_output_is_invalid_json(
    ship, tracer, meter, spans, make_client, responses
):
    client = make_client(responses.returns("Arrr, here be the ship:"))

    result = parse(client, ship, tracer, meter)

    assert result.outcome == INVALID_JSON
    assert result.failed is True
    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.ERROR_TYPE] == INVALID_JSON


def test_a_truncated_object_is_reported_as_truncated(
    ship, tracer, meter, spans, make_client, responses
):
    # finish_reason=length: the JSON stops mid-object, so parsing is pointless.
    client = make_client(
        responses.returns('{"name": "Black Pea', finish_reason="length")
    )

    result = parse(client, ship, tracer, meter)

    assert result.outcome == TRUNCATED
    assert "token limit" in result.error
    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.ERROR_TYPE] == TRUNCATED
    assert span.attributes[sc.RESPONSE_FINISH_REASONS] == ("length",)


# -- refusals are not failures ----------------------------------------------


def test_a_refusal_is_recorded_as_its_own_outcome(
    ship, tracer, meter, make_client, responses
):
    client = make_client(responses.refuses("I can't help with that."))

    result = parse(client, ship, tracer, meter)

    assert result.outcome == REFUSED
    assert result.refusal == "I can't help with that."
    assert result.parsed is None


def test_a_refusal_is_not_an_error(ship, tracer, meter, spans, make_client, responses):
    # A refusal means the model worked. Filing it as an error hides a prompt
    # problem inside the error rate.
    client = make_client(responses.refuses())

    result = parse(client, ship, tracer, meter)

    assert result.failed is False
    (span,) = spans.get_finished_spans()
    assert span.attributes[OUTPUT_OUTCOME] == REFUSED
    assert sc.ERROR_TYPE not in span.attributes
    assert span.status.status_code is not StatusCode.ERROR


# -- metrics -----------------------------------------------------------------


def test_the_outcome_counter_splits_by_outcome(
    ship, tracer, meter, metric_reader, make_client, responses
):
    structured = StructuredChat(
        make_client(responses.returns("{}")), tracer=tracer, meter=meter
    )
    for _ in range(3):
        structured.parse(MESSAGES, model="gpt-4o-mini", schema=ship, strict=False)

    (point,) = points(metric_reader, METRIC_OUTPUTS)
    assert point.attributes[OUTPUT_OUTCOME] == SCHEMA_INVALID
    assert point.value == 3


def test_tokens_are_recorded_even_when_the_shape_is_wrong(
    ship, tracer, meter, metric_reader, make_client, responses
):
    # An unusable answer still cost money.
    client = make_client(responses.returns(REAL_WORLD_JUNK))

    parse(client, ship, tracer, meter, strict=False)

    by_type = {
        p.attributes[sc.TOKEN_TYPE]: p.sum
        for p in points(metric_reader, sc.METRIC_TOKEN_USAGE)
    }
    assert by_type == {"input": 32, "output": 31}


def test_duration_is_keyed_on_the_outcome(
    ship, tracer, meter, metric_reader, client
):
    parse(client, ship, tracer, meter)

    (point,) = points(metric_reader, sc.METRIC_OPERATION_DURATION)
    assert point.attributes[OUTPUT_OUTCOME] == PARSED


def test_an_api_failure_still_raises(ship, tracer, meter, spans, make_client):
    client = make_client(error=RuntimeError("rate limited"))

    with pytest.raises(RuntimeError):
        parse(client, ship, tracer, meter)

    (span,) = spans.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[sc.ERROR_TYPE] == "RuntimeError"
