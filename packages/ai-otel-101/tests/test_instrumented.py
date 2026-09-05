"""What the wrapper must emit, asserted against in-memory spans and metrics."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import FakeClient, FakeResponse
from opentelemetry.trace import SpanKind, StatusCode

from ai_otel_101 import semconv as sc
from ai_otel_101.instrumented import InstrumentedChat

MESSAGES = [
    {"role": "system", "content": "You are a pirate."},
    {"role": "user", "content": "Hello"},
]


def points(metric_reader: Any, metric_name: str) -> list[Any]:
    """Every data point recorded for one instrument."""
    data = metric_reader.get_metrics_data()
    found = []
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == metric_name:
                    found.extend(metric.data.data_points)
    return found


def test_span_is_named_operation_and_model(tracer, meter, spans, client):
    InstrumentedChat(client, tracer=tracer, meter=meter).complete(
        MESSAGES, model="gpt-4o-mini"
    )

    (span,) = spans.get_finished_spans()
    assert span.name == "chat gpt-4o-mini"
    assert span.kind is SpanKind.CLIENT


def test_span_carries_request_and_response_attributes(tracer, meter, spans, client):
    InstrumentedChat(client, tracer=tracer, meter=meter).complete(
        MESSAGES, model="gpt-4o-mini"
    )

    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.OPERATION_NAME] == "chat"
    assert span.attributes[sc.SYSTEM] == "openai"
    assert span.attributes[sc.PROVIDER_NAME] == "openai"
    assert span.attributes[sc.REQUEST_MODEL] == "gpt-4o-mini"
    assert span.attributes[sc.RESPONSE_ID] == "chatcmpl-123"
    assert span.attributes[sc.RESPONSE_MODEL] == "gpt-4o-mini-2024-07-18"
    assert span.attributes[sc.RESPONSE_FINISH_REASONS] == ("stop",)


def test_token_counts_land_on_the_span(tracer, meter, spans, client):
    InstrumentedChat(client, tracer=tracer, meter=meter).complete(
        MESSAGES, model="gpt-4o-mini"
    )

    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.USAGE_INPUT_TOKENS] == 11
    assert span.attributes[sc.USAGE_OUTPUT_TOKENS] == 7


def test_sampling_parameters_are_recorded_only_when_set(tracer, meter, spans, client):
    chat = InstrumentedChat(client, tracer=tracer, meter=meter)

    chat.complete(MESSAGES, model="gpt-4o-mini")
    chat.complete(MESSAGES, model="gpt-4o-mini", temperature=0.2, max_tokens=64)

    plain, tuned = spans.get_finished_spans()
    assert sc.REQUEST_TEMPERATURE not in plain.attributes
    assert tuned.attributes[sc.REQUEST_TEMPERATURE] == 0.2
    assert tuned.attributes[sc.REQUEST_MAX_TOKENS] == 64
    # Extra kwargs are forwarded to the client untouched.
    assert client.calls[1]["temperature"] == 0.2


def test_token_usage_metric_splits_input_and_output(
    tracer, meter, metric_reader, client
):
    InstrumentedChat(client, tracer=tracer, meter=meter).complete(
        MESSAGES, model="gpt-4o-mini"
    )

    by_type = {
        point.attributes[sc.TOKEN_TYPE]: point
        for point in points(metric_reader, sc.METRIC_TOKEN_USAGE)
    }
    assert by_type[sc.TOKEN_TYPE_INPUT].sum == 11
    assert by_type[sc.TOKEN_TYPE_OUTPUT].sum == 7
    # Billing follows the model that answered, so it is on the metric too.
    assert by_type[sc.TOKEN_TYPE_INPUT].attributes[sc.RESPONSE_MODEL] == (
        "gpt-4o-mini-2024-07-18"
    )


def test_duration_metric_is_recorded_once_per_call(tracer, meter, metric_reader, client):
    chat = InstrumentedChat(client, tracer=tracer, meter=meter)

    chat.complete(MESSAGES, model="gpt-4o-mini")
    chat.complete(MESSAGES, model="gpt-4o-mini")

    (point,) = points(metric_reader, sc.METRIC_OPERATION_DURATION)
    assert point.count == 2
    assert point.sum >= 0


def test_missing_usage_is_tolerated(tracer, meter, spans, metric_reader):
    client = FakeClient(response=FakeResponse(usage=None))

    InstrumentedChat(client, tracer=tracer, meter=meter).complete(
        MESSAGES, model="gpt-4o-mini"
    )

    (span,) = spans.get_finished_spans()
    assert sc.USAGE_INPUT_TOKENS not in span.attributes
    assert points(metric_reader, sc.METRIC_TOKEN_USAGE) == []
    # Latency is still measurable without a usage block.
    assert points(metric_reader, sc.METRIC_OPERATION_DURATION)


def test_failure_marks_the_span_and_the_duration_metric(
    tracer, meter, spans, metric_reader
):
    client = FakeClient(error=RuntimeError("rate limited"))
    chat = InstrumentedChat(client, tracer=tracer, meter=meter)

    with pytest.raises(RuntimeError):
        chat.complete(MESSAGES, model="gpt-4o-mini")

    (span,) = spans.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[sc.ERROR_TYPE] == "RuntimeError"
    assert span.events  # the exception is recorded on the span
    (point,) = points(metric_reader, sc.METRIC_OPERATION_DURATION)
    assert point.attributes[sc.ERROR_TYPE] == "RuntimeError"


def test_content_is_not_captured_by_default(tracer, meter, spans, client):
    InstrumentedChat(client, tracer=tracer, meter=meter).complete(
        MESSAGES, model="gpt-4o-mini"
    )

    (span,) = spans.get_finished_spans()
    assert span.events == ()
    assert "Hello" not in str(span.attributes)


def test_content_capture_can_be_opted_into(tracer, meter, spans, client):
    InstrumentedChat(
        client, tracer=tracer, meter=meter, capture_content=True
    ).complete(MESSAGES, model="gpt-4o-mini")

    (span,) = spans.get_finished_spans()
    names = [event.name for event in span.events]
    assert names == [sc.EVENT_SYSTEM_MESSAGE, sc.EVENT_USER_MESSAGE, sc.EVENT_CHOICE]
    assert span.events[1].attributes["content"] == "Hello"
    assert span.events[2].attributes["content"] == "Ahoy, matey!"


def test_capture_content_honors_the_env_var(
    tracer, meter, spans, client, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(sc.CAPTURE_CONTENT_ENV, "true")

    InstrumentedChat(client, tracer=tracer, meter=meter).complete(
        MESSAGES, model="gpt-4o-mini"
    )

    (span,) = spans.get_finished_spans()
    assert span.events


def test_messages_reach_the_client_unchanged(tracer, meter, client):
    InstrumentedChat(client, tracer=tracer, meter=meter).complete(
        MESSAGES, model="gpt-4o-mini"
    )

    assert client.calls[0]["messages"] == MESSAGES
    assert client.calls[0]["model"] == "gpt-4o-mini"
