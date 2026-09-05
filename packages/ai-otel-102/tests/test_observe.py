"""The context-manager instrumentation, checked against in-memory telemetry."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.trace import SpanKind, StatusCode

import ai_otel_102.observe as obs
from ai_otel_102 import ChatTelemetry, build_conversation
from ai_otel_102.observe import PIRATE_SYSTEM_PROMPT, ask_pirate


def points(metric_reader: Any, metric_name: str) -> list[Any]:
    data = metric_reader.get_metrics_data()
    found = []
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == metric_name:
                    found.extend(metric.data.data_points)
    return found


def test_build_conversation_is_system_then_user():
    assert build_conversation("Hello") == [
        {"role": "system", "content": PIRATE_SYSTEM_PROMPT},
        {"role": "user", "content": "Hello"},
    ]


def test_span_is_named_operation_and_model(chat_telemetry, spans, client):
    with chat_telemetry.chat("gpt-4o-mini") as observed:
        observed.record(client.chat.completions.create())

    (span,) = spans.get_finished_spans()
    assert span.name == "chat gpt-4o-mini"
    assert span.kind is SpanKind.CLIENT


def test_recording_a_response_populates_the_span(chat_telemetry, spans, client):
    with chat_telemetry.chat("gpt-4o-mini") as observed:
        observed.record(client.chat.completions.create())

    (span,) = spans.get_finished_spans()
    assert span.attributes[obs.OPERATION_NAME] == "chat"
    assert span.attributes[obs.SYSTEM] == "openai"
    assert span.attributes[obs.PROVIDER_NAME] == "openai"
    assert span.attributes[obs.REQUEST_MODEL] == "gpt-4o-mini"
    assert span.attributes[obs.RESPONSE_ID] == "chatcmpl-123"
    assert span.attributes[obs.RESPONSE_MODEL] == "gpt-4o-mini-2024-07-18"
    assert span.attributes[obs.RESPONSE_FINISH_REASONS] == ("stop",)
    assert span.attributes[obs.USAGE_INPUT_TOKENS] == 11
    assert span.attributes[obs.USAGE_OUTPUT_TOKENS] == 7


def test_sampling_parameters_are_recorded_only_when_set(chat_telemetry, spans, client):
    with chat_telemetry.chat("gpt-4o-mini") as observed:
        observed.record(client.chat.completions.create())
    with chat_telemetry.chat("gpt-4o-mini", temperature=0.2, max_tokens=64) as observed:
        observed.record(client.chat.completions.create())

    plain, tuned = spans.get_finished_spans()
    assert obs.REQUEST_TEMPERATURE not in plain.attributes
    assert tuned.attributes[obs.REQUEST_TEMPERATURE] == 0.2
    assert tuned.attributes[obs.REQUEST_MAX_TOKENS] == 64


def test_token_usage_metric_splits_input_and_output(
    chat_telemetry, metric_reader, client
):
    with chat_telemetry.chat("gpt-4o-mini") as observed:
        observed.record(client.chat.completions.create())

    by_type = {
        point.attributes[obs.TOKEN_TYPE]: point
        for point in points(metric_reader, obs.METRIC_TOKEN_USAGE)
    }
    assert by_type[obs.TOKEN_TYPE_INPUT].sum == 11
    assert by_type[obs.TOKEN_TYPE_OUTPUT].sum == 7
    assert by_type[obs.TOKEN_TYPE_OUTPUT].attributes[obs.RESPONSE_MODEL] == (
        "gpt-4o-mini-2024-07-18"
    )


def test_duration_is_recorded_once_per_block(chat_telemetry, metric_reader, client):
    for _ in range(2):
        with chat_telemetry.chat("gpt-4o-mini") as observed:
            observed.record(client.chat.completions.create())

    (point,) = points(metric_reader, obs.METRIC_OPERATION_DURATION)
    assert point.count == 2
    assert point.sum >= 0


def test_block_that_never_records_still_closes_cleanly(
    chat_telemetry, spans, metric_reader
):
    # An early return, a cache hit -- the span must not depend on record().
    with chat_telemetry.chat("gpt-4o-mini"):
        pass

    (span,) = spans.get_finished_spans()
    assert span.status.status_code is not StatusCode.ERROR
    assert obs.USAGE_INPUT_TOKENS not in span.attributes
    assert points(metric_reader, obs.METRIC_TOKEN_USAGE) == []
    # Latency is still measured.
    assert points(metric_reader, obs.METRIC_OPERATION_DURATION)


def test_response_without_usage_is_tolerated(
    chat_telemetry, spans, metric_reader, make_response
):
    with chat_telemetry.chat("gpt-4o-mini") as observed:
        observed.record(make_response(usage=None))

    (span,) = spans.get_finished_spans()
    assert obs.USAGE_INPUT_TOKENS not in span.attributes
    assert span.attributes[obs.RESPONSE_ID] == "chatcmpl-123"
    assert points(metric_reader, obs.METRIC_TOKEN_USAGE) == []


def test_exception_in_the_block_marks_span_and_duration_metric(
    chat_telemetry, spans, metric_reader
):
    with pytest.raises(RuntimeError):
        with chat_telemetry.chat("gpt-4o-mini"):
            raise RuntimeError("rate limited")

    (span,) = spans.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[obs.ERROR_TYPE] == "RuntimeError"
    assert span.events  # the exception is recorded on the span
    (point,) = points(metric_reader, obs.METRIC_OPERATION_DURATION)
    assert point.attributes[obs.ERROR_TYPE] == "RuntimeError"
    assert points(metric_reader, obs.METRIC_TOKEN_USAGE) == []


def test_observation_text_reads_the_first_choice(chat_telemetry, client):
    with chat_telemetry.chat("gpt-4o-mini") as observed:
        observed.record(client.chat.completions.create())
        assert observed.text() == "Ahoy, matey!"
        assert observed.input_tokens == 11
        assert observed.output_tokens == 7


def test_text_is_empty_before_anything_is_recorded(chat_telemetry):
    with chat_telemetry.chat("gpt-4o-mini") as observed:
        assert observed.text() == ""


def test_ask_pirate_sends_the_conversation_and_returns_the_reply(
    chat_telemetry, spans, client
):
    reply = ask_pirate("Hello", client=client, telemetry=chat_telemetry)

    assert reply == "Ahoy, matey!"
    assert client.calls[0]["messages"] == build_conversation("Hello")
    (span,) = spans.get_finished_spans()
    assert span.attributes[obs.USAGE_INPUT_TOKENS] == 11


def test_resolve_model_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert obs.resolve_model() == obs.DEFAULT_MODEL

    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    assert obs.resolve_model() == "gpt-4o"
    assert obs.resolve_model("gpt-4.1") == "gpt-4.1"
