"""Streaming telemetry: the counts, the first-token latency, and the edges."""

from __future__ import annotations

from typing import Any

import pytest
from ai_otel_101 import semconv as sc
from opentelemetry.trace import SpanKind, StatusCode

from ai_otel_103 import STREAM_CHUNKS, STREAM_COMPLETED, StreamedChat
from ai_otel_103.streaming import METRIC_TIME_TO_FIRST_TOKEN

MESSAGES = [{"role": "user", "content": "Hello"}]


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


def drain(stream) -> list[str]:
    return list(stream)


def test_deltas_are_yielded_and_accumulated(tracer, meter, client):
    with StreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        deltas = drain(stream)

    assert deltas == ["Ahoy", ", ", "matey!"]
    assert stream.text == "Ahoy, matey!"
    assert stream.completed is True


def test_include_usage_is_requested_by_default(tracer, meter, client):
    with StreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        drain(stream)

    call = client.calls[0]
    assert call["stream"] is True
    assert call["stream_options"] == {"include_usage": True}


def test_token_counts_come_off_the_final_usage_chunk(tracer, meter, spans, client):
    with StreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        drain(stream)

    assert (stream.input_tokens, stream.output_tokens) == (35, 13)
    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.USAGE_INPUT_TOKENS] == 35
    assert span.attributes[sc.USAGE_OUTPUT_TOKENS] == 13
    assert span.attributes[sc.RESPONSE_FINISH_REASONS] == ("stop",)


def test_the_usage_chunk_has_no_choices(chunks):
    # The shape that breaks naive chunk.choices[0] code.
    assert chunks.usage().choices == []


def test_without_include_usage_there_are_no_token_counts(
    tracer, meter, spans, metric_reader, make_client, chunks
):
    # A stream that never sends a usage chunk, which is what a provider does
    # when stream_options is missing.
    client = make_client([*chunks.text("Ahoy"), chunks.finish()])

    with StreamedChat(
        client, tracer=tracer, meter=meter, include_usage=False
    ).stream(MESSAGES, model="gpt-4o-mini") as stream:
        drain(stream)

    assert "stream_options" not in client.calls[0]
    assert stream.output_tokens is None
    (span,) = spans.get_finished_spans()
    assert sc.USAGE_OUTPUT_TOKENS not in span.attributes
    assert points(metric_reader, sc.METRIC_TOKEN_USAGE) == []
    # The chunk count is still there as a rough sanity check.
    assert span.attributes[STREAM_CHUNKS] == 2


def test_token_usage_metric_splits_input_and_output(tracer, meter, metric_reader, client):
    with StreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        drain(stream)

    by_type = {
        point.attributes[sc.TOKEN_TYPE]: point.sum
        for point in points(metric_reader, sc.METRIC_TOKEN_USAGE)
    }
    assert by_type == {"input": 35, "output": 13}


def test_time_to_first_token_is_recorded(tracer, meter, metric_reader, client):
    with StreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        drain(stream)

    (ttft,) = points(metric_reader, METRIC_TIME_TO_FIRST_TOKEN)
    (duration,) = points(metric_reader, sc.METRIC_OPERATION_DURATION)
    assert ttft.count == 1
    assert stream.ttft is not None
    # First token cannot be slower than the whole stream.
    assert ttft.sum <= duration.sum


def test_ttft_measures_the_first_visible_token_not_an_empty_chunk(
    tracer, meter, make_client, chunks
):
    # A leading chunk with no content -- role-only openers look like this.
    client = make_client(
        [chunks.finish(None), *chunks.text("Ahoy"), chunks.finish(), chunks.usage()]
    )

    with StreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        drain(stream)

    assert stream.text == "Ahoy"
    assert stream.ttft is not None


def test_no_ttft_when_the_stream_yields_no_text(
    tracer, meter, metric_reader, make_client, chunks
):
    client = make_client([chunks.finish(), chunks.usage()])

    with StreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        drain(stream)

    assert stream.ttft is None
    assert points(metric_reader, METRIC_TIME_TO_FIRST_TOKEN) == []
    # Duration is still recorded.
    assert points(metric_reader, sc.METRIC_OPERATION_DURATION)


def test_abandoning_the_stream_closes_it_and_marks_the_span(
    tracer, meter, spans, metric_reader, client
):
    with StreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        for _ in stream:
            break  # user closed the tab

    (span,) = spans.get_finished_spans()
    assert span.attributes[STREAM_COMPLETED] is False
    assert client.stream.closed is True  # socket released, not leaked
    assert points(metric_reader, sc.METRIC_OPERATION_DURATION)


def test_a_completed_stream_is_not_closed_twice(tracer, meter, client):
    with StreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        drain(stream)

    assert stream.completed is True
    assert client.stream.closed is False


def test_a_failure_opening_the_stream_is_recorded(tracer, meter, spans, make_client):
    client = make_client(error=RuntimeError("rate limited"))
    streamed = StreamedChat(client, tracer=tracer, meter=meter)

    with pytest.raises(RuntimeError):
        with streamed.stream(MESSAGES, model="gpt-4o-mini"):
            pass

    (span,) = spans.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[sc.ERROR_TYPE] == "RuntimeError"


def test_a_failure_mid_stream_is_recorded(
    tracer, meter, spans, metric_reader, make_client, chunks
):
    client = make_client(
        [*chunks.text("Ahoy", ", ", "matey!")],
        stream_error=ConnectionError("dropped"),
    )
    streamed = StreamedChat(client, tracer=tracer, meter=meter)

    with pytest.raises(ConnectionError):
        with streamed.stream(MESSAGES, model="gpt-4o-mini") as stream:
            drain(stream)

    (span,) = spans.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[sc.ERROR_TYPE] == "ConnectionError"
    (point,) = points(metric_reader, sc.METRIC_OPERATION_DURATION)
    assert point.attributes[sc.ERROR_TYPE] == "ConnectionError"


def test_span_shape_matches_the_non_streaming_examples(tracer, meter, spans, client):
    with StreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        drain(stream)

    (span,) = spans.get_finished_spans()
    assert span.name == "chat gpt-4o-mini"
    assert span.kind is SpanKind.CLIENT
    assert span.attributes[sc.OPERATION_NAME] == "chat"
    assert span.attributes[sc.REQUEST_MODEL] == "gpt-4o-mini"


def test_provider_is_derived_from_the_endpoint(tracer, meter, spans, make_client):
    client = make_client()
    client.base_url = "http://localhost:11434/v1"

    with StreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="llama3.2:3b"
    ) as stream:
        drain(stream)

    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.PROVIDER_NAME] == "ollama"
