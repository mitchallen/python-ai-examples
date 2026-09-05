"""Async streaming: the same guarantees as 103, plus cancellation and concurrency."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from ai_otel_101 import semconv as sc
from ai_otel_103.streaming import (
    METRIC_TIME_TO_FIRST_TOKEN,
    STREAM_CHUNKS,
    STREAM_COMPLETED,
)
from opentelemetry.trace import SpanKind, StatusCode

from ai_otel_104 import STREAM_CANCELLED, AsyncStreamedChat, stream_many

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


async def drain(stream) -> list[str]:
    return [delta async for delta in stream]


# -- the basics, mirrored from the sync example ------------------------------


async def test_deltas_stream_and_accumulate(tracer, meter, client):
    async with AsyncStreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        deltas = await drain(stream)

    assert deltas == ["Ahoy", ", ", "matey!"]
    assert stream.text == "Ahoy, matey!"
    assert stream.completed is True


async def test_token_counts_and_span_shape(tracer, meter, spans, client):
    async with AsyncStreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        await drain(stream)

    (span,) = spans.get_finished_spans()
    assert span.name == "chat gpt-4o-mini"
    assert span.kind is SpanKind.CLIENT
    assert span.attributes[sc.USAGE_INPUT_TOKENS] == 35
    assert span.attributes[sc.USAGE_OUTPUT_TOKENS] == 13
    assert span.attributes[STREAM_COMPLETED] is True
    assert (stream.input_tokens, stream.output_tokens) == (35, 13)


async def test_include_usage_is_requested_by_default(tracer, meter, client):
    async with AsyncStreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        await drain(stream)

    assert client.calls[0]["stream_options"] == {"include_usage": True}
    assert client.calls[0]["stream"] is True


async def test_time_to_first_token_is_recorded(tracer, meter, metric_reader, client):
    async with AsyncStreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        await drain(stream)

    (ttft,) = points(metric_reader, METRIC_TIME_TO_FIRST_TOKEN)
    (duration,) = points(metric_reader, sc.METRIC_OPERATION_DURATION)
    assert ttft.count == 1
    assert ttft.sum <= duration.sum


# -- cancellation ------------------------------------------------------------


async def cancel_midway(streamed, delay: float = 0.05):
    """Start a stream in a task, cancel it once deltas are flowing."""
    started = asyncio.Event()

    async def consume():
        async with streamed.stream(MESSAGES, model="gpt-4o-mini") as stream:
            async for _ in stream:
                started.set()
                await asyncio.sleep(delay)  # a slow consumer, like a UI

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return task


async def test_cancellation_is_recorded_not_lost(tracer, meter, spans, client):
    streamed = AsyncStreamedChat(client, tracer=tracer, meter=meter)

    await cancel_midway(streamed)

    (span,) = spans.get_finished_spans()
    assert span.attributes[STREAM_CANCELLED] is True
    assert span.attributes[STREAM_COMPLETED] is False


async def test_cancellation_is_not_an_error(tracer, meter, spans, client):
    # A user closing the tab is nothing to page on, and OTel's span manager
    # only reacts to Exception -- CancelledError is a BaseException.
    streamed = AsyncStreamedChat(client, tracer=tracer, meter=meter)

    await cancel_midway(streamed)

    (span,) = spans.get_finished_spans()
    assert span.status.status_code is not StatusCode.ERROR
    assert sc.ERROR_TYPE not in span.attributes


async def test_cancellation_closes_the_stream(tracer, meter, client):
    streamed = AsyncStreamedChat(client, tracer=tracer, meter=meter)

    await cancel_midway(streamed)

    assert client.stream.closed is True


async def test_cancelled_stream_still_records_duration(
    tracer, meter, metric_reader, client
):
    streamed = AsyncStreamedChat(client, tracer=tracer, meter=meter)

    await cancel_midway(streamed)

    (point,) = points(metric_reader, sc.METRIC_OPERATION_DURATION)
    assert point.attributes[STREAM_CANCELLED] is True


async def test_tokens_seen_before_cancellation_are_still_billed(
    tracer, meter, metric_reader, make_client, chunks
):
    # Usage arrived early, then the consumer went away: those tokens were
    # generated and paid for whether or not anyone read them.
    client = make_client([chunks.usage(), *chunks.text("Ahoy", ", ", "matey!")])
    streamed = AsyncStreamedChat(client, tracer=tracer, meter=meter)

    await cancel_midway(streamed)

    by_type = {
        point.attributes[sc.TOKEN_TYPE]: point.sum
        for point in points(metric_reader, sc.METRIC_TOKEN_USAGE)
    }
    assert by_type == {"input": 35, "output": 13}


# -- concurrency -------------------------------------------------------------


async def test_concurrent_streams_get_their_own_spans(
    tracer, meter, spans, make_client
):
    client = make_client(delay=0.01, shared_stream=False)
    streamed = AsyncStreamedChat(client, tracer=tracer, meter=meter)

    results = await stream_many(streamed, [MESSAGES] * 3, model="gpt-4o-mini")

    assert len(results) == 3
    finished = spans.get_finished_spans()
    assert len(finished) == 3
    assert all(span.name == "chat gpt-4o-mini" for span in finished)
    # Each stream measured its own first token, not one shared number.
    assert all(result.ttft is not None for result in results)


async def test_concurrent_spans_are_siblings_under_the_caller(
    tracer, meter, spans, make_client
):
    client = make_client(delay=0.01, shared_stream=False)
    streamed = AsyncStreamedChat(client, tracer=tracer, meter=meter)

    with tracer.start_as_current_span("batch") as parent:
        await stream_many(streamed, [MESSAGES] * 3, model="gpt-4o-mini")
        parent_id = parent.get_span_context().span_id

    children = [s for s in spans.get_finished_spans() if s.name != "batch"]
    assert len(children) == 3
    # Every task inherited the context that was current when it was created.
    assert {s.parent.span_id for s in children} == {parent_id}


async def test_concurrent_streams_run_in_parallel(tracer, meter, make_client):
    # Three 5-chunk streams at 10ms/chunk: ~50ms concurrently, ~150ms serially.
    client = make_client(delay=0.01, shared_stream=False)
    streamed = AsyncStreamedChat(client, tracer=tracer, meter=meter)

    loop = asyncio.get_running_loop()
    started = loop.time()
    await stream_many(streamed, [MESSAGES] * 3, model="gpt-4o-mini")
    elapsed = loop.time() - started

    assert elapsed < 0.12


# -- failures ----------------------------------------------------------------


async def test_a_failure_opening_the_stream_is_recorded(
    tracer, meter, spans, make_client
):
    client = make_client(error=RuntimeError("rate limited"))
    streamed = AsyncStreamedChat(client, tracer=tracer, meter=meter)

    with pytest.raises(RuntimeError):
        async with streamed.stream(MESSAGES, model="gpt-4o-mini"):
            pass

    (span,) = spans.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[sc.ERROR_TYPE] == "RuntimeError"


async def test_a_failure_mid_stream_is_recorded(
    tracer, meter, spans, metric_reader, make_client, chunks
):
    client = make_client(
        chunks.text("Ahoy", ", ", "matey!"), stream_error=ConnectionError("dropped")
    )
    streamed = AsyncStreamedChat(client, tracer=tracer, meter=meter)

    with pytest.raises(ConnectionError):
        async with streamed.stream(MESSAGES, model="gpt-4o-mini") as stream:
            await drain(stream)

    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.ERROR_TYPE] == "ConnectionError"
    (point,) = points(metric_reader, sc.METRIC_OPERATION_DURATION)
    assert point.attributes[sc.ERROR_TYPE] == "ConnectionError"


async def test_abandoning_a_stream_marks_it_incomplete(tracer, meter, spans, client):
    async with AsyncStreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="gpt-4o-mini"
    ) as stream:
        async for _ in stream:
            break

    (span,) = spans.get_finished_spans()
    assert span.attributes[STREAM_COMPLETED] is False
    assert span.attributes[STREAM_CHUNKS] == 1
    assert client.stream.closed is True


async def test_provider_is_derived_from_the_endpoint(tracer, meter, spans, make_client):
    client = make_client()
    client.base_url = "http://localhost:11434/v1"

    async with AsyncStreamedChat(client, tracer=tracer, meter=meter).stream(
        MESSAGES, model="llama3.2:3b"
    ) as stream:
        await drain(stream)

    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.PROVIDER_NAME] == "ollama"
