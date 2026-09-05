"""Embedding telemetry: batches, caches, and the token that does not exist."""

from __future__ import annotations

from typing import Any

import pytest
from ai_otel_101 import semconv as sc

from ai_otel_107 import (
    CACHE_HITS,
    CACHE_MISSES,
    EMBEDDINGS_DIMENSIONS,
    EMBEDDINGS_INPUTS,
    METRIC_CACHE,
    EmbeddingCache,
    InstrumentedEmbeddings,
)

TEXTS = ["ahoy there", "the sea is calm", "rum is stowed aft"]


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


def embedder(client, tracer, meter, **kwargs) -> InstrumentedEmbeddings:
    return InstrumentedEmbeddings(client, tracer=tracer, meter=meter, **kwargs)


# -- the shape of an embeddings span ----------------------------------------


def test_the_operation_is_embeddings_not_chat(client, tracer, meter, spans):
    embedder(client, tracer, meter).embed(TEXTS, model="text-embedding-3-small")

    (span,) = spans.get_finished_spans()
    assert span.name == "embeddings text-embedding-3-small"
    assert span.attributes[sc.OPERATION_NAME] == "embeddings"


def test_there_are_no_output_tokens(client, tracer, meter, spans):
    # Nothing is generated, so a 0 here would be a fabricated number that
    # drags down every output-token average it lands in.
    embedder(client, tracer, meter).embed(TEXTS, model="m")

    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.USAGE_INPUT_TOKENS] == 21
    assert sc.USAGE_OUTPUT_TOKENS not in span.attributes


def test_only_input_tokens_reach_the_metric(client, tracer, meter, metric_reader):
    embedder(client, tracer, meter).embed(TEXTS, model="m")

    recorded = points(metric_reader, sc.METRIC_TOKEN_USAGE)
    assert {p.attributes[sc.TOKEN_TYPE] for p in recorded} == {"input"}


def test_batch_size_and_width_are_on_the_span(client, tracer, meter, spans):
    embedder(client, tracer, meter).embed(TEXTS, model="m")

    (span,) = spans.get_finished_spans()
    assert span.attributes[EMBEDDINGS_INPUTS] == 3
    assert span.attributes[EMBEDDINGS_DIMENSIONS] == 4


def test_one_call_per_batch_not_per_input(client, tracer, meter):
    embedder(client, tracer, meter).embed(TEXTS, model="m")

    assert len(client.calls) == 1
    assert client.embeddings.inputs_sent == [TEXTS]


def test_vectors_come_back_in_order(client, tracer, meter, vectors):
    batch = embedder(client, tracer, meter).embed(TEXTS, model="m")

    assert batch.vectors == [vectors(text) for text in TEXTS]


def test_embed_one_returns_a_single_vector(client, tracer, meter, vectors):
    vector = embedder(client, tracer, meter).embed_one("ahoy there", model="m")

    assert vector == vectors("ahoy there")


# -- caching -----------------------------------------------------------------


def test_a_repeat_batch_makes_no_api_call(client, tracer, meter):
    embeddings = embedder(client, tracer, meter, cache=EmbeddingCache())

    embeddings.embed(TEXTS, model="m")
    second = embeddings.embed(TEXTS, model="m")

    assert len(client.calls) == 1  # not two
    assert second.hits == 3
    assert second.misses == 0
    assert second.called_the_api is False
    # Nothing was sent, so nothing was billed.
    assert second.input_tokens is None


def test_a_partial_hit_sends_only_the_misses(client, tracer, meter, spans):
    embeddings = embedder(client, tracer, meter, cache=EmbeddingCache())
    embeddings.embed(TEXTS[:2], model="m")

    batch = embeddings.embed([*TEXTS, "a fourth line"], model="m")

    assert (batch.hits, batch.misses) == (2, 2)
    assert client.embeddings.inputs_sent[1] == ["rum is stowed aft", "a fourth line"]
    # Billed for two inputs, not four.
    assert batch.input_tokens == 14
    span = spans.get_finished_spans()[-1]
    assert span.attributes[EMBEDDINGS_INPUTS] == 4
    assert span.attributes[CACHE_HITS] == 2
    assert span.attributes[CACHE_MISSES] == 2


def test_cached_and_fresh_vectors_keep_the_callers_order(client, tracer, meter, vectors):
    embeddings = embedder(client, tracer, meter, cache=EmbeddingCache())
    embeddings.embed(["b"], model="m")

    batch = embeddings.embed(["a", "b", "c"], model="m")

    assert batch.vectors == [vectors("a"), vectors("b"), vectors("c")]


def test_the_cache_is_keyed_by_model(client, tracer, meter):
    embeddings = embedder(client, tracer, meter, cache=EmbeddingCache())

    embeddings.embed(["ahoy"], model="small")
    batch = embeddings.embed(["ahoy"], model="large")

    # Different model, different vector space: a hit here would be a bug.
    assert batch.misses == 1
    assert len(client.calls) == 2


def test_the_cache_counter_splits_hits_from_misses(
    client, tracer, meter, metric_reader
):
    embeddings = embedder(client, tracer, meter, cache=EmbeddingCache())
    embeddings.embed(TEXTS, model="m")
    embeddings.embed(TEXTS, model="m")

    by_result = {
        p.attributes["result"]: p.value for p in points(metric_reader, METRIC_CACHE)
    }
    assert by_result == {"miss": 3, "hit": 3}


def test_without_a_cache_every_input_is_a_miss(client, tracer, meter):
    embeddings = embedder(client, tracer, meter)  # no cache

    embeddings.embed(TEXTS, model="m")
    batch = embeddings.embed(TEXTS, model="m")

    assert batch.hits == 0
    assert len(client.calls) == 2


# -- failures ----------------------------------------------------------------


def test_an_api_failure_is_recorded_and_raised(tracer, meter, spans, make_client):
    client = make_client(error=RuntimeError("rate limited"))

    with pytest.raises(RuntimeError):
        embedder(client, tracer, meter).embed(TEXTS, model="m")

    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.ERROR_TYPE] == "RuntimeError"


def test_an_empty_batch_makes_no_call(client, tracer, meter, spans):
    batch = embedder(client, tracer, meter).embed([], model="m")

    assert batch.vectors == []
    assert client.calls == []
    (span,) = spans.get_finished_spans()
    assert span.attributes[EMBEDDINGS_INPUTS] == 0
