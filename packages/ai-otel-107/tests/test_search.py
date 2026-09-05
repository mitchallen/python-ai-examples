"""Similarity maths and the search span."""

from __future__ import annotations

import math

import pytest

from ai_otel_107 import CORPUS, EmbeddingCache, InstrumentedEmbeddings, SemanticSearch
from ai_otel_107.search import (
    SEARCH_DOCUMENTS,
    SEARCH_TOP_K,
    SEARCH_TOP_SCORE,
    cosine_similarity,
)


# -- the arithmetic ----------------------------------------------------------


def test_identical_vectors_score_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_score_minus_one():
    assert cosine_similarity([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


def test_magnitude_does_not_matter_only_direction():
    # Cosine similarity is about angle: a longer document is not a better match.
    assert cosine_similarity([1.0, 1.0], [10.0, 10.0]) == pytest.approx(1.0)


def test_a_zero_vector_has_no_direction():
    # Not a crash, and not 1.0 -- it simply cannot be similar to anything.
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_mismatched_lengths_are_a_programming_error():
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


# -- the search --------------------------------------------------------------


def search_over(client, tracer, meter, documents=CORPUS) -> SemanticSearch:
    embeddings = InstrumentedEmbeddings(
        client, tracer=tracer, meter=meter, cache=EmbeddingCache()
    )
    return SemanticSearch(embeddings, model="m", documents=documents, tracer=tracer)


def test_indexing_embeds_the_corpus_in_one_call(client, tracer, meter):
    search = search_over(client, tracer, meter)

    tokens = search.index()

    assert len(client.calls) == 1
    assert len(client.embeddings.inputs_sent[0]) == len(CORPUS)
    assert tokens == 7 * len(CORPUS)


def test_searching_before_indexing_is_an_error(client, tracer, meter):
    search = search_over(client, tracer, meter)

    with pytest.raises(RuntimeError, match="index"):
        search.search("where be the rum")


def test_an_exact_document_match_ranks_first(client, tracer, meter):
    documents = ["alpha", "beta", "gamma"]
    search = search_over(client, tracer, meter, documents)
    search.index()

    hits = search.search("beta", top_k=3)

    # The stub embeds identical text identically, so this is exact.
    assert hits[0].document == "beta"
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].index == 1


def test_results_are_sorted_by_descending_score(client, tracer, meter):
    search = search_over(client, tracer, meter)
    search.index()

    hits = search.search("where be the rum", top_k=4)

    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_the_results(client, tracer, meter):
    search = search_over(client, tracer, meter)
    search.index()

    assert len(search.search("rum", top_k=2)) == 2
    assert len(search.search("rum", top_k=99)) == len(CORPUS)


def test_the_search_span_wraps_the_query_embedding(client, tracer, meter, spans):
    search = search_over(client, tracer, meter)
    search.index()
    spans.clear()

    search.search("where be the rum", top_k=3)

    finished = spans.get_finished_spans()
    (parent,) = [s for s in finished if s.name.startswith("search")]
    children = [s for s in finished if s is not parent]
    # Exactly one model call inside a search: the query. Ranking is local.
    assert len(children) == 1
    assert children[0].name == "embeddings m"
    assert children[0].parent.span_id == parent.get_span_context().span_id


def test_the_search_span_records_the_corpus_size_and_best_score(
    client, tracer, meter, spans
):
    search = search_over(client, tracer, meter, ["alpha", "beta"])
    search.index()
    spans.clear()

    search.search("beta", top_k=1)

    (parent,) = [s for s in spans.get_finished_spans() if s.name.startswith("search")]
    assert parent.attributes[SEARCH_DOCUMENTS] == 2
    assert parent.attributes[SEARCH_TOP_K] == 1
    assert parent.attributes[SEARCH_TOP_SCORE] == pytest.approx(1.0)


def test_a_repeated_query_is_served_from_cache(client, tracer, meter):
    search = search_over(client, tracer, meter)
    search.index()
    before = len(client.calls)

    search.search("where be the rum")
    search.search("where be the rum")

    # The corpus call, then one query embedding -- the repeat costs nothing.
    assert len(client.calls) == before + 1
