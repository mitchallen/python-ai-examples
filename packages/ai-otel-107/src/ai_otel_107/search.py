"""Semantic search over a handful of documents.

Here to make one split visible: embedding the query is a network call that costs
tokens, ranking the corpus is arithmetic that costs neither. Which half
dominates decides whether the thing to fix is caching, batching, or nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from opentelemetry import trace
from opentelemetry.trace import SpanKind

from .embeddings import InstrumentedEmbeddings, Vector

# Local: the ranking half of a search has no model call to describe it.
SEARCH_DOCUMENTS = "app.search.documents"
SEARCH_TOP_K = "app.search.top_k"
SEARCH_TOP_SCORE = "app.search.top_score"

CORPUS = [
    "The rum ration is stowed in the aft hold, behind the salt pork.",
    "Careening the hull removes barnacles and keeps her fast.",
    "Tortuga sees squalls most afternoons in the wet season.",
    "A quartermaster divides the plunder and settles disputes.",
    "Scurvy is kept at bay with limes, of which we carry ninety.",
    "The Black Pearl carries forty cannons and a crew of four hundred.",
]


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Cosine of the angle between two vectors; 1.0 is identical direction."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        # An all-zero vector has no direction, so no meaningful similarity.
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass
class Hit:
    document: str
    score: float
    index: int


class SemanticSearch:
    """Embed a corpus once, then rank it against each query."""

    def __init__(
        self,
        embeddings: InstrumentedEmbeddings,
        *,
        model: str,
        documents: Sequence[str] = CORPUS,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._model = model
        self._documents = list(documents)
        self._vectors: list[Vector] = []
        self._tracer = tracer or trace.get_tracer("ai-otel-107")

    def index(self) -> int:
        """Embed the corpus in one batch. Tokens are spent here, once."""
        batch = self._embeddings.embed(self._documents, model=self._model)
        self._vectors = batch.vectors
        return batch.input_tokens or 0

    def search(self, query: str, *, top_k: int = 3) -> list[Hit]:
        """Embed the query, then rank locally."""
        if not self._vectors:
            raise RuntimeError("call index() before search()")

        with self._tracer.start_as_current_span(
            f"search {query!r}",
            kind=SpanKind.INTERNAL,
            attributes={
                SEARCH_DOCUMENTS: len(self._documents),
                SEARCH_TOP_K: top_k,
            },
        ) as span:
            query_vector = self._embeddings.embed_one(query, model=self._model)

            # No span for this: it is arithmetic, and a span per comparison
            # would cost more than the comparison.
            hits = sorted(
                (
                    Hit(document=document, score=cosine_similarity(query_vector, vector), index=index)
                    for index, (document, vector) in enumerate(
                        zip(self._documents, self._vectors)
                    )
                ),
                key=lambda hit: hit.score,
                reverse=True,
            )[:top_k]

            if hits:
                span.set_attribute(SEARCH_TOP_SCORE, hits[0].score)
            return hits
