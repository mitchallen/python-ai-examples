"""Embeddings, instrumented for the things that actually vary.

Three differences from a chat call drive everything here:

* **No output tokens.** Nothing is generated, so ``usage`` has no
  ``completion_tokens``. Recording a zero would pollute every output-token
  statistic that also covers chat calls.
* **The unit of work is a batch.** One call can carry one input or two hundred,
  and the spans are otherwise identical, so the batch size goes on the span.
* **The results are cachable.** The same text always embeds to the same vector,
  so a cache turns repeat work into zero tokens -- and the hit rate is worth
  measuring, because it is money.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ai_otel_101 import semconv as sc
from ai_otel_101.instrumented import provider_from_base_url
from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind

INSTRUMENTATION_NAME = "ai-otel-107"
INSTRUMENTATION_VERSION = "0.1.0"

# Semconv: the operation is "embeddings", not "chat".
OPERATION_EMBEDDINGS = "embeddings"

# Local. Batch size, vector width and cache behaviour have no semconv names,
# and they are what distinguishes one embeddings call from another.
EMBEDDINGS_INPUTS = "app.embeddings.inputs"
EMBEDDINGS_DIMENSIONS = "app.embeddings.dimensions"
CACHE_HITS = "app.embeddings.cache.hits"
CACHE_MISSES = "app.embeddings.cache.misses"
METRIC_CACHE = "app.embeddings.cache"
CACHE_RESULT = "result"
CACHE_HIT = "hit"
CACHE_MISS = "miss"

Vector = list[float]


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class EmbeddingBatch:
    """Vectors for the inputs, plus what they cost."""

    vectors: list[Vector] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    input_tokens: int | None = None

    @property
    def dimensions(self) -> int | None:
        return len(self.vectors[0]) if self.vectors else None

    @property
    def called_the_api(self) -> bool:
        """False when every input came from the cache."""
        return self.misses > 0


class EmbeddingCache:
    """The smallest useful cache: text -> vector, per model."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], Vector] = {}

    def get(self, model: str, text: str) -> Vector | None:
        return self._store.get((model, text))

    def put(self, model: str, text: str, vector: Vector) -> None:
        self._store[(model, text)] = vector

    def __len__(self) -> int:
        return len(self._store)


class InstrumentedEmbeddings:
    """Wraps the embeddings endpoint; batches, caches, and records."""

    def __init__(
        self,
        client: Any,
        *,
        tracer: trace.Tracer | None = None,
        meter: metrics.Meter | None = None,
        provider: str | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._client = client
        self._cache = cache
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
            description="Number of input tokens embedded.",
        )
        self._duration = meter.create_histogram(
            name=sc.METRIC_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration.",
        )
        self._cache_counter = meter.create_counter(
            name=METRIC_CACHE,
            unit="{input}",
            description="Embedding inputs served from cache versus the API.",
        )

    def embed(
        self, inputs: Sequence[str], *, model: str, **kwargs: Any
    ) -> EmbeddingBatch:
        """Embed a batch, sending only what the cache does not already hold."""
        texts = list(inputs)
        cached = self._lookup(texts, model)
        missing = [text for text, vector in zip(texts, cached) if vector is None]

        attributes: dict[str, Any] = {
            sc.OPERATION_NAME: OPERATION_EMBEDDINGS,
            sc.SYSTEM: self._provider,
            sc.PROVIDER_NAME: self._provider,
            sc.REQUEST_MODEL: model,
        }
        batch = EmbeddingBatch(hits=len(texts) - len(missing), misses=len(missing))

        started = time.perf_counter()
        with self._tracer.start_as_current_span(
            f"{OPERATION_EMBEDDINGS} {model}",
            kind=SpanKind.CLIENT,
            attributes=attributes,
        ) as span:
            fresh: list[Vector] = []
            if missing:
                try:
                    response = self._client.embeddings.create(
                        model=model, input=missing, **kwargs
                    )
                except Exception as exc:
                    error_type = type(exc).__qualname__
                    span.set_attribute(sc.ERROR_TYPE, error_type)
                    self._duration.record(
                        time.perf_counter() - started,
                        {**attributes, sc.ERROR_TYPE: error_type},
                    )
                    raise

                fresh = [_get(item, "embedding") for item in _get(response, "data", [])]
                usage = _get(response, "usage")
                if usage is not None:
                    # prompt_tokens only. There is no completion side to an
                    # embedding, and writing a 0 would be a lie that averages.
                    batch.input_tokens = _get(usage, "prompt_tokens")

                for text, vector in zip(missing, fresh):
                    self._store(model, text, vector)

            batch.vectors = self._assemble(cached, fresh)
            self._record(span, batch, attributes, time.perf_counter() - started)
            return batch

    def embed_one(self, text: str, *, model: str, **kwargs: Any) -> Vector:
        """One input, one vector -- the query side of a search."""
        return self.embed([text], model=model, **kwargs).vectors[0]

    # -- cache plumbing -----------------------------------------------------

    def _lookup(self, texts: Sequence[str], model: str) -> list[Vector | None]:
        if self._cache is None:
            return [None] * len(texts)
        return [self._cache.get(model, text) for text in texts]

    def _store(self, model: str, text: str, vector: Vector) -> None:
        if self._cache is not None:
            self._cache.put(model, text, vector)

    @staticmethod
    def _assemble(
        cached: Sequence[Vector | None], fresh: Sequence[Vector]
    ) -> list[Vector]:
        """Rebuild the caller's order from cache hits and fetched vectors."""
        iterator = iter(fresh)
        return [vector if vector is not None else next(iterator) for vector in cached]

    # -- recording ----------------------------------------------------------

    def _record(
        self,
        span: trace.Span,
        batch: EmbeddingBatch,
        attributes: Mapping[str, Any],
        elapsed: float,
    ) -> None:
        total = batch.hits + batch.misses
        span.set_attribute(EMBEDDINGS_INPUTS, total)
        span.set_attribute(CACHE_HITS, batch.hits)
        span.set_attribute(CACHE_MISSES, batch.misses)
        if batch.dimensions is not None:
            span.set_attribute(EMBEDDINGS_DIMENSIONS, batch.dimensions)
        if batch.input_tokens is not None:
            span.set_attribute(sc.USAGE_INPUT_TOKENS, batch.input_tokens)
            self._token_usage.record(
                batch.input_tokens,
                {**attributes, sc.TOKEN_TYPE: sc.TOKEN_TYPE_INPUT},
            )

        for count, result in ((batch.hits, CACHE_HIT), (batch.misses, CACHE_MISS)):
            if count:
                self._cache_counter.add(count, {**attributes, CACHE_RESULT: result})

        self._duration.record(elapsed, dict(attributes))
