"""ai-otel-107: embeddings, batching, caching, and semantic search."""

from .embeddings import (
    CACHE_HIT,
    CACHE_HITS,
    CACHE_MISS,
    CACHE_MISSES,
    EMBEDDINGS_DIMENSIONS,
    EMBEDDINGS_INPUTS,
    METRIC_CACHE,
    OPERATION_EMBEDDINGS,
    EmbeddingBatch,
    EmbeddingCache,
    InstrumentedEmbeddings,
)
from .search import CORPUS, Hit, SemanticSearch, cosine_similarity

__all__ = [
    "CACHE_HIT",
    "CACHE_HITS",
    "CACHE_MISS",
    "CACHE_MISSES",
    "CORPUS",
    "EMBEDDINGS_DIMENSIONS",
    "EMBEDDINGS_INPUTS",
    "EmbeddingBatch",
    "EmbeddingCache",
    "Hit",
    "InstrumentedEmbeddings",
    "METRIC_CACHE",
    "OPERATION_EMBEDDINGS",
    "SemanticSearch",
    "cosine_similarity",
]
