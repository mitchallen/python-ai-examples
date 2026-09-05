"""Embeddings demo: ``make run PKG=ai-otel-107``.

Indexes a small corpus, runs a couple of searches, then repeats one to show a
cache hit costing nothing.
"""

from __future__ import annotations

import os
import sys
import time

from ai_otel_101.telemetry import configure_telemetry
from ai_python_101.chat import create_client
from openai import APIStatusError

from .embeddings import EmbeddingCache, InstrumentedEmbeddings
from .search import CORPUS, SemanticSearch

DEFAULT_MODEL = "text-embedding-3-small"
QUERIES = ["where be the rum", "how do we keep the crew healthy"]


def resolve_embedding_model() -> str:
    """Embeddings need their own model; a chat model will not do."""
    return os.environ.get("OPENAI_EMBEDDING_MODEL") or DEFAULT_MODEL


def explain(error: APIStatusError, model: str) -> str:
    """Turn the two predictable Ollama failures into instructions."""
    detail = str(error)[:150]

    if error.status_code == 404:
        return (
            f"The model {model!r} is not available on this server.\n"
            f"  {detail}\n\n"
            "Embeddings need an embedding model -- a chat model will not do:\n"
            f"    ollama pull {model}\n"
            f"    make run PKG=ai-otel-107 OLLAMA_EMBED_MODEL={model}"
        )

    # 501: a default `ollama serve` refuses embeddings whatever the model.
    return (
        "This server will not serve embeddings at all.\n"
        f"  {detail}\n\n"
        "Ollama has to be started with embeddings enabled:\n"
        "    ollama serve --embeddings\n"
        "    ollama pull nomic-embed-text\n"
        "    make run PKG=ai-otel-107 OLLAMA_EMBED_MODEL=nomic-embed-text"
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    queries = argv or QUERIES

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Either:\n"
            "    make run PKG=ai-otel-107          # falls back to a local Ollama model\n"
            "    export OPENAI_API_KEY=sk-...      # then: make run-openai PKG=ai-otel-107",
            file=sys.stderr,
        )
        return 1

    model = resolve_embedding_model()
    telemetry = configure_telemetry("ai-otel-107")
    cache = EmbeddingCache()
    embeddings = InstrumentedEmbeddings(
        create_client(),
        tracer=telemetry.tracer(),
        meter=telemetry.meter(),
        cache=cache,
    )
    search = SemanticSearch(embeddings, model=model, tracer=telemetry.tracer())

    guidance: str | None = None
    try:
        print(f"model: {model}")
        print(f"corpus: {len(CORPUS)} documents\n")

        started = time.perf_counter()
        try:
            tokens = search.index()
        except APIStatusError as error:
            # 404 (no such model) and 501 (server built without embeddings) are
            # both configuration, not bugs -- say which, and how to fix it.
            # Held until after the exporters flush, or it scrolls away above
            # a screen of span JSON.
            if error.status_code not in (404, 501):
                raise
            guidance = explain(error, model)
            return 2
        elapsed = time.perf_counter() - started
        print(
            f"indexed in {elapsed * 1000:.0f} ms "
            f"({elapsed * 1000 / len(CORPUS):.0f} ms per input), {tokens} tokens"
        )

        for query in queries:
            print(f"\nquery: {query!r}")
            for hit in search.search(query):
                print(f"  {hit.score:.3f}  {hit.document[:62]}")

        # Same query again: every input is already cached.
        repeated = queries[0]
        print(f"\nrepeating {repeated!r} -- now cached")
        before = len(cache)
        batch = embeddings.embed([repeated], model=model)
        print(
            f"  cache hits {batch.hits}, misses {batch.misses}, "
            f"api call: {batch.called_the_api}, tokens: {batch.input_tokens}"
        )
        print(f"  cache holds {before} vectors")
        print("\n--- telemetry ---")
    finally:
        telemetry.shutdown()
        if guidance:
            print(f"\n{guidance}\n", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
