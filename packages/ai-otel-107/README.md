# ai-otel-107

Embeddings. The telemetry looks like the chat examples' until you notice what is
missing from it.

## There are no output tokens

An embeddings response has `usage.prompt_tokens` and `usage.total_tokens` and
**no `completion_tokens`** — nothing is generated, so there is nothing to count.
Copying the chat instrumentation over would record `gen_ai.usage.output_tokens:
0` on every call, which is not zero-cost-because-nothing-happened, it is a
number that should not exist. Averaged into a dashboard beside chat calls it
quietly drags every output-token statistic toward zero.

So this records input tokens only, and a test asserts the output attribute is
absent rather than zero.

## The unit of work is a batch

`embeddings.create` takes a list. One call for 200 documents is not the same
shape of work as one call for one query, but they produce identical spans unless
you say so — and a duration of 900 ms means very different things for the two.

Every span therefore carries `app.embeddings.inputs`, and the demo prints
milliseconds *per input* alongside the total. Batching is the main lever you
have on embedding throughput, and it is invisible without that number.

## Cache hits are the point

The same text always embeds to the same vector, which makes embeddings the most
cachable call in this repo — and re-embedding an unchanged corpus is the most
common way to waste money on one.

The wrapper takes a cache, splits each batch into hits and misses, and sends
**only the misses**. A fully cached batch makes no API call at all. The
`app.embeddings.cache` counter is keyed `hit`/`miss`, so the hit rate is a
query, and every hit is tokens not spent:

```
app.embeddings.inputs=6  app.embeddings.cache.hits=4  app.embeddings.cache.misses=2
gen_ai.usage.input_tokens=19      <- two inputs' worth, not six
```

## What a search actually costs

`search.py` is a small semantic search: embed the corpus once, embed the query,
rank by cosine similarity. It exists to make the split visible in one trace —

```
search "where be the rum"
└── embeddings text-embedding-3-small   inputs=1  in=6 tokens
    (ranking 6 documents: local arithmetic, no tokens, no network)
```

— because the ranking half costs nothing and the embedding half is a network
call, and which one dominates decides whether you should be caching, batching,
or neither.

## Run it

```sh
make run PKG=ai-otel-107
```

**Ollama needs `--embeddings` for this one.** Unlike the chat examples, a
default `ollama serve` answers embeddings requests with
`501 This server does not support embeddings`, whatever model you name. Restart
it with the flag and pull an embedding model:

```sh
ollama serve --embeddings          # in its own terminal
ollama pull nomic-embed-text
make run PKG=ai-otel-107 OLLAMA_EMBED_MODEL=nomic-embed-text
```

The demo detects that 501 and says so rather than failing with a traceback.
Against the hosted API it needs nothing special — `text-embedding-3-small` is
the default.

## Test it

```sh
make test-pkg PKG=ai-otel-107
```

Deterministic stub vectors, so the cache behaviour, the batching, the absent
output tokens, and the similarity ranking are all exact assertions rather than
approximations.
