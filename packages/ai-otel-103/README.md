# ai-otel-103

Streaming, where the token counts are easy to lose.

A non-streaming response hands you `usage` with every call. **A stream does not**
— verified against a local model:

```
stream WITHOUT stream_options : 17 chunks, usage: None
stream WITH  include_usage    : 14 chunks, usage: prompt=35 completion=13
```

Ask for `stream_options={"include_usage": True}` and the provider appends one
final chunk carrying `usage` — with an **empty `choices` list**, which is why
naive `chunk.choices[0]` code crashes on the last chunk of an otherwise working
stream. Forget it and your cost telemetry is silently zero for every streamed
call, which is most of them in a chat UI.

This example asks for usage by default, and records what streaming makes
newly measurable.

## Run it

```sh
make run-ollama PKG=ai-otel-103        # local model, no key, no cost
make run PKG=ai-otel-103               # against OpenAI
make run-ollama PKG=ai-otel-103 ARGS=--no-usage   # see the counts vanish
```

## What gets emitted

Everything [`ai-otel-101`](../ai-otel-101) emits — `gen_ai.*` span attributes,
`gen_ai.client.token.usage`, `gen_ai.client.operation.duration`, provider
derived from the endpoint — plus what only a stream has:

| Signal | Meaning |
| --- | --- |
| `gen_ai.client.time_to_first_token` | Latency to the **first** delta, the number a user actually feels. |
| `app.stream.chunks` | Chunks received. A usable sanity check when `usage` is absent. |
| `app.stream.completed` | `false` when the consumer walked away mid-stream. |

`gen_ai.client.time_to_first_token` is this example's own name: the conventions
define `gen_ai.server.time_to_first_token` for the server side and have no
blessed client-side equivalent. `app.*` attributes are likewise local, not
semconv — named so it's obvious which is which.

Total duration alone is a bad summary of a stream: it grows with the length of
the answer, so a slow start and a long answer look identical. TTFT separates
them.

## The shape

```python
with streamed.stream(build_conversation("Hello"), model="gpt-4o-mini") as stream:
    for delta in stream:
        print(delta, end="", flush=True)

print(stream.text, stream.input_tokens, stream.output_tokens, stream.ttft)
```

The span stays open for the life of the stream, not just the initial call —
otherwise the recorded duration measures how long the request took to *start*.
Break out of that loop and the span still closes, marked incomplete, and the
underlying HTTP response is closed rather than leaked.

## Test it

```sh
make test-pkg PKG=ai-otel-103
```

Fake chunk sequences drive every path: usage present and absent, an abandoned
stream, a mid-stream failure, and the empty-`choices` final chunk that breaks
naive code.
