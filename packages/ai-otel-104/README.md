# ai-otel-104

[`ai-otel-103`](../ai-otel-103) streamed one response. This one does it with
`AsyncOpenAI`, many at a time, and survives the two things async adds.

## Cancellation is not an exception

`asyncio.CancelledError` inherits from **`BaseException`**, not `Exception`. So
this, which looks correct and is what most instrumentation does:

```python
try:
    ...
except Exception as exc:        # never sees a cancelled request
    record_failure(exc)
```

silently misses every cancelled stream — and cancellation is the *normal*
ending for a streaming chat, since the user closing the tab is what cancels it.
Those requests then vanish from telemetry along with the tokens they already
burned.

The other half of the mistake is recording them as errors. A user who navigated
away is nothing to fix; filing it as an error just poisons the rate you page on.
So a cancelled stream here gets `app.stream.cancelled = true`, keeps its span
status unset, still reports whatever tokens arrived, and **always re-raises** —
swallowing `CancelledError` breaks task shutdown.

OpenTelemetry happens to agree: its span context manager reacts only to
`Exception`, so a cancelled span is not marked `ERROR` on its own. There is a
test asserting exactly that, because it is the kind of behaviour that quietly
changes.

## Concurrency: where you open the span decides what it means

Spans live in a `contextvars` context, and every asyncio task gets its own copy
at creation. Open the span **inside** the coroutine doing the work — as
`AsyncStreamedChat.stream` does — and *N* concurrent streams produce *N* sibling
spans, each with its own duration and time-to-first-token.

Wrap one span around a `gather` of many calls instead and you get a single span
whose duration is just "the slowest one", with no per-request latency at all.

```python
async with streamed.stream(conversation, model=model) as stream:
    async for delta in stream:
        print(delta, end="", flush=True)
```

`stream_many(...)` runs a batch that way and hands back one result object per
request.

## Run it

```sh
make run-ollama PKG=ai-otel-104     # local model, no key, no cost
make run PKG=ai-otel-104            # against OpenAI
```

It streams one reply live, then fires three requests concurrently and prints
each one's first-token latency and output tokens.

## What gets emitted

The same signals as `ai-otel-103` — `gen_ai.*` attributes with token counts,
`gen_ai.client.token.usage`, `gen_ai.client.operation.duration`,
`gen_ai.client.time_to_first_token`, `app.stream.chunks`,
`app.stream.completed` — reusing that package's names so sync and async calls
land in the same dashboards. Async adds one: `app.stream.cancelled`.

## Test it

```sh
make test-pkg PKG=ai-otel-104
```

An async fake stream drives the paths that only exist here: real cancellation
mid-stream (via a cancelled task, not a fake exception), concurrent streams
producing sibling spans under one parent, and partial token counts from a stream
nobody finished.
