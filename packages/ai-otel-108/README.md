# ai-otel-108

Retries and rate limits — starting with the reason you cannot see either one.

## The SDK already retried, and you did not see it

```python
>>> OpenAI(api_key="x").max_retries
2
```

The client retries `429` and `5xx` on its own, with backoff, **inside** the call
you wrapped a span around. So one `create()` is up to three HTTP requests, and
your telemetry shows:

- one span,
- no `error.type`,
- and eight seconds of latency.

Rate limiting does not appear as rate limiting. It appears as a slow model, and
you go looking in the wrong place. Worse, the two `429`s that preceded the
success are gone — you cannot alert on a failure you never recorded, and you
cannot tell "we are near the limit" from "the model is slow today".

This example turns the SDK's retries **off** (`max_retries=0`) and does them
itself, so every attempt is a span:

```
chat gpt-4o-mini                    app.retry.attempts=3  app.retry.slept=1.55
├── attempt 1   error.type=RateLimitError  http.response.status_code=429
├── attempt 2   error.type=RateLimitError  app.retry.after_seconds=1.0
└── attempt 3   gen_ai.usage.input_tokens=48
```

Same outcome, same latency — but now the latency has a *reason*, and the 429s
are countable.

## Sleeping is not working

`app.retry.slept` is recorded separately from the call duration, because "slow"
has two meanings that need different fixes. Time in flight means the model is
slow. Time asleep means you are over quota and the fix is a token bucket or a
bigger tier — no amount of model tuning helps.

## The 429 is the symptom; the header is the warning

A response carries your remaining quota:

```
app.ratelimit.remaining_requests=59   app.ratelimit.remaining_tokens=149  …
```

Those are recorded from the response headers on **success**, which is the whole
point: they let you see a limit being approached while calls are still working,
rather than finding out when they stop. This is the cheapest early warning in
the whole repo.

## Retry the retryable, and only that

| Kind | Examples | Retried? |
| --- | --- | --- |
| Rate limit / transient server | `429`, `500`, `502`, `503`, `504`, `408`, `409` | yes |
| Connection, timeout | `APIConnectionError`, `APITimeoutError` | yes |
| Your request is wrong | `400`, `401`, `404`, `422` | **no** |

Retrying a `401` cannot succeed — the key will not become valid — so it just
multiplies the latency of a failure by the retry count and delays the error the
caller needs. The classification is a test, not a comment.

`Retry-After` from the server wins over computed backoff. The server knows when
the window resets; exponential backoff is a guess, and ignoring the header is
how a fleet of clients synchronises into a thundering herd.

## Run it

```sh
make run PKG=ai-otel-108        # a normal call, then a simulated rate-limited one
```

The second call wraps the real client in a shim that raises `429` twice before
letting it through, since provoking a genuine rate limit is neither cheap nor
polite. Everything it exercises — classification, backoff, `Retry-After`,
per-attempt spans — is the same code path a real `429` takes.

## Test it

```sh
make test-pkg PKG=ai-otel-108
```

The sleeper is injected, so backoff schedules are asserted exactly and the suite
still runs in milliseconds.
