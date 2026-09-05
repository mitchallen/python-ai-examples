# python-ai-examples

[![tests](https://github.com/mitchallen/python-ai-examples/actions/workflows/tests.yml/badge.svg)](https://github.com/mitchallen/python-ai-examples/actions/workflows/tests.yml)
[![test count](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmitchallen%2Fpython-ai-examples%2Fmain%2F.github%2Fbadges%2Ftests.json)](https://github.com/mitchallen/python-ai-examples/actions/workflows/tests.yml)

A monorepo of small, self-contained Python AI examples. Each example is its own
package under `packages/`, wired together as a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
one root `uv.lock` and one root `.venv`, but per-example dependencies and entry
points.

## Layout

```
pyproject.toml            # virtual workspace root (members = packages/*)
uv.lock                   # one lock file for the whole workspace
Makefile                  # install / test / run
packages/
  ai-python-101/          # OpenAI client hello-world: a pirate conversation
  ai-otel-101/            # the same call wrapped in OpenTelemetry (tokens, latency)
  ai-otel-102/            # self-contained variant: one module, context-manager shape
  ai-otel-103/            # streaming: token counts, time-to-first-token
  ai-otel-104/            # async streaming: concurrency and cancellation
  ai-otel-105/            # tool calling: agent and tool spans, cost per turn
  ai-otel-106/            # structured outputs: schema outcomes, not just errors
```

## The examples

Read them in order — each one adds a single idea to the one before it.

### [`ai-python-101`](packages/ai-python-101) — the OpenAI client, minimally

Construct `OpenAI()`, send a hard-coded conversation (system: *"you are a
pirate"*, user: *"Hello"*), print the reply. A `Conversation` class adds message
history so follow-up turns carry context.

The one habit worth stealing: every entry point takes an optional `client`. That
single parameter is why the tests need no API key, no network, and no mocking
library — they pass a stub and assert on the request that was built.

```sh
make run PKG=ai-python-101
```

### [`ai-otel-101`](packages/ai-otel-101) — what did that call cost?

The same call, wrapped in OpenTelemetry. `InstrumentedChat` owns the client and
emits, per request, a `chat <model>` CLIENT span carrying the
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
— operation, provider, requested and responding model, finish reasons, and
input/output token counts — plus two histograms, `gen_ai.client.token.usage`
and `gen_ai.client.operation.duration`.

Why the standard names matter: a generic OTel backend can chart cost per model
without knowing anything about this code. The provider is derived from the
endpoint the client actually points at, so running against a local model reports
`ollama`, not a hardcoded `openai` in the very attribute a cost dashboard groups
by. Message content is opt-in, because prompts are user data.

```sh
make run-ollama PKG=ai-otel-101
```

### [`ai-otel-102`](packages/ai-otel-102) — the same idea, standing alone

One module, `observe.py`, with no dependency on anything else in this repo:
copy the file into your own project and it works. A test enforces that, failing
if an intra-repo dependency or sibling import ever appears.

It also inverts the shape. Instead of a wrapper making the call for you, a `with`
block wraps a call **you** make:

```python
with telemetry.chat(model) as observed:
    observed.record(client.chat.completions.create(...))
```

That is what fits a call site already doing something interesting — retries, a
fallback between providers, streaming. Skip `record()` on an early return and
the span still closes cleanly; instrumentation that forces you to restructure
your code does not survive contact with a real call site.

```sh
make run-ollama PKG=ai-otel-102
```

### [`ai-otel-103`](packages/ai-otel-103) — streaming, where the counts go missing

A streamed response carries **no** `usage` block unless the request asks for
`stream_options={"include_usage": True}` — and the chunk that then carries it
arrives with an *empty* `choices` list, which is exactly where naive
`chunk.choices[0]` code crashes on the last chunk of a working stream. Forget
either detail and token telemetry reads zero for every streamed call, which in a
chat UI is most of them:

```
stream WITHOUT stream_options : 17 chunks, usage: None
stream WITH  include_usage    : 14 chunks, usage: prompt=35 completion=13
```

So the span stays open for the whole stream rather than just the opening
request, which also makes time-to-first-token measurable. That matters because
total duration grows with the length of the answer — a slow start and a long
answer look identical — while TTFT is the latency a user actually feels.

```sh
make run-ollama PKG=ai-otel-103
make run-ollama PKG=ai-otel-103 ARGS=--no-usage   # watch the counts vanish
```

### [`ai-otel-104`](packages/ai-otel-104) — async: many at once, and cancelled

The same stream through `AsyncOpenAI`, several at a time. Async adds two traps,
and both are about code that looks correct.

`asyncio.CancelledError` inherits from **`BaseException`**, so `except
Exception` instrumentation never sees a cancelled request — and cancellation is
the *normal* ending for a streaming chat, because closing the tab is what
cancels it. Those requests otherwise vanish from telemetry along with the tokens
they already burned. Recording them as errors is the other half of the mistake:
a user who navigated away is nothing to page on. They get
`app.stream.cancelled`, keep their span status unset, still report their tokens,
and always re-raise.

The second is where you open the span. Spans live in a `contextvars` context
that each asyncio task copies at creation, so opening it *inside* the coroutine
gives N concurrent streams N sibling spans, each with its own first-token
latency. One span around a `gather` would just measure the slowest call.

```sh
make run-ollama PKG=ai-otel-104
```

### [`ai-otel-105`](packages/ai-otel-105) — tool calling, and what a turn costs

One question stops being one model call. The model asks for a function, you run
it, you send the result back, it answers — two round trips minimum, more if it
chains tools. The whole turn becomes one trace: an `invoke_agent` span over
alternating `chat` and `execute_tool` children, so the cost of answering *one
question* is visible in one place.

It also carries a finding worth keeping. The obvious prediction is that round
two costs more input than round one, since it resends everything plus the tool
output. Measured against `llama3.2:3b`, it went the other way — 258 input tokens
down to 173 — because Ollama's reported `prompt_tokens` shrinks as its cached
prefix grows, while OpenAI counts the whole prompt and reports cached tokens as
a subset. So the example reports usage **per round** instead of asserting a
rule: how a growing conversation maps to a bill is your provider's business, and
it is measurable.

Tool failures are recorded on the tool span and handed back to the model as
text, because a model told what went wrong can often recover, while an exception
escaping your instrumentation guarantees it cannot.

```sh
make run PKG=ai-otel-105
```

### [`ai-otel-106`](packages/ai-otel-106) — structured outputs, and what "success" means

"Did the model answer?" and "did it answer in a shape my code can use?" are
different questions, and only the second matters to the caller. Same model, same
prompt, two `response_format` settings:

```
{"type": "json_object"}   ->  { "The Black Pearl: An Infamous Pirate Ship": 1.4 }
{"type": "json_schema"}   ->  { "name": "Black Pearl", "crew": 400, "cannons": 40 }
```

Both parse. The first is useless. So every call is classified — `parsed`,
`refused`, `invalid_json`, `schema_invalid`, `truncated` — and the outcome rides
on the span next to a counter, because a real `--loose` span reports
`finish_reason: "stop"` beside `app.output.outcome: schema_invalid`:
`finish_reason` is the model's opinion of the call, not the caller's.

`refused` is deliberately not an error — a refusal is the model working, and
burying it in the error rate hides a prompt problem inside what looks like an
outage. And a schema buys shape, not sense: one strict run came back perfectly
typed with `crew: 0` and a reputation of `"]["`.

```sh
make run PKG=ai-otel-106
make run PKG=ai-otel-106 ARGS=--loose   # watch a "successful" call be unusable
```

Each example's own README goes deeper. `ai-otel-102` deliberately depends on
nothing; `ai-otel-103` builds on `ai-otel-101`, `ai-otel-104` on `103`, and
`ai-otel-105` reuses `101`'s `InstrumentedChat` for its model calls — so every
example's spans and metrics land in the same dashboards.

## Getting started

```sh
make install                 # uv sync --all-packages
make test                    # run every package's tests
make run PKG=ai-otel-101     # run one example
```

### `make run` picks a backend for you

It needs no configuration and no OpenAI account. Every example is plain
OpenAI-compatible HTTP, so a local model works with nothing in the code
changing — `make run` just decides where to send the request:

| Situation | What `make run` does |
| --- | --- |
| `OPENAI_API_KEY` is exported | Uses the hosted API with it |
| No key, [Ollama](https://ollama.com) answering | Falls back to a local model, free, and says so: `==> no OPENAI_API_KEY set; using local Ollama (llama3.2:3b)` |
| Neither | Prints both ways to fix it and stops — no traceback from inside the SDK |

So the shortest path from a fresh clone to a running example is:

```sh
ollama pull llama3.2:3b
make run PKG=ai-otel-101
```

Force the choice when it matters — an auto-selecting `run` is exactly wrong when
you meant to exercise the real API:

```sh
make run-openai PKG=ai-otel-101    # always the hosted API; refuses early without a key
make run-ollama PKG=ai-otel-101    # always the local model
```

`run-ollama` first checks that the server answers and the model is pulled, so a
model you forgot to pull says so instead of surfacing as a 404. It then sets
`OPENAI_BASE_URL`, `OPENAI_MODEL`, and a placeholder `OPENAI_API_KEY` — the SDK
insists on a non-empty one, and Ollama ignores it. The OTel examples notice
where the request went and report `gen_ai.provider.name: ollama`.

Only the Ollama path defaults to `llama3.2:3b`; the hosted path stays on
`gpt-4o-mini` (via `OPENAI_MODEL`), so exporting a real key never sends OpenAI a
llama model name.

### Targets and variables

| Target | |
| --- | --- |
| `make install` | Create/refresh the workspace `.venv` with every member installed |
| `make test` | Run the whole workspace suite |
| `make test-pkg PKG=…` | Run one package's tests |
| `make run PKG=…` | Run one example, backend chosen as above |
| `make run-openai PKG=…` | Run against the hosted API |
| `make run-ollama PKG=…` | Run against local Ollama |
| `make badge` / `badge-check` | Regenerate / verify the test-count badge |
| `make lint` | Byte-compile every example as a cheap syntax check |
| `make clean` / `distclean` | Remove caches / also the virtualenv |

| Variable | Default | |
| --- | --- | --- |
| `PKG` | `ai-python-101` | Which example to run or test |
| `ARGS` | — | Passed through to the example, e.g. `ARGS=--no-usage` |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model for the Ollama path |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Where Ollama listens |

`make help` prints the same target list.

The test-count badge is generated, not typed: `make badge` rewrites
`.github/badges/tests.json` from pytest's own collection, and CI runs
`make badge-check`, which fails the build if the committed count has drifted.

## Adding an example

```sh
uv init --package packages/<name>
uv add --package <name> <dependency>
```

Declare dependencies in the new package's own `pyproject.toml`, never the root.
If one example depends on another, point at it with
`[tool.uv.sources] <name> = { workspace = true }`.
