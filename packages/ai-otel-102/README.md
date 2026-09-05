# ai-otel-102

The same idea as [`ai-otel-101`](../ai-otel-101) — OpenTelemetry around an
OpenAI chat call — but **self-contained**: one module, `observe.py`, with no
dependency on any other package in this repo. Copy that single file into your
own project and it works.

Two differences from 101, and they are the reason both exist:

| | `ai-otel-101` | `ai-otel-102` |
| --- | --- | --- |
| Shape | `InstrumentedChat` wraps the client and makes the call for you | a `with` block wraps a call **you** make |
| Deps | reuses `ai-python-101` as a workspace sibling | stands alone, `openai` + OTel only |

The context-manager shape is what you want when the call site is already doing
something interesting — retries, a fallback between providers, streaming — and
routing all of that through a wrapper's signature would be the tail wagging the
dog.

## Run it

```sh
export OPENAI_API_KEY=sk-...
make run PKG=ai-otel-102          # from the repo root
```

Telemetry prints to the console. Set `OTEL_EXPORTER_OTLP_ENDPOINT` (and install
the `otlp` extra) to ship it to a real backend instead.

## The pattern

```python
from ai_otel_102 import build_conversation, configure_telemetry
from openai import OpenAI

telemetry = configure_telemetry("my-service")
chat = telemetry.chat_telemetry()
client = OpenAI()

with chat.chat("gpt-4o-mini") as observed:
    observed.record(
        client.chat.completions.create(
            model="gpt-4o-mini", messages=build_conversation("Hello")
        )
    )
    print(observed.text(), observed.input_tokens, observed.output_tokens)

telemetry.shutdown()   # flush; batch exporters drop data otherwise
```

`observed.record(...)` is what lifts the response onto the span. Skip it — an
early return, a cache hit, a call that never happened — and the span still
closes cleanly, just without usage data. Instrumentation that forces you to
restructure your code does not survive contact with a real call site.

There is also a one-liner if you don't need the call site: `ask_pirate("Hello")`.

## The provider is derived, not assumed

`provider_from_base_url()` / `provider_for(client)` name the provider from the
endpoint the client points at: `api.openai.com` → `openai`, `localhost:11434` →
`ollama`, `*.openai.azure.com` → `azure.ai.openai`, anything else speaking the
protocol → `openai_compatible` (generic so internal hostnames stay out of
telemetry).

```sh
make run-ollama PKG=ai-otel-102     # reports provider "ollama"
```

Because this example's whole point is an interesting call site, `provider` can
also be overridden per call — `chat.chat(model, provider="ollama")` — which is
what a block that falls back from one provider to another needs.

## What gets emitted

One CLIENT span named `chat <model>`, carrying the OpenTelemetry
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):
`gen_ai.operation.name`, `gen_ai.system` + `gen_ai.provider.name`,
`gen_ai.request.model`, `gen_ai.response.{id,model,finish_reasons}`,
`gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`, and `error.type`
when the block raises.

Plus the two standard histograms: `gen_ai.client.token.usage` (split by
`gen_ai.token.type`) and `gen_ai.client.operation.duration` (keyed on
`error.type` when it failed).

## Test it

```sh
make test-pkg PKG=ai-otel-102
```

In-memory span exporter and metric reader against a stub client. One test reads
this package's own `pyproject.toml` and fails if an intra-repo dependency ever
sneaks in — self-contained is a property worth enforcing, not just documenting.
