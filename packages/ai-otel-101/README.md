# ai-otel-101

The same OpenAI call as [`ai-python-101`](../ai-python-101), wrapped in
OpenTelemetry so every request produces a span and two metrics — including the
number the finance team asks about, **tokens**.

## Run it

```sh
export OPENAI_API_KEY=sk-...
make run PKG=ai-otel-101          # from the repo root
```

Telemetry goes to the console by default. Point it at a real backend by setting
`OTEL_EXPORTER_OTLP_ENDPOINT` and installing the extra:

```sh
uv add --package ai-otel-101 "opentelemetry-exporter-otlp-proto-http"
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 make run PKG=ai-otel-101
```

## What gets emitted

One CLIENT span per call, named `chat <model>` per the OpenTelemetry
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

| Attribute | Example |
| --- | --- |
| `gen_ai.operation.name` | `chat` |
| `gen_ai.system` / `gen_ai.provider.name` | `openai`, `ollama`, … (both keys, since the name was changed) |
| `gen_ai.request.model` | `gpt-4o-mini` |
| `gen_ai.response.model` | `gpt-4o-mini-2024-07-18` |
| `gen_ai.response.id`, `gen_ai.response.finish_reasons` | `chatcmpl-…`, `["stop"]` |
| `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` | `11` / `7` |
| `error.type` (failures only) | `RateLimitError` |

Plus two histograms, which is what dashboards and alerts actually read:

- **`gen_ai.client.token.usage`** (`{token}`) — split by `gen_ai.token.type`
  = `input` / `output`, keyed on both the requested and responding model.
- **`gen_ai.client.operation.duration`** (`s`) — carries `error.type` when the
  call failed, so success and failure latency stay separable.

Sticking to the standard names is the entire trick: it's why a generic OTel
backend can chart cost per model without knowing anything about this code.

## The provider is derived, not assumed

`gen_ai.provider.name` comes from the endpoint the client actually points at,
via `provider_from_base_url()` — `api.openai.com` → `openai`,
`localhost:11434` → `ollama`, `*.openai.azure.com` → `azure.ai.openai`, and
anything else speaking the protocol → `openai_compatible` (generic on purpose:
the hostname would be accurate but would leak internal infrastructure names into
telemetry that often leaves the network). Hardcoding `openai` would put the
wrong value in the one attribute a cost dashboard groups by, the moment you run
against a local model:

```sh
make run-ollama PKG=ai-otel-101     # from the repo root; reports provider "ollama"
```

Pass `provider="…"` to `InstrumentedChat` when you know better than the URL.

## Message content is opt-in

Prompts and completions are user data, so they are **off** by default. Turn them
on with `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` (or
`InstrumentedChat(..., capture_content=True)`) and they ride as span *events*,
not attributes.

## Use it in your own code

```python
from ai_otel_101 import InstrumentedChat, configure_telemetry
from openai import OpenAI

telemetry = configure_telemetry("my-service")
chat = InstrumentedChat(OpenAI(), tracer=telemetry.tracer(), meter=telemetry.meter())

response = chat.complete(
    [{"role": "user", "content": "Hello"}], model="gpt-4o-mini"
)
telemetry.shutdown()   # flush; batch exporters drop data otherwise
```

Passing `tracer`/`meter` explicitly is optional — omit them and the wrapper uses
the process-global providers.

## Test it

```sh
make test-pkg PKG=ai-otel-101
```

The tests use an in-memory span exporter and metric reader with a stub client:
no network, no API key, no collector. They assert on the emitted attribute
names, because those names are the contract with the backend.
