# ai-otel-105

Tool calling — and the thing tool calling does to your bill.

One user question stops being one model call. The model answers with a *request
to run a function*, you run it, you send the result back, and it answers again.
Two round trips minimum, more if it chains tools — and the conversation grows
every round, because round two resends round one plus the tool output.

None of that is visible in a per-call view, so this example puts the whole turn
in one trace. A real run against `llama3.2:3b`:

```
invoke_agent quartermaster          app.agent.rounds=2  app.agent.tool_calls=2
├── chat llama3.2:3b                in=258  out=29   (asks for both tools)
├── execute_tool get_weather        gen_ai.tool.call.id=call_…
├── execute_tool check_stores       gen_ai.tool.call.id=call_…
└── chat llama3.2:3b                in=173  out=90   (answers)
```

Two model calls and two tool executions for one question, 550 tokens in total.

### Measure the second round, do not assume it

The obvious prediction is that round two costs *more* input than round one: it
carries everything from round one plus two tool results. The messages certainly
grow — 2 messages become 5. But the numbers above go the other way, 258 down to
173, and that is not a typo:

```
round 1: messages=2  prompt_tokens=233  cached=26
round 2: messages=5  prompt_tokens=148  cached=63
```

Ollama's reported `prompt_tokens` shrinks as its cached prefix grows. OpenAI
counts differently — there `prompt_tokens` is the whole prompt, with
`cached_tokens` a subset of it — so the same conversation bills differently
depending on who serves it.

Which is the real lesson, and the reason the example reports usage **per round**
rather than printing a rule: the shape of a tool-calling bill is a property of
your provider, and it is measurable. Do not reason about it from first
principles.

## Run it

```sh
make run PKG=ai-otel-105        # local model by default; no key needed
```

It prints the answer, then the accounting: rounds, tools called, and tokens
totalled across the whole turn.

## What gets emitted

Model calls reuse [`ai-otel-101`](../ai-otel-101)'s `InstrumentedChat`, so every
round is an ordinary `chat <model>` span with token counts, and the standard
histograms already aggregate correctly. This package adds the two spans
tool calling introduces, both named by the
[GenAI conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

| Span | Attributes |
| --- | --- |
| `invoke_agent <name>` | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name`, `app.agent.rounds`, `app.agent.tool_calls` |
| `execute_tool <tool>` | `gen_ai.operation.name=execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type`, `error.type` on failure |

Tool executions also land in `gen_ai.client.operation.duration`, keyed on
`gen_ai.operation.name=execute_tool` — the same instrument as the model calls,
which is how the conventions intend it: one latency metric, split by operation.
A tool that calls a slow database is then visible in the same chart as a slow
model.

**Totals are deliberately not copied onto the parent span.** The agent span
records how many rounds happened, not the summed tokens, because the backend
sums the children — duplicating it there would double-count every turn.

## Two failures the loop has to survive

**A tool that raises.** Killing the turn would be wrong: the model can often
recover if you tell it what went wrong. So the exception is recorded on the tool
span (`error.type`), and the text of the error goes back to the model as the
tool result. The turn continues.

**Arguments that aren't valid JSON.** Models do emit malformed arguments. Same
treatment — recorded, fed back, loop continues — rather than an exception
escaping from inside your instrumentation.

There is also a `max_rounds` ceiling. Hitting it sets `app.agent.truncated` on
the agent span, because a model and a tool talking to each other forever is a
real failure mode and it must be visible rather than merely expensive.

## Test it

```sh
make test-pkg PKG=ai-otel-105
```

Scripted responses drive the loop: a single tool call, a chain of two, a tool
that raises, malformed arguments, and a model that never stops asking for tools.
