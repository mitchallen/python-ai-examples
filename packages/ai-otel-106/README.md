# ai-otel-106

Structured outputs, where "did the model answer?" becomes "did it answer in a
shape my code can use?"

Those are different questions, and only the second one matters to the caller.
A run against `llama3.2:3b` shows how far apart they can be — same model, same
prompt, two `response_format` settings:

```
response_format={"type": "json_object"}   ->  { "The Black Pearl: An Infamous Pirate Ship": 1.4 }
response_format={"type": "json_schema"}   ->  { "name": "Black Pearl", "crew": 400, "cannons": 40 }
```

Both are valid JSON. `json.loads` accepts both. The first is useless — the keys
are prose and the value is a number nobody asked for — and any code expecting
`.name` breaks on it. So "valid JSON" is not the success condition, and a span
that only records `error.type` on exceptions will call that first response a
clean success.

### What a schema does and does not buy

Shape, not sense. A strict run against the same small model came back `parsed`
and perfectly typed:

```
name: "Black Pearl"   crew: 0   cannons: 4   notorious_for: "]["
```

Every field present, every type correct, `crew: 0` and a name for its
reputation that is two brackets. Schema validation is a guarantee about
structure — it is not a quality check, and `parsed` does not mean *right*.
Content evaluation is a separate problem from the one this example solves.

## Five outcomes, not two

This example classifies every structured call and puts the answer on the span as
`app.output.outcome`:

| Outcome | What happened | Error? |
| --- | --- | --- |
| `parsed` | Valid JSON, matched the schema. Usable. | no |
| `refused` | The model declined, via the `refusal` field. | **no** |
| `invalid_json` | Not parseable at all. | yes |
| `schema_invalid` | Parsed, wrong shape — the case above. | yes |
| `truncated` | `finish_reason=length`; the JSON stops mid-object. | yes |

`refused` is deliberately not an error. A refusal is the model working
correctly, and filing it as a failure buries a real signal — a rising refusal
rate is a prompt problem — inside your error rate, where it looks like an
outage.

`schema_invalid` and `truncated` are the two that a naive implementation misses
entirely, and they are the ones that reach production.

Here is a real `--loose` span, and the two attributes to read together:

```
"gen_ai.response.finish_reasons": ["stop"],     <- the model thinks it succeeded
"app.output.outcome": "schema_invalid",         <- the caller got nothing usable
"gen_ai.usage.output_tokens": 187               <- and paid for it
```

`finish_reason` is the model's opinion of the call. It is not the caller's, and
instrumentation that stops there reports a clean success rate over responses no
code can consume.

Alongside the usual token and duration histograms there is a counter,
`app.structured.outputs`, keyed by outcome — so "what fraction of structured
calls came back unusable?" is one query rather than a log-scraping exercise.

## Validate locally anyway

Strict mode is a real guarantee on providers that implement it. This example
still validates every response against the schema on arrival, because the
guarantee evaporates the moment you route through a gateway, fall back to a
second provider, or point at a local model — and the code above is what a local
model actually returned.

The check is cheap. Trusting the wire format is how `schema_invalid` becomes a
production incident instead of a metric.

## Run it

```sh
make run PKG=ai-otel-106                  # strict schema; expects `parsed`
make run PKG=ai-otel-106 ARGS=--loose     # plain JSON mode; usually `schema_invalid`
```

`--loose` reproduces the contrast above live: same prompt, same model, and an
outcome your code cannot use. Exactly *which* useless JSON comes back varies run
to run — `{}` and the prose-keyed object above are both things this model
actually returned — which is the point: without a schema you cannot predict the
shape, so you have to check it.

## The shape

```python
from pydantic import BaseModel

class Ship(BaseModel):
    name: str
    crew: int
    cannons: int

result = structured.parse(messages, model="gpt-4o-mini", schema=Ship)
if result.ok:
    print(result.parsed.name)      # a Ship, not a dict
else:
    print(result.outcome, result.error)
```

`parse()` never raises for a bad answer — a refusal or a malformed shape is data
about the call, not an exception. It returns the outcome and lets the caller
decide, which is also what makes every path testable.

The request itself carries a **strict** JSON schema derived from the Pydantic
model: `strict_schema()` marks every object `additionalProperties: false` and
lists every property as required, which is what strict mode demands. The SDK's
`client.chat.completions.parse()` helper does something similar and raises on
the failure cases instead of classifying them.

## Test it

```sh
make test-pkg PKG=ai-otel-106
```

Every outcome is covered with stubbed responses, including the exact
valid-JSON-wrong-shape payload the local model produced.
