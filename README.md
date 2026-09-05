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
```

## Getting started

```sh
make install          # uv sync --all-packages
make test             # run every package's tests
make run PKG=ai-python-101   # run one example (needs OPENAI_API_KEY)
```

No OpenAI key? Every example is plain OpenAI-compatible HTTP, so a local
[Ollama](https://ollama.com) model works with no code changes:

```sh
ollama pull llama3.2:3b
make run-ollama PKG=ai-otel-101          # no key, no cost
make run-ollama PKG=ai-otel-101 OLLAMA_MODEL=qwen3.5:cloud
```

The target checks that the server answers and the model is pulled, then sets
`OPENAI_BASE_URL`, `OPENAI_MODEL`, and a placeholder `OPENAI_API_KEY` (the SDK
insists on a non-empty one; Ollama ignores it). The OTel examples notice and
report `gen_ai.provider.name: ollama`.

`make help` lists the targets.

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
