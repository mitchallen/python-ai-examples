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
```

## Getting started

```sh
make install          # uv sync --all-packages
make test             # run every package's tests
make run PKG=ai-python-101   # run one example (needs OPENAI_API_KEY)
```

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
