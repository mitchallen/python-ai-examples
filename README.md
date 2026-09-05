# python-ai-examples

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
```

## Getting started

```sh
make install          # uv sync --all-packages
make test             # run every package's tests
make run PKG=ai-python-101   # run one example (needs OPENAI_API_KEY)
```

`make help` lists the targets.

## Adding an example

```sh
uv init --package packages/<name>
uv add --package <name> <dependency>
```

Declare dependencies in the new package's own `pyproject.toml`, never the root.
If one example depends on another, point at it with
`[tool.uv.sources] <name> = { workspace = true }`.
