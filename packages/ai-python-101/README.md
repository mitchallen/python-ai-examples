# ai-python-101

The smallest useful example of the Python OpenAI client: construct `OpenAI()`,
send a hard-coded conversation ("you are a pirate" / "Hello"), print the reply.

## Run it

```sh
export OPENAI_API_KEY=sk-...
make run PKG=ai-python-101          # from the repo root
```

Optional: `OPENAI_MODEL` overrides the default model (`gpt-4o-mini`). Any
arguments become the user turns:

```sh
uv run --package ai-python-101 ai-python-101 "Hello" "Where be the treasure?"
```

## What's in it

| Piece | Purpose |
| --- | --- |
| `create_client()` | Constructs `OpenAI()`; the key comes from `OPENAI_API_KEY`. |
| `build_conversation()` | The hard-coded messages: system pirate prompt + one user turn. |
| `ask_pirate()` | One request, one reply. |
| `Conversation` | Keeps the history so follow-up turns carry context. |

Every entry point accepts an optional `client`, which is the whole trick behind
the tests: they inject a stub, so `make test` needs no API key and makes no
network call.

## Test it

```sh
make test-pkg PKG=ai-python-101
```
