# ai-otel-109

Multi-turn conversations, where the expensive thing is the part you already
paid for.

## Turn ten costs ten times turn one

A chat model has no memory. Every turn resends the entire history, so the
*input* grows with each exchange and the cost of a conversation grows with the
square of its length. Turn 1 sends one question; turn 10 sends nine questions,
nine answers, and a question.

Nobody notices, because each individual call looks fine. The bill notices. A
real 6-turn run against `llama3.2:3b` — same six questions, three strategies:

```
turn   keep-all   sliding window(2)   summarizing
  1      47 in         47 in             47 in
  2     102 in         86 in             81 in
  3     143 in        140 in            115 in
  4     191 in        171 in            158 in
  5     250 in        179 in            166 in   <- compacted
  6     303 in        171 in            212 in
total  1036 in        794 in            890 in
                     (-23%)            (-14%, incl. its compaction call)
```

Input grows every turn under keep-all because history is resent, not because
the questions got harder.

## Four strategies, one interface

| Strategy | Keeps | Costs |
| --- | --- | --- |
| `KeepAll` | everything | grows without bound, then hits the context window |
| `SlidingWindow(n)` | the last *n* exchanges | forgets, silently |
| `TokenBudget(n)` | as much as fits | forgets, but by size rather than count |
| `Summarizing(...)` | a summary plus recent turns | **an extra model call** |

Each records what it did — `app.memory.strategy`, `app.memory.messages_sent`,
`app.memory.messages_dropped` — so "the assistant forgot what I told it" has an
answer in the trace instead of a shrug.

## Compaction is not free, and it should look like it

Summarising old turns costs a model call of its own. It is easy to add and easy
to forget, and then a conversation makes two calls per turn while the dashboard
still says one.

So compaction gets its own span, nested inside the turn that triggered it:

```
turn 5  session=b1f0…  app.memory.strategy=summarizing
├── chat llama3.2:3b   (compaction: 214 in / 41 out)   app.memory.compaction=true
└── chat llama3.2:3b   (the actual answer: 96 in / 38 out)
```

The counter `app.memory.compactions` says how often it happens — and writing this
example is what proved why it needs counting.

The first implementation re-summarised on **every** turn once triggered: four
compaction calls in six turns, 1728 tokens against keep-all's 1015. Compaction
made the conversation *more* expensive than no memory management at all, and
nothing in the code looked wrong. Only the token totals showed it.

It now waits for `recompact_after` messages to age out before paying again, and
messages that have aged out but are not yet in the summary ride along verbatim,
so nothing is dropped unsummarised. One compaction over six turns instead of
four — which is what turns the strategy from a 70% surcharge into the 14% saving
in the table.

The general point survives the fix: compaction pays over long conversations and
taxes short ones, and which side of that line you are on is a measurement, not a
judgement.

## Every turn carries the conversation id

`gen_ai.conversation.id` goes on every span of every turn, so a trace backend
can assemble one user's whole session out of calls that are otherwise
unrelated. Without it, turn 7 of a conversation that went wrong is
indistinguishable from any other request.

## Our estimate versus their count

Trimming has to happen *before* the call, so it works from an estimate —
characters ÷ 4, no tokeniser dependency. The provider's real count arrives
afterwards, so the demo prints the error between them — and it is not small, nor
is it in the flattering direction:

```
turn 1: estimate -51%      turn 6: estimate -31%
```

Characters ÷ 4 **under**-counts, consistently, because every message carries
per-message overhead the character count cannot see, and short messages are
mostly overhead. A budget built on this heuristic will therefore send more than
it thinks — the failure mode is a context-window error, not a saving.

Measure the drift for your own traffic before trusting a budget to it, and never
present the estimate as the number: `app.memory.estimated_tokens` is named as an
estimate on purpose.

## Run it

```sh
make run PKG=ai-otel-109                       # keep-all vs a sliding window
make run PKG=ai-otel-109 ARGS=--summarize      # add the compaction strategy
```

## Test it

```sh
make test-pkg PKG=ai-otel-109
```

Scripted replies, so the trimming, the ordering, the compaction trigger and the
token accounting are exact.
