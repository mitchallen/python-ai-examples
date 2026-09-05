"""The traced conversation: spans per turn, history growth, and compaction."""

from __future__ import annotations

from typing import Any

from ai_otel_101 import semconv as sc

from ai_otel_109 import (
    CONVERSATION_ID,
    ESTIMATED_TOKENS,
    MESSAGES_DROPPED,
    MESSAGES_SENT,
    METRIC_COMPACTIONS,
    METRIC_DROPPED,
    METRIC_MESSAGES,
    KeepAll,
    SlidingWindow,
    Summarizing,
    TracedConversation,
)


def points(metric_reader: Any, name: str) -> list[Any]:
    data = metric_reader.get_metrics_data()
    return [
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def conversation(client, tracer, meter, **kwargs) -> TracedConversation:
    kwargs.setdefault("system_prompt", "You are a pirate.")
    return TracedConversation(
        client, model="gpt-4o-mini", tracer=tracer, meter=meter, **kwargs
    )


def turn_spans(spans):
    return [s for s in spans.get_finished_spans() if s.name.startswith("turn ")]


# -- history ----------------------------------------------------------------


def test_the_reply_is_remembered(client, tracer, meter):
    chat = conversation(client, tracer, meter)

    chat.ask("What's your name?")

    assert [m["role"] for m in chat.history] == ["system", "user", "assistant"]
    assert chat.history[-1]["content"] == "Ahoy!"


def test_each_turn_resends_the_whole_history(client, tracer, meter):
    chat = conversation(client, tracer, meter)

    chat.ask("one")
    chat.ask("two")
    chat.ask("three")

    # 2, 4, 6 messages: the quadratic bill, visible in the requests themselves.
    assert [len(client.messages_sent(i)) for i in range(3)] == [2, 4, 6]


def test_input_tokens_grow_with_every_turn(client, tracer, meter):
    chat = conversation(client, tracer, meter)

    first = chat.ask("one")
    third = (chat.ask("two"), chat.ask("three"))[-1]

    assert third.input_tokens > first.input_tokens


def test_a_sliding_window_stops_the_growth(make_client, tracer, meter):
    chat = conversation(
        make_client(), tracer, meter, strategy=SlidingWindow(max_turns=1)
    )

    for question in ("one", "two", "three", "four"):
        result = chat.ask(question)

    # system + one exchange + the new question, regardless of how long it runs.
    assert result.messages_sent == 4
    assert result.messages_dropped == 4


# -- spans ------------------------------------------------------------------


def test_one_span_per_turn_carrying_the_conversation_id(client, tracer, meter, spans):
    chat = conversation(client, tracer, meter)

    chat.ask("one")
    chat.ask("two")

    turns = turn_spans(spans)
    assert [s.name for s in turns] == ["turn 1", "turn 2"]
    # Without this a backend cannot tell one user's session from any other
    # traffic.
    assert {s.attributes[CONVERSATION_ID] for s in turns} == {chat.conversation_id}


def test_the_model_call_is_a_child_of_the_turn(client, tracer, meter, spans):
    chat = conversation(client, tracer, meter)

    chat.ask("one")

    (turn,) = turn_spans(spans)
    children = [s for s in spans.get_finished_spans() if s is not turn]
    assert [s.name for s in children] == ["chat gpt-4o-mini"]
    assert children[0].parent.span_id == turn.get_span_context().span_id


def test_the_turn_span_records_what_memory_did(make_client, tracer, meter, spans):
    chat = conversation(
        make_client(), tracer, meter, strategy=SlidingWindow(max_turns=1)
    )

    for question in ("one", "two", "three"):
        chat.ask(question)

    last = turn_spans(spans)[-1]
    assert last.attributes["app.memory.strategy"] == "sliding_window"
    assert last.attributes[MESSAGES_SENT] == 4
    assert last.attributes[MESSAGES_DROPPED] == 2
    assert last.attributes[ESTIMATED_TOKENS] > 0


def test_a_given_id_is_used_instead_of_a_generated_one(client, tracer, meter, spans):
    chat = conversation(client, tracer, meter, conversation_id="session-42")

    chat.ask("one")

    assert turn_spans(spans)[0].attributes[CONVERSATION_ID] == "session-42"


# -- compaction --------------------------------------------------------------


def test_compaction_makes_its_own_model_call(make_client, tracer, meter, spans):
    client = make_client()
    chat = conversation(
        client, tracer, meter, strategy=Summarizing(trigger_after=2, keep_turns=1, recompact_after=2)
    )

    for question in ("one", "two", "three"):
        chat.ask(question)

    compactions = [s for s in spans.get_finished_spans() if s.name == "compaction"]
    assert len(compactions) == 1
    # Four calls for three turns: the extra one is the compaction.
    assert len(client.calls) == 4


def test_the_compaction_span_sits_inside_its_turn(make_client, tracer, meter, spans):
    chat = conversation(
        make_client(), tracer, meter, strategy=Summarizing(trigger_after=2, keep_turns=1, recompact_after=2)
    )

    for question in ("one", "two", "three"):
        chat.ask(question)

    (compaction,) = [s for s in spans.get_finished_spans() if s.name == "compaction"]
    turn_ids = {s.get_span_context().span_id for s in turn_spans(spans)}
    assert compaction.parent.span_id in turn_ids
    assert compaction.attributes["app.memory.compaction"] is True


def test_compaction_tokens_are_counted_in_the_total(make_client, tracer, meter):
    chat = conversation(
        make_client(), tracer, meter, strategy=Summarizing(trigger_after=2, keep_turns=1, recompact_after=2)
    )

    for question in ("one", "two", "three"):
        chat.ask(question)

    # Compaction is not free, so its tokens belong in the conversation's bill.
    per_turn = sum(r.input_tokens for r in chat.stats.per_turn)
    assert chat.stats.input_tokens > per_turn
    assert chat.stats.compactions == 1


def test_the_turn_that_compacted_is_flagged(make_client, tracer, meter):
    chat = conversation(
        make_client(), tracer, meter, strategy=Summarizing(trigger_after=2, keep_turns=1, recompact_after=2)
    )

    results = [chat.ask(q) for q in ("one", "two", "three")]

    assert [r.compacted for r in results] == [False, False, True]


# -- metrics and totals ------------------------------------------------------


def test_messages_and_drops_are_metrics(make_client, tracer, meter, metric_reader):
    chat = conversation(
        make_client(), tracer, meter, strategy=SlidingWindow(max_turns=1)
    )

    for question in ("one", "two", "three"):
        chat.ask(question)

    (messages,) = points(metric_reader, METRIC_MESSAGES)
    (dropped,) = points(metric_reader, METRIC_DROPPED)
    assert messages.count == 3
    # Only the third turn drops anything: the window holds one exchange plus
    # the question being asked.
    assert dropped.value == 2


def test_compactions_are_counted(make_client, tracer, meter, metric_reader):
    chat = conversation(
        make_client(), tracer, meter, strategy=Summarizing(trigger_after=2, keep_turns=1, recompact_after=2)
    )

    for question in ("one", "two", "three"):
        chat.ask(question)

    (point,) = points(metric_reader, METRIC_COMPACTIONS)
    assert point.value == 1


def test_running_totals_accumulate(client, tracer, meter):
    chat = conversation(client, tracer, meter, strategy=KeepAll())

    chat.ask("one")
    chat.ask("two")

    assert chat.stats.turns == 2
    assert chat.stats.total_tokens == chat.stats.input_tokens + chat.stats.output_tokens
    assert len(chat.stats.per_turn) == 2


def test_the_estimate_is_reported_against_the_real_count(client, tracer, meter):
    chat = conversation(client, tracer, meter)

    result = chat.ask("one")

    # Named an estimate because it is one; the drift is the caller's to judge.
    assert result.estimated_tokens >= 0
    assert result.estimate_error is not None
