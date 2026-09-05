"""The tool-calling loop: what it sends, what it traces, and how it fails."""

from __future__ import annotations

from typing import Any

import pytest

from ai_otel_105 import ToolCallingChat, semconv as sc
from ai_otel_105.tools import Tool, ToolRegistry

QUESTION = [{"role": "user", "content": "Weather at Tortuga?"}]


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


def spans_named(spans, prefix: str) -> list[Any]:
    return [s for s in spans.get_finished_spans() if s.name.startswith(prefix)]


def agent(client, registry, tracer, meter, **kwargs) -> ToolCallingChat:
    return ToolCallingChat(client, registry, tracer=tracer, meter=meter, **kwargs)


# -- the happy path ----------------------------------------------------------


def test_one_tool_call_takes_two_rounds(
    tracer, meter, spans, registry, make_client, script
):
    client = make_client(
        script.wants(("get_weather", '{"port": "Tortuga"}')),
        script.says("Squalls, matey."),
    )

    turn = agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    assert turn.text == "Squalls, matey."
    assert turn.rounds == 2
    assert turn.tool_calls == ["get_weather"]
    assert turn.truncated is False


def test_the_tool_result_is_sent_back_to_the_model(
    tracer, meter, registry, make_client, script
):
    client = make_client(
        script.wants(("get_weather", '{"port": "Tortuga"}')),
        script.says("Squalls, matey."),
    )

    agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    second_round = client.messages_sent(1)
    # user, assistant(tool_calls), tool
    assert [m["role"] for m in second_round] == ["user", "assistant", "tool"]
    assert second_round[-1]["tool_call_id"] == "call_0"
    assert "squalls" in second_round[-1]["content"]
    # The assistant's request has to go back verbatim or the result orphans.
    assert second_round[1]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_tools_are_offered_on_every_round(tracer, meter, registry, make_client, script):
    client = make_client(
        script.wants(("get_weather", '{"port": "Tortuga"}')),
        script.says("Squalls."),
    )

    agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    for call in client.calls:
        names = {t["function"]["name"] for t in call["tools"]}
        assert names == {"get_weather", "check_stores"}


def test_parallel_tool_calls_in_one_round(
    tracer, meter, spans, registry, make_client, script
):
    client = make_client(
        script.wants(
            ("get_weather", '{"port": "Nassau"}'), ("check_stores", '{"item": "rum"}')
        ),
        script.says("Fog, and 41 units of rum."),
    )

    turn = agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    assert turn.tool_calls == ["get_weather", "check_stores"]
    assert len(spans_named(spans, "execute_tool")) == 2
    assert [m["role"] for m in client.messages_sent(1)][-2:] == ["tool", "tool"]


def test_a_chain_of_tools_across_rounds(tracer, meter, registry, make_client, script):
    client = make_client(
        script.wants(("get_weather", '{"port": "Nassau"}')),
        script.wants(("check_stores", '{"item": "rum"}')),
        script.says("Fog, and rum enough."),
    )

    turn = agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    assert turn.rounds == 3
    assert turn.tool_calls == ["get_weather", "check_stores"]


# -- the trace ---------------------------------------------------------------


def test_the_turn_is_one_trace(tracer, meter, spans, registry, make_client, script):
    client = make_client(
        script.wants(("get_weather", '{"port": "Tortuga"}')),
        script.says("Squalls."),
    )

    agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    finished = spans.get_finished_spans()
    (parent,) = [s for s in finished if s.name.startswith("invoke_agent")]
    children = [s for s in finished if s is not parent]
    assert len(children) == 3  # chat, execute_tool, chat
    assert {s.parent.span_id for s in children} == {parent.get_span_context().span_id}


def test_agent_span_attributes(tracer, meter, spans, registry, make_client, script):
    client = make_client(
        script.wants(("get_weather", '{"port": "Tortuga"}')),
        script.says("Squalls."),
    )

    agent(client, registry, tracer, meter, agent_name="bosun").run(
        QUESTION, model="gpt-4o-mini"
    )

    (parent,) = spans_named(spans, "invoke_agent")
    assert parent.name == "invoke_agent bosun"
    assert parent.attributes[sc.OPERATION_NAME] == "invoke_agent"
    assert parent.attributes[sc.AGENT_NAME] == "bosun"
    assert parent.attributes[sc.AGENT_ROUNDS] == 2
    assert parent.attributes[sc.AGENT_TOOL_CALLS] == 1


def test_totals_are_not_duplicated_onto_the_agent_span(
    tracer, meter, spans, registry, make_client, script
):
    # The children carry usage; the backend sums them. Copying totals up would
    # double-count every turn.
    client = make_client(
        script.wants(("get_weather", '{"port": "Tortuga"}')),
        script.says("Squalls."),
    )

    agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    (parent,) = spans_named(spans, "invoke_agent")
    assert sc.USAGE_INPUT_TOKENS not in parent.attributes
    assert sc.USAGE_OUTPUT_TOKENS not in parent.attributes


def test_tool_span_attributes(tracer, meter, spans, registry, make_client, script):
    client = make_client(
        script.wants(("get_weather", '{"port": "Tortuga"}')),
        script.says("Squalls."),
    )

    agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    (tool_span,) = spans_named(spans, "execute_tool")
    assert tool_span.name == "execute_tool get_weather"
    assert tool_span.attributes[sc.OPERATION_NAME] == "execute_tool"
    assert tool_span.attributes[sc.TOOL_NAME] == "get_weather"
    assert tool_span.attributes[sc.TOOL_CALL_ID] == "call_0"
    assert tool_span.attributes[sc.TOOL_TYPE] == "function"
    assert tool_span.attributes[sc.TOOL_DESCRIPTION]


def test_tool_latency_shares_the_duration_metric(
    tracer, meter, metric_reader, registry, make_client, script
):
    client = make_client(
        script.wants(("get_weather", '{"port": "Tortuga"}')),
        script.says("Squalls."),
    )

    agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    operations = {
        point.attributes[sc.OPERATION_NAME]
        for point in points(metric_reader, sc.METRIC_OPERATION_DURATION)
    }
    # One instrument, split by operation -- as the conventions intend.
    assert operations == {"chat", "execute_tool"}


def test_usage_is_recorded_per_round(tracer, meter, registry, make_client, script):
    client = make_client(
        script.wants(("get_weather", '{"port": "Tortuga"}'), prompt_tokens=233,
                     completion_tokens=29),
        script.says("Squalls.", prompt_tokens=148, completion_tokens=90),
    )

    turn = agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    # Per round, because how a growing conversation maps to billed tokens is a
    # provider's business, not something to assume.
    assert turn.usage_by_round == [(233, 29), (148, 90)]
    assert (turn.input_tokens, turn.output_tokens) == (381, 119)
    assert turn.total_tokens == 500


# -- failures the loop must survive -----------------------------------------


def test_a_raising_tool_is_recorded_and_fed_back(
    tracer, meter, spans, make_client, script
):
    def explode(**_: Any) -> str:
        raise RuntimeError("the hold is flooded")

    registry = ToolRegistry(
        [Tool("count_rum", "Count the rum.", {"type": "object", "properties": {}}, explode)]
    )
    client = make_client(
        script.wants(("count_rum", "{}")), script.says("We'll manage, matey.")
    )

    turn = agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    # The turn survives: the model gets told, and answers.
    assert turn.text == "We'll manage, matey."
    (tool_span,) = spans_named(spans, "execute_tool")
    assert tool_span.attributes[sc.ERROR_TYPE] == "RuntimeError"
    assert "the hold is flooded" in client.messages_sent(1)[-1]["content"]


def test_malformed_arguments_are_recorded_and_fed_back(
    tracer, meter, spans, registry, make_client, script
):
    client = make_client(
        script.wants(("get_weather", "{not json")), script.says("Say again?")
    )

    turn = agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    assert turn.text == "Say again?"
    (tool_span,) = spans_named(spans, "execute_tool")
    assert tool_span.attributes[sc.ERROR_TYPE] == "JSONDecodeError"
    assert "not valid JSON" in client.messages_sent(1)[-1]["content"]


def test_an_unknown_tool_is_recorded_and_fed_back(
    tracer, meter, spans, registry, make_client, script
):
    client = make_client(
        script.wants(("summon_kraken", "{}")), script.says("No such rope aboard.")
    )

    turn = agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    (tool_span,) = spans_named(spans, "execute_tool")
    assert tool_span.attributes[sc.ERROR_TYPE] == "UnknownTool"
    assert "no tool named" in client.messages_sent(1)[-1]["content"]
    assert turn.text == "No such rope aboard."


def test_wrong_arguments_are_recorded_and_fed_back(
    tracer, meter, spans, registry, make_client, script
):
    client = make_client(
        script.wants(("get_weather", '{"harbour": "Tortuga"}')),
        script.says("Which port?"),
    )

    turn = agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    (tool_span,) = spans_named(spans, "execute_tool")
    assert tool_span.attributes[sc.ERROR_TYPE] == "TypeError"
    assert turn.text == "Which port?"


def test_a_runaway_loop_is_capped_and_visible(
    tracer, meter, spans, registry, make_client, script
):
    # A model that never stops asking for tools.
    client = make_client(script.wants(("get_weather", '{"port": "Tortuga"}')))

    turn = agent(client, registry, tracer, meter, max_rounds=3).run(
        QUESTION, model="gpt-4o-mini"
    )

    assert turn.rounds == 3
    assert turn.truncated is True
    (parent,) = spans_named(spans, "invoke_agent")
    assert parent.attributes[sc.AGENT_TRUNCATED] is True


def test_no_tool_call_means_one_round(tracer, meter, spans, registry, make_client, script):
    client = make_client(script.says("Ahoy, no tools needed."))

    turn = agent(client, registry, tracer, meter).run(QUESTION, model="gpt-4o-mini")

    assert (turn.rounds, turn.tool_calls) == (1, [])
    assert spans_named(spans, "execute_tool") == []
