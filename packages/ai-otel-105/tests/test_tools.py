"""The registry and the demo tools."""

from __future__ import annotations

import pytest

from ai_otel_105.tools import Tool, ToolRegistry, check_stores, default_registry, get_weather


def test_schema_is_the_wire_format():
    tool = Tool(
        name="get_weather",
        description="Today's weather at a port.",
        parameters={"type": "object", "properties": {"port": {"type": "string"}}},
        run=get_weather,
    )

    assert tool.schema() == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Today's weather at a port.",
            "parameters": {"type": "object", "properties": {"port": {"type": "string"}}},
        },
    }


def test_registry_lookup_and_schemas():
    registry = default_registry()

    assert len(registry) == 2
    assert "get_weather" in registry
    assert registry.get("nope") is None
    assert {s["function"]["name"] for s in registry.schemas()} == {
        "get_weather",
        "check_stores",
    }


def test_registry_add_replaces_by_name():
    registry = ToolRegistry()
    registry.add(Tool("t", "d", {"type": "object"}, lambda: "one"))
    registry.add(Tool("t", "d", {"type": "object"}, lambda: "two"))

    assert len(registry) == 1
    assert registry.get("t").run() == "two"


@pytest.mark.parametrize(
    ("port", "expected"),
    [("Tortuga", "squalls"), ("port royal", "clear"), ("PORT ROYAL", "clear")],
)
def test_weather_lookup_is_case_insensitive(port, expected):
    assert expected in get_weather(port)


def test_unknown_port_is_a_usable_answer_not_an_error():
    # The model can act on this; an exception would just end the turn.
    assert "no report on file" in get_weather("Atlantis")


def test_stores_lookup():
    assert "41 units of rum" in check_stores("Rum")
    assert "not in the manifest" in check_stores("caviar")
