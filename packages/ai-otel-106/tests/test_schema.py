"""Strict-mode schema generation."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ai_otel_106 import response_format, strict_schema, tighten


class Cannon(BaseModel):
    calibre: int


class Ship(BaseModel):
    name: str
    crew: int
    armament: Cannon


def test_objects_forbid_extra_properties():
    schema = strict_schema(Ship)

    assert schema["additionalProperties"] is False


def test_every_property_is_required():
    # Strict mode has no optional keys; absence is expressed in the type.
    schema = strict_schema(Ship)

    assert set(schema["required"]) == {"name", "crew", "armament"}


def test_nested_definitions_are_tightened_too():
    schema = strict_schema(Ship)

    nested = schema["$defs"]["Cannon"]
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["calibre"]


def test_tighten_leaves_non_objects_alone():
    node = {"type": "array", "items": {"type": "string"}}

    assert tighten(node) == node


def test_tighten_handles_lists_of_schemas():
    node = {"anyOf": [{"type": "object", "properties": {"a": {"type": "string"}}}]}

    result = tighten(node)

    assert result["anyOf"][0]["additionalProperties"] is False
    assert result["anyOf"][0]["required"] == ["a"]


def test_response_format_strict_and_loose():
    strict = response_format(Ship)
    loose = response_format(Ship, strict=False)

    assert strict["type"] == "json_schema"
    assert strict["json_schema"]["name"] == "Ship"
    assert strict["json_schema"]["strict"] is True
    assert loose == {"type": "json_object"}


def test_optional_fields_still_appear_in_required():
    class Loose(BaseModel):
        name: str
        nickname: Optional[str] = None

    schema = strict_schema(Loose)

    # Optionality lives in the type union, not in a missing required entry.
    assert set(schema["required"]) == {"name", "nickname"}
