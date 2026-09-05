"""Turning a Pydantic model into the strict JSON schema the API wants.

Strict mode has two requirements Pydantic does not emit on its own: every
object must say ``additionalProperties: false``, and every property must be
listed in ``required``. Optionality is expressed by allowing ``null`` in the
type, not by leaving a field out.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def tighten(node: Any) -> Any:
    """Recursively apply the strict-mode rules to a JSON schema."""
    if isinstance(node, list):
        return [tighten(item) for item in node]
    if not isinstance(node, dict):
        return node

    result = {key: tighten(value) for key, value in node.items()}
    if result.get("type") == "object":
        properties = result.get("properties", {})
        # Strict mode: no extra keys, and every declared property required.
        result["additionalProperties"] = False
        result["required"] = list(properties)
    return result


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The strict JSON schema for a Pydantic model."""
    return tighten(model.model_json_schema())


def response_format(model: type[BaseModel], *, strict: bool = True) -> dict[str, Any]:
    """The ``response_format`` argument for a chat completion.

    ``strict=False`` falls back to plain JSON mode, which asks for *some* JSON
    and promises nothing about its shape -- the case the tests and the
    ``--loose`` demo exist to show.
    """
    if not strict:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": strict_schema(model),
            "strict": True,
        },
    }
