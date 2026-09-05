"""A tiny tool registry: schemas out, calls in.

Deliberately boring and deterministic -- the point of the example is the
telemetry around tool calls, not the tools themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class Tool:
    """A function the model may ask for, plus the schema describing it."""

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        """The wire format the chat completions API expects in ``tools=[…]``."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Name -> tool, with the schema list the API wants."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


# --- the demo tools --------------------------------------------------------

_WEATHER = {
    "tortuga": "squalls by afternoon, wind out of the east",
    "port royal": "clear and hot, barely a breath of wind",
    "nassau": "fog until midday, then fair",
}

_STORES = {"rum": 41, "biscuit": 12, "salt pork": 6, "limes": 90}


def get_weather(port: str) -> str:
    """Canned forecast, so the example is deterministic."""
    return _WEATHER.get(port.strip().lower(), f"no report on file for {port}")


def check_stores(item: str) -> str:
    key = item.strip().lower()
    if key not in _STORES:
        # A miss the model can act on, rather than an exception.
        return f"{item} is not in the manifest"
    return f"{_STORES[key]} units of {key} remain"


def default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="get_weather",
                description="Today's weather at a port.",
                parameters={
                    "type": "object",
                    "properties": {
                        "port": {"type": "string", "description": "Port name."}
                    },
                    "required": ["port"],
                },
                run=get_weather,
            ),
            Tool(
                name="check_stores",
                description="How much of a provision remains in the hold.",
                parameters={
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "description": "Provision name."}
                    },
                    "required": ["item"],
                },
                run=check_stores,
            ),
        ]
    )
