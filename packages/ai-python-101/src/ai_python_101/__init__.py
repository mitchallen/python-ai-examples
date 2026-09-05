"""ai-python-101: the smallest useful OpenAI chat example."""

from .chat import (
    DEFAULT_MODEL,
    PIRATE_SYSTEM_PROMPT,
    Conversation,
    ask_pirate,
    build_conversation,
    create_client,
)

__all__ = [
    "DEFAULT_MODEL",
    "PIRATE_SYSTEM_PROMPT",
    "Conversation",
    "ask_pirate",
    "build_conversation",
    "create_client",
]
