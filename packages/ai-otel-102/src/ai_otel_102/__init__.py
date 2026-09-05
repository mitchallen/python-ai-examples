"""ai-otel-102: self-contained OpenAI telemetry in a single module."""

from .observe import (
    DEFAULT_MODEL,
    PIRATE_SYSTEM_PROMPT,
    ChatTelemetry,
    Observation,
    Telemetry,
    build_conversation,
    configure_telemetry,
    provider_for,
    provider_from_base_url,
)

__all__ = [
    "DEFAULT_MODEL",
    "PIRATE_SYSTEM_PROMPT",
    "ChatTelemetry",
    "Observation",
    "Telemetry",
    "build_conversation",
    "configure_telemetry",
    "provider_for",
    "provider_from_base_url",
]
