"""ai-otel-106: structured outputs, with the failure modes made visible."""

from .schema import response_format, strict_schema, tighten
from .structured import (
    FAILURES,
    INVALID_JSON,
    METRIC_OUTPUTS,
    OUTPUT_OUTCOME,
    OUTPUT_SCHEMA,
    OUTPUT_TYPE,
    PARSED,
    REFUSED,
    SCHEMA_INVALID,
    TRUNCATED,
    StructuredChat,
    StructuredResult,
)

__all__ = [
    "FAILURES",
    "INVALID_JSON",
    "METRIC_OUTPUTS",
    "OUTPUT_OUTCOME",
    "OUTPUT_SCHEMA",
    "OUTPUT_TYPE",
    "PARSED",
    "REFUSED",
    "SCHEMA_INVALID",
    "StructuredChat",
    "StructuredResult",
    "TRUNCATED",
    "response_format",
    "strict_schema",
    "tighten",
]
