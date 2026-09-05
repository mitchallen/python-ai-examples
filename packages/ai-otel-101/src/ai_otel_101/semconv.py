"""OpenTelemetry GenAI semantic-convention names, spelled out.

These are plain strings on purpose. The names live in the semantic-conventions
spec rather than in the API package, they are still evolving, and seeing them
written out is most of the point of this example -- an OTel backend only knows
how to chart "tokens per model" because everyone agrees on
``gen_ai.usage.input_tokens``.
"""

# --- what happened ---------------------------------------------------------
OPERATION_NAME = "gen_ai.operation.name"  # "chat", "embeddings", ...
OPERATION_CHAT = "chat"

# The provider attribute was renamed from `gen_ai.system` to
# `gen_ai.provider.name`. Backends in the wild read one or the other, so this
# example emits both and lets the collector drop whichever it does not want.
SYSTEM = "gen_ai.system"
PROVIDER_NAME = "gen_ai.provider.name"
PROVIDER_OPENAI = "openai"

# --- the request -----------------------------------------------------------
REQUEST_MODEL = "gen_ai.request.model"
REQUEST_TEMPERATURE = "gen_ai.request.temperature"
REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"

# --- the response ----------------------------------------------------------
RESPONSE_ID = "gen_ai.response.id"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

# --- the part everyone actually wants: cost ---------------------------------
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# --- failures --------------------------------------------------------------
ERROR_TYPE = "error.type"

# --- metric instruments ----------------------------------------------------
METRIC_TOKEN_USAGE = "gen_ai.client.token.usage"
METRIC_OPERATION_DURATION = "gen_ai.client.operation.duration"
TOKEN_TYPE = "gen_ai.token.type"
TOKEN_TYPE_INPUT = "input"
TOKEN_TYPE_OUTPUT = "output"

# --- opt-in message content ------------------------------------------------
# Prompts and replies are user data, so they are off by default and ride as
# span events rather than attributes.
EVENT_USER_MESSAGE = "gen_ai.user.message"
EVENT_SYSTEM_MESSAGE = "gen_ai.system.message"
EVENT_CHOICE = "gen_ai.choice"
CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
