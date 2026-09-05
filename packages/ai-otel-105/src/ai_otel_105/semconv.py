"""Tool-calling names, extending :mod:`ai_otel_101.semconv`.

Only what tool calling adds lives here; everything else (operation, provider,
model, usage, error) comes from the base module unchanged, which is why a
tool-calling turn shows up in the same dashboards as a plain completion.
"""

from ai_otel_101.semconv import *  # noqa: F401,F403  (re-export the base names)

# --- operations ------------------------------------------------------------
OPERATION_INVOKE_AGENT = "invoke_agent"
OPERATION_EXECUTE_TOOL = "execute_tool"

# --- the agent -------------------------------------------------------------
AGENT_NAME = "gen_ai.agent.name"

# --- the tool --------------------------------------------------------------
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_DESCRIPTION = "gen_ai.tool.description"
TOOL_TYPE = "gen_ai.tool.type"
TOOL_TYPE_FUNCTION = "function"

# --- local, not semconv ----------------------------------------------------
# Counts of what happened in the turn. Token totals are deliberately absent:
# the child spans carry those and the backend sums them, so repeating them here
# would double-count.
AGENT_ROUNDS = "app.agent.rounds"
AGENT_TOOL_CALLS = "app.agent.tool_calls"
AGENT_TRUNCATED = "app.agent.truncated"
