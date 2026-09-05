"""Provider naming, self-contained copy."""

from __future__ import annotations

import pytest

import ai_otel_102.observe as obs
from ai_otel_102 import ChatTelemetry, provider_for, provider_from_base_url


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "openai"),
        ("https://api.openai.com/v1", "openai"),
        ("http://localhost:11434/v1", "ollama"),
        ("http://127.0.0.1:11434/v1", "ollama"),
        ("https://my-deployment.openai.azure.com/", "azure.ai.openai"),
        ("http://gateway.corp:8080/v1", "gateway.corp:8080"),
    ],
)
def test_provider_is_derived_from_the_base_url(base_url, expected):
    assert provider_from_base_url(base_url) == expected


def test_provider_for_reads_a_client(make_client):
    client = make_client()
    client.base_url = "http://localhost:11434/v1"

    assert provider_for(client) == "ollama"
    # A stub with no base_url falls back to the SDK's own default endpoint.
    assert provider_for(make_client()) == "openai"


def test_telemetry_reports_the_configured_provider(tracer, meter, spans, client):
    telemetry = ChatTelemetry(tracer=tracer, meter=meter, provider="ollama")

    with telemetry.chat("llama3.2:3b") as observed:
        observed.record(client.chat.completions.create())

    (span,) = spans.get_finished_spans()
    assert span.attributes[obs.SYSTEM] == "ollama"
    assert span.attributes[obs.PROVIDER_NAME] == "ollama"


def test_a_single_call_can_override_the_provider(chat_telemetry, spans, client):
    # The shape a fallback path needs: same telemetry object, second provider.
    with chat_telemetry.chat("gpt-4o-mini") as observed:
        observed.record(client.chat.completions.create())
    with chat_telemetry.chat("llama3.2:3b", provider="ollama") as observed:
        observed.record(client.chat.completions.create())

    primary, fallback = spans.get_finished_spans()
    assert primary.attributes[obs.PROVIDER_NAME] == "openai"
    assert fallback.attributes[obs.PROVIDER_NAME] == "ollama"


def test_ask_pirate_derives_the_provider_from_its_client(tracer, meter, spans, make_client):
    client = make_client()
    client.base_url = "http://localhost:11434/v1"

    obs.ask_pirate(
        "Hello",
        client=client,
        telemetry=ChatTelemetry(tracer=tracer, meter=meter, provider=provider_for(client)),
    )

    (span,) = spans.get_finished_spans()
    assert span.attributes[obs.PROVIDER_NAME] == "ollama"
