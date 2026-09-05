"""The provider name has to follow the endpoint, not a hardcoded guess."""

from __future__ import annotations

import pytest

from ai_otel_101 import semconv as sc
from ai_otel_101.instrumented import InstrumentedChat, provider_from_base_url


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "openai"),
        ("", "openai"),
        ("https://api.openai.com/v1", "openai"),
        ("http://localhost:11434/v1", "ollama"),
        ("http://127.0.0.1:11434/v1", "ollama"),
        ("http://ollama.internal:11434/v1", "ollama"),
        ("https://my-deployment.openai.azure.com/", "azure.ai.openai"),
        # Unknown but real: report where the tokens actually went rather than
        # claiming OpenAI served them.
        ("http://gateway.corp:8080/v1", "gateway.corp:8080"),
    ],
)
def test_provider_is_derived_from_the_base_url(base_url, expected):
    assert provider_from_base_url(base_url) == expected


def test_span_reports_the_client_endpoint(tracer, meter, spans, make_client):
    client = make_client()
    client.base_url = "http://localhost:11434/v1"

    InstrumentedChat(client, tracer=tracer, meter=meter).complete(
        [{"role": "user", "content": "Hello"}], model="llama3.2:3b"
    )

    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.SYSTEM] == "ollama"
    assert span.attributes[sc.PROVIDER_NAME] == "ollama"
    assert span.attributes[sc.REQUEST_MODEL] == "llama3.2:3b"


def test_token_metric_is_keyed_on_the_real_provider(
    tracer, meter, metric_reader, make_client
):
    client = make_client()
    client.base_url = "http://localhost:11434/v1"

    InstrumentedChat(client, tracer=tracer, meter=meter).complete(
        [{"role": "user", "content": "Hello"}], model="llama3.2:3b"
    )

    data = metric_reader.get_metrics_data()
    providers = {
        point.attributes[sc.PROVIDER_NAME]
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        for point in metric.data.data_points
    }
    assert providers == {"ollama"}


def test_an_explicit_provider_wins(tracer, meter, spans, client):
    InstrumentedChat(client, tracer=tracer, meter=meter, provider="vllm").complete(
        [{"role": "user", "content": "Hello"}], model="mistral"
    )

    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.PROVIDER_NAME] == "vllm"


def test_a_client_without_a_base_url_still_works(tracer, meter, spans, client):
    # The stub has no base_url attribute at all.
    InstrumentedChat(client, tracer=tracer, meter=meter).complete(
        [{"role": "user", "content": "Hello"}], model="gpt-4o-mini"
    )

    (span,) = spans.get_finished_spans()
    assert span.attributes[sc.PROVIDER_NAME] == "openai"
