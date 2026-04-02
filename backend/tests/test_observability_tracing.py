import pytest
from fastapi import FastAPI

from src.observability import tracing


pytestmark = pytest.mark.unit


def test_setup_tracing_noops_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing.Config, "OTEL_ENABLED", False)
    monkeypatch.setattr(tracing, "_TRACING_INITIALIZED", False)

    def should_not_load():
        raise AssertionError("OpenTelemetry imports should not run when disabled")

    monkeypatch.setattr(tracing, "_load_otel_modules", should_not_load)

    assert tracing.setup_tracing(FastAPI()) is False


def test_setup_tracing_warns_when_dependencies_missing(monkeypatch, caplog):
    monkeypatch.setattr(tracing.Config, "OTEL_ENABLED", True)
    monkeypatch.setattr(tracing, "_TRACING_INITIALIZED", False)

    def missing_modules():
        raise ImportError("otlp exporter missing")

    monkeypatch.setattr(tracing, "_load_otel_modules", missing_modules)

    with caplog.at_level("WARNING"):
        assert tracing.setup_tracing(FastAPI()) is False

    assert "dependencies are missing" in caplog.text


def test_traced_span_noops_without_otel(monkeypatch):
    def missing_modules():
        raise ImportError("otel missing")

    monkeypatch.setattr(tracing, "_load_otel_modules", missing_modules)

    with tracing.traced_span("rag.test.span") as span:
        assert span is None
