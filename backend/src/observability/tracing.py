"""OpenTelemetry tracing/bootstrap helpers for the FastAPI backend."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from typing import Any

from src.config.settings import Config

logger = logging.getLogger(__name__)

_TRACING_INITIALIZED = False
_LOGGING_INITIALIZED = False


def _build_otlp_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    raw_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "").strip()
    if not raw_headers:
        return headers

    for item in raw_headers.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            headers[key] = value
    return headers


def _build_resource(modules: dict[str, Any]):
    hostname = os.getenv("HOSTNAME") or os.getenv("COMPUTERNAME")

    attributes = {
        "service.name": Config.OTEL_SERVICE_NAME,
        "service.namespace": Config.OTEL_SERVICE_NAMESPACE,
        "service.version": Config.OTEL_SERVICE_VERSION,
        "deployment.environment": Config.OTEL_DEPLOYMENT_ENVIRONMENT,
    }
    if hostname:
        attributes["service.instance.id"] = hostname

    return modules["Resource"].create(attributes)


def _build_span_exporter(modules: dict[str, Any]):
    protocol = Config.OTEL_EXPORTER_OTLP_PROTOCOL.lower()
    endpoint = Config.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip("/")
    headers = _build_otlp_headers()

    if protocol == "grpc":
        return modules["GrpcOTLPSpanExporter"](
            endpoint=endpoint.replace("http://", "").replace("https://", ""),
            insecure=Config.OTEL_EXPORTER_OTLP_INSECURE,
            headers=headers or None,
        )

    if endpoint.endswith("/v1/traces"):
        traces_endpoint = endpoint
    else:
        traces_endpoint = f"{endpoint}/v1/traces"

    return modules["HttpOTLPSpanExporter"](
        endpoint=traces_endpoint,
        headers=headers or None,
    )


def _build_log_exporter(modules: dict[str, Any]):
    protocol = Config.OTEL_EXPORTER_OTLP_PROTOCOL.lower()
    endpoint = Config.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip("/")
    headers = _build_otlp_headers()

    if protocol == "grpc":
        return modules["GrpcOTLPLogExporter"](
            endpoint=endpoint.replace("http://", "").replace("https://", ""),
            insecure=Config.OTEL_EXPORTER_OTLP_INSECURE,
            headers=headers or None,
        )

    if endpoint.endswith("/v1/logs"):
        logs_endpoint = endpoint
    else:
        logs_endpoint = f"{endpoint}/v1/logs"

    return modules["HttpOTLPLogExporter"](
        endpoint=logs_endpoint,
        headers=headers or None,
    )


def _load_otel_modules() -> dict[str, Any]:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
        OTLPLogExporter as GrpcOTLPLogExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GrpcOTLPSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.http._log_exporter import (
        OTLPLogExporter as HttpOTLPLogExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpOTLPSpanExporter,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    from opentelemetry.trace import Status, StatusCode

    return {
        "trace": trace,
        "GrpcOTLPLogExporter": GrpcOTLPLogExporter,
        "GrpcOTLPSpanExporter": GrpcOTLPSpanExporter,
        "HttpOTLPLogExporter": HttpOTLPLogExporter,
        "HttpOTLPSpanExporter": HttpOTLPSpanExporter,
        "FastAPIInstrumentor": FastAPIInstrumentor,
        "LoggerProvider": LoggerProvider,
        "LoggingHandler": LoggingHandler,
        "BatchLogRecordProcessor": BatchLogRecordProcessor,
        "Resource": Resource,
        "TracerProvider": TracerProvider,
        "BatchSpanProcessor": BatchSpanProcessor,
        "ParentBased": ParentBased,
        "TraceIdRatioBased": TraceIdRatioBased,
        "Status": Status,
        "StatusCode": StatusCode,
    }


def setup_otel_logging() -> bool:
    """Export application logs to OTLP so Alloy can forward them to Loki."""

    global _LOGGING_INITIALIZED

    if _LOGGING_INITIALIZED or not Config.OTEL_ENABLED:
        return False

    try:
        modules = _load_otel_modules()
    except ImportError as exc:
        logger.warning(
            "OpenTelemetry logging is enabled but dependencies are missing: %s", exc
        )
        return False

    logger_provider = modules["LoggerProvider"](resource=_build_resource(modules))
    logger_provider.add_log_record_processor(
        modules["BatchLogRecordProcessor"](_build_log_exporter(modules))
    )
    logging_handler = modules["LoggingHandler"](
        level=logging.INFO,
        logger_provider=logger_provider,
    )
    logging_handler.set_name("otel-log-exporter")
    logging.getLogger().addHandler(logging_handler)

    _LOGGING_INITIALIZED = True
    logger.info(
        "OpenTelemetry log export enabled for service=%s endpoint=%s protocol=%s",
        Config.OTEL_SERVICE_NAME,
        Config.OTEL_EXPORTER_OTLP_ENDPOINT,
        Config.OTEL_EXPORTER_OTLP_PROTOCOL,
    )
    return True


def setup_tracing(app) -> bool:
    """Configure OpenTelemetry tracing for the FastAPI app."""

    global _TRACING_INITIALIZED

    if _TRACING_INITIALIZED or not Config.OTEL_ENABLED:
        return False

    try:
        modules = _load_otel_modules()
    except ImportError as exc:
        logger.warning(
            "OpenTelemetry is enabled but dependencies are missing: %s", exc
        )
        return False

    tracer_provider = modules["TracerProvider"](
        resource=_build_resource(modules),
        sampler=modules["ParentBased"](
            modules["TraceIdRatioBased"](Config.OTEL_TRACES_SAMPLER_ARG)
        ),
    )
    tracer_provider.add_span_processor(
        modules["BatchSpanProcessor"](_build_span_exporter(modules))
    )
    modules["trace"].set_tracer_provider(tracer_provider)
    modules["FastAPIInstrumentor"].instrument_app(
        app,
        tracer_provider=tracer_provider,
        excluded_urls="/metrics,/health",
    )
    setup_otel_logging()

    _TRACING_INITIALIZED = True
    logger.info(
        "OpenTelemetry tracing enabled for service=%s endpoint=%s protocol=%s",
        Config.OTEL_SERVICE_NAME,
        Config.OTEL_EXPORTER_OTLP_ENDPOINT,
        Config.OTEL_EXPORTER_OTLP_PROTOCOL,
    )
    return True


@contextmanager
def traced_span(name: str, attributes: dict[str, Any] | None = None):
    """Create a best-effort span around a logical RAG operation."""

    try:
        modules = _load_otel_modules()
        tracer = modules["trace"].get_tracer(Config.OTEL_SERVICE_NAME)
    except ImportError:
        yield None
        return

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(
                modules["Status"](modules["StatusCode"].ERROR, str(exc))
            )
            raise


def get_trace_context() -> tuple[str, str]:
    """Return the current trace and span IDs for log correlation."""

    try:
        from opentelemetry import trace
    except ImportError:
        return ("-", "-")

    context = trace.get_current_span().get_span_context()
    if not context or not context.is_valid:
        return ("-", "-")

    return (f"{context.trace_id:032x}", f"{context.span_id:016x}")


__all__ = ["get_trace_context", "setup_otel_logging", "setup_tracing", "traced_span"]
