"""Observability package for Agentic RAG Backend."""

from src.observability.metrics import (
    increment_active_queries,
    decrement_active_queries,
    observe_request_latency,
    observe_retrieval_latency,
    observe_llm_tokens,
    increment_query_routing,
    increment_retrieval_fallback,
    increment_self_reflection_action,
    observe_chunk_relevance_score,
    increment_error,
    get_metrics_content,
    MetricsErrorType,
)
from src.observability.tracing import (
    get_trace_context,
    setup_otel_logging,
    setup_tracing,
    traced_span,
)

__all__ = [
    "increment_active_queries",
    "decrement_active_queries",
    "get_metrics_content",
    "observe_request_latency",
    "observe_retrieval_latency",
    "observe_llm_tokens",
    "increment_query_routing",
    "increment_retrieval_fallback",
    "increment_self_reflection_action",
    "observe_chunk_relevance_score",
    "increment_error",
    "MetricsErrorType",
    "get_trace_context",
    "setup_otel_logging",
    "setup_tracing",
    "traced_span",
]
