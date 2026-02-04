"""Observability package for Agentic RAG Backend."""

from src.observability.metrics import (
    increment_active_queries,
    decrement_active_queries,
    observe_request_latency,
    observe_retrieval_latency,
    observe_llm_tokens,
    increment_query_routing,
    increment_self_reflection_action,
    observe_chunk_relevance_score,
    increment_error,
    get_metrics_content,
    MetricsErrorType,
)

__all__ = [
    "increment_active_queries",
    "decrement_active_queries",
    "get_metrics_content",
    "observe_request_latency",
    "observe_retrieval_latency",
    "observe_llm_tokens",
    "increment_query_routing",
    "increment_self_reflection_action",
    "observe_chunk_relevance_score",
    "increment_error",
    "MetricsErrorType",
]
