"""
Prometheus Metrics for Agentic RAG Backend.

DEPENDENCY:
    pip install prometheus-client

DATA FLOW:
    This file                  presentation/api/metrics.py         Observability Stack
    ─────────                  ────────────────────────────         ───────────────────
    Define metrics ──────────► /metrics endpoint ──────────────►   Alloy ──► Mimir ──► Grafana

METRIC TYPES:
    - Gauge: Value goes up/down (current count, e.g., active queries)
    - Counter: Value only goes up (total count, e.g., total requests)
    - Histogram: Distribution (for percentiles like P95, e.g., latency)
"""

from prometheus_client import (
    Gauge,
    Histogram,
    Counter,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# =============================================================================
# METRICS DEFINITIONS
# =============================================================================
ACTIVE_QUERIES = Gauge(
    "rag_active_queries", "Number of active queries currently being processed"
)

REQUEST_LATENCY = Histogram(
    "http_server_request_duration_seconds",  # must be the same name as grafana panel metric
    "Http request latency in seconds",
    ["method", "route", "http_status_code"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120],  # in seconds
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Latency of retrieval operations in seconds",
    ["search_type"],
    buckets=[1, 2, 5, 10, 20, 30, 60, 90, 120],
)

LLM_TOKENS_TOTAL = Histogram(
    "rag_llm_tokens_total",
    "Total number of LLM tokens used",
    ["type", "model"],
    buckets=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
)

QUERY_ROUTING_TOTAL = Counter(
    "rag_query_routing_total",
    "Total number of queries by routing type",
    ["route"],
)

SELF_REFLECTION_TOTAL = Counter(
    "rag_reflection_action_total",
    "Total number of self-reflection actions taken",
    ["action"],
)

CHUNK_RELEVANCE_SCORE = Histogram(
    "rag_chunk_relevance_score",
    "Relevance scores of retrieved chunks",
    ["type"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

ERRORS_TOTAL = Counter(
    "rag_errors_total",
    "Total number of errors by type",
    ["error_type"],
)


# =============================================================================
# LABEL CONSTANTS (avoid hardcoding strings throughout codebase)
# =============================================================================
class MetricsErrorType:
    """Error type labels for rag_errors_total metric."""

    RETRIEVAL_FAILED = "retrieval_failed"
    RERANK_FAILED = "rerank_failed"
    LLM_FAILED = "llm_failed"
    TIMEOUT = "timeout"
    WEB_SEARCH_FAILED = "web_search_failed"
    INGESTION_FAILED = "ingestion_failed"


# =============================================================================
# RECORDING FUNCTIONS
# =============================================================================
def increment_active_queries():
    """Call when query STARTS. Integration point: query_supervisor.process_query()"""
    ACTIVE_QUERIES.inc()


def decrement_active_queries():
    """Call when query ENDS (in finally block). Integration point: query_supervisor.process_query()"""
    ACTIVE_QUERIES.dec()


def observe_request_latency(method: str, route: str, status_code: int, duration: float):
    """Call to record request latency. Integration point: presentation/middleware/metrics_middleware.py"""
    REQUEST_LATENCY.labels(
        method=method, route=route, http_status_code=str(status_code)
    ).observe(duration)


def observe_retrieval_latency(search_type: str, duration: float):
    """Call to record retrieval latency. Integration point: services/retrieval.py"""
    RETRIEVAL_LATENCY.labels(search_type=search_type).observe(duration)


def observe_llm_tokens(type: str, model: str, token_count: int):
    """Call to record LLM token usage. Integration point: services/langgraph_nodes.py after LLM calls"""
    LLM_TOKENS_TOTAL.labels(type=type, model=model).observe(token_count)


def increment_query_routing(route: str):
    """Call to record query routing type. Integration point: services/langgraph_routing.py - semantic_route_query()"""
    QUERY_ROUTING_TOTAL.labels(route=route).inc()


def increment_self_reflection_action(action: str):
    """Call to record self-reflection actions. Integration point: services/self_reflection.py"""
    SELF_REFLECTION_TOTAL.labels(action=action).inc()


def observe_chunk_relevance_score(type: str, scores: list[float]):
    """Call to record chunk relevance scores. Integration point: services/retrieval.py after chunk scoring"""
    for score in scores:
        CHUNK_RELEVANCE_SCORE.labels(type=type).observe(score)


def increment_error(error_type: str):
    """
    Call to record an error occurrence.

    Integration points:
        - services/retrieval.py: retrieval_failed
        - services/llm_client.py: llm_failed
        - services/query_supervisor.py: timeout, general errors

    Args:
        error_type: Type of error (retrieval_failed, llm_failed, timeout, etc.)
    """
    ERRORS_TOTAL.labels(error_type=error_type).inc()


# =============================================================================
# HELPER FOR /metrics ENDPOINT
# =============================================================================
def get_metrics_content():
    """
    Generate Prometheus metrics output.

    Called by: presentation/api/metrics.py

    Returns:
        Tuple of (content_bytes, content_type_string)
    """
    return generate_latest(), CONTENT_TYPE_LATEST


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    "increment_active_queries",
    "decrement_active_queries",
    "observe_request_latency",
    "observe_retrieval_latency",
    "observe_llm_tokens",
    "increment_query_routing",
    "increment_self_reflection_action",
    "observe_chunk_relevance_score",
    "increment_error",
    "get_metrics_content",
    "MetricsErrorType",
]
