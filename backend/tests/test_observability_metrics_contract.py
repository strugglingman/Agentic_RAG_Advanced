import pytest

from src.observability import metrics


pytestmark = pytest.mark.unit


def test_retrieval_fallback_metric_is_exported():
    metrics.increment_retrieval_fallback()
    content, _ = metrics.get_metrics_content()

    assert b"rag_retrieval_fallback_total" in content
