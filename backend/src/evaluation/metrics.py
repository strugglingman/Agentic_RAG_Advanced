"""
Custom metrics and metric configuration for RAG evaluation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class MetricType(Enum):
    """Available Ragas metric types."""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    # Custom metrics
    ANSWER_CORRECTNESS = "answer_correctness"
    CONTEXT_ENTITY_RECALL = "context_entity_recall"


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""
    name: str
    score: float
    details: Optional[dict] = None


class RAGMetrics:
    """Configure and manage RAG evaluation metrics."""

    def __init__(self):
        self._enabled_metrics: list[MetricType] = []
        self._custom_metrics: dict[str, Callable] = {}

    def enable_default_metrics(self):
        """Enable standard Ragas metrics."""
        # TODO: Implement
        raise NotImplementedError

    def enable_metric(self, metric: MetricType):
        """Enable a specific metric."""
        # TODO: Implement
        raise NotImplementedError

    def disable_metric(self, metric: MetricType):
        """Disable a specific metric."""
        # TODO: Implement
        raise NotImplementedError

    def add_custom_metric(self, name: str, fn: Callable):
        """Add a custom evaluation metric."""
        # TODO: Implement
        raise NotImplementedError

    def get_ragas_metrics(self):
        """Get list of Ragas metric objects for evaluation."""
        # TODO: Implement - return actual ragas metric objects
        raise NotImplementedError

    def aggregate_results(self, results: list[dict]) -> dict:
        """Aggregate metric results across samples."""
        # TODO: Implement aggregation
        raise NotImplementedError
