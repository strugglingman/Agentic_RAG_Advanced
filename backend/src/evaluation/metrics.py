"""
Custom metrics and metric configuration for RAG evaluation.

This module manages which Ragas metrics to use for evaluation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


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
        # Note: You have duplicates in your list (ANSWER_RELEVANCY, CONTEXT_PRECISION appear twice)
        # Consider cleaning this up to: [FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION, CONTEXT_RECALL]
        self._enabled_metrics: list[MetricType] = [
            MetricType.FAITHFULNESS,
            MetricType.ANSWER_RELEVANCY,
            MetricType.CONTEXT_PRECISION,
            MetricType.CONTEXT_RECALL,
        ]
        self._custom_metrics: dict[str, Callable] = {}

    def enable_metric(self, metric: MetricType):
        """Enable a specific metric."""
        if metric not in self._enabled_metrics:
            self._enabled_metrics.append(metric)

    def disable_metric(self, metric: MetricType):
        """Disable a specific metric."""
        if metric in self._enabled_metrics:
            self._enabled_metrics.remove(metric)

    def add_custom_metric(self, name: str, fn: Callable):
        """Add a custom evaluation metric."""
        if name not in self._custom_metrics:
            self._custom_metrics[name] = fn

    def get_ragas_metrics(self) -> list:
        """Get list of Ragas metric objects for evaluation."""
        enabled_ragas_metrics = []
        for metric in self._enabled_metrics:
            if metric == MetricType.FAITHFULNESS:
                enabled_ragas_metrics.append(faithfulness)
            elif metric == MetricType.ANSWER_RELEVANCY:
                enabled_ragas_metrics.append(answer_relevancy)
            elif metric == MetricType.CONTEXT_PRECISION:
                enabled_ragas_metrics.append(context_precision)
            elif metric == MetricType.CONTEXT_RECALL:
                enabled_ragas_metrics.append(context_recall)

        return enabled_ragas_metrics

    def aggregate_results(self, results: list[dict]) -> dict:
        """
        Aggregate metric results across samples.

        Args:
            results: List of dicts from Ragas evaluation (includes all columns)

        Returns:
            Dict with averaged scores for metric columns only
        """
        if len(results) == 0:
            return {}

        # aggreegate by averaging metric columns
        aggregated = {}
        metric_names = [m.value for m in self._enabled_metrics]
        for m in metric_names:
            values = [
                r[m] for r in results if m in r and isinstance(r[m], (int, float))
            ]
            if values:
                aggregated[m] = sum(values) / len(values)

        return aggregated
