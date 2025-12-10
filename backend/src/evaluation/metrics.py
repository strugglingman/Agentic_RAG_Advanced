"""
Custom metrics and metric configuration for RAG evaluation.

This module manages which Ragas metrics to use for evaluation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from ragas.metrics import (faithfulness, answer_relevancy, context_precision, context_recall)


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
            MetricType.CONTEXT_RECALL
        ]
        self._custom_metrics: dict[str, Callable] = {}

    def enable_metric(self, metric: MetricType):
        """
        Enable a specific metric.

        TODO:
        1. Check if metric is already in self._enabled_metrics
        2. If not, append it to the list
        """
        if metric not in self._enabled_metrics:
            self._enabled_metrics.append(metric)

    def disable_metric(self, metric: MetricType):
        """
        Disable a specific metric.

        TODO:
        1. Check if metric is in self._enabled_metrics
        2. If yes, remove it from the list
        """
        if metric in self._enabled_metrics:
            self._enabled_metrics.remove(metric)

    def add_custom_metric(self, name: str, fn: Callable):
        """
        Add a custom evaluation metric.

        TODO:
        1. Add the callable to self._custom_metrics dict with name as key
        2. This is for metrics not included in Ragas by default
        """
        if name not in self._custom_metrics:
            self._custom_metrics[name] = fn

    def get_ragas_metrics(self) -> list:
        """
        Get list of Ragas metric objects for evaluation.

        TODO: This is the MOST IMPORTANT method!
        1. Import the actual ragas metric objects at the top of file
        2. Create a mapping dict: MetricType -> ragas metric object
           Example: {MetricType.FAITHFULNESS: faithfulness, ...}
        3. Loop through self._enabled_metrics
        4. For each enabled metric, get the corresponding ragas object from mapping
        5. Return a list of ragas metric objects (not MetricType enums!)

        Example return: [faithfulness, answer_relevancy, context_precision]

        NOTE: ragas.evaluate() expects actual metric objects, not strings or enums!
        """
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

        TODO:
        1. Input: List of dicts, each dict contains metric scores for one sample
           Example: [{'faithfulness': 0.9, 'answer_relevancy': 0.8},
                     {'faithfulness': 0.85, 'answer_relevancy': 0.75}]
        2. Calculate average for each metric across all samples
        3. Return dict with aggregated scores
           Example: {'faithfulness': 0.875, 'answer_relevancy': 0.775}

        HINT: Use numpy or simple list comprehension to calculate means
        """
        if len(results) == 0:
            return {}

        keys = results[0].keys()
        aggregated = {}
        for key in keys:
            values = [result[key] for result in results if key in result]
            aggregated[key]= sum(values) / len(values) if values else 0.0

        return aggregated
