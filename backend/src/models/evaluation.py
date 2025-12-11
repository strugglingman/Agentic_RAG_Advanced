"""
Data models for retrieval quality evaluation (Self-Reflection System).

This module defines:
1. EvaluationCriteria - Input: what we're evaluating (query + contexts)
2. EvaluationResult - Output: quality assessment + recommendation
3. ReflectionConfig - Runtime settings for evaluation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class QualityLevel(str, Enum):
    """Quality levels for retrieval evaluation."""

    EXCELLENT = "excellent"
    GOOD = "good"
    PARTIAL = "partial"
    POOR = "poor"


class RecommendationAction(str, Enum):
    """Action recommended based on evaluation result."""

    ANSWER = "answer"
    REFINE = "refine"
    EXTERNAL = "external"
    CLARIFY = "clarify"


class ReflectionMode(str, Enum):
    """Evaluation speed vs accuracy tradeoff modes."""

    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"


@dataclass
class EvaluationCriteria:
    """
    Input data for evaluating retrieval quality.

    Attributes:
        query: User's question
        contexts: Retrieved chunks from search
        search_metadata: Optional metadata (scores, hybrid info, etc.)
        mode: Evaluation mode (fast/balanced/thorough)
    """

    query: str
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    search_metadata: Optional[Dict[str, Any]] = None
    mode: Optional[ReflectionMode] = ReflectionMode.BALANCED

    def __post_init__(self):
        if not self.query or not self.query.strip():
            raise ValueError("Query must be a non-empty string.")


@dataclass
class EvaluationResult:
    """
    Output data from retrieval quality evaluation.

    Attributes:
        quality: Overall quality level (excellent/good/partial/poor)
        confidence: Overall confidence score (0.0-1.0)
        coverage: How well query is covered (0.0-1.0)
        recommendation: What action to take next
        reasoning: Explanation for the recommendation
        relevance_scores: Per-context relevance scores
        issues: Detected problems with retrieval
        missing_aspects: Query aspects not covered
        metrics: Raw scores for debugging
        timestamp: When evaluation occurred
        mode_used: Which evaluation mode was used
    """

    quality: QualityLevel
    confidence: float
    coverage: float
    recommendation: RecommendationAction
    reasoning: str
    relevance_scores: list[float] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    mode_used: ReflectionMode = ReflectionMode.BALANCED

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0.")
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError("Coverage must be between 0.0 and 1.0.")
        for score in self.relevance_scores:
            if not (0.0 <= score <= 1.0):
                raise ValueError("All relevance scores must be between 0.0 and 1.0.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert EvaluationResult to a JSON-serializable dictionary."""
        return {
            "quality": self.quality.value,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "relevance_scores": self.relevance_scores,
            "issues": self.issues,
            "missing_aspects": self.missing_aspects,
            "recommendation": self.recommendation.value,
            "reasoning": self.reasoning,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "mode_used": self.mode_used.value,
        }

    @property
    def should_answer(self) -> bool:
        return self.recommendation == RecommendationAction.ANSWER

    @property
    def should_refine(self) -> bool:
        return self.recommendation == RecommendationAction.REFINE

    @property
    def should_search_external(self) -> bool:
        return self.recommendation == RecommendationAction.EXTERNAL

    @property
    def should_clarify(self) -> bool:
        return self.recommendation == RecommendationAction.CLARIFY


@dataclass
class ReflectionConfig:
    """
    Runtime configuration for retrieval evaluation.

    Attributes:
        mode: Evaluation mode (fast/balanced/thorough)
        auto_refine: Auto-trigger query refinement
        auto_external: Auto-trigger external search
        min_contexts: Minimum contexts needed to proceed
        max_refinement_attempts: Max refinement loops (prevents infinite loops)
        thresholds: Quality level thresholds
    """

    mode: ReflectionMode
    auto_refine: bool
    auto_external: bool
    min_contexts: int
    max_refinement_attempts: int
    thresholds: Dict[str, float]
    avg_score: float
    keyword_overlap: float

    @classmethod
    def from_settings(cls, config_class) -> "ReflectionConfig":
        """Load ReflectionConfig from settings Config class."""
        return cls(
            mode=ReflectionMode(config_class.REFLECTION_MODE),
            thresholds={
                "excellent": config_class.REFLECTION_THRESHOLD_EXCELLENT,
                "good": config_class.REFLECTION_THRESHOLD_GOOD,
                "partial": config_class.REFLECTION_THRESHOLD_PARTIAL,
            },
            auto_refine=config_class.REFLECTION_AUTO_REFINE,
            auto_external=config_class.REFLECTION_AUTO_EXTERNAL,
            min_contexts=config_class.REFLECTION_MIN_CONTEXTS,
            max_refinement_attempts=config_class.REFLECTION_MAX_REFINEMENT_ATTEMPTS,
            avg_score=config_class.REFLECTION_AVG_SCORE,
            keyword_overlap=config_class.REFLECTION_KEYWORD_OVERLAP,
        )

    def get_quality_level(self, confidence: float) -> QualityLevel:
        """Map confidence score to quality level using thresholds."""
        if confidence >= self.thresholds["excellent"]:
            return QualityLevel.EXCELLENT
        elif confidence >= self.thresholds["good"]:
            return QualityLevel.GOOD
        elif confidence >= self.thresholds["partial"]:
            return QualityLevel.PARTIAL
        else:
            return QualityLevel.POOR


if __name__ == "__main__":
    """Test the evaluation models."""
    import json

    print("=" * 70)
    print("TESTING EVALUATION MODELS")
    print("=" * 70)

    # Test 1: Enums
    print("\n[Test 1] Testing Enums...")
    print(f"  QualityLevel.GOOD = {QualityLevel.GOOD.value}")
    print(f"  RecommendationAction.ANSWER = {RecommendationAction.ANSWER.value}")
    print(f"  ReflectionMode.BALANCED = {ReflectionMode.BALANCED.value}")
    print("  [OK] Enums working")

    # Test 2: EvaluationCriteria
    print("\n[Test 2] Testing EvaluationCriteria...")
    try:
        criteria = EvaluationCriteria(
            query="What was Q1 2024 revenue?",
            contexts=[
                {"chunk": "Q1 revenue was $50M...", "score": 0.89},
                {"chunk": "Annual trends...", "score": 0.45},
            ],
            search_metadata={"hybrid": True},
        )
        print(f"  Query: {criteria.query}")
        print(f"  Contexts count: {len(criteria.contexts)}")
        print(f"  Mode: {criteria.mode.value}")
        print("  [OK] EvaluationCriteria working")
    except Exception as e:
        print(f"  [FAIL] EvaluationCriteria failed: {e}")

    # Test 3: Empty query validation
    print("\n[Test 3] Testing validation (empty query)...")
    try:
        bad_criteria = EvaluationCriteria(query="")
        print("  [FAIL] Validation failed - should have raised ValueError")
    except ValueError as e:
        print(f"  [OK] Validation working: {e}")

    # Test 4: EvaluationResult
    print("\n[Test 4] Testing EvaluationResult...")
    try:
        result = EvaluationResult(
            quality=QualityLevel.GOOD,
            confidence=0.82,
            coverage=0.90,
            relevance_scores=[0.95, 0.60],
            issues=["Second context is tangentially related"],
            missing_aspects=[],
            recommendation=RecommendationAction.ANSWER,
            reasoning="Primary context directly answers question",
            metrics={"keyword_overlap": 0.85},
        )
        print(f"  Quality: {result.quality.value}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Recommendation: {result.recommendation.value}")
        print(f"  Should answer: {result.should_answer}")
        print(f"  Should refine: {result.should_refine}")
        print("  [OK] EvaluationResult working")
    except Exception as e:
        print(f"  [FAIL] EvaluationResult failed: {e}")

    # Test 5: Confidence validation
    print("\n[Test 5] Testing confidence validation (>1.0)...")
    try:
        bad_result = EvaluationResult(
            quality=QualityLevel.GOOD,
            confidence=1.5,  # Invalid!
            coverage=0.90,
            recommendation=RecommendationAction.ANSWER,
            reasoning="Test",
        )
        print("  [FAIL] Validation failed - should have raised ValueError")
    except ValueError as e:
        print(f"  [OK] Validation working: {e}")

    # Test 6: to_dict() method
    print("\n[Test 6] Testing to_dict()...")
    try:
        result_dict = result.to_dict()
        print("  [OK] to_dict() working")
        print(f"  JSON preview: {json.dumps(result_dict, indent=2)[:200]}...")
    except Exception as e:
        print(f"  [FAIL] to_dict() failed: {e}")

    # Test 7: ReflectionConfig
    print("\n[Test 7] Testing ReflectionConfig...")
    try:
        config = ReflectionConfig(
            mode=ReflectionMode.BALANCED,
            auto_refine=True,
            auto_external=False,
            min_contexts=1,
            max_refinement_attempts=3,
            thresholds={
                "excellent": 0.85,
                "good": 0.70,
                "partial": 0.50,
            },
            avg_score=0.6,
            keyword_overlap=0.3,
        )
        print(f"  Mode: {config.mode.value}")
        print(f"  Auto refine: {config.auto_refine}")
        print(f"  Thresholds: {config.thresholds}")
        print(f"  Avg score: {config.avg_score}")
        print(f"  Keyword overlap: {config.keyword_overlap}")
        print("  [OK] ReflectionConfig working")
    except Exception as e:
        print(f"  [FAIL] ReflectionConfig failed: {e}")

    # Test 8: get_quality_level()
    print("\n[Test 8] Testing get_quality_level()...")
    try:
        test_scores = [0.92, 0.77, 0.55, 0.35]
        for score in test_scores:
            level = config.get_quality_level(score)
            print(f"  Confidence {score:.2f} -> {level.value}")
        print("  [OK] get_quality_level() working")
    except Exception as e:
        print(f"  [FAIL] get_quality_level() failed: {e}")

    # Test 9: from_settings()
    print("\n[Test 9] Testing from_settings()...")
    try:
        from src.config.settings import Config

        config_from_settings = ReflectionConfig.from_settings(Config)
        print(f"  Loaded mode: {config_from_settings.mode.value}")
        print(f"  Loaded thresholds: {config_from_settings.thresholds}")
        print(f"  Auto refine: {config_from_settings.auto_refine}")
        print("  [OK] from_settings() working")
    except Exception as e:
        print(f"  [FAIL] from_settings() failed: {e}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
    print("\nIf all tests show [OK], your implementation is correct!")
    print("Run this test: python -m src.models.evaluation")
