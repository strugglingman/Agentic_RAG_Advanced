"""
Unit tests for QueryRefiner.

Run with: pytest tests/test_query_refiner.py -v
"""

import pytest
from src.services.query_refiner import QueryRefiner, should_refine, track_refinement
from src.models.evaluation import (
    EvaluationResult,
    QualityLevel,
    RecommendationAction,
)
from src.config.settings import Config


class TestSimpleRefinement:
    """Test simple refinement without LLM."""

    def test_simple_refinement_adds_missing_aspects(self):
        """Test that simple refinement adds missing keywords."""
        refiner = QueryRefiner(openai_client=None)  # No client for simple test

        mock_eval = EvaluationResult(
            quality=QualityLevel.PARTIAL,
            confidence=0.3,
            coverage=0.2,
            recommendation=RecommendationAction.REFINE,
            reasoning="Poor quality",
            issues=["Low keyword overlap"],
            missing_aspects=["vacation", "policy"],
            relevance_scores=[0.3],
            metrics={},
        )

        original = "Tell me about PTO"
        refined = refiner._simple_refinement(original, mock_eval)

        # Should add missing aspects
        assert "vacation" in refined.lower()
        assert "policy" in refined.lower()

    def test_simple_refinement_limits_missing_aspects(self):
        """Test that simple refinement only adds top 3 missing aspects."""
        refiner = QueryRefiner(openai_client=None)

        mock_eval = EvaluationResult(
            quality=QualityLevel.POOR,
            confidence=0.2,
            coverage=0.1,
            recommendation=RecommendationAction.REFINE,
            reasoning="Poor quality",
            issues=[],
            missing_aspects=["one", "two", "three", "four", "five"],
            relevance_scores=[],
            metrics={},
        )

        original = "test query"
        refined = refiner._simple_refinement(original, mock_eval)

        # Should only add first 3
        assert "one" in refined
        assert "two" in refined
        assert "three" in refined
        assert "four" not in refined
        assert "five" not in refined


class TestShouldRefine:
    """Test refinement decision logic."""

    def test_should_refine_on_first_attempt(self):
        """Should refine when count is 0 and recommendation is REFINE."""
        mock_eval = EvaluationResult(
            quality=QualityLevel.PARTIAL,
            confidence=0.3,
            coverage=0.2,
            recommendation=RecommendationAction.REFINE,
            reasoning="",
            issues=[],
            missing_aspects=[],
            relevance_scores=[],
            metrics={},
        )

        context = {"_refinement_count": 0}
        assert should_refine(mock_eval, context) == True

    def test_should_not_refine_after_max_attempts(self):
        """Should not refine when max attempts reached."""
        mock_eval = EvaluationResult(
            quality=QualityLevel.PARTIAL,
            confidence=0.3,
            coverage=0.2,
            recommendation=RecommendationAction.REFINE,
            reasoning="",
            issues=[],
            missing_aspects=[],
            relevance_scores=[],
            metrics={},
        )

        max_attempts = Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS
        context = {"_refinement_count": max_attempts}
        assert should_refine(mock_eval, context) == False

    def test_should_not_refine_when_recommendation_is_answer(self):
        """Should not refine when recommendation is ANSWER."""
        mock_eval = EvaluationResult(
            quality=QualityLevel.GOOD,
            confidence=0.85,
            coverage=0.9,
            recommendation=RecommendationAction.ANSWER,
            reasoning="Good quality",
            issues=[],
            missing_aspects=[],
            relevance_scores=[0.9],
            metrics={},
        )

        context = {"_refinement_count": 0}
        assert should_refine(mock_eval, context) == False

    def test_should_not_refine_when_recommendation_is_clarify(self):
        """Should not refine when recommendation is CLARIFY."""
        mock_eval = EvaluationResult(
            quality=QualityLevel.POOR,
            confidence=0.1,
            coverage=0.0,
            recommendation=RecommendationAction.CLARIFY,
            reasoning="Ambiguous query",
            issues=[],
            missing_aspects=[],
            relevance_scores=[],
            metrics={},
        )

        context = {"_refinement_count": 0}
        assert should_refine(mock_eval, context) == False

    def test_should_refine_with_empty_context(self):
        """Should refine when context has no refinement count (first time)."""
        mock_eval = EvaluationResult(
            quality=QualityLevel.PARTIAL,
            confidence=0.3,
            coverage=0.2,
            recommendation=RecommendationAction.REFINE,
            reasoning="",
            issues=[],
            missing_aspects=[],
            relevance_scores=[],
            metrics={},
        )

        context = {}  # No _refinement_count key
        assert should_refine(mock_eval, context) == True


class TestTrackRefinement:
    """Test refinement tracking."""

    def test_track_refinement_initializes_context(self):
        """First refinement should initialize tracking fields."""
        context = {}

        track_refinement(context, "original query", "refined query")

        assert "_refinement_count" in context
        assert "_original_query" in context
        assert "_refinement_history" in context
        assert context["_refinement_count"] == 1
        assert context["_original_query"] == "original query"

    def test_track_refinement_increments_count(self):
        """Each refinement should increment count."""
        context = {}

        track_refinement(context, "original", "refined v1")
        track_refinement(context, "refined v1", "refined v2")

        assert context["_refinement_count"] == 2

    def test_track_refinement_builds_history(self):
        """Refinement history should contain all attempts."""
        context = {}

        track_refinement(context, "original", "refined v1")
        track_refinement(context, "refined v1", "refined v2")

        assert len(context["_refinement_history"]) == 2
        assert context["_refinement_history"][0]["from"] == "original"
        assert context["_refinement_history"][0]["to"] == "refined v1"
        assert context["_refinement_history"][1]["from"] == "refined v1"
        assert context["_refinement_history"][1]["to"] == "refined v2"

    def test_track_refinement_preserves_original_query(self):
        """Original query should be preserved across multiple refinements."""
        context = {}

        track_refinement(context, "original", "refined v1")
        track_refinement(context, "refined v1", "refined v2")
        track_refinement(context, "refined v2", "refined v3")

        # Original query should still be the first one
        assert context["_original_query"] == "original"


class TestQueryRefinerWithoutClient:
    """Test QueryRefiner behavior when no OpenAI client is provided."""

    def test_refine_query_falls_back_to_simple(self):
        """Without client, should use simple refinement."""
        refiner = QueryRefiner(openai_client=None)

        mock_eval = EvaluationResult(
            quality=QualityLevel.PARTIAL,
            confidence=0.3,
            coverage=0.2,
            recommendation=RecommendationAction.REFINE,
            reasoning="Poor quality",
            issues=[],
            missing_aspects=["keyword1", "keyword2"],
            relevance_scores=[],
            metrics={},
        )

        original = "test query"
        refined = refiner.refine_query(original, mock_eval)

        # Should have added missing aspects via simple refinement
        assert "keyword1" in refined
        assert "keyword2" in refined


class TestBuildRefinementPrompt:
    """Test prompt building for LLM refinement."""

    def test_prompt_includes_original_query(self):
        """Prompt should include the original query."""
        refiner = QueryRefiner(openai_client=None)

        mock_eval = EvaluationResult(
            quality=QualityLevel.PARTIAL,
            confidence=0.3,
            coverage=0.2,
            recommendation=RecommendationAction.REFINE,
            reasoning="Poor quality",
            issues=[],
            missing_aspects=[],
            relevance_scores=[],
            metrics={},
        )

        prompt = refiner._build_refinement_prompt("Tell me about PTO", mock_eval)

        assert "Tell me about PTO" in prompt

    def test_prompt_includes_issues(self):
        """Prompt should include evaluation issues."""
        refiner = QueryRefiner(openai_client=None)

        mock_eval = EvaluationResult(
            quality=QualityLevel.PARTIAL,
            confidence=0.3,
            coverage=0.2,
            recommendation=RecommendationAction.REFINE,
            reasoning="Poor quality",
            issues=["Low keyword overlap", "No direct match"],
            missing_aspects=[],
            relevance_scores=[],
            metrics={},
        )

        prompt = refiner._build_refinement_prompt("test query", mock_eval)

        assert "Low keyword overlap" in prompt
        assert "No direct match" in prompt

    def test_prompt_includes_missing_aspects(self):
        """Prompt should include missing aspects."""
        refiner = QueryRefiner(openai_client=None)

        mock_eval = EvaluationResult(
            quality=QualityLevel.PARTIAL,
            confidence=0.3,
            coverage=0.2,
            recommendation=RecommendationAction.REFINE,
            reasoning="Poor quality",
            issues=[],
            missing_aspects=["vacation", "policy", "accrual"],
            relevance_scores=[],
            metrics={},
        )

        prompt = refiner._build_refinement_prompt("PTO info", mock_eval)

        assert "vacation" in prompt
        assert "policy" in prompt
        assert "accrual" in prompt

    def test_prompt_includes_context_hint(self):
        """Prompt should include context hint when provided."""
        refiner = QueryRefiner(openai_client=None)

        mock_eval = EvaluationResult(
            quality=QualityLevel.PARTIAL,
            confidence=0.3,
            coverage=0.2,
            recommendation=RecommendationAction.REFINE,
            reasoning="Poor quality",
            issues=[],
            missing_aspects=[],
            relevance_scores=[],
            metrics={},
        )

        prompt = refiner._build_refinement_prompt(
            "test query", mock_eval, context_hint="HR policy documents"
        )

        assert "HR policy documents" in prompt

    def test_prompt_includes_guidelines(self):
        """Prompt should include refinement guidelines."""
        refiner = QueryRefiner(openai_client=None)

        mock_eval = EvaluationResult(
            quality=QualityLevel.PARTIAL,
            confidence=0.3,
            coverage=0.2,
            recommendation=RecommendationAction.REFINE,
            reasoning="Poor quality",
            issues=[],
            missing_aspects=[],
            relevance_scores=[],
            metrics={},
        )

        prompt = refiner._build_refinement_prompt("test", mock_eval)

        assert "Guidelines" in prompt
        assert "abbreviations" in prompt.lower()