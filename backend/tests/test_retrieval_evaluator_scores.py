import pytest

from src.config.settings import Config
from src.models.evaluation import (
    EvaluationCriteria,
    ReflectionConfig,
    ReflectionMode,
    RecommendationAction,
)
from src.services.retrieval_evaluator import RetrievalEvaluator


pytestmark = pytest.mark.unit


def _fast_evaluator():
    cfg = ReflectionConfig.from_settings(Config)
    cfg.mode = ReflectionMode.FAST
    return RetrievalEvaluator(config=cfg)


def test_extract_relevance_scores_prioritizes_rerank():
    evaluator = _fast_evaluator()
    contexts = [
        {"sem_sim": 0.2, "hybrid": 0.7, "rerank": 0.0},
        {"sem_sim": 0.1, "hybrid": 0.6, "rerank": 0.8},
    ]
    assert evaluator._extract_relevance_scores(contexts) == [0.0, 0.8]


def test_extract_relevance_scores_prioritizes_hybrid_when_no_rerank():
    evaluator = _fast_evaluator()
    contexts = [
        {"sem_sim": 0.9, "hybrid": 0.2, "rerank": 0.0},
        {"sem_sim": 0.8, "hybrid": 0.4, "rerank": 0.0},
    ]
    assert evaluator._extract_relevance_scores(contexts) == [0.2, 0.4]


def test_extract_relevance_scores_clamps_values_to_unit_interval():
    evaluator = _fast_evaluator()
    contexts = [
        {"sem_sim": -0.5, "hybrid": 0.0, "rerank": 0.0},
        {"sem_sim": 1.6, "hybrid": 0.0, "rerank": 0.0},
    ]
    assert evaluator._extract_relevance_scores(contexts) == [0.0, 1.0]


def test_evaluate_fast_uses_reranker_optimized_formula(monkeypatch):
    evaluator = _fast_evaluator()
    criteria = EvaluationCriteria(
        query="q",
        contexts=[{"rerank": 0.8, "hybrid": 0.0, "sem_sim": 0.1}, {"rerank": 0.6, "hybrid": 0.0, "sem_sim": 0.2}],
        mode=ReflectionMode.FAST,
    )

    monkeypatch.setattr(evaluator, "_extract_keywords", lambda _q: ["q"])
    monkeypatch.setattr(evaluator, "_extract_context_text", lambda _ctx: "context")
    monkeypatch.setattr(evaluator, "_calculate_keyword_overlap", lambda _k, _c: 0.0)
    monkeypatch.setattr(evaluator, "_detect_issues", lambda *_a, **_k: [])
    monkeypatch.setattr(evaluator, "_identify_missing_aspects", lambda *_a, **_k: [])
    monkeypatch.setattr(
        evaluator,
        "_determine_recommendation",
        lambda confidence, _count, _query: (RecommendationAction.ANSWER, f"c={confidence:.3f}"),
    )

    result = evaluator._evaluate_fast(criteria)
    expected = (0.7 * 0.5) + (0.6 * 0.3) + (1.0 * 0.2)
    assert result.confidence == pytest.approx(expected)


def test_evaluate_fast_uses_standard_formula_without_reranker(monkeypatch):
    evaluator = _fast_evaluator()
    criteria = EvaluationCriteria(
        query="q",
        contexts=[{"rerank": 0.0, "hybrid": 0.5, "sem_sim": 0.1}, {"rerank": 0.0, "hybrid": 0.4, "sem_sim": 0.2}],
        mode=ReflectionMode.FAST,
    )

    monkeypatch.setattr(evaluator, "_extract_keywords", lambda _q: ["q"])
    monkeypatch.setattr(evaluator, "_extract_context_text", lambda _ctx: "context")
    monkeypatch.setattr(evaluator, "_calculate_keyword_overlap", lambda _k, _c: 0.25)
    monkeypatch.setattr(evaluator, "_detect_issues", lambda *_a, **_k: [])
    monkeypatch.setattr(evaluator, "_identify_missing_aspects", lambda *_a, **_k: [])
    monkeypatch.setattr(
        evaluator,
        "_determine_recommendation",
        lambda confidence, _count, _query: (RecommendationAction.ANSWER, f"c={confidence:.3f}"),
    )

    result = evaluator._evaluate_fast(criteria)
    # kw*0.4 + avg*0.3 + min*0.2 + presence*0.1
    expected = (0.25 * 0.4) + (0.45 * 0.3) + (0.4 * 0.2) + (1.0 * 0.1)
    assert result.confidence == pytest.approx(expected)

