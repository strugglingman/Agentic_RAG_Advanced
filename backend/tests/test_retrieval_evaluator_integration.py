"""
Manual smoke tests for RetrievalEvaluator across reflection modes.

These checks are intentionally opt-in and may call external APIs
depending on MANUAL_REFLECTION_MODE.
"""

import asyncio
import os

import pytest

from src.config.settings import Config
from src.models.evaluation import EvaluationCriteria, ReflectionConfig, ReflectionMode
from src.services.retrieval_evaluator import RetrievalEvaluator


pytestmark = [pytest.mark.manual, pytest.mark.integration, pytest.mark.external]


@pytest.fixture
def require_manual_opt_in():
    """Require explicit opt-in for manual smoke checks."""
    if os.getenv("RUN_MANUAL_TESTS") != "1":
        pytest.skip("Set RUN_MANUAL_TESTS=1 to run manual smoke tests.")


def _contexts_good():
    return [
        {
            "chunk": (
                "Employees receive 20 days paid vacation annually and can carry over up to "
                "5 unused days."
            ),
            "rerank": 0.92,
        },
        {
            "chunk": "Vacation requests should be submitted at least 2 weeks in advance.",
            "rerank": 0.83,
        },
    ]


def _contexts_poor():
    return [
        {
            "chunk": "The quarterly sales standup is every Monday morning.",
            "rerank": 0.19,
        },
        {
            "chunk": "Expense reports are due by the final business day each month.",
            "rerank": 0.15,
        },
    ]


async def _run_evaluations(mode: ReflectionMode, openai_client):
    config = ReflectionConfig.from_settings(Config)
    config.mode = mode

    evaluator = RetrievalEvaluator(config=config, openai_client=openai_client)

    good = await evaluator.evaluate(
        EvaluationCriteria(
            query="What is the employee vacation policy?",
            contexts=_contexts_good(),
            search_metadata={"hybrid": True, "reranker": True},
            mode=mode,
        )
    )
    poor = await evaluator.evaluate(
        EvaluationCriteria(
            query="What is the employee vacation policy?",
            contexts=_contexts_poor(),
            search_metadata={"hybrid": True, "reranker": True},
            mode=mode,
        )
    )
    empty = await evaluator.evaluate(
        EvaluationCriteria(
            query="What is the employee vacation policy?",
            contexts=[],
            search_metadata={"hybrid": True, "reranker": True},
            mode=mode,
        )
    )
    return good, poor, empty


def test_retrieval_evaluator_quality_ordering(require_manual_opt_in):
    """
    Verify ordering invariants:
    - good contexts should not score below poor/empty contexts
    - no-context path should request fallback action
    """
    mode_name = os.getenv("MANUAL_REFLECTION_MODE", "fast").strip().lower()
    try:
        mode = ReflectionMode(mode_name)
    except ValueError:
        pytest.fail(
            "MANUAL_REFLECTION_MODE must be one of: fast, balanced, thorough"
        )

    client = None
    if mode in {ReflectionMode.BALANCED, ReflectionMode.THOROUGH}:
        if not Config.OPENAI_KEY:
            pytest.skip(
                "OPENAI_API_KEY is required for balanced/thorough manual reflection mode."
            )
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=Config.OPENAI_KEY)

    async def _run():
        try:
            return await _run_evaluations(mode, client)
        finally:
            if client is not None:
                await client.close()

    good, poor, empty = asyncio.run(_run())

    assert good.mode_used == mode
    assert poor.mode_used == mode
    assert empty.mode_used == mode

    assert 0.0 <= good.confidence <= 1.0
    assert 0.0 <= poor.confidence <= 1.0
    assert 0.0 <= empty.confidence <= 1.0

    assert good.confidence >= poor.confidence
    assert good.confidence >= empty.confidence
    assert empty.recommendation.value in {"refine", "external", "clarify"}
