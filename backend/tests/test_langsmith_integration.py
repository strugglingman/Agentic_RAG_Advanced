"""
Manual smoke tests for Ragas + LangSmith integration.

This module is intentionally env-gated and performs external API calls.
It has no import-time side effects.
"""

import os
import pytest


pytestmark = [pytest.mark.manual, pytest.mark.external]


@pytest.fixture
def require_manual_external_env():
    """Require explicit opt-in + API credentials for external manual checks."""
    if os.getenv("RUN_MANUAL_TESTS") != "1":
        pytest.skip("Set RUN_MANUAL_TESTS=1 to run manual external tests.")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required.")
    if not os.getenv("LANGCHAIN_API_KEY"):
        pytest.skip("LANGCHAIN_API_KEY is required.")


def _sample_dataset():
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "question": ["What is the capital of France?"],
            "answer": ["Paris is the capital of France."],
            "contexts": [["Paris is the capital and largest city of France."]],
            "ground_truth": ["Paris"],
        }
    )


def test_ragas_evaluate_dataset_smoke(require_manual_external_env):
    from src.evaluation.ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator()
    results = evaluator.evaluate_dataset(_sample_dataset())
    df = results.to_pandas()

    assert len(df) >= 1
    assert not df.empty


def test_ragas_push_to_langsmith_smoke(require_manual_external_env):
    """
    Optional upload test.

    Set RUN_LANGSMITH_UPLOAD=1 to enable dataset push.
    """
    if os.getenv("RUN_LANGSMITH_UPLOAD") != "1":
        pytest.skip("Set RUN_LANGSMITH_UPLOAD=1 to run LangSmith upload check.")

    from src.evaluation.ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator()
    results = evaluator.evaluate_dataset(_sample_dataset())
    evaluator.push_to_langsmith(
        results,
        dataset_name=os.getenv("LANGSMITH_TEST_DATASET", "rag-test-eval-smoke"),
        description="Manual smoke test for RAG evaluation upload",
    )
