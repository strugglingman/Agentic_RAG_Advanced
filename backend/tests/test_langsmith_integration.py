"""
Test script for Ragas + LangSmith integration.
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from src.evaluation.ragas_evaluator import RagasEvaluator
from datasets import Dataset


# Example 1: Evaluate local dataset and push to LangSmith
def test_push_to_langsmith():
    """Evaluate local dataset and upload results to LangSmith."""

    # Create sample dataset
    data = {
        "question": ["What is the capital of France?", "Who wrote Romeo and Juliet?"],
        "answer": [
            "Paris is the capital of France.",
            "William Shakespeare wrote Romeo and Juliet.",
        ],
        "contexts": [
            ["Paris is the capital and largest city of France."],
            ["Romeo and Juliet is a tragedy written by William Shakespeare."],
        ],
        "ground_truth": ["Paris", "William Shakespeare"],
    }
    dataset = Dataset.from_dict(data)

    # Evaluate
    evaluator = RagasEvaluator()
    results = evaluator.evaluate_dataset(dataset)

    # Push to LangSmith
    evaluator.push_to_langsmith(
        results,
        dataset_name="rag-test-eval",
        description="Test evaluation for RAG system",
    )

    print("\n✅ Results uploaded to LangSmith!")


# Example 2: Evaluate with automatic LangSmith tracing
def test_evaluate_with_tracing():
    """
    Run evaluation with automatic LangSmith tracing.

    LangSmith integration works automatically when these env vars are set:
    - LANGCHAIN_TRACING_V2=true
    - LANGCHAIN_API_KEY=your_key
    - LANGCHAIN_PROJECT=your_project

    Traces will appear in LangSmith dashboard automatically.
    """
    # Create sample dataset
    data = {
        "question": ["What is Python?"],
        "answer": ["Python is a programming language."],
        "contexts": [["Python is a high-level programming language."]],
        "ground_truth": ["A programming language"],
    }
    dataset = Dataset.from_dict(data)

    evaluator = RagasEvaluator()

    try:
        # This automatically logs to LangSmith if env vars are set
        results = evaluator.evaluate_dataset(dataset)
        print("\n✅ Evaluation completed! Check LangSmith for traces.")
        print(results.to_pandas())
    except Exception as e:
        print(f"❌ Error: {e}")


# Example 3: Evaluate single query and upload
def test_single_evaluation():
    """Evaluate a single query and push to LangSmith."""
    evaluator = RagasEvaluator()

    result = evaluator.evaluate_single(
        question="What is machine learning?",
        answer="Machine learning is a subset of AI that enables systems to learn from data.",
        contexts=[
            "Machine learning is a field of artificial intelligence that uses statistical techniques."
        ],
        ground_truth="A subset of artificial intelligence",
    )

    # Push single result
    evaluator.push_to_langsmith(
        result,
        dataset_name="single-query-test",
        description="Single query evaluation test",
    )

    print("\n✅ Single evaluation uploaded to LangSmith!")


if __name__ == "__main__":
    print("=== Ragas + LangSmith Integration Test ===\n")

    # Choose which test to run:
    print("1. Test push_to_langsmith (local dataset → LangSmith)")
    print("2. Test evaluate_with_tracing (auto-traces to LangSmith)")
    print("3. Test single evaluation")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        test_push_to_langsmith()
    elif choice == "2":
        test_evaluate_with_tracing()
    elif choice == "3":
        test_single_evaluation()
    else:
        print("Invalid choice. Running test 1 by default.")
        test_push_to_langsmith()
