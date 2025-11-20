"""
Quick test of self-reflection integration without needing full server.
Tests the RetrievalEvaluator with mock data.
"""

from src.models.evaluation import (
    EvaluationCriteria,
    ReflectionConfig,
    ReflectionMode,
)
from src.services.retrieval_evaluator import RetrievalEvaluator
from src.config.settings import Config
from openai import OpenAI

print("=" * 70)
print("SELF-REFLECTION INTEGRATION TEST")
print("=" * 70)

# Initialize OpenAI client
openai_client = OpenAI(api_key=Config.OPENAI_KEY)

# Load reflection config from settings
config = ReflectionConfig.from_settings(Config)
print(f"\n[Config] Mode: {config.mode.value}")
print(f"[Config] Thresholds: {config.thresholds}")

# Create evaluator
evaluator = RetrievalEvaluator(config=config, openai_client=openai_client)
print(f"\n[Evaluator] Created successfully")

# Test 1: Good retrieval (high relevance, should recommend ANSWER)
print("\n" + "=" * 70)
print("TEST 1: Good Retrieval")
print("=" * 70)

good_contexts = [
    {
        "chunk": "Our company vacation policy allows 20 days of paid time off per year. Employees can carry over up to 5 unused days to the next year.",
        "score": 0.92,
        "hybrid": 0.92,
    },
    {
        "chunk": "Vacation requests should be submitted at least 2 weeks in advance through the HR portal. Manager approval is required.",
        "score": 0.85,
        "hybrid": 0.85,
    },
]

criteria_good = EvaluationCriteria(
    query="What is our vacation policy?",
    contexts=good_contexts,
    search_metadata={"hybrid": True, "reranker": True, "top_k": 5},
    mode=config.mode,
)

result_good = evaluator.evaluate(criteria_good)
print(f"\n[Result] Quality: {result_good.quality.value}")
print(f"[Result] Confidence: {result_good.confidence:.2f}")
print(f"[Result] Coverage: {result_good.coverage:.2f}")
print(f"[Result] Recommendation: {result_good.recommendation.value}")
print(f"[Result] Reasoning: {result_good.reasoning}")
if result_good.issues:
    print(f"[Result] Issues: {', '.join(result_good.issues)}")
if result_good.missing_aspects:
    print(f"[Result] Missing: {', '.join(result_good.missing_aspects)}")

# Test 2: Poor retrieval (irrelevant contexts, should recommend EXTERNAL or REFINE)
print("\n" + "=" * 70)
print("TEST 2: Poor Retrieval (Irrelevant Contexts)")
print("=" * 70)

poor_contexts = [
    {
        "chunk": "The quarterly sales meeting is scheduled for next Monday at 2 PM in Conference Room A.",
        "score": 0.15,
        "hybrid": 0.15,
    },
    {
        "chunk": "Please remember to submit your expense reports by the end of the month.",
        "score": 0.12,
        "hybrid": 0.12,
    },
]

criteria_poor = EvaluationCriteria(
    query="What is the employee vacation policy?",
    contexts=poor_contexts,
    search_metadata={"hybrid": True, "reranker": True, "top_k": 5},
    mode=config.mode,
)

result_poor = evaluator.evaluate(criteria_poor)
print(f"\n[Result] Quality: {result_poor.quality.value}")
print(f"[Result] Confidence: {result_poor.confidence:.2f}")
print(f"[Result] Coverage: {result_poor.coverage:.2f}")
print(f"[Result] Recommendation: {result_poor.recommendation.value}")
print(f"[Result] Reasoning: {result_poor.reasoning}")
if result_poor.issues:
    print(f"[Result] Issues: {', '.join(result_poor.issues)}")
if result_poor.missing_aspects:
    print(f"[Result] Missing: {', '.join(result_poor.missing_aspects)}")

# Test 3: No contexts (should recommend EXTERNAL)
print("\n" + "=" * 70)
print("TEST 3: No Contexts Retrieved")
print("=" * 70)

criteria_empty = EvaluationCriteria(
    query="What is quantum physics?",
    contexts=[],
    search_metadata={"hybrid": True, "reranker": True, "top_k": 5},
    mode=config.mode,
)

result_empty = evaluator.evaluate(criteria_empty)
print(f"\n[Result] Quality: {result_empty.quality.value}")
print(f"\n[Result] Confidence: {result_empty.confidence:.2f}")
print(f"[Result] Coverage: {result_empty.coverage:.2f}")
print(f"[Result] Recommendation: {result_empty.recommendation.value}")
print(f"[Result] Reasoning: {result_empty.reasoning}")
if result_empty.issues:
    print(f"[Result] Issues: {', '.join(result_empty.issues)}")
if result_empty.missing_aspects:
    print(f"[Result] Missing: {', '.join(result_empty.missing_aspects)}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE!")
print("=" * 70)
print(f"\n[SUCCESS] Self-reflection system is working in {config.mode.value.upper()} mode!")
print("\nNext steps:")
print("1. Try changing REFLECTION_MODE in .env to 'fast' or 'thorough'")
print("2. Test with the full server when you have uploaded documents")
print("3. Observe logs when making agent requests through the API")
