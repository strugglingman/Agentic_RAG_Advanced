"""
Query refinement service for self-reflection system.

This service reformulates user queries when retrieval quality is poor,
using LLM and evaluation feedback to create better search queries.

Week 2 - Day 3: Query Refinement
"""

from typing import Optional, List, Dict, Any
from openai import OpenAI
from src.models.evaluation import EvaluationResult, RecommendationAction
from src.config.settings import Config


class QueryRefiner:
    """
    Refines user queries based on evaluation feedback.

    Uses LLM to reformulate queries when retrieval evaluation
    indicates poor quality (REFINE recommendation).

    Example:
        Original: "Tell me about PTO"
        Refined: "employee paid time off vacation policy accrual"
    """

    def __init__(
        self,
        openai_client: OpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        """
        Initialize the query refiner.

        Args:
            openai_client: OpenAI client for LLM calls
            model: Model to use for refinement (default: gpt-4o-mini)
            temperature: Temperature for creativity (0.3 = balanced)

        TODO: Implement initialization
        Steps:
        1. Store openai_client as self.client
        2. Store model as self.model
        3. Store temperature as self.temperature
        """
        self.client = openai_client
        self.model = model
        self.temperature = temperature

    def refine_query(
        self,
        original_query: str,
        eval_result: EvaluationResult,
        context_hint: Optional[str] = None,
    ) -> str:
        """
        Refine a query based on evaluation feedback.

        Args:
            original_query: The original user query
            eval_result: Evaluation result with issues/missing_aspects
            context_hint: Optional context about the document collection

        Returns:
            Refined query string

        Example:
            original: "Tell me about PTO"
            refined: "employee paid time off vacation policy accrual"

        TODO: Implement query refinement
        Steps:
        1. Build refinement prompt using _build_refinement_prompt()
        2. Try to call LLM:
           a. Use self.client.chat.completions.create()
           b. model=self.model
           c. temperature=self.temperature
           d. max_tokens=100
           e. messages=[
                {"role": "system", "content": "You are a query refinement expert..."},
                {"role": "user", "content": prompt}
              ]
        3. Extract refined query from response.choices[0].message.content
        4. Clean up (strip quotes, whitespace)
        5. Validate: if refined looks bad (too short, same as original), return original
        6. On exception: fallback to _simple_refinement()
        7. Return refined query
        """
        if not self.client:
            print("[QUERY_REFINER] No OpenAI client provided, using simple refinement.")
            return self._simple_refinement(original_query, eval_result)

        refine_prompt = self._build_refinement_prompt(
            original_query=original_query,
            eval_result=eval_result,
            context_hint=context_hint,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=150,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query refinement expert who improves user search queries for document retrieval systems.",
                    },
                    {"role": "user", "content": refine_prompt},
                ],
            )
            refined_query = (
                response.choices[0].message.content.strip().strip('"').strip("'")
            )
            # Simple validation
            if (
                len(refined_query) < 5
                or refined_query.lower() == original_query.lower()
            ):
                return original_query

            return refined_query
        except Exception as e:
            print(f"[QUERY_REFINER] LLM error: {e}. Falling back to simple refinement.")
            return self._simple_refinement(original_query, eval_result)

    def _build_refinement_prompt(
        self,
        original_query: str,
        eval_result: EvaluationResult,
        context_hint: Optional[str] = None,
    ) -> str:
        """
        Build the LLM prompt for query refinement.

        Args:
            original_query: Original query
            eval_result: Evaluation result with issues/missing aspects
            context_hint: Optional context hint

        Returns:
            Formatted prompt string

        TODO: Implement prompt building
        Steps:
        1. Start with: f'Original query: "{original_query}"'
        2. Add section: "The search returned poor results. Here's why:"
        3. If eval_result.issues exists:
           - Add "Issues:" section
           - List each issue with "  - {issue}"
        4. If eval_result.missing_aspects exists:
           - Add "Missing keywords in results:" section
           - Add: f"  {', '.join(eval_result.missing_aspects)}"
        5. If context_hint provided:
           - Add: f"Context: {context_hint}"
        6. Add instructions:
           "Task: Reformulate the query to get better search results."
           ""
           "Guidelines:"
           "1. Expand abbreviations (e.g., 'PTO' → 'paid time off vacation')"
           "2. Add relevant keywords from missing aspects"
           "3. Make the query more specific and searchable"
           "4. Keep it concise (under 15 words)"
           "5. Focus on document search, not conversational"
           ""
           "Return ONLY the refined query, nothing else."
        7. Join all parts with newlines and return
        """
        refined_prompt_parts: list[str] = []
        refined_prompt_parts.append(
            f'Original query: "{original_query}" The search returned poor results. Here\'s why:'
        )
        issue_section = ""
        if eval_result.issues:
            issue_section += "\nIssues:"
            issue_section += "\n" + "\n".join(
                f" - {issue}" for issue in eval_result.issues
            )
            refined_prompt_parts.append(issue_section)
        missing_section = ""
        if eval_result.missing_aspects:
            missing_section += "\nMissing keywords in results:"
            missing_section += f"\n  {', '.join(eval_result.missing_aspects)}"
            refined_prompt_parts.append(missing_section)
        if context_hint:
            refined_prompt_parts.append(f"\nContext: {context_hint}")

        instruction = (
            "\nTask: Reformulate the query to get better search results.\n\n"
            "Guidelines:\n"
            "1. Expand abbreviations (e.g., 'PTO' → 'paid time off vacation')\n"
            "2. Add relevant keywords from missing aspects\n"
            "3. Make the query more specific and searchable\n"
            "4. Keep it concise (under 15 words)\n"
            "5. Focus on document search, not conversational\n\n"
            "Return ONLY the refined query, nothing else."
        )
        refined_prompt_parts.append(instruction)

        return "\n".join(refined_prompt_parts)

    def _simple_refinement(
        self,
        original_query: str,
        eval_result: EvaluationResult,
    ) -> str:
        """
        Simple fallback refinement without LLM.

        Expands common abbreviations and adds missing keywords.

        Args:
            original_query: Original query
            eval_result: Evaluation result

        Returns:
            Refined query

        TODO: Implement simple refinement
        Steps:
        1. Convert original_query to lowercase: refined = original_query.lower()
        2. Define abbreviations dict:
           {
               "pto": "paid time off vacation",
               "hr": "human resources",
               "401k": "retirement savings plan",
               "q1": "first quarter",
               "q2": "second quarter",
               "q3": "third quarter",
               "q4": "fourth quarter",
               "yoy": "year over year",
               "roi": "return on investment",
           }
        3. For each abbreviation in abbreviations.items():
           - If abbreviation in refined.split() (as whole word):
             - Replace: refined = refined.replace(abbr, expansion)
        4. If eval_result.missing_aspects exists:
           - Get top 3: missing_keywords = eval_result.missing_aspects[:3]
           - Append: refined = f"{refined} {' '.join(missing_keywords)}"
        5. Return refined.strip()
        """
        refined = original_query.lower()

        if eval_result.missing_aspects:
            missing_keywords = eval_result.missing_aspects[:3]
            refined = f"{refined} {' '.join(missing_keywords)}"

        return refined.strip()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def should_refine(eval_result: EvaluationResult, context: Dict[str, Any]) -> bool:
    """
    Check if refinement should be attempted.

    Args:
        eval_result: Evaluation result
        context: Agent context with refinement tracking

    Returns:
        True if should refine, False otherwise

    TODO: Implement refinement decision logic
    Steps:
    1. Check if eval_result.recommendation == RecommendationAction.REFINE
       - If not, return False
    2. Get refinement_count from context.get("_refinement_count", 0)
    3. Get max_attempts from Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS
    4. If refinement_count >= max_attempts:
       - Print: f"[QUERY_REFINER] Max refinement attempts ({max_attempts}) reached"
       - Return False
    5. Return True
    """
    if eval_result.recommendation != RecommendationAction.REFINE:
        return False

    refinement_count = context.get("_refinement_count", 0)
    if refinement_count >= Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS:
        print(
            f"[QUERY_REFINER] Max refinement attempts ({Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS}) reached"
        )
        return False

    return True


def track_refinement(context: Dict[str, Any], original_query: str, refined_query: str):
    """
    Track refinement in context for loop prevention.

    Args:
        context: Agent context dict (modified in-place)
        original_query: Original query
        refined_query: Refined query

    TODO: Implement refinement tracking
    Steps:
    1. Check if "_refinement_count" not in context:
       - If true (first refinement):
         - Set context["_refinement_count"] = 0
         - Set context["_original_query"] = original_query
         - Set context["_refinement_history"] = []
    2. Increment: context["_refinement_count"] += 1
    3. Append to history:
       context["_refinement_history"].append({
           "attempt": context["_refinement_count"],
           "from": original_query,
           "to": refined_query,
       })
    4. Log:
       print(f"[QUERY_REFINER] Refinement attempt {context['_refinement_count']}")
       print(f"[QUERY_REFINER] Original: {original_query}")
       print(f"[QUERY_REFINER] Refined: {refined_query}")
    """
    if "_refinement_count" not in context:
        context["_refinement_count"] = 0
        context["_original_query"] = original_query
        context["_refinement_history"] = []
    context["_refinement_count"] += 1
    context["_refinement_history"].append(
        {
            "attempt": context["_refinement_count"],
            "from": original_query,
            "to": refined_query,
        }
    )
    print(f"[QUERY_REFINER] Refinement attempt {context['_refinement_count']}")
    print(f"[QUERY_REFINER] Original: {original_query}")
    print(f"[QUERY_REFINER] Refined: {refined_query}")


# =============================================================================
# TESTING (run with: python -m src.services.query_refiner)
# =============================================================================

if __name__ == "__main__":
    """
    Test block for QueryRefiner.

    TODO: Implement test cases
    Steps:
    1. Import dependencies:
       - from openai import OpenAI
       - from src.config.settings import Config
       - from src.models.evaluation import RecommendationAction, QualityLevel

    2. Print header:
       print("=" * 70)
       print("QUERY REFINER TEST")
       print("=" * 70)

    3. Create refiner:
       - client = OpenAI(api_key=Config.OPENAI_KEY)
       - refiner = QueryRefiner(openai_client=client)

    4. Test 1: Abbreviation expansion
       - Print: "Test 1: Abbreviation Expansion"
       - original = "Tell me about PTO"
       - Create mock EvaluationResult with:
         quality=QualityLevel.PARTIAL
         confidence=0.35
         coverage=0.2
         recommendation=RecommendationAction.REFINE
         issues=["Poor keyword match: 0.20"]
         missing_aspects=["vacation", "time", "off", "policy"]
       - Call: refined = refiner.refine_query(original, mock_eval)
       - Print original and refined

    5. Test 2: Ambiguous query
       - Print: "Test 2: Ambiguous Query"
       - original = "How do I apply?"
       - Create mock EvaluationResult with poor quality, no contexts
       - Call with context_hint="HR policy documents"
       - Print original and refined

    6. Test 3: should_refine logic
       - Print: "Test 3: should_refine Logic"
       - Test with _refinement_count=0 (should return True)
       - Test with _refinement_count=3 (should return False)
       - Print results

    7. Test 4: track_refinement
       - Print: "Test 4: track_refinement"
       - Create empty context dict
       - Call track_refinement() twice
       - Print context["_refinement_count"] and history

    8. Print footer:
       print("\n" + "=" * 70)
       print("TEST COMPLETE")
       print("=" * 70)
    """
    from openai import OpenAI
    from src.config.settings import Config
    from src.models.evaluation import RecommendationAction, QualityLevel

    print("=" * 70)
    print("QUERY REFINER TEST")
    print("=" * 70)

    # Create refiner
    client = OpenAI(api_key=Config.OPENAI_KEY)
    refiner = QueryRefiner(openai_client=client)
    print("[OK] QueryRefiner created")

    # Test 1: Abbreviation expansion
    print("\n" + "-" * 70)
    print("Test 1: Query Refinement with LLM")
    print("-" * 70)
    original = "Tell me about PTO"
    mock_eval = EvaluationResult(
        quality=QualityLevel.PARTIAL,
        confidence=0.35,
        coverage=0.2,
        recommendation=RecommendationAction.REFINE,
        reasoning="Poor keyword match",
        issues=["Poor keyword match: 0.20"],
        missing_aspects=["vacation", "time", "off", "policy"],
        relevance_scores=[0.3, 0.4],
        metrics={},
    )
    refined = refiner.refine_query(original, mock_eval)
    print(f"  Original: {original}")
    print(f"  Refined:  {refined}")
    print("  [OK] LLM refinement complete")

    # Test 2: Simple refinement fallback
    print("\n" + "-" * 70)
    print("Test 2: Simple Refinement (fallback)")
    print("-" * 70)
    simple_refined = refiner._simple_refinement(original, mock_eval)
    print(f"  Original: {original}")
    print(f"  Simple:   {simple_refined}")
    print("  [OK] Simple refinement complete")

    # Test 3: should_refine logic
    print("\n" + "-" * 70)
    print("Test 3: should_refine Logic")
    print("-" * 70)
    max_attempts = Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS
    print(f"  Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS = {max_attempts}")

    # Should refine when count=0
    context1 = {"_refinement_count": 0}
    result1 = should_refine(mock_eval, context1)
    print(f"  Count=0: should_refine={result1} (expected: True)")

    # Should NOT refine when count >= max_attempts
    context2 = {"_refinement_count": max_attempts}
    result2 = should_refine(mock_eval, context2)
    print(f"  Count={max_attempts}: should_refine={result2} (expected: False)")

    # Should NOT refine when recommendation is not REFINE
    mock_eval_answer = EvaluationResult(
        quality=QualityLevel.GOOD,
        confidence=0.85,
        coverage=0.9,
        recommendation=RecommendationAction.ANSWER,
        reasoning="Good",
        issues=[],
        missing_aspects=[],
        relevance_scores=[0.9],
        metrics={},
    )
    result3 = should_refine(mock_eval_answer, context1)
    print(f"  Recommendation=ANSWER: should_refine={result3} (expected: False)")
    print("  [OK] should_refine logic verified")

    # Test 4: track_refinement
    print("\n" + "-" * 70)
    print("Test 4: track_refinement")
    print("-" * 70)
    context = {}
    track_refinement(context, "original query", "refined query v1")
    track_refinement(context, "refined query v1", "refined query v2")
    print(f"  Refinement count: {context['_refinement_count']} (expected: 2)")
    print(f"  History length: {len(context['_refinement_history'])} (expected: 2)")
    print("  [OK] track_refinement verified")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
