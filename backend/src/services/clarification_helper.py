"""
Clarification message generation for self-reflection system.

Generates helpful messages when queries fail to retrieve relevant
information after refinement attempts.

Week 2 - Day 4: Clarification & Progressive Fallback
"""

from typing import Optional, List
from openai import OpenAI
from src.models.evaluation import EvaluationResult
from src.config.settings import Config


class ClarificationHelper:
    """
    Generates clarification messages for ambiguous/failed queries.

    When retrieval fails or max refinements reached, this class
    generates helpful messages asking the user for more context.

    Example Usage:
        helper = ClarificationHelper(openai_client=client)
        message = helper.generate_clarification(
            query="How do I apply?",
            eval_result=eval_result,
            max_attempts_reached=True,
        )
    """

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
    ):
        """
        Initialize clarification helper.

        Args:
            openai_client: Optional OpenAI client (for future LLM-based suggestions)
            model: Model to use (default: gpt-4o-mini)
            temperature: Temperature for creativity (0.5 = balanced)

        TODO: Implement initialization
        Steps:
        1. Store openai_client as self.client
        2. Store model as self.model
        3. Store temperature as self.temperature
        """
        self.client = openai_client
        self.model = model
        self.temperature = temperature

    def generate_clarification(
        self,
        query: str,
        eval_result: EvaluationResult,
        refinement_attempted: bool = False,
        max_attempts_reached: bool = False,
        context_hint: Optional[str] = None,
    ) -> str:
        """
        Generate a clarification message based on the scenario.

        Args:
            query: Original user query
            eval_result: Evaluation result with issues/missing_aspects
            refinement_attempted: Whether refinement was tried
            max_attempts_reached: Whether max refinement attempts reached
            context_hint: Optional hint about document collection

        Returns:
            Clarification message string

        TODO: Implement message routing
        Steps:
        1. If max_attempts_reached is True:
           - Return self._max_attempts_message(query, eval_result, context_hint)
        2. Else if eval_result.issues is empty OR "No contexts" in str(eval_result.issues):
           - Return self._no_results_message(query, context_hint)
        3. Else:
           - Return self._ambiguous_query_message(query, eval_result, context_hint)
        """
        pass

    def _no_results_message(
        self,
        query: str,
        context_hint: Optional[str] = None,
    ) -> str:
        """
        Generate message when no contexts were found.

        Args:
            query: User query
            context_hint: Optional hint about document collection

        Returns:
            Helpful message explaining no results found

        TODO: Implement no results message
        Steps:
        1. Create base message:
           f'I couldn't find any relevant information for: "{query}"'
        2. Add suggestions:
           - "The information isn't in the uploaded documents"
           - "Try using different keywords"
           - "Be more specific about what you're looking for"
        3. If context_hint provided:
           - Add: f"Note: I searched in {context_hint}"
        4. Return formatted message
        """
        pass

    def _ambiguous_query_message(
        self,
        query: str,
        eval_result: EvaluationResult,
        context_hint: Optional[str] = None,
    ) -> str:
        """
        Generate message for ambiguous queries with low-quality results.

        Args:
            query: User query
            eval_result: Evaluation result with issues
            context_hint: Optional hint about document collection

        Returns:
            Message explaining issues and suggesting improvements

        TODO: Implement ambiguous query message
        Steps:
        1. Create base message:
           f'Your query "{query}" returned low-quality results.'
        2. If eval_result.issues exists:
           - Add "Issues found:" section
           - List each issue (max 3): f"- {issue}"
        3. If eval_result.missing_aspects exists:
           - Add: f"Missing keywords: {', '.join(eval_result.missing_aspects[:5])}"
        4. Add suggestions section:
           - "Being more specific"
           - "Using different keywords"
           - "Providing more context"
        5. Return formatted message
        """
        pass

    def _max_attempts_message(
        self,
        query: str,
        eval_result: EvaluationResult,
        context_hint: Optional[str] = None,
    ) -> str:
        """
        Generate message after max refinement attempts reached.

        Args:
            query: Original user query
            eval_result: Final evaluation result
            context_hint: Optional hint about document collection

        Returns:
            Message explaining multiple attempts failed

        TODO: Implement max attempts message
        Steps:
        1. Create base message:
           f'After multiple search attempts, I couldn't find highly relevant '
           f'information for: "{query}"'
        2. Add explanation:
           "The best results I found may not fully answer your question."
        3. Add suggestions:
           - "Try rephrasing your question"
           - "Break it into smaller, more specific questions"
           - "Check if the information exists in the uploaded documents"
        4. If eval_result.missing_aspects exists:
           - Add: f"Keywords not found: {', '.join(eval_result.missing_aspects[:5])}"
        5. If context_hint provided:
           - Add: f"Searched in: {context_hint}"
        6. Return formatted message
        """
        pass


# =============================================================================
# TESTING (run with: python -m src.services.clarification_helper)
# =============================================================================

if __name__ == "__main__":
    """
    Test block for ClarificationHelper.

    TODO: Implement test cases
    Steps:
    1. Import dependencies:
       - from src.models.evaluation import EvaluationResult, QualityLevel, RecommendationAction

    2. Print header:
       print("=" * 70)
       print("CLARIFICATION HELPER TEST")
       print("=" * 70)

    3. Create helper:
       helper = ClarificationHelper(openai_client=None)

    4. Test 1: No results message
       - Print: "Test 1: No Results Message"
       - Call: helper._no_results_message("What is quantum physics?", "HR documents")
       - Print the result

    5. Test 2: Ambiguous query message
       - Print: "Test 2: Ambiguous Query Message"
       - Create mock EvaluationResult with:
         quality=QualityLevel.POOR
         confidence=0.2
         coverage=0.1
         recommendation=RecommendationAction.CLARIFY
         issues=["Low average relevance score: 0.20", "Poor keyword match: 0.10"]
         missing_aspects=["apply", "application", "process", "submit"]
       - Call: helper._ambiguous_query_message("How do I apply?", mock_eval)
       - Print the result

    6. Test 3: Max attempts message
       - Print: "Test 3: Max Attempts Message"
       - Call: helper._max_attempts_message("Tell me about xyz", mock_eval, "company documents")
       - Print the result

    7. Test 4: generate_clarification routing
       - Print: "Test 4: generate_clarification Routing"
       - Test with max_attempts_reached=True → should call _max_attempts_message
       - Test with empty issues → should call _no_results_message
       - Test with issues present → should call _ambiguous_query_message
       - Print results

    8. Print footer:
       print("=" * 70)
       print("ALL TESTS COMPLETE!")
       print("=" * 70)
    """
    print("=" * 70)
    print("CLARIFICATION HELPER TEST")
    print("=" * 70)
    print("\nTODO: Implement ClarificationHelper class first")
    print("Then run: python -m src.services.clarification_helper")
    print("=" * 70)
