"""
Clarification message generation for self-reflection system.

Generates helpful messages when queries fail to retrieve relevant
information after refinement attempts.

Week 2 - Day 4: Clarification & Progressive Fallback
"""

from typing import Optional
from openai import OpenAI
from src.models.evaluation import EvaluationResult
from src.config.settings import Config
from langsmith import traceable


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
        model: str = None,
        temperature: float = 0.1,
    ):
        """
        Initialize clarification helper.

        Args:
            openai_client: Optional OpenAI client (for future LLM-based suggestions)
            model: Model to use (default: Config.OPENAI_MODEL)
            temperature: Temperature for creativity (0.5 = balanced)

        TODO: Implement initialization
        Steps:
        1. Store openai_client as self.client
        2. Store model as self.model
        3. Store temperature as self.temperature
        """
        self.client = openai_client
        self.model = model or Config.OPENAI_MODEL
        self.temperature = temperature

    @traceable
    def generate_clarification(
        self,
        query: str,
        eval_result: EvaluationResult,
        max_attempts_reached: bool = False,
        context_hint: Optional[str] = None,
    ) -> str:
        """
        Generate a clarification message based on the scenario.

        Args:
            query: Original user query
            eval_result: Evaluation result with issues/missing_aspects
            max_attempts_reached: Whether max refinement attempts reached (implies refinement was attempted)
            context_hint: Optional hint about document collection

        Returns:
            Clarification message string

        Scenarios:
        1. max_attempts_reached=True: Refinement was tried but failed → _max_attempts_message
        2. No contexts found: Nothing retrieved → _no_results_message
        3. Direct CLARIFY (confidence < 0.5): Poor quality, skip refinement → _ambiguous_query_message

        TODO: Implement message routing
        Steps:
        1. If max_attempts_reached is True:
           - Return self._max_attempts_message(query, eval_result, context_hint)
        2. Else if eval_result.issues is empty OR "No contexts" in str(eval_result.issues):
           - Return self._no_results_message(query, context_hint)
        3. Else:
           - Return self._ambiguous_query_message(query, eval_result, context_hint)
        """
        if max_attempts_reached:
            return self._max_attempts_message(
                query, eval_result=eval_result, context_hint=context_hint
            )
        elif not eval_result.issues or "No contexts" in str(eval_result.issues):
            return self._no_results_message(query, context_hint=context_hint)
        else:
            return self._ambiguous_query_message(
                query, eval_result=eval_result, context_hint=context_hint
            )

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
        message_base = (
            f'I couldn\'t find any relevant information for: "{query}".\n\n'
            "Suggestions to improve your query:\n"
            "- The information might not be in the uploaded documents.\n"
            "- Try using different keywords.\n"
            "- Be more specific about what you're looking for.\n"
        )
        if context_hint:
            message_base += f"\nNote: I have searched in {context_hint}."

        return message_base

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
        message = f'Your query "{query}" returned low-quality results.\n\n'
        if eval_result.issues:
            message += "Issues found:\n"
            for issue in eval_result.issues[:3]:
                message += f"- {issue}\n"
        if eval_result.missing_aspects:
            message += (
                f"\nMissing keywords: {', '.join(eval_result.missing_aspects[:5])}\n"
            )
        message += (
            "\nSuggestions to improve your query:\n"
            "- Be more specific about what you're looking for.\n"
            "- Use different keywords.\n"
            "- Provide more context if possible.\n"
        )
        if context_hint:
            message += f"\nNote: I have searched in {context_hint}."

        return message

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
        message = (
            f"After multiple search attempts, I couldn't find highly relevant "
            f'information for: "{query}".\n\n'
            "The best results I found may not fully answer your question.\n\n"
            "Suggestions to improve your query:\n"
            "- Try rephrasing your question.\n"
            "- Break it into smaller, more specific questions.\n"
            "- Check if the information exists in the uploaded documents.\n"
        )
        if eval_result.missing_aspects:
            message += (
                f"\nKeywords not found: {', '.join(eval_result.missing_aspects[:5])}\n"
            )
        if context_hint:
            message += f"\nI have searched in: {context_hint}."

        return message


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
    from src.models.evaluation import (
        EvaluationResult,
        QualityLevel,
        RecommendationAction,
    )

    print("=" * 70)
    print("CLARIFICATION HELPER TEST")
    print("=" * 70)

    helper = ClarificationHelper(openai_client=None)
    print("[OK] ClarificationHelper created")

    # Test 1: No results message
    print("\n" + "-" * 70)
    print("Test 1: No Results Message")
    print("-" * 70)
    message = helper._no_results_message(
        "What is quantum physics?", context_hint="HR documents"
    )
    print(message)

    # Test 2: Ambiguous query message
    print("\n" + "-" * 70)
    print("Test 2: Ambiguous Query Message")
    print("-" * 70)
    mock_eval = EvaluationResult(
        quality=QualityLevel.POOR,
        confidence=0.2,
        coverage=0.1,
        recommendation=RecommendationAction.CLARIFY,
        reasoning="Poor quality",
        issues=["Low average relevance score: 0.20", "Poor keyword match: 0.10"],
        missing_aspects=["apply", "application", "process", "submit"],
        relevance_scores=[0.1, 0.2],
        metrics={},
    )
    message = helper._ambiguous_query_message(
        "How do I apply?", mock_eval, context_hint="company policies"
    )
    print(message)

    # Test 3: Max attempts message
    print("\n" + "-" * 70)
    print("Test 3: Max Attempts Message")
    print("-" * 70)
    message = helper._max_attempts_message(
        "Tell me about xyz", mock_eval, context_hint="company documents"
    )
    print(message)

    # Test 4: generate_clarification routing
    print("\n" + "-" * 70)
    print("Test 4: generate_clarification Routing")
    print("-" * 70)

    # Test max_attempts_reached=True → _max_attempts_message
    msg1 = helper.generate_clarification(
        query="test query",
        eval_result=mock_eval,
        max_attempts_reached=True,
    )
    print(
        f"  max_attempts_reached=True: {'_max_attempts_message' if 'multiple search attempts' in msg1 else 'WRONG'}"
    )

    # Test empty issues → _no_results_message
    empty_eval = EvaluationResult(
        quality=QualityLevel.POOR,
        confidence=0.1,
        coverage=0.0,
        recommendation=RecommendationAction.CLARIFY,
        reasoning="No contexts",
        issues=[],
        missing_aspects=[],
        relevance_scores=[],
        metrics={},
    )
    msg2 = helper.generate_clarification(
        query="test query",
        eval_result=empty_eval,
        max_attempts_reached=False,
    )
    print(f"  empty issues: {'_no_results_message' if 'couldn' in msg2 else 'WRONG'}")

    # Test with issues present → _ambiguous_query_message
    msg3 = helper.generate_clarification(
        query="test query",
        eval_result=mock_eval,
        max_attempts_reached=False,
    )
    print(
        f"  issues present: {'_ambiguous_query_message' if 'low-quality results' in msg3 else 'WRONG'}"
    )

    print("  [OK] Routing logic verified")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
