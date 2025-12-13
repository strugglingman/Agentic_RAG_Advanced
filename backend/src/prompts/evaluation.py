"""
Evaluation and reflection prompts for quality assessment.

Note: Detailed evaluation prompts are managed by RetrievalEvaluator class.
This module provides high-level prompt coordination.
"""


class EvaluationPrompts:
    """
    Prompts for retrieval quality evaluation and self-reflection.

    Most evaluation logic is handled by src.services.retrieval_evaluator.RetrievalEvaluator
    which has its own comprehensive prompt system based on ReflectionMode (FAST/BALANCED/THOROUGH).

    This class provides complementary prompts for evaluation-related tasks.
    """

    @staticmethod
    def get_evaluation_context_note() -> str:
        """
        Get note about evaluation context for logging/debugging.

        Returns:
            Context note string
        """
        return (
            "Evaluation uses step-specific query from plan, not full original query. "
            "This ensures reflection evaluates the specific retrieval task, not the overall question."
        )

    @staticmethod
    def fallback_reasoning() -> str:
        """
        Get fallback reasoning when evaluation fails.

        Returns:
            Default reasoning string
        """
        return "Reflection failed due to error, proceeding with default assessment."
