"""
Query refinement prompts for improving retrieval quality.

Note: Detailed refinement prompts are managed by QueryRefiner class.
This module provides high-level prompt coordination.
"""


class RefinementPrompts:
    """
    Prompts for query refinement based on evaluation feedback.

    Most refinement logic is handled by src.services.query_refiner.QueryRefiner
    which uses evaluation results to improve queries.

    This class provides complementary prompts for refinement-related tasks.
    """

    @staticmethod
    def get_refinement_context_note() -> str:
        """
        Get note about refinement context for logging/debugging.

        Returns:
            Context note string
        """
        return (
            "Refinement operates on step-specific query (not full query). "
            "Uses evaluation feedback to improve retrieval for that specific step."
        )

    @staticmethod
    def extract_step_query_note() -> str:
        """
        Get note about step query extraction logic.

        Returns:
            Extraction note string
        """
        return (
            "Extract query from plan step format: 'tool_name: query text'. "
            "If no colon, use full step text after removing action keywords."
        )
