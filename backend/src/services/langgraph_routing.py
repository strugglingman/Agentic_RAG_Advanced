"""
LangGraph conditional routing functions.

These functions decide which node to execute next based on state.
"""

from src.services.langgraph_state import AgentState
from src.models.evaluation import EvaluationResult, RecommendationAction
from src.config.settings import Config


def route_after_planning(state: AgentState) -> str:
    """
    Decide what to do after planning.

    Args:
        state: Current agent state

    Returns:
        Next node name: "retrieve", "tool_calculator", "tool_web_search", "generate", or "error"
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if not plan:
        return "error"

    # Check bounds
    if current_step >= len(plan):
        return "error"

    # Check current step to decide which node
    step = plan[current_step].lower()

    # IMPORTANT: Check specific tools BEFORE generic keywords to avoid false matches

    # Web search keywords (check BEFORE generic "search")
    if "web_search" in step or "web search" in step or "internet" in step or "online" in step or "google" in step:
        return "tool_web_search"

    # Calculator/computation keywords
    if "calculate" in step or "calculator" in step or "compute" in step or "math" in step or "sum" in step:
        return "tool_calculator"

    # Document retrieval keywords (after checking web_search to avoid "search" collision)
    if "retrieve" in step or "search_document" in step or "search document" in step or "find" in step or "document" in step:
        return "retrieve"

    # Generic "search" fallback to retrieve (for backward compatibility)
    if "search" in step:
        return "retrieve"

    # Default to generate if no tool needed
    if "answer" in step or "generate" in step or "respond" in step:
        return "generate"

    # Fallback: if unclear, try retrieval first (safer default)
    return "retrieve"


def route_after_reflection(state: AgentState) -> str:
    """
    Decide what to do based on retrieval quality.

    Args:
        state: Current agent state

    Returns:
        Next node name: "generate", "refine", "tool_web_search", or "error"
    """
    # Safety check: prevent infinite loops
    iteration_count = state.get("iteration_count", 0)
    if iteration_count >= Config.LANGGRAPH_MAX_ITERATIONS:
        return "error"

    evaluation_result: EvaluationResult = state.get("evaluation_result", None)
    if not evaluation_result:
        return "error"

    recommendation = evaluation_result.recommendation
    refinement_count = state.get("refinement_count", 0)

    # Max refinement attempts to prevent infinite loops
    if (
        refinement_count >= Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS
        and recommendation == RecommendationAction.REFINE
    ):
        if state.get("retrieved_docs"):
            return "generate"  # Use what we have
        else:
            return "error"  # No results after 3 tries

    if recommendation == RecommendationAction.ANSWER:
        return "generate"
    elif recommendation == RecommendationAction.REFINE:
        return "refine"
    elif recommendation == RecommendationAction.EXTERNAL:
        return "tool_web_search"
    elif recommendation == RecommendationAction.CLARIFY:
        # User needs to clarify the query - generate with clarification message
        return "generate"
    else:
        # Fallback for any unexpected recommendation
        return "generate"


def should_continue(state: AgentState) -> str:
    """
    Decide if we should continue execution or end.

    Args:
        state: Current agent state

    Returns:
        "continue" or "end"
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    iteration_count = state.get("iteration_count", 0)

    # Safety: max iterations to prevent infinite loops
    if iteration_count >= Config.LANGGRAPH_MAX_ITERATIONS:
        return "end"

    # If final_answer is set, end immediately (either CLARIFY or actual answer)
    # CLARIFY needs user input, so don't continue with remaining plan steps
    if state.get("final_answer"):
        return "end"

    # Check if all plan steps completed
    if current_step >= len(plan):
        return "end"

    # Otherwise continue
    return "continue"
