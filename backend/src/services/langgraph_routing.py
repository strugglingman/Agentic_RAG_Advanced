"""
LangGraph conditional routing functions.

These functions decide which node to execute next based on state.
"""

from src.services.langgraph_state import AgentState
from src.models.evaluation import RecommendationAction
from src.config.settings import Config


def route_after_planning(state: AgentState) -> str:
    """
    Decide what to do after planning.

    Args:
        state: Current agent state

    Returns:
        Next node name: "retrieve", "tool_calculator", "tool_web_search",
        "tool_download_file", "tool_send_email", "tool_create_documents",
        "generate", or "error"
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    print(f"[ROUTE_AFTER_PLANNING] plan={plan}, current_step={current_step}")

    if not plan:
        print("[ROUTE_AFTER_PLANNING] No plan, returning error")
        return "error"

    # Check bounds
    if current_step >= len(plan):
        print(
            f"[ROUTE_AFTER_PLANNING] current_step {current_step} >= len(plan) {len(plan)}, returning error"
        )
        return "error"

    # Check current step to decide which node
    step = plan[current_step].lower()
    print(f"[ROUTE_AFTER_PLANNING] step={step}")

    # IMPORTANT: Check specific tools BEFORE generic keywords to avoid false matches
    # Direct answer keywords
    if "direct_answer" in step or "direct answer" in step:
        print("[ROUTE_AFTER_PLANNING] Matched direct_answer, returning direct_answer")
        return "direct_answer"

    # Download file keywords (check BEFORE generic "download")
    if (
        "download_file" in step
        or "download file" in step
        or "download from url" in step
        or "fetch file" in step
    ):
        print(
            "[ROUTE_AFTER_PLANNING] Matched download_file, returning tool_download_file"
        )
        return "tool_download_file"

    # Send email keywords
    if (
        "send_email" in step
        or "send email" in step
        or "email to" in step
        or "mail to" in step
    ):
        print("[ROUTE_AFTER_PLANNING] Matched send_email, returning tool_send_email")
        return "tool_send_email"

    # Create documents keywords (check BEFORE generic "create")
    if (
        "create_document" in step
        or "create document" in step
        or "generate document" in step
        or "create pdf" in step
        or "create docx" in step
        or "create csv" in step
        or "create xlsx" in step
        or "create file" in step
        or "write to file" in step
    ):
        print(
            "[ROUTE_AFTER_PLANNING] Matched create_documents, returning tool_create_documents"
        )
        return "tool_create_documents"

    # Web search keywords (check BEFORE generic "search")
    if (
        "web_search" in step
        or "web search" in step
        or "internet" in step
        or "online" in step
        or "google" in step
    ):
        print("[ROUTE_AFTER_PLANNING] Matched web_search, returning tool_web_search")
        return "tool_web_search"

    # Calculator/computation keywords
    if "calculate" in step or "calculator" in step or "compute" in step:
        print("[ROUTE_AFTER_PLANNING] Matched calculator, returning tool_calculator")
        return "tool_calculator"

    # Document retrieval keywords (after checking web_search to avoid "search" collision)
    if (
        "retrieve" in step
        or "search_document" in step
        or "search document" in step
        or "find in document" in step
    ):
        print("[ROUTE_AFTER_PLANNING] Matched retrieve, returning retrieve")
        return "retrieve"

    # Generic "search" fallback to retrieve (for backward compatibility)
    if "search" in step:
        print("[ROUTE_AFTER_PLANNING] Matched generic search, returning retrieve")
        return "retrieve"

    # Default to generate if no tool needed
    if "answer" in step or "generate" in step or "respond" in step:
        print("[ROUTE_AFTER_PLANNING] Matched generate, returning generate")
        return "generate"

    # Fallback: if unclear, try retrieval first (safer default)
    print("[ROUTE_AFTER_PLANNING] No match, fallback to retrieve")
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

    # evaluation_result is now stored as dict for serialization
    evaluation_result_dict = state.get("evaluation_result", None)
    if not evaluation_result_dict:
        return "error"

    # Access recommendation as string from dict, convert to enum for comparison
    recommendation_str = evaluation_result_dict.get("recommendation")
    if not recommendation_str:
        return "error"

    recommendation = RecommendationAction(recommendation_str)
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
