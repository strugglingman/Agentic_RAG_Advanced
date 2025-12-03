"""
LangGraph conditional routing functions.

These functions decide which node to execute next based on state.
"""

from src.services.langgraph_state import AgentState


def route_after_planning(state: AgentState) -> str:
    """
    Decide what to do after planning.

    Args:
        state: Current agent state

    Returns:
        Next node name: "retrieve", "generate", or "error"
    """
    plan = state.get("plan", [])

    if not plan:
        return "error"

    # Check if first step requires retrieval
    first_step = plan[0].lower()
    if "retrieve" in first_step or "search" in first_step or "find" in first_step:
        return "retrieve"
    else:
        return "generate"


def route_after_reflection(state: AgentState) -> str:
    """
    Decide what to do based on retrieval quality.

    Args:
        state: Current agent state

    Returns:
        Next node name: "generate", "refine", "retrieve", or "error"
    """
    recommendation = state.get("retrieval_recommendation", "ANSWER")
    refinement_count = state.get("refinement_count", 0)

    # Max 3 refinements to prevent infinite loops
    if refinement_count >= 3:
        if state.get("retrieved_docs"):
            return "generate"  # Use what we have
        else:
            return "error"  # No results after 3 tries

    if recommendation == "ANSWER":
        return "generate"
    elif recommendation == "REFINE":
        return "refine"
    elif recommendation == "EXTERNAL":
        # TODO: Future enhancement - trigger web search tool
        return "generate"
    else:
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

    # Safety: max 10 iterations to prevent infinite loops
    if iteration_count >= 10:
        return "end"

    # Check if all plan steps completed
    if current_step >= len(plan):
        return "end"

    # Otherwise continue
    return "continue"
