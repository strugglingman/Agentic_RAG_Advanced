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


def route_after_reflection(state: AgentState) -> str:
    """
    Decide what to do based on retrieval quality.

    Args:
        state: Current agent state

    Returns:
        Next node name: "generate", "refine", "retrieve", or "error"
    """
    # TODO: Implement routing logic after reflection
    pass


def should_continue(state: AgentState) -> str:
    """
    Decide if we should continue execution or end.

    Args:
        state: Current agent state

    Returns:
        "continue" or "end"
    """
    # TODO: Implement continuation logic
    pass
