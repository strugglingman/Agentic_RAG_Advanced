"""Error handler node implementation."""

from typing import Dict, Any

from langchain_core.messages import AIMessage

from src.services.langgraph_state import AgentState

async def error_handler_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle errors gracefully.

    Args:
        state: Current agent state

    Returns:
        Updated state with error message
    """
    error_message = state.get("error", "An unknown error occurred.")
    return {
        "final_answer": f"Error: {error_message}",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": [AIMessage(content=f"Handled error: {error_message}")],
    }
