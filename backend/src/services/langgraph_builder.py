"""
LangGraph agent graph builder.

This file constructs the state machine for the agentic RAG system.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.services.langgraph_state import AgentState
from src.services.langgraph_nodes import (
    plan_node,
    retrieve_node,
    reflect_node,
    refine_node,
    generate_node,
    verify_node,
    error_handler_node,
)
from .langgraph_routing import (
    route_after_reflection,
    route_after_planning,
    should_continue,
)


def build_langgraph_agent():
    """
    Build and compile the LangGraph agent.

    Returns:
        Compiled graph ready for execution
    """

    # Initialize graph with state schema
    graph = StateGraph(AgentState)

    # ==================== ADD NODES ====================
    # Each node is a function that takes state and returns updated state

    # TODO: Add nodes

    # ==================== SET ENTRY POINT ====================

    # TODO: Set entry point

    # ==================== ADD EDGES ====================

    # TODO: Add conditional edges after planning

    # TODO: Add edge from retrieve to reflect

    # TODO: Add conditional edges after reflection

    # TODO: Add edge from refine to retrieve

    # TODO: Add edge from generate to verify

    # TODO: Add conditional edges after verification

    # TODO: Add edge from error to END

    # ==================== COMPILE GRAPH ====================

    # Add checkpointing for memory and resumability
    checkpointer = MemorySaver()

    compiled_graph = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[],  # Can add nodes to interrupt before (for human-in-loop)
        interrupt_after=[],  # Can add nodes to interrupt after
    )

    return compiled_graph


# ==================== VISUALIZATION ====================


def visualize_graph():
    """
    Generate Mermaid diagram of the graph.

    Run this to see the state machine structure:
    python -c "from src.services.langgraph_builder import visualize_graph; print(visualize_graph())"
    """
    graph = build_langgraph_agent()
    return graph.get_graph().draw_mermaid()
