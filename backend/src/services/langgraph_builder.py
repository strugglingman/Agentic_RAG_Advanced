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
    tool_calculator_node,
    tool_web_search_node,
)
from src.services.langgraph_routing import (
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

    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("refine", refine_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("tool_calculator", tool_calculator_node)
    graph.add_node("tool_web_search", tool_web_search_node)
    graph.add_node("error", error_handler_node)

    # ==================== SET ENTRY POINT ====================

    graph.set_entry_point("plan")

    # ==================== ADD EDGES ====================

    # After planning, route to appropriate node based on first step
    graph.add_conditional_edges(
        "plan",
        route_after_planning,
        {
            "retrieve": "retrieve",
            "tool_calculator": "tool_calculator",
            "tool_web_search": "tool_web_search",
            "generate": "generate",
            "error": "error",
        },
    )

    # After retrieve, always reflect on quality
    graph.add_edge("retrieve", "reflect")

    # After reflection, route based on evaluation
    graph.add_conditional_edges(
        "reflect",
        route_after_reflection,
        {
            "generate": "generate",
            "refine": "refine",
            "tool_web_search": "tool_web_search",
            "error": "error",
        },
    )

    # After refine, go back to retrieve with refined query
    graph.add_edge("refine", "retrieve")

    # After generate, verify citations
    graph.add_edge("generate", "verify")

    # After tool execution, generate answer with tool results
    graph.add_edge("tool_calculator", "generate")
    graph.add_edge("tool_web_search", "generate")

    # After verification, check if more plan steps remain
    # This is the key change for Plan-Execute pattern:
    # - If more steps: continue to next step (route back through planning)
    # - If no more steps: end (final_answer is set)
    graph.add_conditional_edges(
        "verify",
        should_continue,
        {
            "continue": "plan",  # More steps remain, go back to planning
            "end": END,  # All steps complete, final_answer is set
        },
    )

    # Error always ends
    graph.add_edge("error", END)

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
