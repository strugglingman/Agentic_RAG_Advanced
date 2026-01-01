"""
LangGraph agent graph builder.

This file constructs the state machine for the agentic RAG system.
"""

from typing import Optional
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from src.services.langgraph_state import AgentState, RuntimeContext
from src.services.langgraph_nodes import (
    create_plan_node,
    create_retrieve_node,
    create_reflect_node,
    create_refine_node,
    create_generate_node,
    create_verify_node,
    error_handler_node,
    create_tool_calculator_node,
    create_tool_web_search_node,
    create_tool_download_file_node,
    create_tool_create_documents_node,
    create_tool_send_email_node,
    create_direct_answer_node,
)
from src.services.langgraph_routing import (
    route_after_reflection,
    route_after_planning,
    should_continue,
)
from src.config.settings import Config

logger = logging.getLogger(__name__)

# Module-level checkpointer instance for persistence across requests
_default_checkpointer: Optional[BaseCheckpointSaver] = None


def get_checkpointer() -> BaseCheckpointSaver:
    """
    Get or create the default checkpointer instance.

    Returns:
        BaseCheckpointSaver instance (MemorySaver by default)
    """
    global _default_checkpointer
    if _default_checkpointer is None:
        _default_checkpointer = MemorySaver()
    return _default_checkpointer


def set_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    """
    Set a custom checkpointer (e.g., SqliteSaver, PostgresSaver).

    Args:
        checkpointer: Custom checkpointer instance
    """
    global _default_checkpointer
    _default_checkpointer = checkpointer


def build_langgraph_agent(
    runtime: RuntimeContext,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """
    Build and compile the LangGraph agent.

    Args:
        runtime: Runtime context with non-serializable objects (vector_db, openai_client, etc.)
        checkpointer: Optional checkpointer for state persistence. If None, uses default MemorySaver.

    Returns:
        Compiled graph ready for execution
    """

    # Initialize graph with state schema
    graph = StateGraph(AgentState)

    # ==================== ADD NODES ====================
    # Each node is created via factory function with runtime context bound via closure

    graph.add_node("plan", create_plan_node(runtime))
    graph.add_node("retrieve", create_retrieve_node(runtime))
    graph.add_node("reflect", create_reflect_node(runtime))
    graph.add_node("refine", create_refine_node(runtime))
    graph.add_node("generate", create_generate_node(runtime))
    graph.add_node("verify", create_verify_node(runtime))
    graph.add_node("tool_calculator", create_tool_calculator_node(runtime))
    graph.add_node("tool_web_search", create_tool_web_search_node(runtime))
    graph.add_node("tool_download_file", create_tool_download_file_node(runtime))
    graph.add_node("tool_create_documents", create_tool_create_documents_node(runtime))
    graph.add_node("tool_send_email", create_tool_send_email_node(runtime))
    graph.add_node("direct_answer", create_direct_answer_node(runtime))
    graph.add_node("error", error_handler_node)  # No runtime needed

    # ==================== SET ENTRY POINT ====================

    graph.set_entry_point("plan")

    # ==================== ADD EDGES ====================

    # After planning, route to appropriate node based on first step
    graph.add_conditional_edges(
        "plan",
        route_after_planning,
        {
            "direct_answer": "direct_answer",
            "retrieve": "retrieve",
            "tool_calculator": "tool_calculator",
            "tool_web_search": "tool_web_search",
            "tool_download_file": "tool_download_file",
            "tool_create_documents": "tool_create_documents",
            "tool_send_email": "tool_send_email",
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

    # After file/email tools, go directly to verify (no generation needed - action complete)
    # These tools produce direct results (download links, confirmation messages)
    graph.add_edge("tool_download_file", "verify")
    graph.add_edge("tool_create_documents", "verify")
    graph.add_edge("tool_send_email", "verify")

    # After direct answer, verify the answer directly
    graph.add_edge("direct_answer", "verify")

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

    # Use provided checkpointer or default MemorySaver
    # Checkpointing now works because AgentState is fully serializable
    # (runtime objects are passed via closure, not stored in state)
    if Config.CHECKPOINT_ENABLED:
        active_checkpointer = (
            checkpointer if checkpointer is not None else get_checkpointer()
        )
        logger.info(
            f"[DEBUG] Using checkpointer used in langgraph_builder: {type(active_checkpointer).__name__}"
        )
    else:
        active_checkpointer = None
        logger.info("[DEBUG] Checkpointing disabled in langgraph_builder")

    compiled_graph = (
        graph.compile(active_checkpointer) if active_checkpointer else graph.compile()
    )

    return compiled_graph


# ==================== VISUALIZATION ====================


def visualize_graph():
    """
    Generate Mermaid diagram of the graph.

    Run this to see the state machine structure:
    python -c "from src.services.langgraph_builder import visualize_graph; print(visualize_graph())"
    """
    from src.services.langgraph_state import create_runtime_context

    # Create dummy runtime for visualization (nodes won't be executed)
    dummy_runtime = create_runtime_context()
    graph = build_langgraph_agent(dummy_runtime)
    return graph.get_graph().draw_mermaid()
