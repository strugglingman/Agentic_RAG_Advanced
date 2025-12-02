"""
LangGraph agent state definition.

This defines all possible states the agent can be in during execution.
"""

from typing import TypedDict, List, Optional, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from operator import add


class AgentState(TypedDict):
    """
    State schema for the LangGraph agent.

    This is the single source of truth for agent execution state.
    All nodes read from and write to this state.
    """

    # Core
    messages: Annotated[Sequence[BaseMessage], add]  # Conversation history
    query: str  # Original user query
    conversation_id: Optional[str]  # For memory persistence

    # Planning
    plan: Optional[List[str]]  # Step-by-step execution plan
    current_step: int  # Which plan step we're executing

    # Retrieval
    retrieved_docs: List[dict]  # Documents from ChromaDB
    retrieval_quality: Optional[float]  # Self-reflection score
    retrieval_recommendation: Optional[str]  # ANSWER, REFINE, EXTERNAL, CLARIFY

    # Query Refinement
    original_query: str  # Store original for reference
    refined_query: Optional[str]  # After refinement
    refinement_count: int  # How many times we refined

    # Tool Execution
    tools_used: List[str]  # Track which tools were called
    tool_results: dict  # Store tool outputs

    # Generation
    draft_answer: Optional[str]  # Answer before verification
    final_answer: Optional[str]  # Verified answer with citations
    confidence: Optional[float]  # Agent's confidence (0-1)

    # Control Flow
    next_step: Optional[str]  # Which node to execute next
    iteration_count: int  # Prevent infinite loops
    error: Optional[str]  # Track errors for graceful handling


def create_initial_state(
    query: str, conversation_id: Optional[str] = None
) -> AgentState:
    """
    Factory function to create initial state.

    Args:
        query: User's question
        conversation_id: Optional conversation ID for memory

    Returns:
        Initial agent state
    """
    agent_state = AgentState(
        messages=[HumanMessage(content=query)],
        query=query,
        conversation_id=conversation_id,
        plan=None,
        current_step=0,
        retrieved_docs=[],
        retrieval_quality=None,
        retrieval_recommendation=None,
        original_query=query,
        refined_query=None,
        refinement_count=0,
        tools_used=[],
        tool_results={},
        draft_answer=None,
        final_answer=None,
        confidence=None,
        next_step="plan",
        iteration_count=0,
        error=None,
    )

    return agent_state
