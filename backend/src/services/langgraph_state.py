"""
LangGraph agent state definition.

This defines all possible states the agent can be in during execution.
"""

from operator import add
from typing import TypedDict, List, Optional, Annotated, Sequence, Any
from langchain_core.messages import BaseMessage, HumanMessage
from src.models.evaluation import EvaluationResult


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
    evaluation_result: Optional[EvaluationResult]  # Details from RetrievalEvaluator

    # Query Refinement
    original_query: str  # Store original for reference
    refined_query: Optional[str]  # After refinement
    refinement_count: int  # How many times we refined

    # Tool Execution
    tools_used: List[str]  # Track which tools were called
    tool_results: dict  # Store tool outputs

    # Per-Step Context Isolation (Option C implementation)
    step_contexts: dict  # Contexts indexed by step number {step: {type, docs/result, plan_step}}
    step_answers: List[dict]  # Verified answers per step [{step, question, answer, context_count}]

    # Generation
    draft_answer: Optional[str]  # Answer before verification
    final_answer: Optional[str]  # Verified answer with citations
    confidence: Optional[float]  # Agent's confidence (0-1)

    # Control Flow
    next_step: Optional[str]  # Which node to execute next
    iteration_count: int  # Prevent infinite loops
    error: Optional[str]  # Track errors for graceful handling

    # Runtime Context (passed during initialization, not modified by nodes)
    collection: Optional[Any]  # ChromaDB collection
    dept_id: Optional[str]  # Department ID for filtering
    user_id: Optional[str]  # User ID for filtering
    request_data: Optional[dict]  # Original request payload
    openai_client: Optional[Any]  # OpenAI client instance
    conversation_history: Optional[List[dict]]  # Pre-loaded conversation history


def create_initial_state(
    query: str,
    conversation_id: Optional[str] = None,
    collection: Optional[Any] = None,
    dept_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_data: Optional[dict] = None,
    openai_client: Optional[Any] = None,
    conversation_history: Optional[List[dict]] = None,
) -> AgentState:
    """
    Factory function to create initial state.

    Args:
        query: User's question
        conversation_id: Optional conversation ID for memory
        collection: ChromaDB collection for retrieval
        dept_id: Department ID for filtering
        user_id: User ID for filtering
        openai_client: OpenAI client instance
        conversation_history: Pre-loaded conversation history (avoids async issues in nodes)

    Returns:
        Initial agent state with runtime context
    """
    agent_state = AgentState(
        # Core
        messages=[HumanMessage(content=query)],
        query=query,
        conversation_id=conversation_id,
        # Planning
        plan=None,
        current_step=0,
        # Retrieval
        retrieved_docs=[],
        evaluation_result=None,
        # Query Refinement
        original_query=query,
        refined_query=None,
        refinement_count=0,
        # Tool Execution
        tools_used=[],
        tool_results={},
        # Per-Step Context Isolation
        step_contexts={},
        step_answers=[],
        # Generation
        draft_answer=None,
        final_answer=None,
        confidence=None,
        # Control Flow
        next_step="plan",
        iteration_count=0,
        error=None,
        # Runtime Context
        collection=collection,
        dept_id=dept_id,
        user_id=user_id,
        request_data=request_data,
        openai_client=openai_client,
        conversation_history=conversation_history or [],
    )

    return agent_state
