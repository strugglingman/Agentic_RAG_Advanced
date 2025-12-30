"""
LangGraph agent state definition.

This defines all possible states the agent can be in during execution.
State is split into:
- AgentState: Serializable, modified by nodes, checkpointed
- RuntimeContext: Non-serializable, read-only, passed separately
"""

from operator import add
from typing import TypedDict, List, Optional, Annotated, Sequence, Any, Tuple
from langchain_core.messages import BaseMessage, HumanMessage


class RuntimeContext(TypedDict):
    """
    Runtime context for the LangGraph agent.

    These are non-serializable objects passed to nodes but NOT checkpointed.
    Nodes should treat these as read-only.
    """

    vector_db: Optional[Any]  # VectorDB instance for retrieval
    openai_client: Optional[Any]  # OpenAI client instance
    dept_id: Optional[str]  # Department ID for filtering
    user_id: Optional[str]  # User ID for filtering
    request_data: Optional[dict]  # Original request payload
    conversation_history: Optional[List[dict]]  # Pre-loaded conversation history
    file_service: Optional[Any]  # FileService for file operations (async)


class AgentState(TypedDict):
    """
    State schema for the LangGraph agent.

    This is the single source of truth for agent execution state.
    All nodes read from and write to this state.
    All fields must be JSON-serializable for checkpointing.
    """

    # Core
    messages: Annotated[Sequence[BaseMessage], add]  # Conversation history
    query: str  # Original user query

    # Planning
    plan: Optional[List[str]]  # Step-by-step execution plan
    current_step: int  # Which plan step we're executing

    # Retrieval
    retrieved_docs: List[dict]  # Documents from ChromaDB
    evaluation_result: Optional[dict]  # Serialized EvaluationResult (was EvaluationResult object)

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


def create_runtime_context(
    vector_db: Optional[Any] = None,
    openai_client: Optional[Any] = None,
    dept_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_data: Optional[dict] = None,
    conversation_history: Optional[List[dict]] = None,
    file_service: Optional[Any] = None,
) -> RuntimeContext:
    """
    Factory function to create runtime context.

    Args:
        vector_db: VectorDB instance for retrieval
        openai_client: OpenAI client instance
        dept_id: Department ID for filtering
        user_id: User ID for filtering
        request_data: Original request payload
        conversation_history: Pre-loaded conversation history
        file_service: FileService for file operations (async)

    Returns:
        RuntimeContext with non-serializable objects
    """
    return RuntimeContext(
        vector_db=vector_db,
        openai_client=openai_client,
        dept_id=dept_id,
        user_id=user_id,
        request_data=request_data,
        conversation_history=conversation_history or [],
        file_service=file_service,
    )


def create_initial_state(query: str) -> AgentState:
    """
    Factory function to create initial agent state.

    Args:
        query: User's question

    Returns:
        Initial agent state (serializable, checkpointable)
    """
    return AgentState(
        # Core
        messages=[HumanMessage(content=query)],
        query=query,
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
    )


def create_initial_state_with_context(
    query: str,
    vector_db: Optional[Any] = None,
    openai_client: Optional[Any] = None,
    dept_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_data: Optional[dict] = None,
    conversation_history: Optional[List[dict]] = None,
    file_service: Optional[Any] = None,
) -> Tuple[AgentState, RuntimeContext]:
    """
    Factory function to create both initial state and runtime context.

    This is a convenience function that creates both in one call.

    Args:
        query: User's question
        vector_db: VectorDB instance for retrieval
        openai_client: OpenAI client instance
        dept_id: Department ID for filtering
        user_id: User ID for filtering
        request_data: Original request payload
        conversation_history: Pre-loaded conversation history
        file_service: FileService for file operations (async)

    Returns:
        Tuple of (AgentState, RuntimeContext)
    """
    state = create_initial_state(query)
    runtime = create_runtime_context(
        vector_db=vector_db,
        openai_client=openai_client,
        dept_id=dept_id,
        user_id=user_id,
        request_data=request_data,
        conversation_history=conversation_history,
        file_service=file_service,
    )
    return state, runtime
