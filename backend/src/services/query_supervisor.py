"""
Query Supervisor - Routes queries to appropriate execution engine.

This module classifies incoming queries and routes them to either:
1. AgentService (ReAct pattern) - for simple, single-step queries
2. LangGraph (Plan-Execute pattern) - for complex, multi-step queries

Architecture:
    User Query → LLM Classifier → Route Decision → Execute → Response

Usage:
    supervisor = QuerySupervisor(openai_client)
    answer, contexts = await supervisor.process_query(query, context)
"""

from enum import Enum
from typing import Tuple, List, Dict, Any
import json
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
import psycopg
from src.config.settings import Config
from src.services.langgraph_state import (
    create_initial_state,
    create_runtime_context,
)
from src.services.agent_service import AgentService
from src.services.langgraph_builder import build_langgraph_agent, set_checkpointer


class ExecutionRoute(str, Enum):
    """Execution engine options."""

    AGENT_SERVICE = "agent_service"  # ReAct pattern - simple queries
    LANGGRAPH = "langgraph"  # Plan-Execute pattern - complex queries


class QuerySupervisor:
    """
    Supervisor that classifies queries and routes to appropriate execution engine.

    Attributes:
        openai_client: OpenAI client for LLM classification
        agent_service: ReAct-based agent for simple queries
        langgraph_agent: Plan-Execute agent for complex queries
    """

    def __init__(self, openai_client):
        """
        Initialize the supervisor.

        Args:
            openai_client: OpenAI client instance
        """
        self.openai_client = openai_client
        self.agent_service = AgentService(openai_client=openai_client)
        # Note: LangGraph agent is built per-request with RuntimeContext

        # Initialize PostgreSQL checkpointer (cached for all requests)
        self._checkpointer = None
        self._connection_pool = None  # Store connection pool for cleanup
        if Config.CHECKPOINT_POSTGRES_DATABASE_URL and PostgresSaver and psycopg:
            try:
                # Setup tables first (needs autocommit for CREATE INDEX CONCURRENTLY)
                with psycopg.connect(
                    Config.CHECKPOINT_POSTGRES_DATABASE_URL, autocommit=True
                ) as setup_conn:
                    temp_saver = PostgresSaver(setup_conn)
                    temp_saver.setup()

                # Create connection pool with autocommit for checkpoint operations
                # This ensures each checkpoint write is committed immediately
                connection_kwargs = {
                    "autocommit": True,
                    "prepare_threshold": 0,  # Disable prepared statements for pgbouncer compatibility
                    "row_factory": dict_row,  # Required: PostgresSaver accesses rows as dicts
                }
                self._connection_pool = ConnectionPool(
                    conninfo=Config.CHECKPOINT_POSTGRES_DATABASE_URL,
                    max_size=20,
                    kwargs=connection_kwargs,
                )
                self._checkpointer = PostgresSaver(self._connection_pool)

                print("✓ PostgreSQL checkpointer initialized with connection pool")
            except Exception as e:
                print(f"⚠ Could not initialize PostgreSQL checkpointer: {e}")
                print("  Falling back to in-memory checkpointer")

    def close(self):
        """Close connection pool when shutting down."""
        if self._connection_pool:
            try:
                self._connection_pool.close()
                print("✓ PostgreSQL connection pool closed")
            except Exception as e:
                print(f"⚠ Error closing connection pool: {e}")

    async def process_query(
        self, query: str, context: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main entry point - classify and route query to appropriate engine.

        Args:
            query: User's question
            context: Execution context (collection, dept_id, user_id, etc.)

        Returns:
            Tuple of (answer, contexts)
        """
        route = self._classify_query(query)
        if route == ExecutionRoute.AGENT_SERVICE:
            return await self._execute_agent_service(query, context)
        else:
            return await self._execute_langgraph(query, context)

    def _classify_query(self, query: str) -> ExecutionRoute:
        """
        Use LLM to classify query complexity.

        Classification criteria:
        - AGENT_SERVICE: Single question, direct lookup, simple calculation
        - LANGGRAPH: Multi-step reasoning, multiple tools, comparison, aggregation

        Args:
            query: User's question

        Returns:
            ExecutionRoute enum value
        """
        prompt = self._get_classification_prompt(query)
        response = self.openai_client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
            response_format={"type": "json_object"},
        )
        if not response.choices or len(response.choices) == 0:
            raise ValueError("LLM classification returned no choices")

        content = response.choices[0].message.content.strip()
        classification_data = json.loads(content)
        print("----------- Query Classification, Decide what way to route ----------")
        print(classification_data)
        classification = classification_data.get("classification", "simple")

        return (
            ExecutionRoute.LANGGRAPH
            if classification == "complex" and Config.USE_LANGGRAPH
            else ExecutionRoute.AGENT_SERVICE
        )

    def _get_classification_prompt(self, query: str) -> str:
        """
        Build the classification prompt for LLM.

        Args:
            query: User's question

        Returns:
            Formatted prompt string
        """
        return f"""You are a query complexity classifier for a RAG (Retrieval-Augmented Generation) system.

    Analyze the user's query and classify it into one of two categories:

    ## Categories

    **SIMPLE** - Use ReAct pattern (single-step, fast):
    - Direct factual questions with single answer
    - Simple lookups from documents
    - Single calculations
    - Questions that can be answered from one source
    - No comparison or aggregation needed

    **COMPLEX** - Use Plan-Execute pattern (multi-step, thorough):
    - Questions requiring multiple information sources
    - Comparison between different items/concepts
    - Aggregation or synthesis of multiple facts
    - Multi-part questions (contains "and", "also", "as well as")
    - Questions requiring calculations on retrieved data
    - Questions needing both internal documents AND external web search
    - Analysis or reasoning across multiple steps

    ## Examples

    SIMPLE queries:
    - "What is the company vacation policy?"
    - "How much revenue did we have in Q3?"
    - "What is 15% of 2500?"
    - "Who is the CEO?"

    COMPLEX queries:
    - "Compare our Q3 and Q4 revenue and calculate the growth rate"
    - "What are our top 3 products and how do they compare to competitors?"
    - "Summarize the vacation policy and calculate how many days I have left"
    - "Find the latest industry trends and compare with our strategy"
    - "What is our budget for marketing and how does it compare to last year?"

    ## User Query
    {query}

    ## Response Format
    Respond with ONLY a JSON object:
    {{"classification": "simple" | "complex", "reasoning": "brief explanation"}}
    """

    async def _execute_agent_service(
        self, query: str, context: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute query using AgentService (ReAct pattern).

        Args:
            query: User's question
            context: Execution context (must include pre-loaded conversation_history)

        Returns:
            Tuple of (answer, contexts)
        """
        # Use pre-loaded history from context (loaded in chat.py, avoids async issues)
        messages_history = context.get("conversation_history", [])
        final_answer, retrieved_contexts = self.agent_service.run(
            query, context, messages_history=messages_history
        )

        return final_answer, retrieved_contexts

    async def _execute_langgraph(
        self, query: str, context: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute query using LangGraph (Plan-Execute pattern).

        Args:
            query: User's question
            context: Execution context (should include 'thread_id' for checkpointing)

        Returns:
            Tuple of (answer, contexts)
        """
        # Create runtime context with non-serializable objects
        runtime = create_runtime_context(
            collection=context.get("collection"),
            openai_client=self.openai_client,
            dept_id=context.get("dept_id", ""),
            user_id=context.get("user_id", ""),
            request_data=context.get("request_data", {}),
            conversation_history=context.get("conversation_history", []),
        )

        # Build graph with runtime context bound to nodes, using cached checkpointer
        print(
            f"[DEBUG] Using checkpointer: {type(self._checkpointer).__name__ if self._checkpointer else 'None (will use MemorySaver)'}"
        )
        langgraph_agent = build_langgraph_agent(
            runtime, checkpointer=self._checkpointer
        )

        # Create serializable initial state (no runtime objects)
        agent_state = self._build_langgraph_initial_state(query)

        # Use conversation_id as thread_id for checkpointing (enables multi-turn conversations and state recovery)
        thread_id = (
            context.get("thread_id") or context.get("conversation_id") or "default"
        )
        config = {"configurable": {"thread_id": thread_id}}

        final_state = langgraph_agent.invoke(agent_state, config=config)
        return self._extract_langgraph_results(final_state)

    def _build_langgraph_initial_state(self, query: str) -> Dict[str, Any]:
        """
        Build initial state for LangGraph execution.

        Args:
            query: User's question

        Returns:
            Initial state dictionary matching AgentState schema (serializable)
        """
        # create_initial_state now only takes query - runtime objects are in RuntimeContext
        agent_state = create_initial_state(query=query)
        return {**agent_state}

    def _extract_langgraph_results(
        self, final_state: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract answer and contexts from LangGraph final state.

        Args:
            final_state: Final state after graph execution

        Returns:
            Tuple of (answer, contexts)
        """
        final_answer = final_state.get("final_answer") or "No answer generated."

        # Extract contexts from step_contexts (per-step isolation)
        # This ensures we return the actual contexts used in the answer
        step_contexts = final_state.get("step_contexts", {})
        all_contexts = []

        for step_num in sorted(step_contexts.keys()):
            step_ctx = step_contexts[step_num]
            ctx_type = step_ctx.get("type", "")

            if ctx_type == "retrieval":
                # Add retrieved documents from this step
                docs = step_ctx.get("docs", [])
                all_contexts.extend(docs)
            elif ctx_type == "tool":
                # Add tool result as a context entry
                tool_context = {
                    "tool_name": step_ctx.get("tool_name", "unknown"),
                    "result": step_ctx.get("result", ""),
                    "args": step_ctx.get("args", {}),
                    "step": step_num,
                }
                all_contexts.append(tool_context)

        # Fallback: if step_contexts is empty, use legacy approach
        if not all_contexts:
            retrieved_docs = final_state.get("retrieved_docs", [])
            tool_results_dict = final_state.get("tool_results", {})
            tool_results_list = []
            for results in tool_results_dict.values():
                tool_results_list.extend(results)
            all_contexts = retrieved_docs + tool_results_list

        return final_answer, all_contexts


# ==================== FACTORY FUNCTION ====================


def create_supervisor(openai_client) -> QuerySupervisor:
    """
    Factory function to create a configured QuerySupervisor.

    Args:
        openai_client: OpenAI client instance

    Returns:
        Configured QuerySupervisor instance
    """
    return QuerySupervisor(openai_client=openai_client)
