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
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import json
import logging
import re
import uuid
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import psycopg
from src.config.settings import Config
from src.services.langgraph_state import (
    create_initial_state,
    create_runtime_context,
)
from src.services.agent_service import AgentService
from src.services.langgraph_builder import build_langgraph_agent
from src.services.llm_client import chat_completion_json
from src.observability.metrics import (
    increment_active_queries,
    decrement_active_queries,
    increment_error,
    MetricsErrorType,
)

logger = logging.getLogger(__name__)


# ==================== HITL DATA STRUCTURES ====================


@dataclass
class HITLInterruptResult:
    """Result when graph is interrupted for human confirmation."""

    status: str  # "awaiting_confirmation"
    action: str  # "send_email", etc.
    thread_id: str  # Thread ID for resumption
    details: Dict[str, Any]  # Action-specific details for confirmation UI
    previous_steps: List[Dict[str, Any]]  # Results from completed steps


@dataclass
class QueryResult:
    """
    Result from process_query that can be either:
    - A complete answer (answer + contexts)
    - An HITL interrupt requiring confirmation
    """

    answer: str
    contexts: List[Dict[str, Any]]
    hitl_interrupt: Optional[HITLInterruptResult] = None

    @property
    def is_interrupted(self) -> bool:
        """Check if this result requires human confirmation."""
        return self.hitl_interrupt is not None


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

        # Initialize PostgreSQL checkpointer settings
        # We use AsyncPostgresSaver for async operations with ainvoke
        self._checkpoint_conn_string = None
        self._checkpoint_enabled = False

        if Config.CHECKPOINT_ENABLED:
            if Config.CHECKPOINT_POSTGRES_DATABASE_URL and psycopg:
                try:
                    # Setup tables first using sync connection (needs autocommit for CREATE INDEX CONCURRENTLY)
                    with psycopg.connect(
                        Config.CHECKPOINT_POSTGRES_DATABASE_URL, autocommit=True
                    ) as setup_conn:
                        temp_saver = PostgresSaver(setup_conn)
                        temp_saver.setup()

                    # Store connection string for async checkpointer creation
                    self._checkpoint_conn_string = (
                        Config.CHECKPOINT_POSTGRES_DATABASE_URL
                    )
                    self._checkpoint_enabled = True

                    print(
                        "[OK] PostgreSQL checkpoint tables initialized, async checkpointer ready"
                    )
                except Exception as e:
                    print(f"[WARN] Could not initialize PostgreSQL checkpointer: {e}")
                    print("  Falling back to in-memory checkpointer")

    def close(self):
        """Cleanup resources when shutting down."""
        # AsyncPostgresSaver manages its own connections via context manager
        pass

    async def _clear_checkpoint_for_thread(self, thread_id: str) -> None:
        """
        Clear checkpoint data for a thread before starting a NEW query.

        Why this is needed:
        - Checkpoints track state within a single workflow execution (for HITL)
        - Each new query should start fresh - conversation history is passed separately
        - Old checkpoint data can cause "invalid memory alloc" errors in PostgreSQL

        Note: Do NOT call this for resume operations (HITL confirm/cancel).

        Args:
            thread_id: The thread ID to clear checkpoints for
        """
        if not self._checkpoint_enabled or not self._checkpoint_conn_string:
            return

        try:
            async with asyncio.timeout(Config.LLM_CONNECT_TIMEOUT):
                async with await psycopg.AsyncConnection.connect(
                    self._checkpoint_conn_string
                ) as conn:
                    async with conn.cursor() as cur:
                        # Delete from checkpoint tables for this thread_id
                        await cur.execute(
                            "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                            (thread_id,),
                        )
                        await cur.execute(
                            "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                            (thread_id,),
                        )
                        await cur.execute(
                            "DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,)
                        )
                    await conn.commit()
                    logger.debug(f"[CHECKPOINT] Cleared old data for thread_id={thread_id}")
        except TimeoutError:
            logger.warning(
                "[CHECKPOINT] Clear timed out after %ds for thread_id=%s",
                Config.LLM_CONNECT_TIMEOUT, thread_id,
            )
            increment_error(MetricsErrorType.TIMEOUT)
        except Exception as e:
            logger.warning(
                f"[CHECKPOINT] Failed to clear for thread_id={thread_id}: {e}"
            )

    async def process_query(self, query: str, context: Dict[str, Any]) -> QueryResult:
        """
        Main entry point - classify and route query to appropriate engine.

        Args:
            query: User's question
            context: Execution context (vector_db, dept_id, user_id, etc.)

        Returns:
            QueryResult containing answer, contexts, and optional HITL interrupt
        """
        try:
            # Track active queries for observability
            logger.debug("[Process Query] Incrementing active queries counter")
            increment_active_queries()

            route = await self._classify_query(query)
            if route == ExecutionRoute.AGENT_SERVICE:
                answer, contexts = await self._execute_agent_service(query, context)
                return QueryResult(answer=answer, contexts=contexts)

            answer, contexts, hitl_interrupt = await self._execute_langgraph(
                query, context
            )
            return QueryResult(
                answer=answer, contexts=contexts, hitl_interrupt=hitl_interrupt
            )
        finally:
            # Decrement active queries counter
            logger.debug("[Process Query] Decrementing active queries counter")
            decrement_active_queries()

    async def resume_workflow(
        self, thread_id: str, context: Dict[str, Any], confirmed: bool = True
    ) -> QueryResult:
        """
        Resume an interrupted workflow after human confirmation.

        Args:
            thread_id: Thread ID of the interrupted workflow
            context: Execution context (same as process_query)
            confirmed: Whether user confirmed the action

        Returns:
            QueryResult with final answer or another HITL interrupt
        """
        if not confirmed:
            # User cancelled - return current state without executing pending action
            return await self._get_cancelled_result(thread_id, context)

        # Create runtime context for resumption
        runtime = create_runtime_context(
            vector_db=context.get("vector_db"),
            openai_client=self.openai_client,
            dept_id=context.get("dept_id", ""),
            user_id=context.get("user_id", ""),
            conversation_id=context.get("conversation_id", ""),
            request_data=context.get("request_data", {}),
            conversation_history=context.get("conversation_history", []),
            file_service=context.get("file_service"),
            available_files=context.get("available_files", []),
            attachment_file_ids=context.get("attachment_file_ids", []),
        )

        config = {"configurable": {"thread_id": thread_id}}

        # Use AsyncPostgresSaver context manager for async operations
        if self._checkpoint_enabled and self._checkpoint_conn_string:
            async with AsyncPostgresSaver.from_conn_string(
                self._checkpoint_conn_string
            ) as checkpointer:
                return await self._resume_with_checkpointer(
                    runtime, config, thread_id, checkpointer
                )
        else:
            return await self._resume_with_checkpointer(
                runtime, config, thread_id, None
            )

    async def _resume_with_checkpointer(
        self,
        runtime,
        config: Dict[str, Any],
        thread_id: str,
        checkpointer,
    ) -> QueryResult:
        """
        Resume workflow execution with the given checkpointer.
        """
        # Build graph with checkpointer
        langgraph_agent = build_langgraph_agent(runtime, checkpointer=checkpointer)

        try:
            # Track active queries for observability
            increment_active_queries()

            # Resume from checkpoint using astream - pass None to continue from saved state
            logger.info(f"[HITL] Resuming workflow with thread_id: {thread_id}")
            final_state = None
            async with asyncio.timeout(Config.AGENT_TIMEOUT):
                async for event in langgraph_agent.astream(
                    None, config=config, stream_mode="values"
                ):
                    final_state = event

            logger.info("[HITL] Resume stream completed, checking for interrupt...")

            # Check if interrupted again
            snapshot = await langgraph_agent.aget_state(config)
            if snapshot.next:
                pending_node = snapshot.next[0] if snapshot.next else None
                logger.info(
                    f"[HITL] Graph interrupted again before node: {pending_node}"
                )

                hitl_result = self._extract_hitl_details(
                    pending_node, snapshot.values, thread_id
                )
                if hitl_result:
                    answer, contexts = self._extract_langgraph_results(snapshot.values)
                    return QueryResult(
                        answer=answer, contexts=contexts, hitl_interrupt=hitl_result
                    )

            # Normal completion
            if final_state is None:
                final_state = snapshot.values
            answer, contexts = self._extract_langgraph_results(final_state)
            return QueryResult(answer=answer, contexts=contexts)

        except TimeoutError:
            logger.warning(
                "[HITL] Resume timed out after %ds for thread_id=%s",
                Config.AGENT_TIMEOUT,
                thread_id,
            )
            increment_error(MetricsErrorType.TIMEOUT)
            if final_state:
                answer, contexts = self._extract_langgraph_results(final_state)
                return QueryResult(
                    answer=f"{answer}\n\n(Response timed out after {Config.AGENT_TIMEOUT}s. "
                    f"This is a partial result.)",
                    contexts=contexts,
                )
            return QueryResult(
                answer=f"The request timed out after {Config.AGENT_TIMEOUT} seconds. "
                f"Please try a simpler question or try again later.",
                contexts=[],
            )
        except Exception as e:
            logger.error(f"[HITL] Error resuming workflow: {e}")
            raise
        finally:
            # Decrement active queries counter
            decrement_active_queries()

    async def _get_cancelled_result(
        self, thread_id: str, context: Dict[str, Any]
    ) -> QueryResult:
        """
        Build result when user cancels a pending action.

        Returns completed step results without executing the cancelled action.
        """
        # Create runtime to get graph
        runtime = create_runtime_context(
            vector_db=context.get("vector_db"),
            openai_client=self.openai_client,
            dept_id=context.get("dept_id", ""),
            user_id=context.get("user_id", ""),
            conversation_id=context.get("conversation_id", ""),
            request_data=context.get("request_data", {}),
            conversation_history=context.get("conversation_history", []),
            file_service=context.get("file_service"),
            available_files=context.get("available_files", []),
            attachment_file_ids=context.get("attachment_file_ids", []),
        )

        config = {"configurable": {"thread_id": thread_id}}

        # Use AsyncPostgresSaver context manager for async operations
        if self._checkpoint_enabled and self._checkpoint_conn_string:
            async with AsyncPostgresSaver.from_conn_string(
                self._checkpoint_conn_string
            ) as checkpointer:
                return await self._build_cancelled_result(runtime, config, checkpointer)
        else:
            return await self._build_cancelled_result(runtime, config, None)

    async def _build_cancelled_result(
        self,
        runtime,
        config: Dict[str, Any],
        checkpointer,
    ) -> QueryResult:
        """
        Build cancelled result with the given checkpointer.
        """
        langgraph_agent = build_langgraph_agent(runtime, checkpointer=checkpointer)

        snapshot = await langgraph_agent.aget_state(config)

        # Build answer from completed steps
        step_answers = snapshot.values.get("step_answers", [])
        if step_answers:
            answer_parts = []
            for step_ans in step_answers:
                task = step_ans.get("question", f"Step {step_ans.get('step', 0)}")
                answer_parts.append(f"**{task}**\n{step_ans['answer']}")
            final_answer = "\n\n".join(answer_parts)
            final_answer += "\n\n*(Email action cancelled by user)*"
        else:
            final_answer = "Action cancelled by user."

        _, contexts = self._extract_langgraph_results(snapshot.values)
        return QueryResult(answer=final_answer, contexts=contexts)

    async def _classify_query(self, query: str) -> ExecutionRoute:
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
        try:
            async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
                response = await chat_completion_json(
                    client=self.openai_client,
                    messages=[{"role": "user", "content": prompt}],
                    model=Config.OPENAI_MODEL,
                    temperature=0,
                    max_tokens=150,
                )
        except TimeoutError:
            logger.warning(
                "[CLASSIFIER] LLM classification timed out after %ds, defaulting to AGENT_SERVICE",
                Config.AGENT_TOOL_TIMEOUT,
            )
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            return ExecutionRoute.AGENT_SERVICE

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
            context: Execution context (must include pre-loaded conversation_history and file_service)

        Returns:
            Tuple of (answer, contexts)
        """
        # Use pre-loaded history from context (loaded in chat.py, avoids async issues)
        try:
            async with asyncio.timeout(Config.AGENT_TIMEOUT):
                final_answer, retrieved_contexts = await self.agent_service.run(
                    query, context
                )
            return final_answer, retrieved_contexts
        except TimeoutError:
            logger.warning(
                "[AgentService] Execution timed out after %ds",
                Config.AGENT_TIMEOUT,
            )
            increment_error(MetricsErrorType.TIMEOUT)
            return (
                f"The request timed out after {Config.AGENT_TIMEOUT} seconds. "
                f"Please try a simpler question or try again later.",
                [],
            )

    async def _execute_langgraph(
        self, query: str, context: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], Optional[HITLInterruptResult]]:
        """
        Execute query using LangGraph (Plan-Execute pattern).

        Args:
            query: User's question
            context: Execution context (should include 'thread_id' for checkpointing)

        Returns:
            Tuple of (answer, contexts, hitl_interrupt)
            - hitl_interrupt is None if completed normally
            - hitl_interrupt contains details if interrupted for human confirmation
        """
        # Create runtime context with non-serializable objects
        runtime = create_runtime_context(
            vector_db=context.get("vector_db"),
            openai_client=self.openai_client,
            dept_id=context.get("dept_id", ""),
            user_id=context.get("user_id", ""),
            conversation_id=context.get("conversation_id", ""),
            request_data=context.get("request_data", {}),
            conversation_history=context.get("conversation_history", []),
            file_service=context.get("file_service"),
            available_files=context.get("available_files", []),
            attachment_file_ids=context.get("attachment_file_ids", []),
        )

        # Create serializable initial state (no runtime objects)
        agent_state = self._build_langgraph_initial_state(query)

        # Use conversation_id as thread_id for checkpointing (enables multi-turn conversations and state recovery)
        thread_id = context.get("thread_id") or context.get("conversation_id")
        if not thread_id:
            raise ValueError("thread_id or conversation_id is required in context")
        config = {"configurable": {"thread_id": thread_id}}

        # Clear old checkpoint data before starting new query
        # This prevents "invalid memory alloc" errors from corrupted checkpoint blobs
        # Note: resume_workflow() does NOT call this - it needs existing checkpoints
        await self._clear_checkpoint_for_thread(thread_id)

        # Use AsyncPostgresSaver context manager for async operations
        if self._checkpoint_enabled and self._checkpoint_conn_string:
            async with AsyncPostgresSaver.from_conn_string(
                self._checkpoint_conn_string
            ) as checkpointer:
                return await self._run_langgraph_with_checkpointer(
                    runtime, agent_state, config, thread_id, checkpointer
                )
        else:
            # No checkpointer - run without persistence
            return await self._run_langgraph_with_checkpointer(
                runtime, agent_state, config, thread_id, None
            )

    async def _run_langgraph_with_checkpointer(
        self,
        runtime,
        agent_state: Dict[str, Any],
        config: Dict[str, Any],
        thread_id: str,
        checkpointer,
    ) -> Tuple[str, List[Dict[str, Any]], Optional[HITLInterruptResult]]:
        """
        Execute LangGraph with the given checkpointer.

        Args:
            runtime: Runtime context
            agent_state: Initial state
            config: Config with thread_id
            thread_id: Thread ID for error messages
            checkpointer: AsyncPostgresSaver instance or None

        Returns:
            Tuple of (answer, contexts, hitl_interrupt)
        """
        # Build graph with runtime context bound to nodes
        langgraph_agent = build_langgraph_agent(runtime, checkpointer=checkpointer)

        try:
            # Use astream for better interrupt handling
            # ainvoke() can hang when interrupt_before is triggered
            logger.info(f"[HITL] Starting graph execution with thread_id: {thread_id}")
            final_state = None
            async with asyncio.timeout(Config.AGENT_TIMEOUT):
                async for event in langgraph_agent.astream(
                    agent_state, config=config, stream_mode="values"
                ):
                    final_state = event

            logger.info("[HITL] Graph stream completed, checking for interrupt...")

            # Check if graph was interrupted (HITL)
            snapshot = await langgraph_agent.aget_state(config)
            if snapshot.next:
                # Graph was interrupted before a node
                pending_node = snapshot.next[0] if snapshot.next else None
                logger.info(f"[HITL] Graph interrupted before node: {pending_node}")

                # Extract HITL details based on pending node
                hitl_result = self._extract_hitl_details(
                    pending_node, snapshot.values, thread_id
                )
                if hitl_result:
                    # Return partial results with HITL interrupt
                    # Build answer from completed steps (final_answer is not set yet)
                    answer = self._build_partial_answer(snapshot.values)
                    _, contexts = self._extract_langgraph_results(snapshot.values)
                    return answer, contexts, hitl_result
            else:
                logger.info("[HITL] Graph completed without interruption")

            # Normal completion - no interrupt
            if final_state is None:
                final_state = snapshot.values
            return self._extract_langgraph_results(final_state) + (None,)

        except TimeoutError:
            logger.warning(
                "[LangGraph] Execution timed out after %ds for thread_id=%s",
                Config.AGENT_TIMEOUT,
                thread_id,
            )
            increment_error(MetricsErrorType.TIMEOUT)
            if final_state:
                answer, contexts = self._extract_langgraph_results(final_state)
                return (
                    f"{answer}\n\n(Response timed out after {Config.AGENT_TIMEOUT}s. "
                    f"This is a partial result.)",
                    contexts,
                    None,
                )
            return (
                f"The request timed out after {Config.AGENT_TIMEOUT} seconds. "
                f"Please try a simpler question or try again later.",
                [],
                None,
            )

        except Exception as e:
            # Auto-recovery for stale/corrupted checkpoints
            error_str = str(e).lower()
            error_type = type(e).__name__

            is_checkpoint_error = any(
                [
                    "unexpected end" in error_str,
                    "eoferror" in error_type.lower(),
                    "pickle" in error_str,
                    "deserialization" in error_str,
                    "corrupt" in error_str,
                    "invalid" in error_str and "state" in error_str,
                    "invalid memory alloc" in error_str,  # PostgreSQL blob corruption
                    "keyerror" in error_type.lower(),  # Missing state keys
                    "typeerror" in error_type.lower()
                    and "state" in error_str,  # State type mismatch
                ]
            )

            if is_checkpoint_error:
                logger.warning(
                    f"[CHECKPOINT_RECOVERY] Stale/corrupted checkpoint detected for "
                    f"thread_id={thread_id}: {error_type}: {e}"
                )
                logger.info(
                    "[CHECKPOINT_RECOVERY] Auto-recovering with fresh thread_id..."
                )

                # Generate new thread_id to bypass corrupted checkpoint
                new_thread_id = f"recovery_{uuid.uuid4()}"
                new_config = {"configurable": {"thread_id": new_thread_id}}

                # Retry with fresh state
                final_state = None
                try:
                    async with asyncio.timeout(Config.AGENT_TIMEOUT):
                        async for event in langgraph_agent.astream(
                            agent_state, config=new_config, stream_mode="values"
                        ):
                            final_state = event
                except TimeoutError:
                    logger.warning(
                        "[CHECKPOINT_RECOVERY] Recovery timed out after %ds for thread_id=%s",
                        Config.AGENT_TIMEOUT, new_thread_id,
                    )
                    increment_error(MetricsErrorType.TIMEOUT)
                    if final_state:
                        answer, contexts = self._extract_langgraph_results(final_state)
                        return (
                            f"{answer}\n\n(Recovery timed out after {Config.AGENT_TIMEOUT}s. "
                            f"This is a partial result.)",
                            contexts,
                            None,
                        )
                    return (
                        f"The request timed out after {Config.AGENT_TIMEOUT} seconds during recovery. "
                        f"Please try again later.",
                        [],
                        None,
                    )

                # Check for HITL after recovery
                snapshot = await langgraph_agent.aget_state(new_config)
                if snapshot.next:
                    pending_node = snapshot.next[0] if snapshot.next else None
                    logger.info(
                        f"[CHECKPOINT_RECOVERY] HITL interrupt after recovery: {pending_node}"
                    )
                    hitl_result = self._extract_hitl_details(
                        pending_node, snapshot.values, new_thread_id
                    )
                    if hitl_result:
                        answer = self._build_partial_answer(snapshot.values)
                        _, contexts = self._extract_langgraph_results(snapshot.values)
                        return answer, contexts, hitl_result

                if final_state is None:
                    final_state = snapshot.values
                return self._extract_langgraph_results(final_state) + (None,)
            else:
                # Re-raise non-checkpoint errors
                raise

    def _extract_hitl_details(
        self, pending_node: str, state: Dict[str, Any], thread_id: str
    ) -> Optional[HITLInterruptResult]:
        """
        Extract HITL details based on the pending node.

        Args:
            pending_node: Name of the node that was about to execute
            state: Current graph state
            thread_id: Thread ID for resumption

        Returns:
            HITLInterruptResult if this is an HITL interrupt, None otherwise
        """
        # Map node names to action types
        node_to_action = {
            "tool_send_email": "send_email",
            # Future: add more tools as needed
            # "tool_download_file": "download_file",
            # "tool_create_documents": "create_documents",
        }

        action = node_to_action.get(pending_node)
        if not action:
            return None

        # Extract action-specific details
        details = {}
        if action == "send_email":
            details = self._extract_send_email_details(state)

        # Get previous step results
        previous_steps = state.get("step_answers", [])

        return HITLInterruptResult(
            status="awaiting_confirmation",
            action=action,
            thread_id=thread_id,
            details=details,
            previous_steps=previous_steps,
        )

    def _extract_send_email_details(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract email details from state for confirmation UI.

        Args:
            state: Current graph state

        Returns:
            Dict with email details (recipient, subject, attachments, etc.)
        """
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        step_task = plan[current_step] if current_step < len(plan) else ""

        # Extract file_ids from previous steps (download_file, create_documents results)
        tool_results = state.get("tool_results", {})
        available_file_ids = []
        for step_results in tool_results.values():
            if isinstance(step_results, list):
                for result in step_results:
                    if isinstance(result, dict) and "file_id" in result:
                        available_file_ids.append(result["file_id"])

        # Try to parse email address from task description
        recipient = None
        email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
        matches = re.findall(email_pattern, step_task)
        if matches:
            recipient = matches[0]

        return {
            "task": step_task,
            "recipient": recipient,
            "available_attachments": available_file_ids,
            "step_answers": state.get("step_answers", []),
        }

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
            step_ctx_list = step_contexts[step_num]  # Now a list of contexts

            # Iterate through all contexts for this step (e.g., retrieve + web_search)
            for step_ctx in step_ctx_list:
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
                elif ctx_type == "direct_answer":
                    # Direct answer doesn't have contexts to return
                    pass

        # Fallback: if step_contexts is empty, use legacy approach
        if not all_contexts:
            retrieved_docs = final_state.get("retrieved_docs", [])
            tool_results_dict = final_state.get("tool_results", {})
            tool_results_list = []
            for results in tool_results_dict.values():
                tool_results_list.extend(results)
            all_contexts = retrieved_docs + tool_results_list

        return final_answer, all_contexts

    def _build_partial_answer(self, state: Dict[str, Any]) -> str:
        """
        Build a partial answer from completed steps when graph is interrupted for HITL.

        This is used when the graph pauses for human confirmation - we want to show
        what has been accomplished so far, not "No answer generated".

        Args:
            state: Current graph state with step_answers

        Returns:
            Partial answer string summarizing completed steps
        """
        step_answers = state.get("step_answers", [])

        if not step_answers:
            # No steps completed yet - this shouldn't happen normally
            return ""

        # Build answer from completed steps
        answer_parts = []
        for step_ans in step_answers:
            task = step_ans.get("question", f"Step {step_ans.get('step', 0)}")
            answer = step_ans.get("answer", "")
            if answer:
                answer_parts.append(f"**{task}** {answer}")

        if answer_parts:
            return "\n\n".join(answer_parts)

        return ""


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
