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

from typing import Tuple, List, Dict, Any, Optional
from langgraph.graph.state import CompiledStateGraph
from src.config.settings import Config
from src.services.agent_service import AgentService
from src.services.langgraph_builder import build_langgraph_agent
from enum import Enum


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

    def __init__(
        self,
        openai_client,
        agent_service: AgentService = None,
        langgraph_agent: CompiledStateGraph = None,
    ):
        """
        Initialize the supervisor.

        Args:
            openai_client: OpenAI client instance
            agent_service: Optional AgentService instance (lazy loaded if not provided)
            langgraph_agent: Optional LangGraph agent (lazy loaded if not provided)
        """
        self.openai_client = openai_client
        self.agent_service = agent_service or AgentService(openai_client=openai_client)
        self.langgraph_agent = langgraph_agent or build_langgraph_agent()

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
        )
        if not response.choices or len(response.choices) == 0:
            raise ValueError("LLM classification returned no choices")

        content = response.choices[0].message.content.strip()
        classification = content.lower().get("classification", "simple")

        return (
            ExecutionRoute.LANGGRAPH
            if classification == "complex"
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
            context: Execution context

        Returns:
            Tuple of (answer, contexts)
        """
        self.agent_service.run_stream(query, context)
        # TODO: Initialize AgentService if needed
        # TODO: Call agent_service.process_query()
        # TODO: Return results
        pass

    async def _execute_langgraph(
        self, query: str, context: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute query using LangGraph (Plan-Execute pattern).

        Args:
            query: User's question
            context: Execution context

        Returns:
            Tuple of (answer, contexts)
        """
        # TODO: Build initial state from context
        # TODO: Invoke langgraph agent
        # TODO: Extract final_answer and contexts from final state
        # TODO: Return results
        pass

    def _build_langgraph_initial_state(
        self, query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build initial state for LangGraph execution.

        Args:
            query: User's question
            context: Execution context

        Returns:
            Initial state dictionary matching AgentState schema
        """
        # TODO: Map context to AgentState fields
        # TODO: Initialize empty lists/dicts for accumulating fields
        # TODO: Set counters to 0
        pass

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
        # TODO: Get final_answer from state
        # TODO: Combine retrieved_docs and tool_results into contexts
        # TODO: Format contexts to match expected output format
        pass


# ==================== FACTORY FUNCTION ====================


def create_supervisor(openai_client) -> QuerySupervisor:
    """
    Factory function to create a configured QuerySupervisor.

    Args:
        openai_client: OpenAI client instance

    Returns:
        Configured QuerySupervisor instance
    """
    # TODO: Create and return supervisor with dependencies
    pass
