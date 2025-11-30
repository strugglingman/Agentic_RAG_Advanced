# Agentic RAG Implementation Guide - Hybrid Approach

**Project:** RAG Chatbot with LangGraph Integration
**Approach:** Option C - Hybrid Architecture
**Timeline:** 3-4 weeks
**Updated:** 2025-11-30

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Phase 1: Setup & Infrastructure](#phase-1-setup--infrastructure)
4. [Phase 2: Query Routing System](#phase-2-query-routing-system)
5. [Phase 3: LangGraph Agent Implementation](#phase-3-langgraph-agent-implementation)
6. [Phase 4: Integration & Testing](#phase-4-integration--testing)
7. [Phase 5: Observability & Monitoring](#phase-5-observability--monitoring)
8. [Migration Strategy](#migration-strategy)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Troubleshooting Guide](#troubleshooting-guide)

---

## Executive Summary

### Current State
- **Semi-Agentic RAG**: Custom ReAct agent with 3 tools (search_documents, calculator, web_search)
- **Self-Reflection**: 3-mode evaluation system (FAST, BALANCED, THOROUGH)
- **Limitations**: No state management, no planning, sequential tool execution

### Target State
- **Hybrid System**: Keep existing agent for simple queries + LangGraph for complex workflows
- **Benefits**:
  - 80% of queries use fast existing system
  - 20% complex queries get sophisticated multi-step reasoning
  - Gradual migration path with no breaking changes

### Success Metrics
| Metric | Current | Target |
|--------|---------|--------|
| Simple Query Latency | 2-3s | 2-3s (unchanged) |
| Complex Query Success Rate | 60% | 85% |
| Multi-Step Query Support | ❌ No | ✅ Yes |
| Debugging Visibility | ⚠️ Limited | ✅ Full trace |
| Conversation Memory | ❌ No | ✅ Persistent |

---

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Complexity Router                       │
│  - Analyze query structure, intent, and requirements            │
│  - Decision: Simple vs Complex                                  │
└─────────────────────────────────────────────────────────────────┘
                    ▼                           ▼
        ┌───────────────────┐         ┌───────────────────┐
        │  SIMPLE QUERIES   │         │  COMPLEX QUERIES  │
        │    (80% of queries)│         │   (20% of queries)│
        └───────────────────┘         └───────────────────┘
                    ▼                           ▼
        ┌───────────────────┐         ┌───────────────────┐
        │  Existing Agent   │         │  LangGraph Agent  │
        │  - Fast (2-3s)    │         │  - Planning       │
        │  - Proven         │         │  - State Machine  │
        │  - Simple RAG     │         │  - Multi-Step     │
        └───────────────────┘         └───────────────────┘
                    ▼                           ▼
        ┌───────────────────────────────────────────────┐
        │           Response Formatter                   │
        │  - Unified format regardless of agent used    │
        └───────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Query Router (`backend/src/services/query_router.py`)
**Purpose:** Classify queries as simple or complex

**Responsibilities:**
- Analyze query structure (single vs multi-part)
- Detect planning requirements
- Identify comparison/aggregation needs
- Route to appropriate agent

**Example Routing Logic:**
```python
Simple Queries → Existing Agent:
- "What is our PTO policy?"
- "Find documents about vacation days"
- "Calculate 15% of 200"

Complex Queries → LangGraph Agent:
- "Compare PTO policies across departments and recommend changes"
- "Find all pricing documents, summarize them, and create a table"
- "Monitor these 3 documents for changes over the next week"
```

#### 2. Existing Agent (Keep As-Is)
**Location:** `backend/src/services/agent_service.py`
**Usage:** 80% of queries (fast path)

**Do NOT modify** - This is your proven, production-stable system

#### 3. LangGraph Agent (New)
**Location:** `backend/src/services/langgraph_agent.py`
**Usage:** 20% of queries (complex workflows)

**Features:**
- State machine with explicit transitions
- Planning before execution
- Parallel tool execution
- Conversation memory
- Checkpointing for long-running tasks

---

## Phase 1: Setup & Infrastructure

**Duration:** Week 1 (5 days)
**Goal:** Install dependencies, set up LangGraph, create basic graph

### Day 1-2: Installation & Configuration

#### Step 1.1: Install LangChain/LangGraph

```bash
# Backend dependencies
cd backend
pip install langgraph langchain langchain-openai langchain-community
pip install langsmith  # For observability

# Update requirements.txt
pip freeze > requirements.txt
```

#### Step 1.2: Environment Variables

Add to `backend/.env`:
```bash
# LangSmith (for visualization)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=rag-chatbot-agentic

# Agent Configuration
USE_LANGGRAPH=true
LANGGRAPH_MAX_ITERATIONS=10
LANGGRAPH_TIMEOUT=120  # seconds
```

#### Step 1.3: Configuration Class Updates

**File:** `backend/src/config/settings.py`

```python
class Config:
    # ... existing config ...

    # LangGraph Settings
    USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "true").lower() in {"1", "true", "yes"}
    LANGGRAPH_MAX_ITERATIONS = int(os.getenv("LANGGRAPH_MAX_ITERATIONS", "10"))
    LANGGRAPH_TIMEOUT = int(os.getenv("LANGGRAPH_TIMEOUT", "120"))

    # Query Routing
    COMPLEXITY_THRESHOLD = float(os.getenv("COMPLEXITY_THRESHOLD", "0.6"))
    ENABLE_QUERY_ROUTING = os.getenv("ENABLE_QUERY_ROUTING", "true").lower() in {"1", "true", "yes"}
```

### Day 3-4: Create Basic LangGraph Structure

#### Step 1.4: Define Agent State

**File:** `backend/src/services/langgraph_state.py`

```python
"""
LangGraph agent state definition.

This defines all possible states the agent can be in during execution.
"""

from typing import TypedDict, List, Optional, Annotated, Sequence
from langchain_core.messages import BaseMessage
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


def create_initial_state(query: str, conversation_id: Optional[str] = None) -> AgentState:
    """
    Factory function to create initial state.

    Args:
        query: User's question
        conversation_id: Optional conversation ID for memory

    Returns:
        Initial agent state
    """
    from langchain_core.messages import HumanMessage

    return AgentState(
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
        next_step="plan",  # Start with planning
        iteration_count=0,
        error=None
    )
```

#### Step 1.5: Create Graph Builder

**File:** `backend/src/services/langgraph_builder.py`

```python
"""
LangGraph agent graph builder.

This file constructs the state machine for the agentic RAG system.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .langgraph_state import AgentState
from .langgraph_nodes import (
    plan_node,
    retrieve_node,
    reflect_node,
    refine_node,
    generate_node,
    verify_node,
    error_handler_node
)
from .langgraph_routing import (
    route_after_reflection,
    route_after_planning,
    should_continue
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

    graph.add_node("plan", plan_node)           # Create execution plan
    graph.add_node("retrieve", retrieve_node)   # Retrieve documents
    graph.add_node("reflect", reflect_node)     # Evaluate retrieval quality
    graph.add_node("refine", refine_node)       # Refine query if needed
    graph.add_node("generate", generate_node)   # Generate answer
    graph.add_node("verify", verify_node)       # Verify citations
    graph.add_node("error", error_handler_node) # Handle errors gracefully

    # ==================== SET ENTRY POINT ====================
    graph.set_entry_point("plan")

    # ==================== ADD EDGES ====================

    # After planning, decide what to do
    graph.add_conditional_edges(
        "plan",
        route_after_planning,  # Function that decides next step
        {
            "retrieve": "retrieve",     # If plan requires retrieval
            "generate": "generate",     # If no retrieval needed
            "error": "error"            # If planning failed
        }
    )

    # After retrieval, always evaluate quality
    graph.add_edge("retrieve", "reflect")

    # After reflection, decide based on quality
    graph.add_conditional_edges(
        "reflect",
        route_after_reflection,
        {
            "generate": "generate",     # Quality good → generate answer
            "refine": "refine",         # Quality poor → refine query
            "retrieve": "retrieve",     # Try retrieval again
            "error": "error"            # Max refinements exceeded
        }
    )

    # After refinement, retrieve again
    graph.add_edge("refine", "retrieve")

    # After generation, verify citations
    graph.add_edge("generate", "verify")

    # After verification, check if we should continue or end
    graph.add_conditional_edges(
        "verify",
        should_continue,
        {
            "continue": "plan",  # Multi-step plan not complete
            "end": END           # All steps complete
        }
    )

    # Error handler always ends execution
    graph.add_edge("error", END)

    # ==================== COMPILE GRAPH ====================

    # Add checkpointing for memory and resumability
    checkpointer = MemorySaver()

    compiled_graph = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[],  # Can add nodes to interrupt before (for human-in-loop)
        interrupt_after=[]    # Can add nodes to interrupt after
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


# ==================== USAGE EXAMPLE ====================

"""
Example usage:

from src.services.langgraph_builder import build_langgraph_agent
from src.services.langgraph_state import create_initial_state

# Build graph
agent = build_langgraph_agent()

# Create initial state
state = create_initial_state(
    query="Compare PTO policies across departments",
    conversation_id="conv_123"
)

# Execute
result = agent.invoke(
    state,
    config={"configurable": {"thread_id": "conv_123"}}
)

# Access final answer
print(result["final_answer"])
"""
```

### Day 5: Create Node Skeletons

**File:** `backend/src/services/langgraph_nodes.py`

```python
"""
LangGraph agent node implementations.

Each node is a pure function: state → updated state
"""

from typing import Dict, Any
from .langgraph_state import AgentState
from langchain_core.messages import AIMessage


# ==================== PLANNING NODE ====================

def plan_node(state: AgentState) -> Dict[str, Any]:
    """
    Create execution plan before taking action.

    This is the "thinking" step - decompose complex query into steps.

    Args:
        state: Current agent state

    Returns:
        Updated state with plan
    """
    from openai import OpenAI
    from src.config.settings import Config

    query = state["query"]

    # TODO: Implement planning logic
    # For now, simple single-step plan
    plan = [f"Retrieve documents for: {query}", "Generate answer"]

    return {
        "plan": plan,
        "current_step": 0,
        "messages": state["messages"] + [
            AIMessage(content=f"Plan created: {len(plan)} steps")
        ]
    }


# ==================== RETRIEVAL NODE ====================

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieve documents from ChromaDB.

    Args:
        state: Current agent state

    Returns:
        Updated state with retrieved documents
    """
    # TODO: Implement retrieval
    # Use existing retrieval service

    query = state.get("refined_query") or state["query"]

    # Placeholder
    retrieved_docs = []

    return {
        "retrieved_docs": retrieved_docs,
        "tools_used": state["tools_used"] + ["search_documents"]
    }


# ==================== REFLECTION NODE ====================

def reflect_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluate retrieval quality (self-reflection).

    Args:
        state: Current agent state

    Returns:
        Updated state with quality assessment
    """
    # TODO: Implement reflection
    # Use existing RetrievalEvaluator

    quality = 0.8  # Placeholder
    recommendation = "ANSWER"  # Placeholder

    return {
        "retrieval_quality": quality,
        "retrieval_recommendation": recommendation
    }


# ==================== REFINEMENT NODE ====================

def refine_node(state: AgentState) -> Dict[str, Any]:
    """
    Refine query based on reflection feedback.

    Args:
        state: Current agent state

    Returns:
        Updated state with refined query
    """
    # TODO: Implement refinement
    # Use existing QueryRefiner

    refinement_count = state["refinement_count"] + 1

    return {
        "refined_query": state["query"] + " (refined)",
        "refinement_count": refinement_count
    }


# ==================== GENERATION NODE ====================

def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate answer from retrieved documents.

    Args:
        state: Current agent state

    Returns:
        Updated state with generated answer
    """
    # TODO: Implement generation

    draft_answer = "Generated answer placeholder"

    return {
        "draft_answer": draft_answer,
        "confidence": 0.85
    }


# ==================== VERIFICATION NODE ====================

def verify_node(state: AgentState) -> Dict[str, Any]:
    """
    Verify citations and finalize answer.

    Args:
        state: Current agent state

    Returns:
        Updated state with verified answer
    """
    # TODO: Implement verification

    final_answer = state["draft_answer"]  # Placeholder

    return {
        "final_answer": final_answer
    }


# ==================== ERROR HANDLER NODE ====================

def error_handler_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle errors gracefully.

    Args:
        state: Current agent state

    Returns:
        Updated state with error message
    """
    error_msg = state.get("error", "Unknown error occurred")

    return {
        "final_answer": f"I apologize, but I encountered an error: {error_msg}",
        "confidence": 0.0
    }
```

**File:** `backend/src/services/langgraph_routing.py`

```python
"""
LangGraph conditional routing functions.

These functions decide which node to execute next based on state.
"""

from .langgraph_state import AgentState


def route_after_planning(state: AgentState) -> str:
    """
    Decide what to do after planning.

    Args:
        state: Current agent state

    Returns:
        Next node name: "retrieve", "generate", or "error"
    """
    plan = state.get("plan", [])

    if not plan:
        return "error"

    # Check if first step requires retrieval
    first_step = plan[0].lower()
    if "retrieve" in first_step or "search" in first_step:
        return "retrieve"
    else:
        return "generate"


def route_after_reflection(state: AgentState) -> str:
    """
    Decide what to do based on retrieval quality.

    Args:
        state: Current agent state

    Returns:
        Next node name: "generate", "refine", "retrieve", or "error"
    """
    recommendation = state.get("retrieval_recommendation", "ANSWER")
    refinement_count = state.get("refinement_count", 0)

    # Max 3 refinements
    if refinement_count >= 3:
        if state.get("retrieved_docs"):
            return "generate"  # Use what we have
        else:
            return "error"  # No results after 3 tries

    if recommendation == "ANSWER":
        return "generate"
    elif recommendation == "REFINE":
        return "refine"
    elif recommendation == "EXTERNAL":
        # TODO: Trigger web search tool
        return "generate"
    else:
        return "generate"


def should_continue(state: AgentState) -> str:
    """
    Decide if we should continue execution or end.

    Args:
        state: Current agent state

    Returns:
        "continue" or "end"
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    iteration_count = state.get("iteration_count", 0)

    # Safety: max 10 iterations
    if iteration_count >= 10:
        return "end"

    # Check if all plan steps completed
    if current_step >= len(plan):
        return "end"

    # Otherwise continue
    return "continue"
```

---

## Phase 2: Query Routing System

**Duration:** Week 2 (5 days)
**Goal:** Implement query complexity classification and routing logic

### Day 6-7: Query Complexity Analyzer

**File:** `backend/src/services/query_analyzer.py`

```python
"""
Query complexity analyzer.

Classifies queries as simple or complex to route to appropriate agent.
"""

from typing import Dict, Literal
from dataclasses import dataclass
import re


QueryComplexity = Literal["simple", "complex"]


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    complexity: QueryComplexity
    confidence: float  # 0-1
    reasons: list[str]  # Why this classification
    estimated_steps: int  # How many steps needed


class QueryAnalyzer:
    """
    Analyzes query complexity to route to appropriate agent.

    Simple queries → Existing agent (fast)
    Complex queries → LangGraph agent (sophisticated)
    """

    def __init__(self):
        # Patterns that indicate complexity
        self.complex_indicators = {
            "comparison": r"\b(compare|versus|vs|difference between|contrast)\b",
            "multi_part": r"\b(and then|after that|next|followed by)\b",
            "aggregation": r"\b(summarize|aggregate|combine|merge|total)\b",
            "analysis": r"\b(analyze|evaluate|assess|determine|identify)\b",
            "planning": r"\b(plan|strategy|approach|steps to|how to)\b",
            "temporal": r"\b(monitor|track|watch|notify|alert when)\b",
            "multi_doc": r"\b(all documents|multiple files|across|throughout)\b",
            "conditional": r"\b(if|unless|depending on|based on)\b",
        }

        # Patterns that indicate simplicity
        self.simple_indicators = {
            "definition": r"\b(what is|define|explain|meaning of)\b",
            "single_fact": r"\b(who|when|where|which)\b",
            "calculation": r"\b(calculate|compute|what is \d+)\b",
        }

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query complexity.

        Args:
            query: User's question

        Returns:
            QueryAnalysis with classification and reasoning
        """
        query_lower = query.lower()

        # Check for complex indicators
        complex_matches = []
        for category, pattern in self.complex_indicators.items():
            if re.search(pattern, query_lower):
                complex_matches.append(category)

        # Check for simple indicators
        simple_matches = []
        for category, pattern in self.simple_indicators.items():
            if re.search(pattern, query_lower):
                simple_matches.append(category)

        # Count question marks (multiple questions = complex)
        question_count = query.count("?")

        # Count sentences (multiple sentences = complex)
        sentence_count = len(re.split(r'[.!?]', query))

        # Calculate complexity score
        complexity_score = 0.0
        reasons = []

        # Complex indicators add to score
        if complex_matches:
            complexity_score += 0.3 * len(complex_matches)
            reasons.append(f"Complex patterns: {', '.join(complex_matches)}")

        # Simple indicators subtract from score
        if simple_matches:
            complexity_score -= 0.2 * len(simple_matches)
            reasons.append(f"Simple patterns: {', '.join(simple_matches)}")

        # Multiple questions add to score
        if question_count > 1:
            complexity_score += 0.4
            reasons.append(f"Multiple questions ({question_count})")

        # Multiple sentences add to score
        if sentence_count > 2:
            complexity_score += 0.3
            reasons.append(f"Multiple sentences ({sentence_count})")

        # Query length (very long queries are usually complex)
        if len(query.split()) > 20:
            complexity_score += 0.2
            reasons.append(f"Long query ({len(query.split())} words)")

        # Classify based on score
        if complexity_score > 0.5:
            complexity = "complex"
            confidence = min(complexity_score, 1.0)
            estimated_steps = max(3, len(complex_matches) * 2)
        else:
            complexity = "simple"
            confidence = 1.0 - min(abs(complexity_score), 0.5)
            estimated_steps = 1

        return QueryAnalysis(
            complexity=complexity,
            confidence=confidence,
            reasons=reasons if reasons else ["Default classification"],
            estimated_steps=estimated_steps
        )


# ==================== USAGE EXAMPLE ====================

"""
Example:

analyzer = QueryAnalyzer()

# Simple query
result = analyzer.analyze("What is our PTO policy?")
print(result.complexity)  # "simple"

# Complex query
result = analyzer.analyze(
    "Compare PTO policies across all departments, "
    "identify inconsistencies, and recommend changes"
)
print(result.complexity)  # "complex"
print(result.reasons)  # ["Complex patterns: comparison, analysis", ...]
"""
```

### Day 8-9: Query Router Implementation

**File:** `backend/src/services/query_router.py`

```python
"""
Query router - decides which agent to use.

Routes queries to:
- Existing agent (simple queries - fast path)
- LangGraph agent (complex queries - sophisticated path)
"""

from typing import Dict, Any, Optional
from .query_analyzer import QueryAnalyzer, QueryAnalysis
from .agent_service import Agent  # Your existing agent
from .langgraph_agent import LangGraphAgent  # New LangGraph agent
from src.config.settings import Config
import time


class QueryRouter:
    """
    Routes queries to appropriate agent based on complexity.
    """

    def __init__(
        self,
        openai_client,
        collection,
        dept_id: str,
        user_id: str
    ):
        """
        Initialize router with both agents.

        Args:
            openai_client: OpenAI client instance
            collection: ChromaDB collection
            dept_id: Department ID for scoping
            user_id: User ID for scoping
        """
        self.analyzer = QueryAnalyzer()

        # Existing agent (simple queries)
        self.simple_agent = Agent(
            openai_client=openai_client,
            max_iterations=5,
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE
        )

        # LangGraph agent (complex queries)
        self.complex_agent = LangGraphAgent(
            openai_client=openai_client,
            collection=collection,
            dept_id=dept_id,
            user_id=user_id
        )

        self.dept_id = dept_id
        self.user_id = user_id
        self.collection = collection
        self.openai_client = openai_client

    def route_and_execute(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        request_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze query, route to appropriate agent, and execute.

        Args:
            query: User's question
            conversation_id: Optional conversation ID for memory
            request_data: Additional request context (filters, etc.)

        Returns:
            {
                "answer": str,
                "contexts": List[dict],
                "agent_used": "simple" | "complex",
                "analysis": QueryAnalysis,
                "execution_time": float,
                "metadata": dict
            }
        """
        start_time = time.time()

        # Step 1: Analyze query complexity
        analysis = self.analyzer.analyze(query)

        # Step 2: Route to appropriate agent
        if Config.ENABLE_QUERY_ROUTING and analysis.complexity == "complex":
            # Use LangGraph agent for complex queries
            result = self._execute_complex_agent(
                query=query,
                conversation_id=conversation_id,
                analysis=analysis,
                request_data=request_data
            )
            agent_used = "complex"
        else:
            # Use existing agent for simple queries
            result = self._execute_simple_agent(
                query=query,
                request_data=request_data
            )
            agent_used = "simple"

        execution_time = time.time() - start_time

        # Step 3: Format unified response
        return {
            "answer": result.get("answer", ""),
            "contexts": result.get("contexts", []),
            "agent_used": agent_used,
            "analysis": {
                "complexity": analysis.complexity,
                "confidence": analysis.confidence,
                "reasons": analysis.reasons,
                "estimated_steps": analysis.estimated_steps
            },
            "execution_time": execution_time,
            "metadata": result.get("metadata", {})
        }

    def _execute_simple_agent(
        self,
        query: str,
        request_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute existing agent for simple queries.

        Args:
            query: User's question
            request_data: Additional context

        Returns:
            Agent response
        """
        # Build context for existing agent
        context = {
            "collection": self.collection,
            "dept_id": self.dept_id,
            "user_id": self.user_id,
            "request_data": request_data or {},
            "use_hybrid": Config.USE_HYBRID,
            "use_reranker": Config.USE_RERANKER,
            "openai_client": self.openai_client,
            "model": Config.OPENAI_MODEL,
            "temperature": Config.OPENAI_TEMPERATURE
        }

        # Run existing agent
        answer, contexts = self.simple_agent.run(query, context)

        return {
            "answer": answer,
            "contexts": contexts,
            "metadata": {
                "iterations": self.simple_agent.iteration_count,
                "tools_called": self.simple_agent.tools_used
            }
        }

    def _execute_complex_agent(
        self,
        query: str,
        conversation_id: Optional[str],
        analysis: QueryAnalysis,
        request_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute LangGraph agent for complex queries.

        Args:
            query: User's question
            conversation_id: For conversation memory
            analysis: Complexity analysis result
            request_data: Additional context

        Returns:
            Agent response
        """
        # Run LangGraph agent
        result = self.complex_agent.run(
            query=query,
            conversation_id=conversation_id,
            estimated_steps=analysis.estimated_steps,
            request_data=request_data
        )

        return result


# ==================== USAGE EXAMPLE ====================

"""
Example in chat route:

from src.services.query_router import QueryRouter

# In chat endpoint
router = QueryRouter(
    openai_client=openai_client,
    collection=collection,
    dept_id=dept_id,
    user_id=user_id
)

result = router.route_and_execute(
    query=user_query,
    conversation_id=conversation_id,
    request_data=payload
)

# result contains:
# - answer: Final answer
# - contexts: Retrieved documents
# - agent_used: "simple" or "complex"
# - analysis: Why this routing decision
# - execution_time: How long it took
"""
```

### Day 10: Integration Testing

Create test file for query routing:

**File:** `backend/tests/test_query_routing.py`

```python
"""
Tests for query routing system.
"""

import pytest
from src.services.query_analyzer import QueryAnalyzer
from src.services.query_router import QueryRouter


class TestQueryAnalyzer:
    """Test query complexity analysis."""

    def setup_method(self):
        self.analyzer = QueryAnalyzer()

    def test_simple_query_classification(self):
        """Test that simple queries are classified correctly."""
        queries = [
            "What is our PTO policy?",
            "Define machine learning",
            "When was the company founded?",
            "Calculate 15% of 200"
        ]

        for query in queries:
            result = self.analyzer.analyze(query)
            assert result.complexity == "simple", f"Failed for: {query}"

    def test_complex_query_classification(self):
        """Test that complex queries are classified correctly."""
        queries = [
            "Compare PTO policies across all departments and recommend changes",
            "Find all pricing documents, summarize them, and create a comparison table",
            "Analyze sales trends and identify top performing products",
            "Monitor these documents for changes and notify me when updated"
        ]

        for query in queries:
            result = self.analyzer.analyze(query)
            assert result.complexity == "complex", f"Failed for: {query}"
            assert result.estimated_steps > 1

    def test_multi_question_complexity(self):
        """Test that multiple questions indicate complexity."""
        query = "What is our PTO policy? How does it compare to competitors? Should we change it?"
        result = self.analyzer.analyze(query)

        assert result.complexity == "complex"
        assert "Multiple questions" in str(result.reasons)


class TestQueryRouter:
    """Test query routing logic."""

    # TODO: Add router tests
    pass


# ==================== MANUAL TESTING ====================

"""
Run manual tests:

python -m pytest backend/tests/test_query_routing.py -v

Or test individual queries:

from src.services.query_analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()
result = analyzer.analyze("Your query here")
print(f"Complexity: {result.complexity}")
print(f"Confidence: {result.confidence}")
print(f"Reasons: {result.reasons}")
"""
```

---

## Phase 3: LangGraph Agent Implementation

**Duration:** Week 3 (5 days)
**Goal:** Implement full LangGraph agent with all nodes

### Day 11-12: Implement Core Nodes

Update `backend/src/services/langgraph_nodes.py` with full implementations:

```python
# ... (previous skeleton code) ...

# ==================== FULL IMPLEMENTATIONS ====================

def plan_node(state: AgentState) -> Dict[str, Any]:
    """
    Create execution plan using LLM.

    This is critical - good planning leads to better execution.
    """
    from openai import OpenAI
    from src.config.settings import Config
    from langchain_core.messages import AIMessage

    query = state["query"]

    # Use LLM to create plan
    client = OpenAI()

    planning_prompt = f"""You are a planning assistant for a RAG system.

User Query: {query}

Available Actions:
1. Retrieve documents from internal knowledge base
2. Search the web for external information
3. Perform calculations
4. Analyze and compare information
5. Summarize and synthesize findings

Create a step-by-step plan to answer this query.
Return ONLY a JSON array of steps, like:
["Step 1 description", "Step 2 description", ...]

Keep plans concise (max 5 steps).
"""

    try:
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": planning_prompt}],
            temperature=0.3,  # Lower temperature for more deterministic planning
            response_format={"type": "json_object"}
        )

        import json
        plan_data = json.loads(response.choices[0].message.content)
        plan = plan_data.get("steps", [query])  # Fallback to single step

    except Exception as e:
        print(f"Planning failed: {e}")
        plan = [f"Retrieve documents for: {query}", "Generate answer"]

    return {
        "plan": plan,
        "current_step": 0,
        "messages": state["messages"] + [
            AIMessage(content=f"📋 Plan created with {len(plan)} steps:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan)))
        ]
    }


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieve documents using existing retrieval service.
    """
    from src.services.retrieval import retrieve_hybrid
    from src.config.settings import Config

    query = state.get("refined_query") or state["query"]

    # Get context from state (passed from router)
    # In practice, you'll need to pass this through state initialization
    # For now, placeholder

    # TODO: Get these from state initialization
    collection = None  # state.get("collection")
    dept_id = None  # state.get("dept_id")
    user_id = None  # state.get("user_id")

    if not collection:
        return {
            "retrieved_docs": [],
            "error": "Collection not available in state"
        }

    # Use existing retrieval function
    try:
        contexts = retrieve_hybrid(
            collection=collection,
            query=query,
            dept_id=dept_id,
            user_id=user_id,
            top_k=Config.TOP_K,
            use_reranker=Config.USE_RERANKER
        )

        return {
            "retrieved_docs": contexts,
            "tools_used": state["tools_used"] + ["search_documents"]
        }

    except Exception as e:
        return {
            "retrieved_docs": [],
            "error": f"Retrieval failed: {str(e)}"
        }


def reflect_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluate retrieval quality using existing evaluator.
    """
    from src.services.retrieval_evaluator import RetrievalEvaluator
    from src.config.settings import Config
    from openai import OpenAI

    query = state.get("refined_query") or state["query"]
    contexts = state.get("retrieved_docs", [])

    # Use existing evaluator
    evaluator = RetrievalEvaluator(
        openai_client=OpenAI(),
        mode=Config.REFLECTION_MODE  # "balanced" by default
    )

    try:
        eval_result = evaluator.evaluate(query, contexts)

        return {
            "retrieval_quality": eval_result.confidence_score,
            "retrieval_recommendation": eval_result.recommendation.value
        }

    except Exception as e:
        # Fallback to basic heuristic
        if len(contexts) >= 3:
            return {
                "retrieval_quality": 0.7,
                "retrieval_recommendation": "ANSWER"
            }
        else:
            return {
                "retrieval_quality": 0.3,
                "retrieval_recommendation": "REFINE"
            }


def refine_node(state: AgentState) -> Dict[str, Any]:
    """
    Refine query using existing query refiner.
    """
    from src.services.query_refiner import QueryRefiner
    from openai import OpenAI
    from src.config.settings import Config

    query = state.get("refined_query") or state["query"]
    contexts = state.get("retrieved_docs", [])

    # Use existing refiner
    refiner = QueryRefiner(
        openai_client=OpenAI(),
        mode="simple"  # Use simple mode for speed
    )

    try:
        refined = refiner.refine_query(query, contexts)

        return {
            "refined_query": refined,
            "refinement_count": state["refinement_count"] + 1
        }

    except Exception as e:
        # Fallback: add context hints
        return {
            "refined_query": query + " (please provide specific details)",
            "refinement_count": state["refinement_count"] + 1
        }


def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate answer using retrieved contexts.
    """
    from openai import OpenAI
    from src.config.settings import Config

    query = state["query"]
    contexts = state.get("retrieved_docs", [])

    # Format contexts
    context_str = "\n\n".join([
        f"Context {i+1} (Source: {c.get('source', 'Unknown')}, Page: {c.get('page', '?')}):\n{c.get('chunk', '')}"
        for i, c in enumerate(contexts)
    ])

    # Generate answer
    client = OpenAI()

    generation_prompt = f"""You are a helpful assistant answering questions based on provided context.

User Question: {query}

Context:
{context_str if context_str else "No context available. Use general knowledge."}

Instructions:
1. Answer the question accurately based on the context
2. Cite sources using [1], [2], etc. to reference context numbers
3. If context is insufficient, acknowledge limitations
4. Be concise but comprehensive

Answer:"""

    try:
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": generation_prompt}],
            temperature=Config.OPENAI_TEMPERATURE
        )

        draft_answer = response.choices[0].message.content

        # Calculate confidence based on context quality
        if contexts:
            avg_score = sum(c.get('hybrid', 0) for c in contexts) / len(contexts)
            confidence = min(avg_score, 0.95)
        else:
            confidence = 0.5  # Lower confidence without context

        return {
            "draft_answer": draft_answer,
            "confidence": confidence
        }

    except Exception as e:
        return {
            "draft_answer": f"Error generating answer: {str(e)}",
            "confidence": 0.0,
            "error": str(e)
        }


def verify_node(state: AgentState) -> Dict[str, Any]:
    """
    Verify citations in the answer.
    """
    from src.utils.safety import check_citations_present

    draft_answer = state.get("draft_answer", "")
    contexts = state.get("retrieved_docs", [])

    # Check if citations are present when contexts exist
    if contexts:
        has_citations = check_citations_present(draft_answer)

        if not has_citations:
            # Add warning to answer
            final_answer = draft_answer + "\n\n⚠️ Note: Answer generated from provided context but citations may be incomplete."
        else:
            final_answer = draft_answer
    else:
        final_answer = draft_answer

    # Move to next plan step
    current_step = state.get("current_step", 0) + 1

    return {
        "final_answer": final_answer,
        "current_step": current_step,
        "iteration_count": state.get("iteration_count", 0) + 1
    }
```

### Day 13-14: Create LangGraph Agent Wrapper

**File:** `backend/src/services/langgraph_agent.py`

```python
"""
LangGraph Agent - wrapper for executing the graph.
"""

from typing import Dict, Any, Optional
from .langgraph_builder import build_langgraph_agent
from .langgraph_state import create_initial_state


class LangGraphAgent:
    """
    Wrapper for LangGraph agent execution.

    This class provides a clean interface to run the LangGraph agent
    and is compatible with the existing agent interface.
    """

    def __init__(
        self,
        openai_client,
        collection,
        dept_id: str,
        user_id: str
    ):
        """
        Initialize LangGraph agent.

        Args:
            openai_client: OpenAI client instance
            collection: ChromaDB collection
            dept_id: Department ID for scoping
            user_id: User ID for scoping
        """
        self.openai_client = openai_client
        self.collection = collection
        self.dept_id = dept_id
        self.user_id = user_id

        # Build the graph (compiled once, reused)
        self.graph = build_langgraph_agent()

    def run(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        estimated_steps: int = 3,
        request_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run the LangGraph agent.

        Args:
            query: User's question
            conversation_id: For conversation memory
            estimated_steps: How many steps we expect (from analyzer)
            request_data: Additional context (filters, etc.)

        Returns:
            {
                "answer": str,
                "contexts": List[dict],
                "metadata": {
                    "plan": List[str],
                    "iterations": int,
                    "tools_used": List[str],
                    "confidence": float
                }
            }
        """
        # Create initial state
        state = create_initial_state(query, conversation_id)

        # Add execution context to state
        # Note: LangGraph state is immutable, so we create enriched initial state
        state["collection"] = self.collection
        state["dept_id"] = self.dept_id
        state["user_id"] = self.user_id
        state["openai_client"] = self.openai_client
        state["request_data"] = request_data or {}

        # Execute graph
        try:
            config = {
                "configurable": {
                    "thread_id": conversation_id or "default"
                }
            }

            result = self.graph.invoke(state, config)

            # Extract results
            return {
                "answer": result.get("final_answer", "No answer generated"),
                "contexts": result.get("retrieved_docs", []),
                "metadata": {
                    "plan": result.get("plan", []),
                    "iterations": result.get("iteration_count", 0),
                    "tools_used": result.get("tools_used", []),
                    "confidence": result.get("confidence", 0.0),
                    "refinement_count": result.get("refinement_count", 0)
                }
            }

        except Exception as e:
            # Graceful error handling
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "contexts": [],
                "metadata": {
                    "error": str(e),
                    "plan": [],
                    "iterations": 0,
                    "tools_used": [],
                    "confidence": 0.0
                }
            }


# ==================== USAGE EXAMPLE ====================

"""
Example:

from src.services.langgraph_agent import LangGraphAgent
from openai import OpenAI

agent = LangGraphAgent(
    openai_client=OpenAI(),
    collection=chroma_collection,
    dept_id="engineering",
    user_id="user@example.com"
)

result = agent.run(
    query="Compare PTO policies across departments",
    conversation_id="conv_123"
)

print(result["answer"])
print(f"Used {result['metadata']['iterations']} iterations")
print(f"Plan: {result['metadata']['plan']}")
"""
```

### Day 15: Testing & Debugging

Create comprehensive test file:

**File:** `backend/tests/test_langgraph_agent.py`

```python
"""
Tests for LangGraph agent.
"""

import pytest
from src.services.langgraph_agent import LangGraphAgent
from src.services.langgraph_builder import build_langgraph_agent
from src.services.langgraph_state import create_initial_state


class TestLangGraphState:
    """Test state creation and management."""

    def test_initial_state_creation(self):
        """Test creating initial state."""
        state = create_initial_state(
            query="What is our PTO policy?",
            conversation_id="conv_123"
        )

        assert state["query"] == "What is our PTO policy?"
        assert state["conversation_id"] == "conv_123"
        assert state["iteration_count"] == 0
        assert state["refinement_count"] == 0
        assert state["next_step"] == "plan"


class TestLangGraphBuilder:
    """Test graph construction."""

    def test_graph_compilation(self):
        """Test that graph compiles without errors."""
        graph = build_langgraph_agent()
        assert graph is not None

    def test_graph_has_required_nodes(self):
        """Test that all required nodes exist."""
        graph = build_langgraph_agent()
        graph_structure = graph.get_graph()

        required_nodes = [
            "plan", "retrieve", "reflect",
            "refine", "generate", "verify", "error"
        ]

        for node in required_nodes:
            assert node in graph_structure.nodes


class TestLangGraphAgent:
    """Test agent execution."""

    # TODO: Add full agent tests with mocked OpenAI/ChromaDB
    pass


# ==================== INTEGRATION TESTS ====================

"""
Manual integration test (requires real OpenAI API key):

from src.services.langgraph_agent import LangGraphAgent
from openai import OpenAI
import chromadb

# Set up
client = chromadb.Client()
collection = client.get_or_create_collection("test")

agent = LangGraphAgent(
    openai_client=OpenAI(),
    collection=collection,
    dept_id="test",
    user_id="test@example.com"
)

# Test simple query
result = agent.run("What is 2+2?")
print(result["answer"])

# Test complex query
result = agent.run(
    "Find all documents about pricing, summarize them, and create a comparison table"
)
print(result["answer"])
print(result["metadata"]["plan"])
"""
```

---

## Phase 4: Integration & Testing

**Duration:** Week 4 (5 days)
**Goal:** Integrate router into chat endpoint, test end-to-end

### Day 16-17: Update Chat Endpoint

**File:** `backend/src/routes/chat.py`

Add router integration:

```python
# ... existing imports ...
from src.services.query_router import QueryRouter

@chat_bp.post("/chat/agent")
@require_identity
async def chat_agent(request: Request):
    """
    Enhanced chat endpoint with query routing.

    Routes queries to appropriate agent based on complexity.
    """
    # ... existing auth and validation code ...

    # Build agent context (existing code)
    agent_context = {
        "collection": collection,
        "dept_id": dept_id,
        "user_id": user_id,
        "request_data": payload,
        "use_hybrid": Config.USE_HYBRID,
        "use_reranker": Config.USE_RERANKER,
        "openai_client": openai_client,
        "model": Config.OPENAI_MODEL,
        "temperature": Config.OPENAI_TEMPERATURE,
    }

    # NEW: Use router instead of direct agent call
    router = QueryRouter(
        openai_client=openai_client,
        collection=collection,
        dept_id=dept_id,
        user_id=user_id
    )

    # Route and execute
    result = router.route_and_execute(
        query=query,
        conversation_id=conversation_id,
        request_data=payload
    )

    # Extract results
    final_answer = result["answer"]
    contexts = result["contexts"]
    agent_used = result["agent_used"]

    # Log which agent was used (for monitoring)
    print(f"Query routed to: {agent_used} agent")
    print(f"Complexity analysis: {result['analysis']}")

    # ... rest of streaming response code ...
```

### Day 18-19: End-to-End Testing

Create test scenarios:

**File:** `backend/tests/test_e2e_routing.py`

```python
"""
End-to-end tests for query routing system.
"""

import pytest
from src.services.query_router import QueryRouter


class TestE2ERouting:
    """Test complete routing flow."""

    @pytest.fixture
    def setup_router(self):
        """Set up router with test dependencies."""
        # TODO: Set up test OpenAI client, ChromaDB collection
        pass

    def test_simple_query_flow(self, setup_router):
        """Test that simple query uses existing agent."""
        # TODO: Implement
        pass

    def test_complex_query_flow(self, setup_router):
        """Test that complex query uses LangGraph agent."""
        # TODO: Implement
        pass


# ==================== MANUAL TESTING CHECKLIST ====================

"""
Test these scenarios manually:

1. Simple Queries (should use existing agent):
   - "What is our PTO policy?"
   - "Define machine learning"
   - "Calculate 15% of 200"

2. Complex Queries (should use LangGraph agent):
   - "Compare PTO policies across all departments and recommend changes"
   - "Find all pricing documents, summarize them, and create a table"
   - "Analyze sales trends and identify top performers"

3. Edge Cases:
   - Very long queries
   - Multiple questions in one query
   - Queries with conditional logic

4. Performance:
   - Measure latency for simple vs complex queries
   - Check LangSmith traces for LangGraph execution
   - Verify caching works correctly

5. Error Handling:
   - Query with no results
   - OpenAI API timeout
   - ChromaDB unavailable
"""
```

### Day 20: Performance Testing

**File:** `backend/tests/test_performance.py`

```python
"""
Performance benchmarks for routing system.
"""

import time
import pytest
from src.services.query_router import QueryRouter


class TestPerformance:
    """Benchmark query routing performance."""

    def test_simple_query_latency(self):
        """Test that simple queries stay fast (< 3s)."""
        # TODO: Benchmark existing agent
        pass

    def test_complex_query_correctness(self):
        """Test that complex queries improve success rate."""
        # TODO: Compare LangGraph vs existing agent on complex queries
        pass

    def test_routing_overhead(self):
        """Test that routing decision is fast (< 100ms)."""
        # TODO: Measure QueryAnalyzer.analyze() performance
        pass


# ==================== BENCHMARK RESULTS ====================

"""
Expected performance:

Metric                      | Target    | Actual
----------------------------|-----------|--------
Simple query latency        | < 3s      | ?
Complex query latency       | < 10s     | ?
Routing decision time       | < 100ms   | ?
Complex query success rate  | > 85%     | ?
LangGraph planning time     | < 1s      | ?

Run benchmarks:
python -m pytest backend/tests/test_performance.py -v --benchmark
"""
```

---

## Phase 5: Observability & Monitoring

**Duration:** Days 21-25
**Goal:** Set up monitoring, logging, and debugging tools

### Day 21-22: LangSmith Integration

Already partially done with @traceable decorators, but enhance:

**File:** `backend/src/services/langsmith_config.py`

```python
"""
LangSmith configuration for observability.
"""

import os
from langsmith import Client
from functools import wraps


def get_langsmith_client():
    """Get LangSmith client if enabled."""
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        return Client()
    return None


def trace_execution(name: str):
    """
    Decorator to trace function execution in LangSmith.

    Usage:
        @trace_execution("retrieve_documents")
        def retrieve(query):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            client = get_langsmith_client()
            if client:
                # Create trace
                with client.trace(name=name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator
```

### Day 23-24: Metrics Collection

**File:** `backend/src/services/metrics.py`

```python
"""
Metrics collection for agent performance.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json


@dataclass
class AgentMetrics:
    """Metrics for agent execution."""

    query: str
    agent_used: str  # "simple" or "complex"
    execution_time: float
    success: bool
    confidence: float
    iterations: int
    tools_used: list[str]
    error: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self):
        """Convert to dict for logging."""
        return {
            "query": self.query[:100],  # Truncate for privacy
            "agent_used": self.agent_used,
            "execution_time": self.execution_time,
            "success": self.success,
            "confidence": self.confidence,
            "iterations": self.iterations,
            "tools_used": self.tools_used,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }


class MetricsCollector:
    """Collect and aggregate agent metrics."""

    def __init__(self):
        self.metrics = []

    def record(self, metrics: AgentMetrics):
        """Record metrics."""
        self.metrics.append(metrics)

        # Log to file (in production, send to monitoring service)
        with open("agent_metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")

    def get_stats(self) -> dict:
        """Get aggregated statistics."""
        if not self.metrics:
            return {}

        simple_metrics = [m for m in self.metrics if m.agent_used == "simple"]
        complex_metrics = [m for m in self.metrics if m.agent_used == "complex"]

        return {
            "total_queries": len(self.metrics),
            "simple_queries": len(simple_metrics),
            "complex_queries": len(complex_metrics),
            "avg_latency_simple": sum(m.execution_time for m in simple_metrics) / len(simple_metrics) if simple_metrics else 0,
            "avg_latency_complex": sum(m.execution_time for m in complex_metrics) / len(complex_metrics) if complex_metrics else 0,
            "success_rate": sum(1 for m in self.metrics if m.success) / len(self.metrics),
            "avg_confidence": sum(m.confidence for m in self.metrics) / len(self.metrics)
        }


# Global metrics collector
metrics_collector = MetricsCollector()
```

### Day 25: Dashboard & Visualization

Create simple monitoring endpoint:

**File:** `backend/src/routes/monitoring.py`

```python
"""
Monitoring endpoints for agent performance.
"""

from flask import Blueprint, jsonify
from src.middleware.auth import require_identity
from src.services.metrics import metrics_collector


monitoring_bp = Blueprint("monitoring", __name__)


@monitoring_bp.get("/metrics/agent")
@require_identity
async def get_agent_metrics():
    """
    Get agent performance metrics.

    Returns aggregated statistics about agent usage.
    """
    stats = metrics_collector.get_stats()
    return jsonify(stats)


@monitoring_bp.get("/metrics/routing")
@require_identity
async def get_routing_stats():
    """
    Get query routing statistics.

    Shows distribution of simple vs complex queries.
    """
    # TODO: Implement routing-specific metrics
    pass
```

---

## Migration Strategy

### Gradual Rollout Plan

#### Week 1: Internal Testing
- Enable routing only for specific test users
- Monitor metrics closely
- Compare results with existing agent

#### Week 2: Beta (20% of users)
- Roll out to 20% of users
- Collect feedback
- Monitor error rates

#### Week 3: Majority (80% of users)
- If metrics look good, expand to 80%
- Keep 20% on old system for comparison

#### Week 4: Full Deployment
- 100% on new routing system
- Keep existing agent as fallback

### Feature Flags

Add to `backend/src/config/settings.py`:

```python
class Config:
    # ... existing config ...

    # Gradual Rollout
    ROUTING_ENABLED_USERS = os.getenv("ROUTING_ENABLED_USERS", "").split(",")
    ROUTING_ROLLOUT_PERCENTAGE = int(os.getenv("ROUTING_ROLLOUT_PERCENTAGE", "100"))
```

### Rollback Plan

If issues occur:

1. **Immediate**: Set `ENABLE_QUERY_ROUTING=false` in .env
2. **All queries** go back to existing agent
3. **No breaking changes** - system degrades gracefully

---

## Performance Benchmarks

### Expected Metrics

| Scenario | Existing Agent | LangGraph Agent | Target |
|----------|----------------|-----------------|---------|
| Simple Q&A | 2-3s | N/A (not used) | < 3s |
| Complex multi-step | 5-10s (often fails) | 8-15s | < 15s |
| Planning overhead | N/A | 1-2s | < 2s |
| Success rate (simple) | 90% | N/A | > 90% |
| Success rate (complex) | 60% | 85% | > 85% |

---

## Troubleshooting Guide

### Common Issues

#### Issue: LangGraph agent slower than existing agent
**Solution:**
- Check if planning is taking too long
- Consider caching plans for similar queries
- Use faster LLM for planning (GPT-3.5 instead of GPT-4)

#### Issue: Queries routed incorrectly
**Solution:**
- Adjust `COMPLEXITY_THRESHOLD` in settings
- Review `QueryAnalyzer` patterns
- Check LangSmith traces to see why classification failed

#### Issue: LangGraph agent gets stuck in loops
**Solution:**
- Check `LANGGRAPH_MAX_ITERATIONS` setting
- Review conditional routing logic
- Add more exit conditions to `should_continue()`

#### Issue: Memory not persisting across conversations
**Solution:**
- Verify checkpointer is configured
- Check `thread_id` is being passed correctly
- Ensure MemorySaver is not being reset

---

## Next Steps After Implementation

### Advanced Features to Add

1. **Parallel Tool Execution**
   - Modify graph to run independent tools concurrently
   - Merge results intelligently

2. **Human-in-the-Loop**
   - Add interrupt points for low-confidence answers
   - Allow users to provide feedback mid-execution

3. **Multi-Turn Planning**
   - Replan after each step based on results
   - Adaptive planning that learns from failures

4. **Tool Result Caching**
   - Cache tool outputs for identical calls
   - Reduce redundant OpenAI API calls

5. **Conversation Memory**
   - Use checkpointer to maintain state across messages
   - Reference previous turns in planning

---

## Conclusion

This hybrid approach gives you:

✅ **Best of both worlds**: Fast existing agent + sophisticated LangGraph agent
✅ **Gradual migration**: No breaking changes, rollback-friendly
✅ **Improved success rate**: Complex queries handled properly
✅ **Full observability**: LangSmith traces show agent's thinking
✅ **Production-ready**: Error handling, monitoring, testing

**Estimated Total Effort:** 3-4 weeks for full implementation

**Key Success Factors:**
1. Start with good test coverage
2. Monitor metrics closely during rollout
3. Iterate on query routing patterns based on real usage
4. Keep existing agent as reliable fallback

Good luck with implementation! 🚀