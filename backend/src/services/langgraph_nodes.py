"""
LangGraph agent node implementations.

Each node is a pure function: state → updated state
"""

from typing import Dict, Any
import json
from langchain_core.messages import AIMessage
from openai import OpenAI
from src.services.langgraph_state import AgentState
from src.services.retrieval import retrieve, build_where
from src.services.retrieval_evaluator import RetrievalEvaluator
from src.config.settings import Config


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
    query = state.get("query", "")
    planning_prompt = f"""
    You are a planning assistant. Create a step-by-step plan to answer the following query:
        Query: {query}

    Available Tools:
    1. search_documents: Retrieve relevant documents from internal knowledge base and self-reflection and generate tool results.
    2. calculator: Use calculator tool if numerical computation is required and also generate tool results.
    3. web_search: Use web search tool if up-to-date info is needed, then generate tool results.

    Provide the plan as a numbered list of steps, Return ONLY a list of steps in JSON format as follows:
    {
        "steps": [
            "Step 1: description",
            "Step 2: description",
            ...
        ]
    }
    Try to keep the number of steps minimal and concise (max 5 steps).
    """
    try:
        client = OpenAI(api_key=Config.OPENAI_KEY)
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": planning_prompt}],
            temperature=Config.OPENAI_TEMPERATURE,
            response_format={"type": "json_object"},  # Expecting JSON array response
        )

        plan_data = {}
        if response.choices and response.choices[0].message:
            plan_data = json.loads(response.choices[0].message.content)

        print("444444444444444444444444")
        print("Plan node response:", plan_data)
        plans = []
        if plan_data:
            plans = plan_data.get(
                "steps", [f"Retrieve documents for: {query}", "Generate answer"]
            )
    except Exception as e:
        print(f"Planning node error: {e}")
        plans = [
            f"Retrieve documents for the query: {query}",
            "Generate answer from retrieved documents",
        ]  # Fallback plan

    return {
        "plan": plans,
        "current_step": 0,
        "messages": state.get("messages", [])
        + [
            AIMessage(
                content=f"""
                     Plan created with {len(plans)} steps:\n
                     {"\n".join(["step " + str(i+1) + ": " + p for i, p in enumerate(plans)])}
                     """
            )
        ],
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
    dept_id = state.get("dept_id", "")
    user_id = state.get("user_id", "")
    collection = state.get("collection", None)
    if not collection:
        print("No collection found in state.")
        return {
            "retrieved_docs": [],
            "messages": state.get("messages", [])
            + [AIMessage(content="No document collection available for retrieval.")],
        }
    if not dept_id or not user_id:
        print("Department ID or User ID missing in state.")
        return {
            "retrieved_docs": [],
            "messages": state.get("messages", [])
            + [AIMessage(content="Missing department or user context for retrieval.")],
        }

    try:
        request_data = state.get("request_data", None)
        where = build_where(request_data, dept_id, user_id)
        ctx, _ = retrieve(
            query=state.get("refined_query") or state.get("query"),
            collection=state.get("collection"),
            dept_id=state.get("dept_id"),
            user_id=state.get("user_id"),
            top_k=Config.TOP_K,
            where=where,
            use_hybrid=Config.USE_HYBRID,
            use_reranker=Config.USE_RERANKER,
        )
        if not ctx:
            print("No documents retrieved.")
            return {
                "retrieved_docs": [],
                "messages": state.get("messages", [])
                + [AIMessage(content="No relevant documents found.")],
            }

        return {
            "retrieved_docs": ctx,
            "tools_used": state.get("tools_used", []) + ["search_documents"],
            "messages": state.get("messages", [])
            + [AIMessage(content=f"Retrieved {len(ctx)} documents.")],
        }
    except Exception as e:
        print(f"Retrieval node error: {e}")
        return {
            "retrieved_docs": [],
            "messages": state.get("messages", [])
            + [AIMessage(content="Error during document retrieval.")],
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
    pass


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
    pass


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
    pass


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
    pass


# ==================== ERROR HANDLER NODE ====================


def error_handler_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle errors gracefully.

    Args:
        state: Current agent state

    Returns:
        Updated state with error message
    """
    # TODO: Implement error handling
    pass
