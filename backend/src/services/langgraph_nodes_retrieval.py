"""Retrieval/reflection/refinement node implementations."""

from typing import Dict, Any, Callable, Coroutine
import asyncio
import logging

from langchain_core.messages import AIMessage

from src.services.langgraph_state import AgentState, RuntimeContext
from src.services.retrieval_qdrant import build_where
from src.services.retrieval_decomposition import retrieve_with_decomposition
from src.services.retrieval_evaluator import RetrievalEvaluator
from src.models.evaluation import (
    EvaluationCriteria,
    ReflectionMode,
    ReflectionConfig,
    EvaluationResult,
    QualityLevel,
    RecommendationAction,
)
from src.services.query_refiner import QueryRefiner
from src.config.settings import Config
from src.observability.metrics import increment_query_routing, increment_error, MetricsErrorType
from src.observability.tracing import traced_span
from src.services.langgraph_nodes_common import (
    _clone_step_contexts,
    _optimize_step_query,
    evaluation_result_to_dict,
    dict_to_evaluation_result,
)

logger = logging.getLogger(__name__)

def create_retrieve_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create retrieve_node with runtime context bound.

    Args:
        runtime: RuntimeContext with vector_db, dept_id, user_id, request_data

    Returns:
        Async retrieve_node function
    """

    async def retrieve_node(state: AgentState) -> Dict[str, Any]:
        """
        Retrieve documents from ChromaDB.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved documents
        """
        increment_query_routing("retrieve")
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        is_detour = state.get("evaluation_result") is not None
        if not plan:
            return {
                "retrieved_docs": [],
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content="No plan or over max steps in plan.")],
            }
        if current_step >= len(plan):
            return {
                "retrieved_docs": state.get("retrieved_docs", []),
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content="Over maximum steps in plan.")],
            }
        action = plan[current_step].lower()
        if (
            "retrieve" not in action
            and "search" not in action
            and "document" not in action
            and "find" not in action
        ):
            return {
                "retrieved_docs": state.get("retrieved_docs", []),
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content="Current step is not to retrieve documents.")],
            }

        # Get runtime context
        vector_db = runtime.get("vector_db")
        dept_id = runtime.get("dept_id", "")
        user_id = runtime.get("user_id", "")
        request_data = runtime.get("request_data")

        if not vector_db:
            return {
                "retrieved_docs": state.get("retrieved_docs", []),
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content="No vector database available for retrieval.")],
            }
        if not dept_id or not user_id:
            return {
                "retrieved_docs": [],
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [
                    AIMessage(
                        content="Missing department or user context for retrieval."
                    )
                ],
            }

        try:
            with traced_span(
                "rag.node.retrieve",
                {
                    "rag.plan.current_step": current_step,
                    "rag.plan.length": len(plan),
                    "rag.refinement.is_detour": is_detour,
                },
            ) as span:
                async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
                    # Extract query from current plan step
                    current_plan_step = plan[current_step]

                    if ":" in current_plan_step:
                        step_query = current_plan_step.split(":", 1)[1].strip()
                    else:
                        step_query = current_plan_step
                        for keyword in ["retrieve", "search", "find", "document", "documents"]:
                            step_query = step_query.replace(keyword, "").strip()

                    openai_client = runtime.get("openai_client")
                    if state.get("refined_query"):
                        query = state.get("refined_query")
                        logger.info(
                            f"[RETRIEVE] Using refined_query (semantic refinement): '{query}'"
                        )
                    else:
                        logger.info(f"[RETRIEVE] Optimizing step_query: '{step_query}'")
                        query = await _optimize_step_query(
                            step_query, "retrieve", openai_client
                        ) or state.get("query")
                        logger.info(f"[RETRIEVE] Optimized query: '{query}'")

                    if span is not None:
                        span.set_attribute("rag.query.text", query)
                        span.set_attribute("rag.retrieval.use_hybrid", Config.USE_HYBRID)
                        span.set_attribute("rag.retrieval.use_reranker", Config.USE_RERANKER)

                    where = build_where(request_data, dept_id, user_id)
                    ctx, _ = await retrieve_with_decomposition(
                        vector_db=vector_db,
                        openai_client=openai_client,
                        query=query,
                        dept_id=dept_id,
                        user_id=user_id,
                        top_k=Config.TOP_K,
                        where=where,
                        use_hybrid=Config.USE_HYBRID,
                        use_reranker=Config.USE_RERANKER,
                    )
                    if span is not None:
                        span.set_attribute("rag.retrieval.result_count", len(ctx))

                    if not ctx:
                        return {
                            "retrieved_docs": [],
                            "current_step": current_step,
                            "iteration_count": state.get("iteration_count", 0) + 1,
                            "messages": [AIMessage(content="No relevant documents found.")],
                        }

                    step_contexts = _clone_step_contexts(state)
                    if current_step not in step_contexts:
                        step_contexts[current_step] = []

                    step_contexts[current_step] = [
                        ctx
                        for ctx in step_contexts[current_step]
                        if ctx.get("type") != "retrieval"
                    ]

                    step_contexts[current_step].append(
                        {
                            "type": "retrieval",
                            "docs": ctx,
                            "plan_step": plan[current_step] if current_step < len(plan) else "",
                        }
                    )

                    return {
                        "retrieved_docs": ctx,
                        "step_contexts": step_contexts,
                        "tools_used": state.get("tools_used", []) + ["search_documents"],
                        "current_step": current_step,
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "messages": [AIMessage(content=f"Retrieved {len(ctx)} documents.")],
                    }
        except TimeoutError:
            logger.warning("[RETRIEVE] Node timed out after %ds", Config.AGENT_TOOL_TIMEOUT)
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            return {
                "retrieved_docs": [],
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content=f"Retrieval timed out after {Config.AGENT_TOOL_TIMEOUT}s.")],
            }
        except Exception as e:
            return {
                "retrieved_docs": [],
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content="Error during document retrieval.")],
            }

    return retrieve_node

def create_reflect_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create reflect_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client

    Returns:
        Async reflect_node function
    """

    async def reflect_node(state: AgentState) -> Dict[str, Any]:
        """
        Evaluate retrieval quality (self-reflection).

        Args:
            state: Current agent state

        Returns:
            Updated state with quality assessment (evaluation_result as dict)
        """
        try:
            with traced_span("rag.node.reflect") as span:
                async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
                    plan = state.get("plan", [])
                    current_step = state.get("current_step", 0)

                    step_query = state.get("query", "")
                    if plan and current_step < len(plan):
                        current_plan_step = plan[current_step]
                        if ":" in current_plan_step:
                            step_query = current_plan_step.split(":", 1)[1].strip()
                        else:
                            step_query = current_plan_step
                            for keyword in [
                                "retrieve",
                                "search",
                                "find",
                                "document",
                                "documents",
                            ]:
                                step_query = step_query.replace(keyword, "").strip()

                    query = state.get("refined_query") or step_query
                    retrieved_docs = state.get("retrieved_docs", [])
                    if span is not None:
                        span.set_attribute("rag.query.text", query)
                        span.set_attribute("rag.retrieval.result_count", len(retrieved_docs))

                    evaluator_criteria = EvaluationCriteria(
                        query=query,
                        contexts=retrieved_docs,
                        mode=ReflectionMode.BALANCED,
                    )
                    reflection_config = ReflectionConfig.from_settings(Config)
                    openai_client = runtime.get("openai_client")

                    if not openai_client:
                        raise ValueError("OpenAI client is required for reflection node.")

                    evaluator = RetrievalEvaluator(
                        config=reflection_config,
                        openai_client=openai_client,
                    )
                    evaluation_result = await evaluator.evaluate(evaluator_criteria)
                    if span is not None:
                        span.set_attribute(
                            "rag.reflection.recommendation",
                            evaluation_result.recommendation.value,
                        )
                        span.set_attribute(
                            "rag.reflection.confidence", evaluation_result.confidence
                        )

                    return {
                        "evaluation_result": evaluation_result_to_dict(evaluation_result),
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "messages": [
                            AIMessage(
                                content=f"Retrieval quality: {evaluation_result.quality.value} (confidence: {evaluation_result.confidence:.2f}). Recommendation: {evaluation_result.recommendation.value}."
                            )
                        ],
                    }
        except TimeoutError:
            logger.warning("[REFLECT] Node timed out after %ds", Config.AGENT_TOOL_TIMEOUT)
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            fallback_result = EvaluationResult(
                quality=QualityLevel.PARTIAL,
                confidence=0.5,
                coverage=0.5,
                recommendation=RecommendationAction.ANSWER,
                reasoning="Reflection timed out, proceeding with default assessment.",
            )
            return {
                "evaluation_result": evaluation_result_to_dict(fallback_result),
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content=f"Reflection timed out after {Config.AGENT_TOOL_TIMEOUT}s.")],
            }
        except Exception as e:
            # Fallback to default values (as dict)
            fallback_result = EvaluationResult(
                quality=QualityLevel.PARTIAL,
                confidence=0.5,
                coverage=0.5,
                recommendation=RecommendationAction.ANSWER,
                reasoning="Reflection failed due to error, proceeding with default assessment.",
            )
            return {
                "evaluation_result": evaluation_result_to_dict(fallback_result),
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [
                    AIMessage(
                        content=f"Reflection failed: {str(e)}. Proceeding with default assessment."
                    )
                ],
            }

    return reflect_node

def create_refine_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create refine_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client

    Returns:
        Async refine_node function
    """

    async def refine_node(state: AgentState) -> Dict[str, Any]:
        """
        Refine query based on reflection feedback.

        Args:
            state: Current agent state

        Returns:
            Updated state with refined query
        """
        # Use step-specific query from plan, not full original query
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        # Extract step-specific query from plan (current_step points directly to executing step)
        step_query = state.get("query", "")  # Default to full query
        if plan and current_step < len(plan):
            current_plan_step = plan[current_step]
            # Extract query after colon (e.g., "retrieve: The Man Called Ove" → "The Man Called Ove")
            if ":" in current_plan_step:
                step_query = current_plan_step.split(":", 1)[1].strip()
            else:
                # Fallback: use the step text as-is, removing action keywords
                step_query = current_plan_step
                for keyword in ["retrieve", "search", "find", "document", "documents"]:
                    step_query = step_query.replace(keyword, "").strip()

        # Use existing refined query if available, otherwise use step-specific query
        current_query = state.get("refined_query") or step_query
        try:
            with traced_span(
                "rag.node.refine", {"rag.query.original": current_query}
            ) as span:
                async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
                    openai_client = runtime.get("openai_client")
                    if not openai_client:
                        raise ValueError("OpenAI client is required for refinement node.")

                    refiner = QueryRefiner(
                        openai_client=openai_client,
                        model=Config.OPENAI_MODEL,
                        temperature=Config.OPENAI_TEMPERATURE,
                    )
                    evaluation_result_dict = state.get("evaluation_result")
                    if not evaluation_result_dict:
                        raise ValueError("Evaluation result is required for query refinement.")

                    evaluation_result = dict_to_evaluation_result(evaluation_result_dict)

                    refined_query = await refiner.refine_query(
                        original_query=current_query,
                        eval_result=evaluation_result,
                    )
                    if span is not None:
                        span.set_attribute("rag.query.refined", refined_query)
                        span.set_attribute(
                            "rag.reflection.recommendation",
                            evaluation_result.recommendation.value,
                        )

                    current_refinement_count = state.get("refinement_count", 0)
                    logger.info(
                        f"[REFINED_QUERY] Original: '{current_query}' → Refined: '{refined_query}'"
                    )
                    return {
                        "refined_query": refined_query,
                        "refinement_count": current_refinement_count + 1,
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "messages": [AIMessage(content=f"Refined query to: {refined_query}")],
                    }
        except TimeoutError:
            logger.warning("[REFINE] Node timed out after %ds", Config.AGENT_TOOL_TIMEOUT)
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            return {
                "refined_query": current_query,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content=f"Query refinement timed out after {Config.AGENT_TOOL_TIMEOUT}s.")],
            }
        except Exception as e:
            return {
                "refined_query": current_query,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content="Error during query refinement.")],
            }

    return refine_node
