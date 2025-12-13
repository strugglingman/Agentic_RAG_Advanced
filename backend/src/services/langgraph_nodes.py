"""
LangGraph agent node implementations.

Each node is created via a factory function that binds RuntimeContext.
Pattern: create_xxx_node(runtime) -> node_function(state) -> updated_state

This allows nodes to access non-serializable objects (collection, openai_client)
without storing them in the checkpointable AgentState.
"""

from typing import Dict, Any, Callable
import json
import logging
from langchain_core.messages import AIMessage
from src.services.langgraph_state import AgentState, RuntimeContext
from src.services.retrieval import retrieve, build_where
from src.services.agent_tools import execute_tool_call
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
from src.utils.safety import enforce_citations
from src.utils.sanitizer import sanitize_text
from src.config.settings import Config
from src.prompts import PlanningPrompts, GenerationPrompts, ToolPrompts
from src.prompts.generation import ContextType

logger = logging.getLogger(__name__)

# ==================== HELPER: EvaluationResult <-> dict ====================


def evaluation_result_to_dict(result: EvaluationResult) -> dict:
    """Convert EvaluationResult object to serializable dict."""
    return {
        "quality": result.quality.value,
        "confidence": result.confidence,
        "coverage": result.coverage,
        "recommendation": result.recommendation.value,
        "reasoning": result.reasoning,
    }


def dict_to_evaluation_result(d: dict) -> EvaluationResult:
    """Convert dict back to EvaluationResult object."""
    if d is None:
        return None
    return EvaluationResult(
        quality=QualityLevel(d["quality"]),
        confidence=d["confidence"],
        coverage=d["coverage"],
        recommendation=RecommendationAction(d["recommendation"]),
        reasoning=d["reasoning"],
    )


# ==================== PLANNING NODE ====================


def create_plan_node(runtime: RuntimeContext) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Factory function to create plan_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client

    Returns:
        plan_node function
    """

    def plan_node(state: AgentState) -> Dict[str, Any]:
        """
        Create execution plan before taking action.

        This is the "thinking" step - decompose complex query into steps.

        IMPORTANT: Only creates plan on first call. On subsequent calls (when looping back
        from verify_node), it returns existing plan without re-planning.

        Args:
            state: Current agent state

        Returns:
            Updated state with plan
        """
        # Check if plan already exists (subsequent call from loop)
        existing_plan = state.get("plan")
        if existing_plan:
            # Plan already exists, don't re-plan, just return state as-is
            return {
                "plan": existing_plan,
                "current_step": state.get("current_step", 0),
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Continuing with existing plan, step {state.get('current_step', 0) + 1}/{len(existing_plan)}"
                    )
                ],
            }

        query = state.get("query", "")
        planning_prompt = PlanningPrompts.create_plan(query)
        try:
            client = runtime.get("openai_client")
            if not client:
                logger.debug("[PLAN_NODE] No OpenAI client found in runtime context.")
                raise ValueError("OpenAI client is required for planning node.")

            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=Config.OPENAI_TEMPERATURE,
                response_format={"type": "json_object"},
            )

            plan_data = {}
            if response.choices and response.choices[0].message:
                plan_data = json.loads(response.choices[0].message.content)

            logger.debug("000000000000000000000000000000000")
            logger.debug(f"Plan node response: {plan_data}")
            logger.debug(f"current step: {state.get('current_step', 0)}")
            plans = []
            if plan_data:
                plans = plan_data.get(
                    "steps", [f"Retrieve documents for: {query}", "Generate answer"]
                )
        except Exception as e:
            plans = [
                f"Retrieve documents for the query: {query}",
                "Generate answer from retrieved documents",
            ]  # Fallback plan

        return {
            "plan": plans,
            "current_step": 0,
            "iteration_count": state.get("iteration_count", 0) + 1,
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

    return plan_node


# ==================== RETRIEVAL NODE ====================


def create_retrieve_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Factory function to create retrieve_node with runtime context bound.

    Args:
        runtime: RuntimeContext with collection, dept_id, user_id, request_data

    Returns:
        retrieve_node function
    """

    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        """
        Retrieve documents from ChromaDB.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved documents
        """
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        logger.debug(f"In tretrieve node and current step is {current_step}")
        is_detour = state.get("evaluation_result") is not None
        if not plan:
            return {
                "retrieved_docs": [],
                "current_step": current_step + 1 if not is_detour else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="No plan or over max steps in plan.")],
            }
        if current_step >= len(plan):
            print("In retrieve node but over max steps in plan.")
            print(f"current step: {current_step}, plan length: {len(plan)}")
            return {
                "retrieved_docs": state.get("retrieved_docs", []),
                "current_step": current_step + 1 if not is_detour else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Over maximum steps in plan.")],
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
                "current_step": current_step + 1 if not is_detour else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Current step is not to retrieve documents.")],
            }

        # Get runtime context
        collection = runtime.get("collection")
        dept_id = runtime.get("dept_id", "")
        user_id = runtime.get("user_id", "")
        request_data = runtime.get("request_data")

        if not collection:
            print("[RETRIEVE_NODE] No collection found in runtime context.")
            return {
                "retrieved_docs": state.get("retrieved_docs", []),
                "current_step": current_step + 1 if not is_detour else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(content="No document collection available for retrieval.")
                ],
            }
        if not dept_id or not user_id:
            print(
                "[RETRIEVE_NODE] Department ID or User ID missing in runtime context."
            )
            return {
                "retrieved_docs": [],
                "current_step": current_step + 1 if not is_detour else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content="Missing department or user context for retrieval."
                    )
                ],
            }

        try:
            # Extract query from current plan step
            # Format: "retrieve: search for information about X" or just the action text
            current_plan_step = plan[current_step]

            # Try to extract query after colon (e.g., "retrieve: query text")
            if ":" in current_plan_step:
                step_query = current_plan_step.split(":", 1)[1].strip()
            else:
                # Fallback: use the step text as-is, removing action keywords
                step_query = current_plan_step
                for keyword in ["retrieve", "search", "find", "document", "documents"]:
                    step_query = step_query.replace(keyword, "").strip()

            # Use refined query if available (from refinement loop), otherwise use step-specific query
            query = state.get("refined_query") or step_query or state.get("query")

            where = build_where(request_data, dept_id, user_id)
            ctx, err = retrieve(
                query=query,
                collection=collection,
                dept_id=dept_id,
                user_id=user_id,
                top_k=Config.TOP_K,
                where=where,
                use_hybrid=Config.USE_HYBRID,
                use_reranker=Config.USE_RERANKER,
            )
            print("********************************************************")
            print(f"Retrieved {len(ctx)} documents for query: {query}")
            print("Retrieval error:", err)
            if not ctx:
                print("No documents retrieved.")
                return {
                    "retrieved_docs": [],
                    "current_step": current_step + 1 if not is_detour else current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [AIMessage(content="No relevant documents found.")],
                }

            print(f"Retrieved {len(ctx)} documents.")

            # Store retrieved docs PER STEP to avoid mixing contexts from different questions
            step_contexts = state.get("step_contexts", {})
            step_contexts[current_step] = {
                "type": "retrieval",
                "docs": ctx,
                "plan_step": plan[current_step] if current_step < len(plan) else "",
            }

            return {
                "retrieved_docs": ctx,  # Keep for backward compatibility with reflection
                "step_contexts": step_contexts,
                "tools_used": state.get("tools_used", []) + ["search_documents"],
                "current_step": current_step + 1 if not is_detour else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Retrieved {len(ctx)} documents.")],
            }
        except Exception as e:
            print(f"Retrieval node error: {str(e)}")
            return {
                "retrieved_docs": [],
                "current_step": current_step + 1 if not is_detour else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Error during document retrieval.")],
            }

    return retrieve_node


# ==================== REFLECTION NODE ====================


def create_reflect_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Factory function to create reflect_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client

    Returns:
        reflect_node function
    """

    def reflect_node(state: AgentState) -> Dict[str, Any]:
        """
        Evaluate retrieval quality (self-reflection).

        Args:
            state: Current agent state

        Returns:
            Updated state with quality assessment (evaluation_result as dict)
        """
        try:
            print("In reflect node")
            print(f"current step: {state.get('current_step', 0)}")

            # Use step-specific query from plan, not full original query
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)

            # IMPORTANT: retrieve_node already incremented current_step
            # So we need to look at current_step - 1 to get the step that was just executed
            actual_step = current_step - 1 if current_step > 0 else 0

            # Extract step-specific query from plan
            step_query = state.get("query", "")  # Default to full query
            if plan and actual_step < len(plan):
                current_plan_step = plan[actual_step]
                # Extract query after colon (e.g., "retrieve: The Man Called Ove" → "The Man Called Ove")
                if ":" in current_plan_step:
                    step_query = current_plan_step.split(":", 1)[1].strip()
                else:
                    # Fallback: use the step text as-is, removing action keywords
                    step_query = current_plan_step
                    for keyword in [
                        "retrieve",
                        "search",
                        "find",
                        "document",
                        "documents",
                    ]:
                        step_query = step_query.replace(keyword, "").strip()

            # Use refined query if available (from refinement loop), otherwise use step-specific query
            query = state.get("refined_query") or step_query
            retrieved_docs = state.get("retrieved_docs", [])

            print(f"[REFLECT_NODE] Actual step being evaluated: {actual_step}")
            print(f"[REFLECT_NODE] Evaluating with query: {query}")

            # Create evaluation criteria
            evaluator_criteria = EvaluationCriteria(
                query=query,
                contexts=retrieved_docs,
                mode=ReflectionMode.BALANCED,
            )
            reflection_config = ReflectionConfig.from_settings(Config)
            openai_client = runtime.get("openai_client")

            if not openai_client:
                print("[REFLECT_NODE] No OpenAI client found in runtime context.")
                raise ValueError("OpenAI client is required for reflection node.")

            evaluator = RetrievalEvaluator(
                config=reflection_config,
                openai_client=openai_client,
            )
            evaluation_result = evaluator.evaluate(evaluator_criteria)

            print(f"Evaluation result: {evaluation_result}")

            # Convert EvaluationResult to dict for serialization
            return {
                "evaluation_result": evaluation_result_to_dict(evaluation_result),
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Retrieval quality: {evaluation_result.quality.value} (confidence: {evaluation_result.confidence:.2f}). Recommendation: {evaluation_result.recommendation.value}."
                    )
                ],
            }
        except Exception as e:
            print(f"[REFLECT_NODE] Error during reflection: {e}")
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
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Reflection failed: {str(e)}. Proceeding with default assessment."
                    )
                ],
            }

    return reflect_node


# ==================== TOOL EXECUTOR NODE ====================


def create_tool_calculator_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Factory function to create tool_calculator_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client, collection, dept_id, user_id

    Returns:
        tool_calculator_node function
    """

    def tool_calculator_node(state: AgentState) -> Dict[str, Any]:
        """
        Execute non-retrieval tools (calculator) using LLM function calling.

        This node uses OpenAI function calling to let the LLM decide tool arguments,
        similar to agent_service.py approach.

        This node can be called in two ways:
        1. PLANNED: From route_after_planning (part of the original plan) → increments current_step
        2. DETOUR: From route_after_reflection (ad-hoc tool call) → does NOT increment current_step

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        try:
            print("22222222222222222222222222")
            print("In tool_calculator_node")
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)
            query = state.get("query", "")

            # Get OpenAI client from runtime
            client = runtime.get("openai_client")
            if not client:
                print("[TOOL_CALCULATOR] No OpenAI client found in runtime context.")
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,  # Don't increment on error
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "error": "OpenAI client required for tool execution",
                }

            # Determine if this is a DETOUR call BEFORE building prompt
            is_detour = state.get("evaluation_result") is not None

            # Build prompt for LLM tool calling
            # For planned calls: use plan[current_step]
            # For detour calls: prefer refined_query (specific to current step), else plan[current_step - 1]
            if plan and current_step < len(plan) and not is_detour:
                # PLANNED call - use current step
                action_step = plan[current_step]
                prompt = ToolPrompts.calculator_prompt(action_step, is_detour=False)
            elif is_detour:
                # DETOUR call - prefer refined_query (from refine_node), else previous step
                refined_query = state.get("refined_query")
                if refined_query:
                    # refined_query is specific to the step we're supplementing
                    task_query = refined_query
                elif plan and current_step > 0:
                    # Fallback to previous step (the one we're supplementing)
                    task_query = plan[current_step - 1]
                else:
                    # Final fallback
                    task_query = query
                prompt = ToolPrompts.calculator_prompt(task_query, is_detour=True)
            else:
                # True fallback - no plan, no detour
                prompt = ToolPrompts.fallback_prompt(query, "calculator")

            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                tools=TOOL_CALCULATOR,
                tool_choice="auto",
                temperature=0.1,
            )

            # is_detour already determined above for prompt building
            is_planned_call = not is_detour

            # Check if LLM called a tool
            if not response.choices[0].message.tool_calls:
                print("[TOOL_CALCULATOR] No tool call detected in LLM response.")
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": (
                        current_step + 1 if is_planned_call else current_step
                    ),
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [AIMessage(content="No tool was called by the LLM.")],
                }

            # Execute the tool call (same pattern as agent_service.py)
            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Build context for tool execution from runtime
            context = {
                "collection": runtime.get("collection"),
                "dept_id": runtime.get("dept_id"),
                "user_id": runtime.get("user_id"),
                "openai_client": client,
                "request_data": runtime.get("request_data", {}),
            }

            # Execute the tool
            result = execute_tool_call(tool_name, tool_args, context)

            # Update tool results
            tool_results = state.get("tool_results", {})
            tool_key = f"{tool_name}_step_{current_step}"
            if tool_key not in tool_results:
                tool_results[tool_key] = []
            tool_results[tool_key].append(
                {
                    "step": current_step,
                    "args": tool_args,
                    "result": result,
                    "query": query,
                }
            )

            print(f"Tool {tool_name} executed with result: {result[:200]}...")

            # Store tool result PER STEP to avoid mixing contexts
            step_contexts = state.get("step_contexts", {})
            step_contexts[current_step] = {
                "type": "tool",
                "tool_name": tool_name,
                "result": result,
                "args": tool_args,
                "plan_step": (
                    plan[current_step] if plan and current_step < len(plan) else ""
                ),
            }

            return {
                "tools_used": state.get("tools_used", []) + [tool_name],
                "tool_results": tool_results,
                "step_contexts": step_contexts,
                "current_step": current_step + 1 if is_planned_call else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Executed {tool_name} with result: {result[:200]}..."
                    )
                ],
            }

        except Exception as e:
            print(f"[TOOL_CALCULATOR] Error: {e}")
            # On error, check if planned to decide increment
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)
            evaluation_result = state.get("evaluation_result", None)
            is_planned_call = evaluation_result is None
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "current_step": current_step + 1 if is_planned_call else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Tool execution failed: {str(e)}",
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Tool execution failed: {str(e)}")],
            }

    return tool_calculator_node


def create_tool_web_search_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Factory function to create tool_web_search_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client, collection, dept_id, user_id

    Returns:
        tool_web_search_node function
    """

    def tool_web_search_node(state: AgentState) -> Dict[str, Any]:
        """
        Execute web search tool using LLM function calling.

        This node can be called in two ways:
        1. PLANNED: From route_after_planning → increments current_step
        2. DETOUR: From route_after_reflection → does NOT increment current_step

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        try:
            print("33333333333333333333333333")
            print("In tool_web_search_node")
            print(f"current step: {state.get('current_step', 0)}")
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)
            query = state.get("query", "")

            # Get OpenAI client from runtime
            client = runtime.get("openai_client")
            if not client:
                print("[TOOL_WEB_SEARCH] No OpenAI client found in runtime context.")
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "error": "OpenAI client required for tool execution",
                }

            # Determine if this is a DETOUR call BEFORE building prompt
            is_detour = state.get("evaluation_result") is not None

            # Build prompt for LLM tool calling
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Full Plan:", plan)
            print(f"Current step: {current_step}")
            print(f"Is detour: {is_detour}")
            if plan and current_step < len(plan) and not is_detour:
                action_step = plan[current_step]
                prompt = ToolPrompts.web_search_prompt(action_step, is_detour=False)
            elif is_detour:
                refined_query = state.get("refined_query")
                if refined_query:
                    task_query = refined_query
                elif plan and current_step > 0:
                    task_query = plan[current_step - 1]
                else:
                    task_query = query
                prompt = ToolPrompts.web_search_prompt(task_query, is_detour=True)
            else:
                prompt = ToolPrompts.fallback_prompt(query, "web_search")

            print("Prompt for web search tool calling:", prompt)

            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                tools=TOOL_WEB_SEARCH,
                tool_choice="auto",
                temperature=0.1,
            )

            is_planned_call = not is_detour

            if not response.choices[0].message.tool_calls:
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": (
                        current_step + 1 if is_planned_call else current_step
                    ),
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [AIMessage(content="No tool was called by the LLM.")],
                }

            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Build context for tool execution from runtime
            context = {
                "collection": runtime.get("collection"),
                "dept_id": runtime.get("dept_id"),
                "user_id": runtime.get("user_id"),
                "openai_client": client,
                "request_data": runtime.get("request_data", {}),
            }

            result = execute_tool_call(tool_name, tool_args, context)

            tool_results = state.get("tool_results", {})
            tool_key = f"{tool_name}_step_{current_step}"
            if tool_key not in tool_results:
                tool_results[tool_key] = []
            tool_results[tool_key].append(
                {
                    "step": current_step,
                    "args": tool_args,
                    "result": result,
                    "query": query,
                }
            )

            print(f"Tool {tool_name} executed with result: {result}...")

            step_contexts = state.get("step_contexts", {})
            step_contexts[current_step] = {
                "type": "tool",
                "tool_name": tool_name,
                "result": result,
                "args": tool_args,
                "plan_step": (
                    plan[current_step] if plan and current_step < len(plan) else ""
                ),
            }

            return {
                "tools_used": state.get("tools_used", []) + [tool_name],
                "tool_results": tool_results,
                "step_contexts": step_contexts,
                "current_step": current_step + 1 if is_planned_call else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Executed {tool_name} with result: {result[:200]}..."
                    )
                ],
            }

        except Exception as e:
            print(f"[TOOL_WEB_SEARCH] Error: {e}")
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)
            evaluation_result = state.get("evaluation_result", None)
            is_planned_call = evaluation_result is None
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "current_step": current_step + 1 if is_planned_call else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Tool execution failed: {str(e)}",
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Tool execution failed: {str(e)}")],
            }

    return tool_web_search_node


# ==================== DIRECT ANSWER NODE ====================
# TODO: Implement direct_answer node for handling general knowledge questions
# This node should:
# 1. Extract question from plan step (format: "direct_answer: question")
# 2. Call LLM directly without external tools (no retrieval/web search)
# 3. Store result in tool_results and step_contexts like other tool nodes
# 4. Follow the same pattern as create_tool_calculator_node and create_tool_web_search_node
# 5. Increment current_step after completion
#
def create_direct_answer_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Factory function to create direct_answer_node with runtime context bound.
    """
    def direct_answer_node(state: AgentState) -> Dict[str, Any]:
        # TODO: Implement the logic here
        # - Get OpenAI client from runtime
        # - Extract question from plan[current_step]
        # - Call LLM with simple prompt (no tools, no retrieval)
        # - Store result in step_contexts with type="direct_answer"
        # - Return updated state
        pass
    return direct_answer_node


# ==================== REFINEMENT NODE ====================
def create_refine_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Factory function to create refine_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client

    Returns:
        refine_node function
    """

    def refine_node(state: AgentState) -> Dict[str, Any]:
        """
        Refine query based on reflection feedback.

        Args:
            state: Current agent state

        Returns:
            Updated state with refined query
        """
        print("In refine_node")
        print(f"current step: {state.get('current_step', 0)}")

        # Use step-specific query from plan, not full original query
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        # IMPORTANT: retrieve_node already incremented current_step
        # So we need to look at current_step - 1 to get the step that was just executed
        actual_step = current_step - 1 if current_step > 0 else 0

        # Extract step-specific query from plan
        step_query = state.get("query", "")  # Default to full query
        if plan and actual_step < len(plan):
            current_plan_step = plan[actual_step]
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
        print(f"[REFINE_NODE] Actual step being refined: {actual_step}")
        print(f"[REFINE_NODE] Refining query: {current_query}")
        try:
            openai_client = runtime.get("openai_client")
            if not openai_client:
                print("[REFINE_NODE] No OpenAI client found in runtime context.")
                raise ValueError("OpenAI client is required for refinement node.")

            refiner = QueryRefiner(
                openai_client=openai_client,
                model=Config.OPENAI_MODEL,
                temperature=Config.OPENAI_TEMPERATURE,
            )
            # Convert dict back to EvaluationResult for refiner
            evaluation_result_dict = state.get("evaluation_result")
            if not evaluation_result_dict:
                print("[REFINE_NODE] No evaluation result found in state.")
                raise ValueError("Evaluation result is required for query refinement.")

            evaluation_result = dict_to_evaluation_result(evaluation_result_dict)

            refined_query = refiner.refine_query(
                original_query=current_query,
                eval_result=evaluation_result,
            )

            current_refinement_count = state.get("refinement_count", 0)
            print(
                f"------------------------------------Refinement count: {current_refinement_count}"
            )
            print(f"------------------------------------Refined query: {refined_query}")
            return {
                "refined_query": refined_query,
                "refinement_count": current_refinement_count + 1,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Refined query to: {refined_query}")],
            }
        except Exception as e:
            print(f"Refinement node error: {e}")
            return {
                "refined_query": current_query,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Error during query refinement.")],
            }

    return refine_node


# ==================== GENERATION NODE ====================


def create_generate_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Factory function to create generate_node with runtime context.

    Args:
        runtime: Runtime context with non-serializable objects

    Returns:
        generate_node function with runtime bound via closure
    """

    def generate_node(state: AgentState) -> Dict[str, Any]:
        """
        Generate answer from retrieved documents.

        Args:
            state: Current agent state

        Returns:
            Updated state with generated answer
        """
        try:
            print("In generate_node")
            print(f"current step: {state.get('current_step', 0)}")
            openai_client = runtime.get("openai_client", None)
            if not openai_client:
                print("No OpenAI client found in runtime context.")
                raise ValueError("OpenAI client is required for generation node.")

            # Check if this is a CLARIFY recommendation from reflection
            evaluation_result_dict = state.get("evaluation_result")
            evaluation_result = dict_to_evaluation_result(evaluation_result_dict)
            if (
                evaluation_result
                and evaluation_result.recommendation == RecommendationAction.CLARIFY
            ):
                # Generate clear clarification request message using prompt registry
                clarification_message = GenerationPrompts.clarification_message(
                    evaluation_result.reasoning
                )
                return {
                    "draft_answer": clarification_message,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [AIMessage(content=clarification_message)],
                }

            # Get ONLY current step's context (per-step isolation)
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)
            step_contexts = state.get("step_contexts", {})

            # Get context for the running step (current_step was already incremented by the executor node)
            target_step = current_step - 1

            if target_step not in step_contexts:
                print(f"[GENERATE] No context found for step {target_step}")
                return {
                    "draft_answer": "",
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [
                        AIMessage(
                            content="No context available to generate answer from."
                        )
                    ],
                }

            step_ctx = step_contexts[target_step]
            plan_step_desc = step_ctx.get("plan_step", "")

            # Detect if this is a web search result
            is_web_search = (
                step_ctx.get("type") == "tool"
                and step_ctx.get("tool_name") == "web_search"
            )

            print(
                f"[GENERATE] Generating answer for step {target_step}: {plan_step_desc}"
            )
            print(f"[GENERATE] Is web search: {is_web_search}")

            # Build numbered context from ONLY this step's data
            contexts = []
            context_num = 1

            print(
                "*******************In generate node, show step_context (comment temprarily):"
            )
            # print("Step context:", step_ctx)

            if step_ctx["type"] == "retrieval":
                # Retrieved documents from this step only
                docs = step_ctx.get("docs", [])
                for doc in docs:
                    chunk = doc.get("chunk", str(doc))
                    source = doc.get("source", "unknown")
                    page = doc.get("page", 0)

                    header = f"Context {context_num} (Source: {source}"
                    if page > 0:
                        header += f", Page: {page}"
                    header += "):\n"

                    score_info = ""
                    if doc.get("hybrid") is not None:
                        score_info += f"Hybrid score: {doc['hybrid']:.2f}"
                    if doc.get("rerank") is not None:
                        if score_info:
                            score_info += ", "
                        score_info += f"Rerank score: {doc['rerank']:.2f}"

                    context_entry = f"{header}{chunk}"
                    if score_info:
                        context_entry += f"\n{score_info}"

                    contexts.append(context_entry)
                    context_num += 1

            elif step_ctx["type"] == "tool":
                # Tool result from this step only
                tool_name = step_ctx.get("tool_name", "unknown")
                result_text = step_ctx.get("result", "")
                args = step_ctx.get("args", {})

                header = (
                    f"Context {context_num} (Tool: {tool_name}, Step: {target_step}):\n"
                )
                if args:
                    header += f"Arguments: {args}\n"

                context_entry = f"{header}Result: {result_text}"
                contexts.append(context_entry)
                context_num += 1

            # TODO: Add handling for direct_answer type
            # elif step_ctx["type"] == "direct_answer":
            #     # Direct answer from LLM's built-in knowledge
            #     result_text = step_ctx.get("result", "")
            #     question = step_ctx.get("question", "")
            #     header = f"Context {context_num} (Direct Answer for: {question}):\n"
            #     context_entry = f"{header}{result_text}"
            #     contexts.append(context_entry)
            #     context_num += 1

            if not contexts:
                print("No context available for answer generation.")
                raise ValueError(
                    "No context available from retrieved documents or tool results."
                )

            final_context = "\n\n".join(contexts)

            # Build system prompt - different rules for web search vs document retrieval
            # Use prompt registry for context-aware prompts
            if is_web_search:
                context_type = ContextType.WEB_SEARCH
            else:
                context_type = ContextType.DOCUMENT

            system_prompt = GenerationPrompts.get_system_prompt(context_type)

            # Use the SPECIFIC plan step as the question (not the full multi-part query)
            # This ensures the LLM answers ONLY what this step is about
            refined_query = state.get("refined_query", None)

            # Extract the task from plan step (format: "tool_name: description")
            step_question = plan_step_desc
            if ":" in plan_step_desc:
                step_question = plan_step_desc.split(":", 1)[1].strip()

            # Build user message using prompt registry
            user_message_with_context = GenerationPrompts.build_user_message(
                question=step_question,
                context=final_context,
                refined_query=refined_query,
            )
            # Comment out too much citation prompt
            # Include bracket citations [n] for every sentence that uses information.
            # At the end of your answer, cite the sources you used. For each source file, list the specific page numbers
            # from the contexts you referenced. Format: 'Sources: filename.pdf (pages 15, 23), filename2.pdf (page 7)'

            # Build messages list: system + conversation_history + current query with contexts
            openai_messages = [{"role": "system", "content": system_prompt}]

            # Use pre-loaded conversation history from runtime (loaded in chat.py, avoids async issues)
            conversation_history = runtime.get("conversation_history", [])
            if conversation_history:
                for h in conversation_history:
                    sanitized_msg = {
                        "role": h.get("role", "user"),
                        "content": sanitize_text(h.get("content", ""), max_length=5000),
                    }
                    openai_messages.append(sanitized_msg)
                print(
                    f"[GENERATE] Using {len(conversation_history)} messages from pre-loaded conversation history"
                )

            # Add current query with contexts
            openai_messages.append(
                {"role": "user", "content": user_message_with_context}
            )

            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print(
                "Sending messages to OpenAI for answer generation (comment temprarily)..."
            )
            # print(f"Messages: {openai_messages}")
            response = openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=openai_messages,
                max_tokens=Config.CHAT_MAX_TOKENS,
                temperature=Config.OPENAI_TEMPERATURE,
            )
            draft_answer = ""
            if response.choices and response.choices[0].message:
                draft_answer = response.choices[0].message.content

            print("Generated draft answer successfully.")
            print(f"Draft answer: {draft_answer[:200]}...")
            return {
                "draft_answer": draft_answer,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Generated answer successfully.")],
            }

        except Exception as e:
            print(f"Generation node error: {e}")
            return {
                "draft_answer": "",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Error during answer generation.")],
            }

    return generate_node


# ==================== VERIFICATION NODE ====================


def create_verify_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Factory function to create verify_node with runtime context.

    Args:
        runtime: Runtime context with non-serializable objects

    Returns:
        verify_node function with runtime bound via closure
    """

    def verify_node(state: AgentState) -> Dict[str, Any]:
        """
        Verify citations and route to next step or finalize answer.

        This node handles two scenarios:
        1. INTERMEDIATE: More plan steps remain → verify and continue to next step
        2. FINAL: All plan steps complete → verify and create final_answer

        The key insight from Plan-Execute pattern:
        - Data gathering phase: Execute plan steps, accumulate results
        - Generation phase: ONE final answer at end synthesizing all data

        Args:
            state: Current agent state

        Returns:
            Updated state with verified answer
        """
        draft_answer = state.get("draft_answer", "")
        print("--------------------In VERIFY NODE")
        print(f"current step: {state.get('current_step', 0)}")
        print(f"draft answer: {draft_answer[:200]}...")
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        # Check if there are more plan steps remaining
        has_more_steps = plan and current_step < len(plan)

        if not draft_answer:
            return {
                "final_answer": (
                    "" if not has_more_steps else None
                ),  # Only set final_answer at end
                "evaluation_result": None,  # Clear evaluation_result for next cycle
                "refined_query": None,  # Clear refined_query for next step
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="No draft answer to verify.")],
            }

        # Check if this is a CLARIFY recommendation - pass through without verification
        evaluation_result_dict = state.get("evaluation_result")
        evaluation_result = dict_to_evaluation_result(evaluation_result_dict)
        if (
            evaluation_result
            and evaluation_result.recommendation == RecommendationAction.CLARIFY
        ):
            # Clarification messages don't need citation enforcement
            # CLARIFY always ends the flow (user needs to provide input)
            return {
                "final_answer": draft_answer,  # CLARIFY always creates final_answer
                "evaluation_result": None,  # Clear evaluation_result for next cycle
                "refined_query": None,  # Clear refined_query for next step
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Clarification request prepared.")],
            }

        try:
            # Calculate valid context IDs from both sources
            retrieved_docs = state.get("retrieved_docs", [])
            tool_results = state.get("tool_results", {})

            valid_ids = []
            context_num = 1

            # Add IDs for retrieved documents
            for _ in retrieved_docs:
                valid_ids.append(context_num)
                context_num += 1

            # Add IDs for tool results
            for _, results in tool_results.items():
                for _ in results:
                    valid_ids.append(context_num)
                    context_num += 1

            if not valid_ids:
                print("[VERIFY] No contexts available, cannot verify citations")
                error_message = "I apologize, but I couldn't find any relevant information to answer your question. Please try rephrasing your query or providing more details."
                return {
                    "final_answer": error_message,  # Always set final_answer to END
                    "evaluation_result": None,  # Clear evaluation_result for next cycle
                    "refined_query": None,  # Clear refined_query for next step
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [
                        AIMessage(
                            content="Warning: No contexts to verify citations against."
                        )
                    ],
                }

            # Optional: Enforce citations - drops sentences without valid citations
            clean_answer, all_supported = (
                enforce_citations(draft_answer, valid_ids)
                if Config.ENFORCE_CITATIONS
                else (draft_answer, True)
            )

            if not all_supported:
                print(f"[VERIFY] Dropped unsupported sentences from answer")
                print(f"[VERIFY] Valid context IDs: {valid_ids}")

            # Store answer for THIS STEP
            target_step = current_step - 1  # Answer was for previous step
            step_answers = state.get("step_answers", [])
            step_contexts = state.get("step_contexts", {})

            step_ctx = step_contexts.get(target_step, {})
            plan_step_desc = step_ctx.get("plan_step", f"Step {target_step}")

            step_answers.append(
                {
                    "step": target_step,
                    "question": plan_step_desc,
                    "answer": clean_answer,
                    "context_count": len(valid_ids),
                }
            )

            print(f"[VERIFY] Stored answer for step {target_step}: {plan_step_desc}")
            print(f"Has more steps: {has_more_steps}")

            # If all steps complete, concatenate all step answers
            if not has_more_steps:
                # Build final answer from all step answers
                if len(step_answers) == 1:
                    # Single step - return answer directly
                    final_answer = step_answers[0]["answer"]
                else:
                    # Multiple steps - format with sections
                    answer_parts = []
                    for step_ans in step_answers:
                        # Extract task description (remove tool name prefix)
                        task = step_ans["question"]
                        if ":" in task:
                            task = task.split(":", 1)[1].strip()

                        answer_parts.append(
                            f"**{task.capitalize()}**\n{step_ans['answer']}"
                        )

                    final_answer = "\n\n".join(answer_parts)

                print(
                    f"[VERIFY] Final answer composed from {len(step_answers)} step(s)"
                )
            else:
                final_answer = None

            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Output step answers and final answer (if complete).")
            print(f"Step answers: {step_answers}")
            print(f"Final answer: {final_answer}")
            return {
                "final_answer": final_answer,
                "step_answers": step_answers,
                "evaluation_result": None,  # Clear evaluation_result for next cycle
                "refined_query": None,  # Clear refined_query for next step
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Answer verified and citations checked.")],
            }

        except Exception as e:
            print(f"Verification node error: {e}")
            return {
                "final_answer": draft_answer if not has_more_steps else None,
                "evaluation_result": None,  # Clear evaluation_result for next cycle
                "refined_query": None,  # Clear refined_query for next step
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Error during verification: {str(e)}")],
            }

    return verify_node


# ==================== ERROR HANDLER NODE ====================
def error_handler_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle errors gracefully.

    Args:
        state: Current agent state

    Returns:
        Updated state with error message
    """
    print("------------------In ERROR HANDLER NODE-----------------------")
    error_message = state.get("error", "An unknown error occurred.")
    print(f"Error message: {error_message}")
    msgs = state.get("messages", [])
    print(msgs[0:20])
    # print(state.get("messages", []))
    return {
        "final_answer": f"Error: {error_message}",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": state.get("messages", [])
        + [AIMessage(content=f"Handled error: {error_message}")],
    }


# Define available tools for function calling
TOOL_CALCULATOR = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations. Evaluates mathematical expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '15 * 0.2', '(100 + 50) / 2')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

TOOL_WEB_SEARCH = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use when internal documents don't have the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding information on the web",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]
