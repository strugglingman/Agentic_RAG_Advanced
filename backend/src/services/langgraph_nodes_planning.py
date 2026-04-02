"""Planning node implementation."""

from typing import Dict, Any, Callable, Coroutine
import json
import asyncio
import logging

from langchain_core.messages import AIMessage

from src.services.langgraph_state import AgentState, RuntimeContext
from src.services.langgraph_routing import semantic_route_query
from src.services.llm_client import chat_completion_structured
from src.config.settings import Config
from src.prompts import PlanningPrompts
from src.observability.metrics import increment_error, MetricsErrorType

logger = logging.getLogger(__name__)

def create_plan_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create plan_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client

    Returns:
        Async plan_node function
    """

    async def plan_node(state: AgentState) -> Dict[str, Any]:
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
                "messages": [
                    AIMessage(
                        content=f"Continuing with existing plan, step {state.get('current_step', 0) + 1}/{len(existing_plan)}"
                    )
                ],
            }

        query = state.get("query", "")

        # ==================== SEMANTIC ROUTING (FAST PATH) ====================
        # Try semantic router first for high-confidence deterministic routing
        # This skips the LLM call entirely for obvious cases like travel queries
        semantic_route, confidence = semantic_route_query(
            query, confidence_threshold=0.6
        )
        if semantic_route and confidence >= 0.6:
            # High confidence - use semantic route directly, skip LLM planning
            logger.info(
                f"[PLAN] Semantic router matched: route={semantic_route}, confidence={confidence:.3f}"
            )

            # Map semantic route to plan format
            if semantic_route == "web_search":
                plans = [f"web_search: {query}"]
            elif semantic_route == "retrieve":
                plans = [f"retrieve: {query}"]
            elif semantic_route == "direct_answer":
                plans = [f"direct_answer: {query}"]
            else:
                plans = [f"{semantic_route}: {query}"]

            return {
                "plan": plans,
                "current_step": 0,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [
                    AIMessage(
                        content=f"[Semantic Router] Plan: {plans[0]} (confidence: {confidence:.2f})"
                    )
                ],
            }

        # ==================== LLM PLANNING (FALLBACK) ====================
        # Semantic router didn't match with high confidence, use LLM planning
        logger.info(
            f"[PLAN] Semantic router not used or low confidence ({confidence:.3f}), using LLM planning"
        )

        # Build conversation context summary for reference resolution
        conversation_history = runtime.get("conversation_history", [])
        conversation_context = ""
        if conversation_history:
            # Build a concise summary of recent conversation for context
            # Only need recent messages for reference resolution (e.g., "these", "those", "it")
            recent_messages = conversation_history[-Config.PLANNING_CONTEXT_LIMIT :]
            context_parts = []
            for msg in recent_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")[:1500]  # Truncate long messages
                context_parts.append(f"{role}: {content}")
            conversation_context = "\n".join(context_parts)

        # Get available files and attachments from runtime context
        # Note: Attachment metadata is handled by PlanningPrompts._build_files_section()
        # Actual content extraction happens in tool nodes (e.g., create_documents)
        available_files = runtime.get("available_files", [])
        attachment_file_ids = runtime.get("attachment_file_ids", [])

        planning_prompt = PlanningPrompts.create_plan(
            query=query,
            conversation_context=conversation_context,
            available_files=available_files,
            attachment_file_ids=attachment_file_ids,
        )
        try:
            async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
                client = runtime.get("openai_client")
                if not client:
                    raise ValueError("OpenAI client is required for planning node.")

                # Use OpenAI Structured Outputs with strict schema for reliable planning
                # This guarantees 100% schema compliance - LLM cannot generate invalid plans
                plan_schema = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "execution_plan",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "steps": {
                                    "type": "array",
                                    "description": "List of execution steps. Use multiple steps when user requests chained actions (e.g., 'search and download', 'create and email').",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "tool": {
                                                "type": "string",
                                                "enum": [
                                                    "retrieve",
                                                    "web_search",
                                                    "direct_answer",
                                                    "download_file",
                                                    "send_email",
                                                    "create_documents",
                                                    "code_execution",
                                                ],
                                                "description": "The tool to use for this step",
                                            },
                                            "query": {
                                                "type": "string",
                                                "description": "The query or instruction for this tool",
                                            },
                                        },
                                        "required": ["tool", "query"],
                                        "additionalProperties": False,
                                    },
                                    "minItems": 1,
                                    "maxItems": 3,  # Prevent redundant multi-step plans
                                },
                            },
                            "required": ["steps"],
                            "additionalProperties": False,
                        },
                    },
                }

                response = await chat_completion_structured(
                    client=client,
                    messages=[{"role": "user", "content": planning_prompt}],
                    schema=plan_schema,
                    model=Config.OPENAI_MODEL,
                    temperature=Config.OPENAI_TEMPERATURE,
                )

                plan_data = {}
                if response.choices and response.choices[0].message:
                    plan_data = json.loads(response.choices[0].message.content)

                plans = []
                if plan_data and "steps" in plan_data:
                    # Convert structured format to string format for routing
                    # e.g., {"tool": "retrieve", "query": "..."} -> "retrieve: ..."
                    for step in plan_data["steps"]:
                        tool = step.get("tool", "retrieve")
                        step_query = step.get("query", query)
                        plans.append(f"{tool}: {step_query}")

                if not plans:
                    plans = [f"retrieve: {query}"]  # Default fallback

                logger.info(f"[PLAN] Structured plan created: {plans}")

        except TimeoutError:
            logger.warning("[PLAN] Node timed out after %ds", Config.AGENT_TOOL_TIMEOUT)
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            plans = [f"retrieve: {query}"]
        except Exception as e:
            logger.warning(f"[PLAN] Structured output failed, using fallback: {e}")
            plans = [
                f"retrieve: {query}",
            ]  # Fallback plan - single retrieve step

        return {
            "plan": plans,
            "current_step": 0,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": [
                AIMessage(
                    content=f"""
                         Plan created with {len(plans)} steps:\n
                         {"\n".join(["step " + str(i+1) + ": " + p for i, p in enumerate(plans)])}
                         """
                )
            ],
        }

    return plan_node
