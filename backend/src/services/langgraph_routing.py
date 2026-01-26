"""
LangGraph conditional routing functions.

These functions decide which node to execute next based on state.
Includes semantic router for deterministic tool selection.
"""

import logging
from typing import Optional, Tuple

from src.services.langgraph_state import AgentState
from src.models.evaluation import RecommendationAction
from src.config.settings import Config

logger = logging.getLogger(__name__)

# ==================== SEMANTIC ROUTER ====================
# Deterministic routing using embedding similarity instead of LLM decisions

_semantic_router = None  # Lazy-loaded singleton


def _init_semantic_router():
    """
    Initialize the semantic router with predefined routes.
    Uses Aurelio's semantic-router library for fast, deterministic routing.
    """
    global _semantic_router
    if _semantic_router is not None:
        return _semantic_router

    try:
        from semantic_router import Route, SemanticRouter
        from semantic_router.encoders import HuggingFaceEncoder

        # Use a small, fast encoder for efficiency
        encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

        # Define routes with example utterances
        # The router will match queries to the most similar route

        web_search_route = Route(
            name="web_search",
            utterances=[
                # Real-time information
                "what's the weather today",
                "what is the weather in Beijing",
                "today's weather forecast",
                "current temperature",
                "weather tomorrow",
                # Travel/tourism (general, not about specific uploaded docs)
                "places to visit in Suzhou",
                "travel recommendations for Shanghai",
                "tourist attractions in Nanjing",
                "best restaurants in Hangzhou",
                "hotels near the airport",
                "things to do in Tokyo",
                "travel guide for Paris",
                "vacation spots in Thailand",
                # Current events and news
                "latest news about",
                "recent events",
                "what happened today",
                "current stock price",
                "live scores",
                # Product/service lookups
                "price of iPhone",
                "reviews of",
                "where to buy",
                "compare prices",
            ],
        )

        retrieve_route = Route(
            name="retrieve",
            utterances=[
                # Document-specific queries
                "what does the document say about",
                "find in my files",
                "search my documents",
                "in the uploaded file",
                "according to the report",
                "based on the PDF",
                "from my documents",
                "what's in my files about",
                # Company/internal data
                "our company policy on",
                "internal guidelines for",
                "Q3 revenue report",
                "quarterly earnings",
                "employee handbook says",
                "meeting notes from",
                "project documentation",
                # Book/content analysis (when user has uploaded)
                "tell me about the book",
                "character analysis",
                "summary of the document",
                "key points from the file",
            ],
        )

        direct_answer_route = Route(
            name="direct_answer",
            utterances=[
                # General knowledge
                "what is photosynthesis",
                "explain quantum physics",
                "how does gravity work",
                "history of the Roman Empire",
                "who invented the telephone",
                # How-to and explanations
                "how to cook pasta",
                "how do I learn Python",
                "explain machine learning",
                "what are the steps to",
                "guide to",
                # Cultural/factual knowledge
                "capital of France",
                "population of China",
                "when was World War 2",
                "who wrote Romeo and Juliet",
                # Greetings and simple queries
                "hello",
                "hi there",
                "thanks",
                "thank you",
                "goodbye",
            ],
        )

        calculator_route = Route(
            name="calculator",
            utterances=[
                "calculate 15 + 27",
                "what is 100 divided by 5",
                "compute the sum of",
                "multiply 12 by 8",
                "percentage of",
                "convert celsius to fahrenheit",
                "square root of 144",
                "calculate compound interest",
            ],
        )

        # Create the semantic router
        # auto_sync="local" builds the index immediately from route utterances
        _semantic_router = SemanticRouter(
            encoder=encoder,
            routes=[
                web_search_route,
                retrieve_route,
                direct_answer_route,
                calculator_route,
            ],
            auto_sync="local",
        )

        logger.info("[SEMANTIC_ROUTER] Initialized successfully with 4 routes")
        return _semantic_router

    except ImportError as e:
        logger.warning(f"[SEMANTIC_ROUTER] Failed to import semantic-router: {e}")
        return None
    except Exception as e:
        logger.warning(f"[SEMANTIC_ROUTER] Failed to initialize: {e}")
        return None


def semantic_route_query(
    query: str, confidence_threshold: float = 0.5
) -> Tuple[Optional[str], float]:
    """
    Route a query using semantic similarity.

    Args:
        query: The user's query
        confidence_threshold: Minimum confidence to return a route (0-1)

    Returns:
        Tuple of (route_name, confidence) or (None, 0) if no confident match
    """
    if not Config.USE_SEMANTIC_ROUTER:
        return None, 0.0

    router = _init_semantic_router()
    if router is None:
        return None, 0.0

    try:
        result = router(query)
        if result and result.name:
            # semantic-router returns similarity score
            confidence = getattr(result, "similarity", 0.7)  # Default if not available
            logger.info(
                f"[SEMANTIC_ROUTER] Query='{query[:50]}...' -> Route={result.name}, Confidence={confidence:.3f}"
            )

            if confidence >= confidence_threshold:
                return result.name, confidence
            else:
                logger.info(
                    f"[SEMANTIC_ROUTER] Confidence {confidence:.3f} below threshold {confidence_threshold}"
                )
                return None, confidence
        return None, 0.0
    except Exception as e:
        logger.warning(f"[SEMANTIC_ROUTER] Error routing query: {e}")
        return None, 0.0


def route_after_planning(state: AgentState) -> str:
    """
    Decide what to do after planning.

    ENTERPRISE RAG: On the first step (current_step == 0), always route to retrieve
    to check internal documents first, regardless of what the planner decided.

    Plan format is "tool_name: description", so we extract the tool name
    by splitting on ":" and do exact match.

    Args:
        state: Current agent state

    Returns:
        Next node name: "retrieve", "tool_calculator", "tool_web_search",
        "tool_download_file", "tool_send_email", "tool_create_documents",
        "generate", or "error"
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    logger.info(f"[ROUTE_AFTER_PLANNING] plan={plan}, current_step={current_step}")

    # ENTERPRISE RAG: Force retrieval on first step to check internal documents first
    if current_step == 0:
        logger.info("[ROUTE_AFTER_PLANNING] Enterprise RAG: Forcing retrieve on first step")
        return "retrieve"

    if not plan:
        logger.info("[ROUTE_AFTER_PLANNING] No plan, returning error")
        return "error"

    # Check bounds
    if current_step >= len(plan):
        logger.info(
            f"[ROUTE_AFTER_PLANNING] current_step {current_step} >= len(plan) {len(plan)}, returning error"
        )
        return "error"

    # Extract tool name from "tool_name: description" format
    step = plan[current_step]
    if ":" in step:
        tool_name = step.split(":")[0].strip().lower()
    else:
        tool_name = step.strip().lower()

    logger.info(
        f"[ROUTE_AFTER_PLANNING] step='{step}', extracted tool_name='{tool_name}'"
    )

    # Exact match on tool name (defined in planning.py Available Tools)
    tool_to_node = {
        "direct_answer": "direct_answer",
        "retrieve": "retrieve",
        "calculator": "tool_calculator",
        "web_search": "tool_web_search",
        "download_file": "tool_download_file",
        "send_email": "tool_send_email",
        "create_documents": "tool_create_documents",
    }

    if tool_name in tool_to_node:
        node = tool_to_node[tool_name]
        logger.info(f"[ROUTE_AFTER_PLANNING] Matched '{tool_name}' -> {node}")
        return node

    # Fallback: if unclear, try retrieval first (safer default)
    logger.info(
        f"[ROUTE_AFTER_PLANNING] No exact match for '{tool_name}', fallback to retrieve"
    )
    return "retrieve"


def route_after_reflection(state: AgentState) -> str:
    """
    Decide what to do based on retrieval quality.

    Args:
        state: Current agent state

    Returns:
        Next node name: "generate", "refine", "tool_web_search", or "error"
    """
    # Safety check: prevent infinite loops
    iteration_count = state.get("iteration_count", 0)
    if iteration_count >= Config.LANGGRAPH_MAX_ITERATIONS:
        logger.info(
            f"[ROUTE_AFTER_REFLECTION] Max iterations reached ({iteration_count}), returning error"
        )
        return "error"

    # evaluation_result is now stored as dict for serialization
    evaluation_result_dict = state.get("evaluation_result", None)
    if not evaluation_result_dict:
        logger.info("[ROUTE_AFTER_REFLECTION] No evaluation_result, returning error")
        return "error"

    # Access recommendation as string from dict, convert to enum for comparison
    recommendation_str = evaluation_result_dict.get("recommendation")
    confidence = evaluation_result_dict.get("confidence", 0)
    quality = evaluation_result_dict.get("quality", "unknown")

    logger.info(
        f"[ROUTE_AFTER_REFLECTION] quality={quality}, confidence={confidence:.3f}, recommendation={recommendation_str}"
    )

    if not recommendation_str:
        logger.info("[ROUTE_AFTER_REFLECTION] No recommendation, returning error")
        return "error"

    recommendation = RecommendationAction(recommendation_str)
    refinement_count = state.get("refinement_count", 0)

    # Max refinement attempts to prevent infinite loops
    if (
        refinement_count >= Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS
        and recommendation == RecommendationAction.REFINE
    ):
        logger.info(
            f"[ROUTE_AFTER_REFLECTION] Max refinements reached ({refinement_count}/{Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS})"
        )
        if state.get("retrieved_docs"):
            logger.info("[ROUTE_AFTER_REFLECTION] Has docs, proceeding to generate")
            return "generate"  # Use what we have
        else:
            logger.info(
                "[ROUTE_AFTER_REFLECTION] No docs after max refinements, returning error"
            )
            return "error"  # No results after 3 tries

    if recommendation == RecommendationAction.ANSWER:
        logger.info("[ROUTE_AFTER_REFLECTION] → GENERATE (quality sufficient)")
        return "generate"
    elif recommendation == RecommendationAction.REFINE:
        logger.info(
            f"[ROUTE_AFTER_REFLECTION] → REFINE (attempt {refinement_count + 1}/{Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS})"
        )
        return "refine"
    elif recommendation == RecommendationAction.EXTERNAL:
        logger.info("[ROUTE_AFTER_REFLECTION] → WEB_SEARCH (external search needed)")
        return "tool_web_search"
    elif recommendation == RecommendationAction.CLARIFY:
        logger.info(
            "[ROUTE_AFTER_REFLECTION] → CLARIFY (asking user for clarification)"
        )
        # User needs to clarify the query - generate with clarification message
        return "generate"
    else:
        logger.info(
            f"[ROUTE_AFTER_REFLECTION] → GENERATE (fallback for {recommendation})"
        )
        # Fallback for any unexpected recommendation
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

    # Safety: max iterations to prevent infinite loops
    if iteration_count >= Config.LANGGRAPH_MAX_ITERATIONS:
        return "end"

    # If final_answer is set, end immediately (either CLARIFY or actual answer)
    # CLARIFY needs user input, so don't continue with remaining plan steps
    if state.get("final_answer"):
        return "end"

    # Check if all plan steps completed
    if current_step >= len(plan):
        return "end"

    # Otherwise continue
    return "continue"
