"""
LangGraph agent node implementations.

Each node is a pure function: state → updated state
"""

from typing import Dict, Any
import json
from langchain_core.messages import AIMessage
from src.services.langgraph_state import AgentState
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
from src.services.conversation_service import ConversationService


# ==================== PLANNING NODE ====================


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
    planning_prompt = f"""
    You are a planning assistant. Create a step-by-step plan to answer the following query:
        Query: {query}

    Available Tools:
    1. search_documents: Search internal company documents and knowledge base. Use for questions about company data, policies, reports, or uploaded documents.
    2. calculator: Perform mathematical calculations and numerical operations. Use for arithmetic, percentages, or any numerical computation.
    3. web_search: Search the web for current, up-to-date information. Use when internal documents don't have the answer or when external/recent information is needed.

    Instructions:
    - Create steps that use these tools appropriately
    - Each step should be clear and actionable
    - Combine tools if needed (e.g., retrieve data, then calculate)

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
        client = state.get("openai_client", None)
        if not client:
            print("No OpenAI client found in state.")
            raise ValueError("OpenAI client is required for planning node.")

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


# ==================== RETRIEVAL NODE ====================


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
    if not plan or current_step >= len(plan):
        return {
            "retrieved_docs": [],
            "current_step": current_step + 1,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": state.get("messages", [])
            + [AIMessage(content="No plan or over max steps in plan.")],
        }
    action = plan[current_step].lower()
    if (
        "retrieve" not in action
        and "search" not in action
        and "document" not in action
        and "find" not in action
    ):
        return {
            "retrieved_docs": [],
            "current_step": current_step + 1,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": state.get("messages", [])
            + [AIMessage(content="Current step is not to retrieve documents.")],
        }

    dept_id = state.get("dept_id", "")
    user_id = state.get("user_id", "")
    collection = state.get("collection", None)
    if not collection:
        print("No collection found in state.")
        return {
            "retrieved_docs": [],
            "current_step": current_step + 1,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": state.get("messages", [])
            + [AIMessage(content="No document collection available for retrieval.")],
        }
    if not dept_id or not user_id:
        print("Department ID or User ID missing in state.")
        return {
            "retrieved_docs": [],
            "current_step": current_step + 1,
            "iteration_count": state.get("iteration_count", 0) + 1,
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
                "current_step": current_step + 1,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="No relevant documents found.")],
            }

        return {
            "retrieved_docs": ctx,
            # Log for conceptual consistency since this retrieval is not from a tool call like web search and calculator
            "tools_used": state.get("tools_used", []) + ["search_documents"],
            "current_step": current_step + 1,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": state.get("messages", [])
            + [AIMessage(content=f"Retrieved {len(ctx)} documents.")],
        }
    except Exception as e:
        print(f"Retrieval node error: {e}")
        return {
            "retrieved_docs": [],
            "current_step": state.get("current_step", 0) + 1,
            "iteration_count": state.get("iteration_count", 0) + 1,
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
    try:
        query = state.get("query", "")
        retrieved_docs = state.get("retrieved_docs", [])

        # Create evaluation criteria
        evaluator_criteria = EvaluationCriteria(
            query=query,
            contexts=retrieved_docs,  # ✅ Correct parameter name
            mode=ReflectionMode.BALANCED,
        )

        reflection_config = ReflectionConfig.from_settings(Config)
        openai_client = state.get("openai_client", None)

        if not openai_client:
            print("[REFLECT_NODE] No OpenAI client found in state.")
            raise ValueError("OpenAI client is required for reflection node.")

        evaluator = RetrievalEvaluator(
            config=reflection_config,
            openai_client=openai_client,
        )
        evaluation_result = evaluator.evaluate(evaluator_criteria)

        return {
            "evaluation_result": evaluation_result,  # Store full evaluation result
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
        # Fallback to default values
        return {
            "evaluation_result": EvaluationResult(
                quality=QualityLevel.PARTIAL,
                confidence=0.5,
                coverage=0.5,
                recommendation=RecommendationAction.ANSWER,
                reasoning="Reflection failed due to error, proceeding with default assessment.",
            ),
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": state.get("messages", [])
            + [
                AIMessage(
                    content=f"Reflection failed: {str(e)}. Proceeding with default assessment."
                )
            ],
        }


# ==================== TOOL EXECUTOR NODE ====================


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
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        query = state.get("query", "")

        # Detect if this is a DETOUR call from reflection
        # Detour indicators: evaluation_result exists (set by reflect_node, not cleared until verify_node)
        # NOTE: We cannot rely on evaluation_result existence alone, as it persists
        # Instead, we skip the plan bounds check for now and let iteration_count provide safety
        # The tool will work for both planned and detour calls

        # Get OpenAI client from state
        client = state.get("openai_client")
        if not client:
            print("[TOOL_CALCULATOR] No OpenAI client found in state.")
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "current_step": current_step,  # Don't increment on error
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": "OpenAI client required for tool execution",
            }

        # Build prompt for LLM tool calling
        # For planned calls: use plan[current_step]
        # For detour calls: use generic prompt
        if plan and current_step < len(plan):
            action_step = plan[current_step]
            prompt = f"Based on the user's query and the current plan step, call the appropriate tool.\n\nUser Query: {query}\n\nCurrent Step: {action_step}"
        else:
            # Detour or no plan - generic prompt
            prompt = f"Based on the user's query, call the appropriate calculator tool.\n\nUser Query: {query}"

        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            tools=TOOL_CALCULATOR,
            tool_choice="auto",
            temperature=0.1,
        )

        # Determine if this is a DETOUR call using evaluation_result
        # If evaluation_result exists → DETOUR (came from reflect_node)
        # If evaluation_result is None → PLANNED (came from route_after_planning, after verify cleared it)
        # This works because verify_node clears evaluation_result at the end of each reflection cycle
        is_detour = state.get("evaluation_result") is not None
        is_planned_call = not is_detour

        # Check if LLM called a tool
        if not response.choices[0].message.tool_calls:
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "current_step": current_step + 1 if is_planned_call else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="No tool was called by the LLM.")],
            }

        # Execute the tool call (same pattern as agent_service.py)
        tool_call = response.choices[0].message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        # Build context for tool execution
        context = {
            "collection": state.get("collection"),
            "dept_id": state.get("dept_id"),
            "user_id": state.get("user_id"),
            "openai_client": client,
            "request_data": state.get("request_data", {}),
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

        return {
            "tools_used": state.get("tools_used", []) + [tool_name],
            "tool_results": tool_results,
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


def tool_web_search_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute web search tool using LLM function calling.

    This node uses OpenAI function calling to let the LLM decide tool arguments,
    similar to agent_service.py approach.

    This node can be called in two ways:
    1. PLANNED: From route_after_planning (part of the original plan) → increments current_step
    2. DETOUR: From route_after_reflection with EXTERNAL recommendation → does NOT increment current_step

    Args:
        state: Current agent state

    Returns:
        Updated state with tool results
    """
    try:
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        query = state.get("query", "")

        # Get OpenAI client from state
        client = state.get("openai_client")
        if not client:
            print("[TOOL_WEB_SEARCH] No OpenAI client found in state.")
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "current_step": current_step,  # Don't increment on error
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": "OpenAI client required for tool execution",
            }

        # Build prompt for LLM tool calling
        # For planned calls: use plan[current_step]
        # For detour calls: use generic prompt
        if plan and current_step < len(plan):
            action_step = plan[current_step]
            prompt = f"Based on the user's query and the current plan step, call the web_search tool.\n\nUser Query: {query}\n\nCurrent Step: {action_step}"
        else:
            # Detour or no plan - generic prompt
            prompt = f"Based on the user's query, search the web for relevant information.\n\nUser Query: {query}"

        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            tools=TOOL_WEB_SEARCH,
            tool_choice="auto",
            temperature=0.1,
        )

        # Determine if this is a DETOUR call using evaluation_result
        # If evaluation_result exists → DETOUR (came from reflect_node)
        # If evaluation_result is None → PLANNED (came from route_after_planning, after verify cleared it)
        # This works because verify_node clears evaluation_result at the end of each reflection cycle
        is_detour = state.get("evaluation_result") is not None
        is_planned_call = not is_detour

        # Check if LLM called a tool
        if not response.choices[0].message.tool_calls:
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "current_step": current_step + 1 if is_planned_call else current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="No tool was called by the LLM.")],
            }

        # Execute the tool call (same pattern as agent_service.py)
        tool_call = response.choices[0].message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        # Build context for tool execution
        context = {
            "collection": state.get("collection"),
            "dept_id": state.get("dept_id"),
            "user_id": state.get("user_id"),
            "openai_client": client,
            "request_data": state.get("request_data", {}),
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

        return {
            "tools_used": state.get("tools_used", []) + [tool_name],
            "tool_results": tool_results,
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


# ==================== REFINEMENT NODE ====================


def refine_node(state: AgentState) -> Dict[str, Any]:
    """
    Refine query based on reflection feedback.

    Args:
        state: Current agent state

    Returns:
        Updated state with refined query
    """
    current_query = state.get("refined_query") or state.get("query", "")

    try:
        openai_client = state.get("openai_client", None)
        if not openai_client:
            print("No OpenAI client found in state.")
            raise ValueError("OpenAI client is required for refinement node.")

        refiner = QueryRefiner(
            openai_client=openai_client,
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE,
        )
        evaluation_result: EvaluationResult = state.get("evaluation_result", None)
        if not evaluation_result:
            print("No evaluation result found in state.")
            raise ValueError("Evaluation result is required for query refinement.")

        refined_query = refiner.refine_query(
            original_query=current_query,
            eval_result=evaluation_result,
        )

        current_refinement_count = state.get("refinement_count", 0)
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


# ==================== GENERATION NODE ====================


def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate answer from retrieved documents.

    Args:
        state: Current agent state

    Returns:
        Updated state with generated answer
    """
    try:
        openai_client = state.get("openai_client", None)
        if not openai_client:
            print("No OpenAI client found in state.")
            raise ValueError("OpenAI client is required for generation node.")

        # Check if this is a CLARIFY recommendation from reflection
        evaluation_result = state.get("evaluation_result")
        if (
            evaluation_result
            and evaluation_result.recommendation == RecommendationAction.CLARIFY
        ):
            # Generate clear clarification request message
            clarification_message = (
                "I need more information to answer your question accurately. "
                f"{evaluation_result.reasoning}\n\n"
                "Could you please provide more specific details or rephrase your question?"
            )
            return {
                "draft_answer": clarification_message,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content=clarification_message)],
            }

        # Check if we have any context to generate from
        retrieved_docs = state.get("retrieved_docs", [])
        tool_results = state.get("tool_results", {})

        if not retrieved_docs and not tool_results:
            # No context available - cannot generate
            return {
                "draft_answer": "",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="No context available to generate answer from.")],
            }

        # Build numbered context from both sources with rich formatting
        contexts = []
        context_num = 1

        # Add retrieved documents with detailed metadata
        if retrieved_docs:
            for doc in retrieved_docs:
                chunk = doc.get("chunk", str(doc))
                source = doc.get("source", "unknown")
                page = doc.get("page", 0)

                # Build context header similar to agent_tools.py
                header = f"Context {context_num} (Source: {source}"
                if page > 0:
                    header += f", Page: {page}"
                header += "):\n"

                # Add optional scores if available
                score_info = ""
                if doc.get("hybrid") is not None:
                    score_info += f"Hybrid score: {doc['hybrid']:.2f}"
                if doc.get("rerank") is not None:
                    if score_info:
                        score_info += ", "
                    score_info += f"Rerank score: {doc['rerank']:.2f}"

                # Combine all parts
                context_entry = f"{header}{chunk}"
                if score_info:
                    context_entry += f"\n{score_info}"

                contexts.append(context_entry)
                context_num += 1

        # Add tool results with clear labeling
        if tool_results:
            for tool_key, results in tool_results.items():
                for res in results:
                    tool_name = (
                        tool_key.split("_step_")[0]
                        if "_step_" in tool_key
                        else tool_key
                    )
                    step = res.get("step", "")
                    query_used = res.get("query", "")
                    result_text = res.get("result", "")

                    # Format tool result context
                    header = f"Context {context_num} (Tool: {tool_name}"
                    if step:
                        header += f", Step: {step}"
                    header += "):\n"

                    if query_used:
                        header += f"Query used: {query_used}\n"

                    context_entry = f"{header}Result: {result_text}"
                    contexts.append(context_entry)
                    context_num += 1

        if not contexts:
            print("No context available for answer generation.")
            raise ValueError(
                "No context available from retrieved documents or tool results."
            )

        final_context = "\n\n".join(contexts)

        # Build system prompt (similar to agent_service.py)
        system_prompt = """You are a helpful assistant that answers questions using provided contexts and tool results.

STRICT RULES:
1. Use ONLY information from the numbered contexts below - DO NOT use external knowledge
2. Include bracket citations [n] for every sentence that uses information (e.g., [1], [2])
3. Synthesize information from multiple contexts when relevant
4. At the end of your answer, cite the sources you used:
   - For document sources: List the filename and specific page numbers (e.g., "Sources: report.pdf (pages 15, 23)")
   - For tool sources: List the tool names used (e.g., "Tools: web_search, calculator")
5. If the contexts don't contain sufficient information to answer, say: "I don't have enough information to answer that based on the available context."
6. Be concise, accurate, and professional"""

        # Use BOTH original and refined query for best LLM understanding
        original_query = state.get("query", "")
        refined_query = state.get("refined_query", None)

        # Build question section - show both original and refined if refinement occurred
        question_section = f"Question: {original_query}"
        if refined_query and refined_query != original_query:
            question_section += f"\n(Refined as: {refined_query})"

        # Build user message with contexts (similar to build_prompt in retrieval.py)
        user_message_with_context = f"""{question_section}

Context:
{final_context}

Instructions: Answer the question concisely by synthesizing information from the contexts above.
Include bracket citations [n] for every sentence that uses information.
At the end of your answer, cite the sources you used. For each source file, list the specific page numbers
from the contexts you referenced. Format: 'Sources: filename.pdf (pages 15, 23), filename2.pdf (page 7)'"""

        # Build messages list: system + conversation_history + current query with contexts
        openai_messages = [{"role": "system", "content": system_prompt}]

        # Load previous conversation history from database if conversation_id exists
        conversation_id = state.get("conversation_id", None)
        if conversation_id:
            try:
                # Load conversation history (similar to chat.py)
                conversation_service = ConversationService()
                import asyncio

                history = asyncio.run(
                    conversation_service.get_message_history(
                        conversation_id, Config.REDIS_CACHE_LIMIT
                    )
                )

                # Add sanitized history to messages
                for h in history:
                    sanitized_msg = {
                        "role": h["role"],
                        "content": sanitize_text(h["content"], max_length=5000),
                    }
                    openai_messages.append(sanitized_msg)

                print(
                    f"[GENERATE] Loaded {len(history)} messages from conversation history"
                )
            except Exception as e:
                print(f"[GENERATE] Failed to load conversation history: {e}")
                # Continue without history

        # Add current query with contexts
        openai_messages.append({"role": "user", "content": user_message_with_context})

        response = openai_client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=openai_messages,
            temperature=Config.OPENAI_TEMPERATURE,
        )
        draft_answer = ""
        if response.choices and response.choices[0].message:
            draft_answer = response.choices[0].message.content

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


# ==================== VERIFICATION NODE ====================


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
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": state.get("messages", [])
            + [AIMessage(content="No draft answer to verify.")],
        }

    # Check if this is a CLARIFY recommendation - pass through without verification
    evaluation_result = state.get("evaluation_result")
    if (
        evaluation_result
        and evaluation_result.recommendation == RecommendationAction.CLARIFY
    ):
        # Clarification messages don't need citation enforcement
        # CLARIFY always ends the flow (user needs to provide input)
        return {
            "final_answer": draft_answer,  # CLARIFY always creates final_answer
            "evaluation_result": None,  # Clear evaluation_result for next cycle
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
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content="Warning: No contexts to verify citations against."
                    )
                ],
            }

        # Enforce citations - drops sentences without valid citations
        clean_answer, all_supported = enforce_citations(draft_answer, valid_ids)

        if not all_supported:
            print(f"[VERIFY] Dropped unsupported sentences from answer")
            print(f"[VERIFY] Valid context IDs: {valid_ids}")

        # If more plan steps remain, don't set final_answer yet
        # The intermediate result is stored in draft_answer and contexts accumulate in state
        return {
            "final_answer": clean_answer if not has_more_steps else None,
            "evaluation_result": None,  # Clear evaluation_result for next cycle
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": state.get("messages", [])
            + [AIMessage(content="Answer verified and citations checked.")],
        }

    except Exception as e:
        print(f"Verification node error: {e}")
        return {
            "final_answer": draft_answer if not has_more_steps else None,
            "evaluation_result": None,  # Clear evaluation_result for next cycle
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": state.get("messages", [])
            + [AIMessage(content=f"Error during verification: {str(e)}")],
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
    error_message = state.get("error", "An unknown error occurred.")
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
