"""
Agent tools for the ReAct agent.

This module defines:
1. Tool schemas (what the LLM sees - OpenAI function calling format)
2. Tool execution functions (what actually runs when LLM calls a tool)
3. Tool registry (maps tool names to execution functions)

Each tool has two parts:
- SCHEMA: Describes the tool to the LLM (name, description, parameters)
- execute_xxx: The Python function that actually executes the tool
"""

import re
from typing import Dict, Any
from src.services.retrieval import retrieve, build_where
from src.config.settings import Config
from src.utils.safety import scrub_context
from src.services.retrieval_evaluator import RetrievalEvaluator
from src.models.evaluation import ReflectionConfig, EvaluationCriteria
from src.services.query_refiner import QueryRefiner
from src.services.clarification_helper import ClarificationHelper
from src.services.web_search import WebSearchService

# ============================================================================
# TOOL 1: SEARCH DOCUMENTS
# ============================================================================

SEARCH_DOCUMENTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search internal company documents for information. "
            "Use this when the user asks about company data, reports, policies, "
            "or any information that might be in uploaded documents. "
            "Returns relevant document chunks with sources and page numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific and use keywords from the user's question. "
                        "For example: 'Q1 2024 revenue', 'employee vacation policy', 'sales targets'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Number of document chunks to retrieve. "
                        "Use 3-5 for quick lookups, 8-10 for comprehensive answers. Default is 5."
                    ),
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def execute_search_documents(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Execute the search_documents tool.

    Args:
        args: Arguments from LLM containing:
            - query (str): Search query
            - top_k (int, optional): Number of results to return

        context: System context containing:
            - collection: ChromaDB collection
            - dept_id: Department ID from auth
            - user_id: User ID from auth
            - request: Flask request object (for filters)

    Returns:
        Formatted string result for LLM containing document chunks

    Flow with refinement loop:
    1. Retrieve contexts with original query
    2. Evaluate quality
    3. If REFINE recommended and attempts < MAX:
       - Refine query
       - Retrieve again
       - Re-evaluate
       - Repeat until ANSWER or max attempts
    4. Return best contexts found
    """
    query = args.get("query", "")
    top_k = args.get("top_k", 5)
    collection = context.get("collection", None)
    dept_id = context.get("dept_id", "")
    user_id = context.get("user_id", "")
    use_hybrid = context.get("use_hybrid", Config.USE_HYBRID)
    use_reranker = context.get("use_reranker", Config.USE_RERANKER)
    request_data = context.get("request_data", {})
    if not dept_id or not user_id:
        return "Error: No organization ID or user ID provided"
    if not collection:
        return "Error: No document collection available"
    try:
        # Build where clause
        where = build_where(request_data, dept_id, user_id)

        # Initial retrieval
        current_query = query
        ctx, _ = retrieve(
            collection=collection,
            query=current_query,
            dept_id=dept_id,
            user_id=user_id,
            top_k=top_k,
            where=where,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )

        if not ctx:
            return "No relevant documents found."

        # Store raw contexts in context dict for agent to access later
        context["_retrieved_contexts"] = ctx

        # Initialize messages (set by self-reflection based on quality)
        clarification_msg = None
        external_msg = None

        if Config.USE_SELF_REFLECTION:
            config = ReflectionConfig.from_settings(Config)
            client = context.get("openai_client", None)
            evaluator = RetrievalEvaluator(config=config, openai_client=client)

            # Initial evaluation
            criteria = EvaluationCriteria(
                query=current_query,
                contexts=ctx,
                search_metadata={
                    "hybrid": use_hybrid,
                    "reranker": use_reranker,
                    "top_k": top_k,
                },
                mode=config.mode,
            )
            eval_result = evaluator.evaluate(criteria)
            context["_last_evaluation"] = eval_result

            # Log initial evaluation
            print(
                f"[SELF-REFLECTION] Quality: {eval_result.quality.value}, "
                f"Confidence: {eval_result.confidence:.2f}, "
                f"Recommendation: {eval_result.recommendation.value}"
            )
            print(f"[SELF-REFLECTION] Reasoning: {eval_result.reasoning}")
            if eval_result.issues:
                print(f"[SELF-REFLECTION] Issues: {', '.join(eval_result.issues)}")
            if eval_result.missing_aspects:
                print(
                    f"[SELF-REFLECTION] Missing aspects: {', '.join(eval_result.missing_aspects)}"
                )

            # =================================================================
            # Handle Direct CLARIFY (confidence < 0.5) - skip refinement
            # =================================================================
            if eval_result.should_clarify:
                clarifier = ClarificationHelper(openai_client=client)
                clarification_msg = clarifier.generate_clarification(
                    query=query,
                    eval_result=eval_result,
                    max_attempts_reached=False,
                    context_hint="uploaded documents",
                )
                print(f"[CLARIFICATION] Direct clarify (confidence < 0.5)")

            # =================================================================
            # Handle EXTERNAL Recommendation (Week 3 - Day 6)
            # =================================================================
            # Signal agent to use web_search tool (don't execute automatically)
            if eval_result.should_search_external:
                print(f"[SELF-REFLECTION] EXTERNAL recommended - suggesting web search")
                external_msg = (
                    f"[EXTERNAL SEARCH SUGGESTED]\n"
                    f'The query "{query}" appears to require external information '
                    f"not found in the uploaded documents.\n\n"
                    f"Reason: {eval_result.reasoning}\n\n"
                    f"Recommendation: Use the web_search tool to find current or external information.\n\n"
                    f"---\n\n"
                )

            # Refinement loop with local counter (only if not already CLARIFY)
            refinement_count = 0
            if Config.REFLECTION_AUTO_REFINE and eval_result.should_refine:
                max_attempts = Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS
                openai_model = context.get("model", Config.OPENAI_MODEL)
                temperature = context.get("temperature", Config.OPENAI_TEMPERATURE)
                query_refiner = QueryRefiner(
                    openai_client=client, model=openai_model, temperature=temperature
                )

                # Track refinement history for this tool call (for logging)
                refinement_history = []

                while refinement_count < max_attempts:

                    refinement_count += 1
                    print(
                        f"[QUERY_REFINER] Refinement attempt {refinement_count}/{max_attempts}"
                    )

                    # Refine the query (use current_query, not original)
                    refined_query = query_refiner.refine_query(
                        original_query=current_query,
                        eval_result=eval_result,
                        context_hint=" ".join([c.get("chunk", "")[:100] for c in ctx]),
                    )

                    # Track for logging
                    refinement_history.append(
                        {
                            "attempt": refinement_count,
                            "from": current_query,
                            "to": refined_query,
                        }
                    )
                    print(f"[QUERY_REFINER] '{current_query}' -> '{refined_query}'")

                    # Update current query for next iteration
                    current_query = refined_query

                    # Retrieve with refined query
                    ctx, _ = retrieve(
                        collection=collection,
                        query=current_query,
                        dept_id=dept_id,
                        user_id=user_id,
                        top_k=top_k,
                        where=where,
                        use_hybrid=use_hybrid,
                        use_reranker=use_reranker,
                    )

                    if not ctx:
                        print(
                            f"[QUERY_REFINER] No results for refined query, keeping previous results"
                        )
                        # Restore previous contexts
                        ctx = context["_retrieved_contexts"]
                        break

                    # Update stored contexts
                    context["_retrieved_contexts"] = ctx

                    # Re-evaluate
                    criteria = EvaluationCriteria(
                        query=current_query,
                        contexts=ctx,
                        search_metadata={
                            "hybrid": use_hybrid,
                            "reranker": use_reranker,
                            "top_k": top_k,
                        },
                        mode=config.mode,
                    )
                    eval_result = evaluator.evaluate(criteria)
                    context["_last_evaluation"] = eval_result

                    # Log refined evaluation
                    print(
                        f"[SELF-REFLECTION] (Attempt {refinement_count}) Quality: {eval_result.quality.value}, "
                        f"Confidence: {eval_result.confidence:.2f}, "
                        f"Recommendation: {eval_result.recommendation.value}"
                    )

                # Log final status
                if refinement_count > 0:
                    if refinement_count >= max_attempts:
                        print(
                            f"[QUERY_REFINER] Max refinement attempts ({max_attempts}) reached"
                        )

                        # =================================================================
                        # Progressive Fallback: max attempts reached → clarification
                        # =================================================================
                        clarifier = ClarificationHelper(openai_client=client)
                        clarification_msg = clarifier.generate_clarification(
                            query=query,  # Original query (not refined)
                            eval_result=eval_result,
                            max_attempts_reached=True,
                            context_hint="uploaded documents",
                        )
                        print(
                            f"[CLARIFICATION] Max attempts reached, returning clarification"
                        )
                    else:
                        print(
                            f"[QUERY_REFINER] Refinement complete after {refinement_count} attempt(s)"
                        )

                    # Store history in context for debugging (optional)
                    context["_refinement_history"] = refinement_history

        # Continue with normal flow: format and return contexts
        for c in ctx:
            c["chunk"] = scrub_context(c.get("chunk", ""))

        # Format contexts for LLM using the same high-quality formatting as build_prompt()
        # This creates detailed context headers with source, page, and scores
        context_str = "\n\n".join(
            (
                f"Context {i+1} (Source: {c.get('source', 'unknown')}"
                + (f", Page: {c['page']}" if c.get("page", 0) > 0 else "")
                + f"):\n{c.get('chunk', '')}\n"
                + (
                    f"Hybrid score: {c['hybrid']:.2f}"
                    if c.get("hybrid") is not None
                    else ""
                )
                + (
                    f", Rerank score: {c['rerank']:.2f}"
                    if c.get("rerank") is not None
                    else ""
                )
            )
            for i, c in enumerate(ctx)
        )

        # Return formatted contexts with instructions for the LLM
        result = (
            f"Found {len(ctx)} relevant document(s):\n\n"
            f"{context_str}\n\n"
            f"Instructions: Answer the question concisely by synthesizing information from the contexts above. "
            f"Include bracket citations [n] for every sentence (e.g., [1], [2]). "
            f"At the end of your answer, cite the sources you used. For each source file, list the specific page numbers "
            f"from the contexts you referenced (look at the 'Page:' information in each context header). "
            f"Format: 'Sources: filename1.pdf (pages 15, 23), filename2.pdf (page 7)'"
        )

        # Prepend external search suggestion if recommended
        if external_msg:
            result = external_msg + result

        # Prepend clarification message if quality was poor
        if clarification_msg:
            result = (
                f"[QUALITY WARNING]\n{clarification_msg}\n\n" f"---\n\n" f"{result}"
            )

        return result
    except Exception as e:
        return f"Error during document retrieval: {str(e)}"


# ============================================================================
# TOOL 2: CALCULATOR
# ============================================================================

CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Perform mathematical calculations. "
            "Use this for arithmetic operations, percentages, or any numerical computation. "
            "Supports basic operations (+, -, *, /), percentages, and more."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression to evaluate. "
                        "Examples: '2+2', '15% of 100', '(100-80)/80 * 100', '1000 * 1.15'"
                    ),
                },
            },
            "required": ["expression"],
        },
    },
}


def execute_calculator(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Execute the calculator tool.

    Args:
        args: Arguments from LLM containing:
            - expression (str): Mathematical expression to evaluate

        context: System context (not used for calculator)

    Returns:
        String result of the calculation
    """
    expression = args.get("expression", "").strip()

    if not expression:
        return "Error: No expression provided"

    try:
        # Handle percentage expressions
        # "15% of 100" -> "0.15 * 100"
        # "20%" -> "0.20"
        processed = _handle_percentage(expression)

        # Safely evaluate the expression
        result = _safe_eval(processed)

        # Format result nicely
        if isinstance(result, float):
            # Remove unnecessary decimals for whole numbers
            if result.is_integer():
                return str(int(result))
            else:
                return f"{result:.6f}".rstrip("0").rstrip(".")
        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: Invalid mathematical expression - {str(e)}"


def _handle_percentage(expr: str) -> str:
    """
    Convert percentage expressions to Python math.

    Examples:
        "15% of 100" -> "0.15 * 100"
        "20%" -> "0.20"
        "increase by 10%" -> handled in context
    """
    # Pattern: "X% of Y" -> "X/100 * Y"
    expr = re.sub(
        r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)",
        r"(\1/100) * \2",
        expr,
        flags=re.IGNORECASE,
    )

    # Pattern: "X%" -> "X/100"
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)

    return expr


def _safe_eval(expr: str) -> float:
    """
    Safely evaluate a mathematical expression.

    Security: Only allows numbers and basic math operators.
    """
    # Whitelist: only allow numbers, operators, parentheses, dots, spaces
    if not re.match(r"^[\d\s\+\-\*\/\(\)\.\%]+$", expr):
        raise ValueError("Expression contains invalid characters")

    # Additional safety: limit length
    if len(expr) > 200:
        raise ValueError("Expression too long")

    # Evaluate using Python's eval (safe because we validated input)
    # Note: In production, consider using numexpr or ast-based parser
    result = eval(expr, {"__builtins__": {}}, {})

    return float(result)


# ============================================================================
# TOOL 3: WEB SEARCH
# ============================================================================
WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for up-to-date information when not found in internal documents. "
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific and use keywords from the user's question. "
                        "For example: 'Q1 2024 revenue', 'employee vacation policy', 'sales targets'"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        "Number of web search results to return. "
                        "Use 3-5 for quick lookups, 8-10 for comprehensive answers. Default is 5."
                    ),
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def execute_web_search(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Execute the web_search tool.

    Args:
        args: Arguments from LLM containing:
            - query (str): Search query
            - top_k (int, optional): Number of results to return
    """
    query = args.get("query", "")
    max_results = args.get("max_results", 5) or Config.WEB_SEARCH_MAX_RESULTS
    try:
        web_search_service = WebSearchService(
            provider=Config.WEB_SEARCH_PROVIDER,
            max_results=max_results,
            tavily_api_key=Config.TAVILY_API_KEY,
        )
        web_search_results = web_search_service.search(
            query=query, max_results=max_results
        )
        if web_search_results:
            return web_search_service.format_for_agent(web_search_results, query)
        else:
            print("[WEB_SEARCH] No results found, falling back to document contexts")
            return "No relevant web results found."
    except Exception as e:
        return f"Error during web search: {str(e)}"


# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOL_REGISTRY = {
    "search_documents": execute_search_documents,
    "web_search": execute_web_search,
    "calculator": execute_calculator,
}
"""
Maps tool names to their execution functions.
When LLM calls a tool, we lookup the function here and execute it.
"""


# ============================================================================
# ALL TOOLS LIST
# ============================================================================

ALL_TOOLS = [
    SEARCH_DOCUMENTS_SCHEMA,
    CALCULATOR_SCHEMA,
    WEB_SEARCH_SCHEMA,
]
"""
List of all tool schemas to send to OpenAI API.
This tells the LLM what tools are available.
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_tool_executor(tool_name: str):
    """
    Get the execution function for a tool by name.

    Args:
        tool_name: Name of the tool (e.g., "search_documents")

    Returns:
        Execution function or None if tool not found

    Example:
        executor = get_tool_executor("calculator")
        result = executor({"expression": "2+2"}, {})
    """
    return TOOL_REGISTRY.get(tool_name)


def execute_tool_call(
    tool_name: str, tool_args: Dict[str, Any], context: Dict[str, Any]
) -> str:
    """
    Execute a tool call from the LLM.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments from LLM (parsed from JSON)
        context: System context (collection, auth, etc.)

    Returns:
        String result from tool execution

    Raises:
        ValueError: If tool doesn't exist
    """
    # Look up tool executor
    executor = get_tool_executor(tool_name)

    if not executor:
        raise ValueError(
            f"Tool '{tool_name}' not found in registry. Available tools: {list(TOOL_REGISTRY.keys())}"
        )

    # Ensure args is a dict
    executor_args = tool_args if tool_args else {}

    try:
        # Execute the tool
        result = executor(executor_args, context)
        return result
    except Exception as e:
        # Log error and return error message
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        print(error_msg)  # For debugging
        return error_msg
