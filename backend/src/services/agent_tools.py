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

import json
import re
from typing import Dict, Any, List
from src.services.retrieval import retrieve, build_where
from src.config.settings import Config
from src.utils.safety import looks_like_injection, scrub_context

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

    TODO: Implement this function
    Steps:
    1. Extract query and top_k from args
    2. Get collection, dept_id, user_id from context
    3. Build where clause using build_where()
    4. Call retrieve() with all parameters
    5. Format results as readable text for LLM
    6. Handle errors gracefully
    """
    query = args.get("query", "")
    top_k = args.get("top_k", 5)
    collection = context.get("collection", None)
    dept_id = context.get("dept_id", "")
    user_id = context.get("user_id", "")
    use_hybrid = context.get("use_hybrid", Config.USE_HYBRID)
    use_reranker = context.get("use_reranker", Config.USE_RERANKER)
    request = context.get("request", None)
    if not dept_id or not user_id:
        return "Error: No organization ID or user ID provided"
    if not collection:
        return "Error: No document collection available"
    try:
        # Build where clause
        where = build_where(request, dept_id, user_id)
        ctx, _ = retrieve(
            collection=collection,
            query=query,
            dept_id=dept_id,
            user_id=user_id,
            top_k=top_k,
            where=where,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )

        if not ctx:
            return "No relevant documents found."
        for c in ctx:
            c["chunk"] = scrub_context(c.get("chunk", ""))

        # Format contexts for LLM
        all_chunk_tex = ""
        for i, c in enumerate(ctx, 1):  # Start from 1
            chunk = c.get("chunk", "")
            page = c.get("page", 0)
            source = c.get("source", "unknown")
            sem_sim = c.get("sem_sim", 0)
            hybrid = c.get("hybrid", 0)
            rerank = c.get("rerank", 0)  # Fixed: was "reranker"

            # Build metadata parts
            metadata_parts = []
            if source:
                metadata_parts.append(f"Source: {source}")
            if page > 0:
                metadata_parts.append(f"Page: {page}")
            if hybrid > 0:
                metadata_parts.append(f"Hybrid: {hybrid:.2f}")
            elif sem_sim > 0:
                metadata_parts.append(f"Similarity: {sem_sim:.2f}")
            if rerank > 0:
                metadata_parts.append(f"Rerank: {rerank:.2f}")

            # Format chunk with clean metadata line
            metadata_line = (
                ", ".join(metadata_parts) if metadata_parts else "No metadata"
            )
            chunk_text = f"[{i}] {metadata_line}\n{chunk}\n\n"

            all_chunk_tex += chunk_text

        return f"Found {len(ctx)} relevant document(s):\n\n" + all_chunk_tex
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
# TOOL REGISTRY
# ============================================================================

TOOL_REGISTRY = {
    "search_documents": execute_search_documents,
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
