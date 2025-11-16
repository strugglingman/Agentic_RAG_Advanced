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
from typing import Dict, Any, List
from src.services.retrieval import retrieve, build_where
from src.config.settings import Config


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
    # TODO: Implementation here
    pass


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

    TODO: Implement this function
    Steps:
    1. Extract expression from args
    2. Handle percentage expressions (e.g., "15% of 100")
    3. Safely evaluate expression (use ast.literal_eval or similar)
    4. Return result as string
    5. Handle errors (invalid expressions, division by zero, etc.)

    Security Note: NEVER use eval() directly - use safe evaluation
    """
    # TODO: Implementation here
    pass


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


def execute_tool_call(tool_name: str, tool_args: Dict[str, Any], context: Dict[str, Any]) -> str:
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

    TODO: Implement this function
    Steps:
    1. Look up tool in TOOL_REGISTRY
    2. If not found, raise error
    3. Execute the tool with args and context
    4. Return result
    5. Handle any exceptions from tool execution
    """
    # TODO: Implementation here
    pass
