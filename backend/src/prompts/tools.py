"""
Tool calling prompts for calculator and web search.
"""


class ToolPrompts:
    """Prompts for LLM function calling with tools."""

    @staticmethod
    def calculator_prompt(task: str, is_detour: bool = False) -> str:
        """
        Generate prompt for calculator tool calling.

        Args:
            task: The calculation task description
            is_detour: Whether this is an ad-hoc call (True) or planned (False)

        Returns:
            Prompt string for calculator function calling
        """
        context_note = ""
        if is_detour:
            context_note = "\n\nNote: This is a supplementary calculation based on previous results."

        return f"""You need to call the calculator tool for this specific task.

Task: {task}

IMPORTANT: Extract the calculation request ONLY from the task description above. Do NOT use unrelated parts from other tasks.{context_note}

Call the calculator tool with the appropriate expression."""

    @staticmethod
    def web_search_prompt(task: str, is_detour: bool = False) -> str:
        """
        Generate prompt for web search tool calling.

        Args:
            task: The search task description
            is_detour: Whether this is an ad-hoc call (True) or planned (False)

        Returns:
            Prompt string for web_search function calling
        """
        context_note = ""
        if is_detour:
            context_note = (
                "\n\nNote: This is a supplementary search based on previous results."
            )

        return f"""You need to call the web_search tool for this specific task.

Task: {task}

IMPORTANT: Extract the search query ONLY from the task description above. Do NOT use unrelated parts from other tasks.{context_note}

Call the web_search tool with the appropriate query."""

    @staticmethod
    def fallback_prompt(query: str, tool_type: str) -> str:
        """
        Fallback prompt when no specific task is available.

        Args:
            query: User's original query
            tool_type: Type of tool ("calculator" or "web_search")

        Returns:
            Generic tool calling prompt
        """
        if tool_type == "calculator":
            return f"Based on the user's query, call the appropriate calculator tool.\n\nUser Query: {query}"
        elif tool_type == "web_search":
            return f"Based on the user's query, search the web for relevant information.\n\nUser Query: {query}"
        else:
            return f"Use the appropriate tool to help answer this query: {query}"
