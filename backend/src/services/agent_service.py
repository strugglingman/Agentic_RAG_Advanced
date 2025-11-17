"""
Agent service implementing the ReAct (Reason + Act) pattern.

This module provides:
1. Agent class - orchestrates the ReAct loop
2. Tool execution - calls tools based on LLM decisions
3. State management - tracks conversation and tool results

ReAct Loop:
    User Query → LLM (with tools) → Tool Call → Execute Tool →
    Tool Result → LLM (continue) → Final Answer
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
from src.services.agent_tools import ALL_TOOLS, execute_tool_call
from src.config.settings import Config


class Agent:
    """
    ReAct agent that uses tools to answer questions.

    The agent follows this loop:
    1. Receive user query
    2. Call LLM with available tools
    3. If LLM wants to use tools:
       a. Execute tools
       b. Send results back to LLM
       c. Repeat from step 2
    4. If LLM has final answer:
       a. Return answer to user
    """

    def __init__(
        self,
        openai_client: OpenAI,
        max_iterations: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        """
        Initialize the agent.

        Args:
            openai_client: OpenAI client instance
            max_iterations: Maximum number of tool calls to prevent infinite loops
            model: OpenAI model to use (default: gpt-4o-mini)
            temperature: LLM temperature (default: 0.1 for factual responses)
        """
        self.client = openai_client
        self.max_iterations = max_iterations
        self.model = model
        self.temperature = temperature
        self.tools = ALL_TOOLS  # Tool schemas from agent_tools.py

    def run(
        self,
        query: str,
        context: Dict[str, Any],
        messages_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Run the agent on a user query.

        Args:
            query: User's question
            context: System context (collection, dept_id, user_id, etc.)
            messages_history: Previous conversation messages

        Returns:
            Final answer as string

        TODO: Implement the ReAct loop
        Steps:
        1. Build initial messages list (system + history + user query)
        2. Start iteration loop (max_iterations)
        3. Call LLM with tools
        4. Check if LLM wants to use tools or has final answer
        5. If tools: execute them, add results to messages, continue loop
        6. If final answer: return it
        7. If max iterations reached: return error message
        """
        # TODO: Implementation here
        pass

    def _call_llm(self, messages: List[Dict[str, str]]) -> Any:
        """
        Call OpenAI API with tools.

        Args:
            messages: Conversation history

        Returns:
            OpenAI chat completion response

        TODO: Implement this method
        Steps:
        1. Call openai_client.chat.completions.create()
        2. Pass model, messages, tools, temperature
        3. Set tool_choice="auto" (let LLM decide)
        4. Return response
        """
        # TODO: Implementation here
        pass

    def _has_tool_calls(self, response: Any) -> bool:
        """
        Check if LLM response contains tool calls.

        Args:
            response: OpenAI API response

        Returns:
            True if response has tool_calls, False otherwise

        TODO: Implement this method
        Steps:
        1. Access response.choices[0].message
        2. Check if message.tool_calls exists and is not None
        3. Return boolean
        """
        # TODO: Implementation here
        pass

    def _execute_tools(
        self,
        response: Any,
        context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Execute all tool calls from LLM response.

        Args:
            response: OpenAI API response with tool_calls
            context: System context to pass to tools

        Returns:
            List of tool result messages in OpenAI format

        TODO: Implement this method
        Steps:
        1. Extract tool_calls from response.choices[0].message
        2. For each tool_call:
           a. Get tool name
           b. Parse arguments (JSON string to dict)
           c. Call execute_tool_call(name, args, context)
           d. Build tool result message
        3. Return list of tool result messages

        Tool result message format:
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_string
        }
        """
        # TODO: Implementation here
        pass

    def _build_messages(
        self,
        query: str,
        messages_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build initial messages list for the agent.

        Args:
            query: User's question
            messages_history: Previous conversation messages

        Returns:
            List of messages in OpenAI format

        TODO: Implement this method
        Steps:
        1. Create system message with agent instructions
        2. Add messages_history if provided
        3. Add user query message
        4. Return combined list

        System message should instruct the agent to:
        - Use tools when needed
        - Be concise and accurate
        - Cite sources when using search_documents
        """
        # TODO: Implementation here
        pass

    def _get_final_answer(self, response: Any) -> str:
        """
        Extract final answer from LLM response.

        Args:
            response: OpenAI API response

        Returns:
            Final answer text

        TODO: Implement this method
        Steps:
        1. Access response.choices[0].message.content
        2. Return content or default message if None
        """
        # TODO: Implementation here
        pass

    def _append_tool_results(
        self,
        messages: List[Dict[str, str]],
        assistant_message: Any,
        tool_results: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Append assistant's tool calls and tool results to messages.

        Args:
            messages: Current messages list
            assistant_message: The assistant message with tool_calls
            tool_results: List of tool result messages

        Returns:
            Updated messages list

        TODO: Implement this method
        Steps:
        1. Convert assistant_message to dict format
        2. Append to messages
        3. Extend messages with tool_results
        4. Return updated messages

        Note: The assistant message must include tool_calls for OpenAI API
        """
        # TODO: Implementation here
        pass


# ============================================================================
# HELPER FUNCTIONS (Optional)
# ============================================================================

def format_tool_call_for_logging(tool_call) -> str:
    """
    Format a tool call for logging/debugging.

    Args:
        tool_call: OpenAI tool call object

    Returns:
        Human-readable string

    Example output:
        "calculator(expression='2+2')"
        "search_documents(query='revenue Q1')"
    """
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    args_str = ", ".join(f"{k}='{v}'" for k, v in args.items())
    return f"{name}({args_str})"