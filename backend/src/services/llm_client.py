"""
Centralized LLM client wrapper.

Simple wrapper functions for chat completions that can be easily swapped
to different providers (OpenAI, Anthropic, etc.).

Usage:
    from src.services.llm_client import chat_completion, chat_completion_with_tools

    # Simple chat
    response = chat_completion(
        client=openai_client,
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o-mini",
    )
    print(response.choices[0].message.content)

    # With tools
    response = chat_completion_with_tools(
        client=openai_client,
        messages=messages,
        tools=tools,
        model="gpt-4o-mini",
    )
"""

from typing import Any, Optional


def chat_completion(
    client,
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    stop: Optional[list[str]] = None,
) -> Any:
    """Basic chat completion.

    Args:
        client: OpenAI client instance
        messages: List of message dicts [{"role": "user", "content": "..."}]
        model: Model name (e.g., "gpt-4o-mini")
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        stop: Optional stop sequences

    Returns:
        OpenAI ChatCompletion response
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stop=stop,
    )


def chat_completion_with_tools(
    client,
    messages: list[dict],
    tools: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    tool_choice: str = "auto",
    stop: Optional[list[str]] = None,
) -> Any:
    """Chat completion with tool/function calling.

    Args:
        client: OpenAI client instance
        messages: List of message dicts
        tools: List of tool definitions
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        tool_choice: "auto", "none", or specific tool
        stop: Optional stop sequences

    Returns:
        OpenAI ChatCompletion response
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stop=stop,
    )


def chat_completion_json(
    client,
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Any:
    """Chat completion with JSON response format.

    Args:
        client: OpenAI client instance
        messages: List of message dicts
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate

    Returns:
        OpenAI ChatCompletion response with JSON content
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
    )


def chat_completion_structured(
    client,
    messages: list[dict],
    schema: dict,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Any:
    """Chat completion with structured output (JSON schema).

    Args:
        client: OpenAI client instance
        messages: List of message dicts
        schema: Full response_format dict with type and json_schema, e.g.:
                {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate

    Returns:
        OpenAI ChatCompletion response matching schema
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        response_format=schema,
    )


def chat_completion_stream(
    client,
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Any:
    """Chat completion with streaming response.

    Args:
        client: OpenAI client instance
        messages: List of message dicts
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate

    Returns:
        OpenAI streaming response iterator
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stream=True,
    )


# ============ Helper functions ============


def get_content(response) -> Optional[str]:
    """Extract text content from response."""
    if response.choices and response.choices[0].message:
        return response.choices[0].message.content
    return None


def get_tool_calls(response) -> list:
    """Extract tool calls from response."""
    if response.choices and response.choices[0].message:
        return response.choices[0].message.tool_calls or []
    return []


def has_tool_calls(response) -> bool:
    """Check if response has tool calls."""
    return len(get_tool_calls(response)) > 0
