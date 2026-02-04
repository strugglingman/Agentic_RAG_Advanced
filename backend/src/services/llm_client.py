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
from httpx import Timeout
from langsmith import traceable
from src.observability.metrics import observe_llm_tokens, increment_error, MetricsErrorType


# Default timeout: 120 seconds for LLM requests (can be slow for complex queries)
DEFAULT_TIMEOUT = Timeout(120.0, connect=10.0)


@traceable(run_type="llm", name="chat_completion")
def chat_completion(
    client,
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    stop: Optional[list[str]] = None,
    timeout: Optional[Timeout] = None,
) -> Any:
    """Basic chat completion.

    Args:
        client: OpenAI client instance
        messages: List of message dicts [{"role": "user", "content": "..."}]
        model: Model name (e.g., "gpt-4o-mini")
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        stop: Optional stop sequences
        timeout: Request timeout (default: 120s total, 10s connect)

    Returns:
        OpenAI ChatCompletion response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stop=stop,
            timeout=timeout or DEFAULT_TIMEOUT,
        )

        if response.usage:
            observe_llm_tokens("input", model, response.usage.prompt_tokens)
            observe_llm_tokens("output", model, response.usage.completion_tokens)

        return response
    except Exception:
        increment_error(MetricsErrorType.LLM_FAILED)
        raise


@traceable(run_type="llm", name="chat_completion_with_tools")
def chat_completion_with_tools(
    client,
    messages: list[dict],
    tools: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    tool_choice: str = "auto",
    stop: Optional[list[str]] = None,
    parallel_tool_calls: bool = False,
    timeout: Optional[Timeout] = None,
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
        parallel_tool_calls: Whether to allow parallel tool calls
        timeout: Request timeout (default: 120s total, 10s connect)

    Returns:
        OpenAI ChatCompletion response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stop=stop,
            parallel_tool_calls=parallel_tool_calls,
            timeout=timeout or DEFAULT_TIMEOUT,
        )

        if response.usage:
            observe_llm_tokens("input", model, response.usage.prompt_tokens)
            observe_llm_tokens("output", model, response.usage.completion_tokens)

        return response
    except Exception:
        increment_error(MetricsErrorType.LLM_FAILED)
        raise


@traceable(run_type="llm", name="chat_completion_json")
def chat_completion_json(
    client,
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: Optional[Timeout] = None,
) -> Any:
    """Chat completion with JSON response format.

    Args:
        client: OpenAI client instance
        messages: List of message dicts
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        timeout: Request timeout (default: 120s total, 10s connect)

    Returns:
        OpenAI ChatCompletion response with JSON content
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
            timeout=timeout or DEFAULT_TIMEOUT,
        )

        if response.usage:
            observe_llm_tokens("input", model, response.usage.prompt_tokens)
            observe_llm_tokens("output", model, response.usage.completion_tokens)

        return response
    except Exception:
        increment_error(MetricsErrorType.LLM_FAILED)
        raise


@traceable(run_type="llm", name="chat_completion_structured")
def chat_completion_structured(
    client,
    messages: list[dict],
    schema: dict,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: Optional[Timeout] = None,
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
        timeout: Request timeout (default: 120s total, 10s connect)

    Returns:
        OpenAI ChatCompletion response matching schema
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            response_format=schema,
            timeout=timeout or DEFAULT_TIMEOUT,
        )

        if response.usage:
            observe_llm_tokens("input", model, response.usage.prompt_tokens)
            observe_llm_tokens("output", model, response.usage.completion_tokens)

        return response
    except Exception:
        increment_error(MetricsErrorType.LLM_FAILED)
        raise


@traceable(run_type="llm", name="chat_completion_stream")
def chat_completion_stream(
    client,
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: Optional[Timeout] = None,
) -> Any:
    """Chat completion with streaming response.

    Args:
        client: OpenAI client instance
        messages: List of message dicts
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        timeout: Request timeout (default: 120s total, 10s connect)

    Returns:
        OpenAI streaming response iterator
    """
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
            timeout=timeout or DEFAULT_TIMEOUT,
        )
    except Exception:
        increment_error(MetricsErrorType.LLM_FAILED)
        raise


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
