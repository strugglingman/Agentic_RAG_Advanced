"""
Centralized LLM client wrapper (async).

Uses AsyncOpenAI for non-blocking LLM calls that don't hold the event loop.

Usage:
    from src.services.llm_client import chat_completion, chat_completion_with_tools

    # Simple chat
    response = await chat_completion(
        client=openai_client,
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o-mini",
    )
    print(response.choices[0].message.content)

    # With tools
    response = await chat_completion_with_tools(
        client=openai_client,
        messages=messages,
        tools=tools,
        model="gpt-4o-mini",
    )
"""

import asyncio
import logging
import time
from typing import Any, Optional

from httpx import Timeout
from langsmith import traceable
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from src.observability.metrics import (
    observe_llm_tokens,
    increment_error,
    MetricsErrorType,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = Timeout(40.0, connect=10.0)

_RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class CircuitOpenError(Exception):
    """LLM provider is down — fail-fast."""

    pass


class AsyncCircuitBreaker:
    """CLOSED → (N failures) → OPEN → (cooldown) → HALF_OPEN → (1 test) → CLOSED."""

    def __init__(self, threshold: int = 5, recovery: int = 30):
        self._threshold = threshold
        self._recovery = recovery
        self._failures = 0
        self._opened_at: float = 0.0
        self._state = "closed"
        self._lock = asyncio.Lock()

    async def check(self) -> None:
        async with self._lock:
            if self._state == "closed":
                return
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self._recovery:
                    self._state = "half_open"
                    logger.info("[CircuitBreaker] OPEN → HALF_OPEN (allowing one test request)")
                    return
                raise CircuitOpenError(f"LLM circuit open ({self._failures} failures)")
            # half_open: only one test request allowed; block the rest
            if self._state == "half_open":
                raise CircuitOpenError("LLM circuit half-open, test request in progress")

    async def record_success(self) -> None:
        async with self._lock:
            if self._state == "half_open":
                logger.info("[CircuitBreaker] HALF_OPEN → CLOSED")
            self._failures = 0
            self._state = "closed"

    async def record_failure(self, error: Exception) -> None:
        async with self._lock:
            # Any failure during HALF_OPEN probe → back to OPEN immediately,
            # regardless of error type, to avoid getting stuck in HALF_OPEN.
            if self._state == "half_open":
                self._state = "open"
                self._opened_at = time.monotonic()
                logger.warning(
                    "[CircuitBreaker] HALF_OPEN → OPEN (probe failed: %s)",
                    type(error).__name__,
                )
                return
            if not isinstance(error, _RETRYABLE):
                return
            self._failures += 1
            if self._failures >= self._threshold:
                self._state = "open"
                self._opened_at = time.monotonic()
                logger.warning("[CircuitBreaker] → OPEN (%d failures)", self._failures)


_cb: Optional[AsyncCircuitBreaker] = None


def _get_cb() -> AsyncCircuitBreaker:
    global _cb
    if _cb is None:
        from src.config.settings import Config

        _cb = AsyncCircuitBreaker(
            Config.LLM_CB_FAILURE_THRESHOLD, Config.LLM_CB_RECOVERY_TIMEOUT
        )
    return _cb


_fallback_client: Optional[AsyncOpenAI] = None


def _get_fallback_client() -> Optional[tuple[AsyncOpenAI, str]]:
    """Lazy-init cross-provider fallback client. Returns (client, model) or None."""
    global _fallback_client
    from src.config.settings import Config

    if not (
        Config.LLM_FALLBACK_BASE_URL
        and Config.LLM_FALLBACK_API_KEY
        and Config.LLM_FALLBACK_MODEL
    ):
        return None
    if _fallback_client is None:
        _fallback_client = AsyncOpenAI(
            api_key=Config.LLM_FALLBACK_API_KEY,
            base_url=Config.LLM_FALLBACK_BASE_URL,
            max_retries=Config.LLM_MAX_RETRIES,
            timeout=Timeout(Config.LLM_TIMEOUT, connect=Config.LLM_CONNECT_TIMEOUT),
        )
        logger.info(
            "[LLM] Fallback client ready → %s (%s)",
            Config.LLM_FALLBACK_MODEL,
            Config.LLM_FALLBACK_BASE_URL,
        )
    return (_fallback_client, Config.LLM_FALLBACK_MODEL)


async def _try_fallback(
    primary_model: str, error: Exception, create_kwargs: dict
) -> Any:
    """Retry once with fallback provider if primary fails with retryable error or circuit open."""
    if not isinstance(error, (*_RETRYABLE, CircuitOpenError)):
        return None
    fallback = _get_fallback_client()
    if fallback is None:
        return None
    fb_client, fb_model = fallback
    logger.warning(
        "[LLM] %s on %s → fallback to %s @ %s",
        type(error).__name__,
        primary_model,
        fb_model,
        fb_client.base_url,
    )
    try:
        response = await fb_client.chat.completions.create(
            model=fb_model, **create_kwargs
        )
        if hasattr(response, "usage") and response.usage:
            observe_llm_tokens("input", fb_model, response.usage.prompt_tokens)
            observe_llm_tokens("output", fb_model, response.usage.completion_tokens)
        return response
    except Exception as fb_err:
        logger.warning(
            "[LLM] Fallback also failed (%s → %s): %s",
            primary_model,
            fb_model,
            fb_err,
        )
        return None


@traceable(run_type="llm", name="chat_completion")
async def chat_completion(
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
        client: AsyncOpenAI client instance
        messages: List of message dicts [{"role": "user", "content": "..."}]
        model: Model name (e.g., "gpt-4o-mini")
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        stop: Optional stop sequences
        timeout: Request timeout (default: 120s total, 10s connect)

    Returns:
        OpenAI ChatCompletion response
    """
    cb = _get_cb()
    create_kwargs = dict(
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stop=stop,
        timeout=timeout or DEFAULT_TIMEOUT,
    )
    try:
        await cb.check()
        response = await client.chat.completions.create(model=model, **create_kwargs)

        if response.usage:
            observe_llm_tokens("input", model, response.usage.prompt_tokens)
            observe_llm_tokens("output", model, response.usage.completion_tokens)

        await cb.record_success()
        return response
    except Exception as e:
        if not isinstance(e, CircuitOpenError):
            await cb.record_failure(e)
        fallback_resp = await _try_fallback(model, e, create_kwargs)
        if fallback_resp is not None:
            return fallback_resp
        increment_error(MetricsErrorType.LLM_FAILED)
        raise


@traceable(run_type="llm", name="chat_completion_with_tools")
async def chat_completion_with_tools(
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
        client: AsyncOpenAI client instance
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
    cb = _get_cb()
    create_kwargs = dict(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stop=stop,
        parallel_tool_calls=parallel_tool_calls,
        timeout=timeout or DEFAULT_TIMEOUT,
    )
    try:
        await cb.check()
        response = await client.chat.completions.create(model=model, **create_kwargs)

        if response.usage:
            observe_llm_tokens("input", model, response.usage.prompt_tokens)
            observe_llm_tokens("output", model, response.usage.completion_tokens)

        await cb.record_success()
        return response
    except Exception as e:
        if not isinstance(e, CircuitOpenError):
            await cb.record_failure(e)
        fallback_resp = await _try_fallback(model, e, create_kwargs)
        if fallback_resp is not None:
            return fallback_resp
        increment_error(MetricsErrorType.LLM_FAILED)
        raise


@traceable(run_type="llm", name="chat_completion_json")
async def chat_completion_json(
    client,
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: Optional[Timeout] = None,
) -> Any:
    """Chat completion with JSON response format.

    Args:
        client: AsyncOpenAI client instance
        messages: List of message dicts
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        timeout: Request timeout (default: 120s total, 10s connect)

    Returns:
        OpenAI ChatCompletion response with JSON content
    """
    cb = _get_cb()
    create_kwargs = dict(
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
        timeout=timeout or DEFAULT_TIMEOUT,
    )
    try:
        await cb.check()
        response = await client.chat.completions.create(model=model, **create_kwargs)

        if response.usage:
            observe_llm_tokens("input", model, response.usage.prompt_tokens)
            observe_llm_tokens("output", model, response.usage.completion_tokens)

        await cb.record_success()
        return response
    except Exception as e:
        if not isinstance(e, CircuitOpenError):
            await cb.record_failure(e)
        fallback_resp = await _try_fallback(model, e, create_kwargs)
        if fallback_resp is not None:
            return fallback_resp
        increment_error(MetricsErrorType.LLM_FAILED)
        raise


@traceable(run_type="llm", name="chat_completion_structured")
async def chat_completion_structured(
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
        client: AsyncOpenAI client instance
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
    cb = _get_cb()
    create_kwargs = dict(
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        response_format=schema,
        timeout=timeout or DEFAULT_TIMEOUT,
    )
    try:
        await cb.check()
        response = await client.chat.completions.create(model=model, **create_kwargs)

        if response.usage:
            observe_llm_tokens("input", model, response.usage.prompt_tokens)
            observe_llm_tokens("output", model, response.usage.completion_tokens)

        await cb.record_success()
        return response
    except Exception as e:
        if not isinstance(e, CircuitOpenError):
            await cb.record_failure(e)
        fallback_resp = await _try_fallback(model, e, create_kwargs)
        if fallback_resp is not None:
            return fallback_resp
        increment_error(MetricsErrorType.LLM_FAILED)
        raise


@traceable(run_type="llm", name="chat_completion_stream")
async def chat_completion_stream(
    client,
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: Optional[Timeout] = None,
) -> Any:
    """Chat completion with streaming response.

    Args:
        client: AsyncOpenAI client instance
        messages: List of message dicts
        model: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        timeout: Request timeout (default: 120s total, 10s connect)

    Returns:
        OpenAI async streaming response iterator
    """
    cb = _get_cb()
    create_kwargs = dict(
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stream=True,
        timeout=timeout or DEFAULT_TIMEOUT,
    )
    try:
        await cb.check()
        stream = await client.chat.completions.create(model=model, **create_kwargs)
        await cb.record_success()
        return stream
    except Exception as e:
        if not isinstance(e, CircuitOpenError):
            await cb.record_failure(e)
        fallback_resp = await _try_fallback(model, e, create_kwargs)
        if fallback_resp is not None:
            return fallback_resp
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
