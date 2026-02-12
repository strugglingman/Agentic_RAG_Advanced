"""Utility functions for streaming text responses"""

import json
import time
from typing import Any, Generator, Union


def stream_text(
    text: str,
    chunk_mode: str = "token_like",
    chunk_size: int = 1,
    delay_ms: int = 0,
) -> Generator[str, None, None]:
    """
    Stream text in chunks similar to LLM streaming responses.

    Args:
        text: The full text to stream
        chunk_mode: How to split the text - "char", "word", or "token_like"
            - "char": Character by character (most granular)
            - "word": Word by word (smooth, natural)
            - "token_like": 1-3 characters per chunk (mimics LLM behavior)
        chunk_size: Number of units per chunk (for char/word modes)
        delay_ms: Delay in milliseconds between chunks (0 for no delay)

    Yields:
        str: Text chunks

    Examples:
        # Character by character
        for chunk in stream_text("Hello world", chunk_mode="char"):
            yield chunk

        # Word by word
        for chunk in stream_text("Hello world", chunk_mode="word"):
            yield chunk

        # Token-like (mimics LLM, 1-3 chars per chunk)
        for chunk in stream_text("Hello world", chunk_mode="token_like"):
            yield chunk
    """
    if not text:
        return

    delay_seconds = delay_ms / 1000.0 if delay_ms > 0 else 0

    if chunk_mode == "char":
        # Stream character by character
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            yield chunk
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    elif chunk_mode == "word":
        # Stream word by word (preserving spaces)
        words = text.split(" ")
        for i in range(0, len(words), chunk_size):
            # Get chunk_size words
            word_chunk = words[i : i + chunk_size]
            # Join with spaces and add trailing space if not last chunk
            chunk = " ".join(word_chunk)
            if i + chunk_size < len(words):
                chunk += " "
            yield chunk
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    elif chunk_mode == "token_like":
        # Mimic LLM token streaming (1-3 characters per chunk, varies)
        # This creates a more "natural" streaming feel like real LLMs
        import random

        i = 0
        while i < len(text):
            # Randomly choose 1-3 characters (weighted toward 1-2)
            chars_to_take = random.choices([1, 2, 3], weights=[50, 35, 15])[0]
            chunk = text[i : i + chars_to_take]
            yield chunk
            i += chars_to_take
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    else:
        raise ValueError(
            f"Invalid chunk_mode: {chunk_mode}. Use 'char', 'word', or 'token_like'"
        )


def stream_text_smart(text: str, delay_ms: int = 20) -> Generator[str, None, None]:
    """
    Smart streaming that chooses chunk size based on text length.
    Mimics LLM streaming behavior with variable chunk sizes.

    This is the recommended function for most use cases.

    Args:
        text: The full text to stream
        delay_ms: Delay in milliseconds between chunks (default: 20ms)

    Yields:
        str: Text chunks of varying sizes
    """
    import random

    if not text:
        return

    delay_seconds = delay_ms / 1000.0 if delay_ms > 0 else 0

    i = 0
    while i < len(text):
        # Variable chunk size (1-5 characters, weighted toward smaller)
        # This mimics how LLMs stream tokens
        chunk_size = random.choices([1, 2, 3, 4, 5], weights=[30, 30, 20, 15, 5])[0]

        # Don't break in the middle of a word if possible
        chunk = text[i : i + chunk_size]

        # If we're in the middle of a word and it's not the end, try to complete it
        if i + chunk_size < len(text) and text[i + chunk_size] not in [
            " ",
            "\n",
            ".",
            ",",
            "!",
            "?",
        ]:
            # Look ahead to find next space/punctuation
            next_break = i + chunk_size
            while next_break < len(text) and next_break < i + chunk_size + 10:
                if text[next_break] in [" ", "\n", ".", ",", "!", "?"]:
                    break
                next_break += 1

            # If break is close, extend to it
            if next_break < i + chunk_size + 5:
                chunk = text[i:next_break]

        yield chunk
        i += len(chunk)

        if delay_seconds > 0:
            time.sleep(delay_seconds)


def sse_event(event: str, data: Union[str, dict, list, Any]) -> str:
    """
    Format a Server-Sent Event.

    Args:
        event: Event type (e.g., "text", "hitl", "context", "progress")
        data: Payload — str for text chunks, dict/list will be JSON-serialized.
              Multiline data is handled per SSE spec (multiple data: lines).

    Returns:
        SSE-formatted string ending with double newline.
    """
    serialized = data if isinstance(data, str) else json.dumps(data)
    # SSE spec: multiline data needs separate "data:" lines
    lines = [f"event: {event}"]
    for line in serialized.split("\n"):
        lines.append(f"data: {line}")
    lines.append("")  # blank line terminates event
    return "\n".join(lines) + "\n"
