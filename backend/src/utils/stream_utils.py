"""Utility functions for streaming text responses via SSE."""

import asyncio
import json
from typing import Any, AsyncGenerator, Union


async def stream_text_smart(text: str, delay_ms: int = 20) -> AsyncGenerator[str, None]:
    """
    Smart streaming that chooses chunk size based on text length.
    Mimics LLM streaming behavior with variable chunk sizes.

    Uses ``await asyncio.sleep`` instead of ``time.sleep`` so the event
    loop stays free to serve other requests between chunks.

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
            await asyncio.sleep(delay_seconds)


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
