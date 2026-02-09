"""
Shared utility for determining how many conversation history messages are needed.

Used by:
- prisma_message_repository.py (Web UI: fetches from PostgreSQL)
- slack_adapter.py (Slack: fetches from Slack API)
"""

import json
import logging
from typing import Any

from src.config.settings import Config

logger = logging.getLogger(__name__)


def determine_message_count(
    query: str,
    openai_client: Any,
    fallback_limit: int = Config.REDIS_CACHE_LIMIT,
) -> int:
    """
    Use LLM to determine how many conversation history messages are needed for a query.

    Args:
        query: User's query to analyze
        openai_client: OpenAI client instance
        fallback_limit: Default limit if LLM call fails

    Returns:
        Number of messages needed (0-200)
    """
    if not openai_client:
        logger.info(
            "[history_utils] No OpenAI client provided, "
            f"using fallback limit {fallback_limit}"
        )
        return fallback_limit

    try:
        from src.services.llm_client import chat_completion_json

        prompt = f"""Analyze this user query and determine how many previous conversation messages are needed to answer it accurately.

Query: "{query}"

Consider:
- Does it reference previous conversation? (e.g., "what did we discuss?")
- Is it a follow-up? (e.g., "what about...", "and also...")
- Does it need context? (e.g., "based on what you said...")
- Is it asking for a summary? (e.g., "summarize our chat")
- Or is it standalone? (e.g., "what is Python?")

Respond with ONLY a JSON object:
{{
  "messages_needed": <number between 0 and 200>,
  "reasoning": "<brief explanation>"
}}

Examples:
- "What is Python?" → {{"messages_needed": 0, "reasoning": "Standalone general knowledge"}}
- "What about Java?" → {{"messages_needed": 3, "reasoning": "Follow-up to recent topic"}}
- "What did we discuss about databases?" → {{"messages_needed": 20, "reasoning": "Search recent history for topic"}}
- "Summarize our conversation" → {{"messages_needed": 200, "reasoning": "Full conversation summary"}}
"""

        response = chat_completion_json(
            client=openai_client,
            model=Config.OPENAI_SIMPLE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )

        result = json.loads(response.choices[0].message.content)
        messages_needed = min(result["messages_needed"], 200)  # Cap at 200

        logger.info(
            f"[LLM Intent] Query needs {messages_needed} messages - "
            f"Reason: {result['reasoning']}"
        )

        return messages_needed

    except Exception as e:
        logger.warning(
            f"[LLM Intent] Failed to determine message count: {e}, "
            f"falling back to {fallback_limit}"
        )
        return fallback_limit
