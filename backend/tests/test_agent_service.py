"""
Manual smoke tests for AgentService against live model calls.

These tests are opt-in and excluded from default/CI lanes.
"""

import asyncio
import os

import pytest

from src.config.settings import Config
from src.services.agent_service import AgentService


pytestmark = [pytest.mark.manual, pytest.mark.integration, pytest.mark.external]


@pytest.fixture
def require_manual_agent_env():
    if os.getenv("RUN_MANUAL_TESTS") != "1":
        pytest.skip("Set RUN_MANUAL_TESTS=1 to run manual agent smoke tests.")
    if not Config.OPENAI_KEY:
        pytest.skip("OPENAI_API_KEY is required for manual agent smoke tests.")


def _is_openai_connectivity_error(exc: Exception) -> bool:
    text = f"{exc!r}\n{exc}"
    return (
        "APIConnectionError" in text
        or "ConnectError" in text
        or "Connection error" in text
    )


def test_agent_service_direct_answer_smoke(require_manual_agent_env):
    """Validate that AgentService can return a direct factual answer."""
    from openai import AsyncOpenAI

    async def _run_query():
        client = AsyncOpenAI(api_key=Config.OPENAI_KEY)
        try:
            agent = AgentService(openai_client=client, max_iterations=3)
            return await agent.run(
                query="What is the capital of France? Answer in one short sentence.",
                context={},
            )
        finally:
            await client.close()

    try:
        answer, contexts = asyncio.run(_run_query())
    except Exception as exc:
        if _is_openai_connectivity_error(exc):
            pytest.skip(
                "OpenAI endpoint is unreachable from this environment. "
                "Retry when network/API access is available."
            )
        raise

    assert isinstance(answer, str)
    assert answer.strip()
    assert "paris" in answer.lower()
    assert isinstance(contexts, list)
