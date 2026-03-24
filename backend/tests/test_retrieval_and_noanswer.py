import pytest
import asyncio

from src.utils.safety import coverage_ok
from src.services import agent_tools
from src.services.retrieval_qdrant import build_prompt


pytestmark = pytest.mark.unit


def test_coverage_ok_thresholds():
    assert coverage_ok([0.6, 0.5, 0.4], topk=3, score_avg=0.28, score_min=0.38) is True
    assert coverage_ok([0.3, 0.2], topk=3, score_avg=0.35, score_min=0.5) is False


def test_search_documents_returns_no_results_when_retrieval_is_empty(monkeypatch):
    async def fake_retrieve_with_decomposition(**kwargs):
        return [], None

    monkeypatch.setattr(
        agent_tools, "retrieve_with_decomposition", fake_retrieve_with_decomposition
    )

    context = {
        "vector_db": object(),
        "openai_client": object(),
        "dept_id": "engineering",
        "user_id": "user@example.com",
        "request_data": {},
        "use_hybrid": False,
        "use_reranker": False,
    }
    result = asyncio.run(
        agent_tools.execute_search_documents({"query": "any query"}, context)
    )
    assert result == "No relevant documents found."


def test_search_documents_no_context_prompt_defaults_to_i_dont_know():
    system_prompt, user_prompt = build_prompt(
        query="What is our policy?",
        ctx=[],
        use_ctx=True,
    )
    assert "Use ONLY the provided CONTEXT" in system_prompt
    assert "I don't know." in user_prompt
