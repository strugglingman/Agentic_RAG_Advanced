import pytest

from src.services import langgraph_routing


pytestmark = pytest.mark.unit


def test_route_after_reflection_external_increments_retrieval_fallback(monkeypatch):
    calls: list[str] = []

    def _fake_increment():
        calls.append("fallback")

    monkeypatch.setattr(langgraph_routing, "increment_retrieval_fallback", _fake_increment)

    state = {
        "iteration_count": 0,
        "refinement_count": 0,
        "evaluation_result": {
            "recommendation": "external",
            "confidence": 0.72,
            "quality": "partial",
        },
    }

    assert langgraph_routing.route_after_reflection(state) == "tool_web_search"
    assert calls == ["fallback"]


def test_route_after_reflection_answer_does_not_increment_fallback(monkeypatch):
    calls: list[str] = []

    def _fake_increment():
        calls.append("fallback")

    monkeypatch.setattr(langgraph_routing, "increment_retrieval_fallback", _fake_increment)

    state = {
        "iteration_count": 0,
        "refinement_count": 0,
        "evaluation_result": {
            "recommendation": "answer",
            "confidence": 0.91,
            "quality": "good",
        },
    }

    assert langgraph_routing.route_after_reflection(state) == "generate"
    assert calls == []
