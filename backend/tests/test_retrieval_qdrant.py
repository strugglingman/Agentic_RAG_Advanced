import asyncio

import pytest

from src.services import retrieval_qdrant as rq


pytestmark = pytest.mark.unit


class _FakeVectorDB:
    def __init__(self, *, hybrid_result=None, semantic_result=None):
        self.hybrid_result = hybrid_result
        self.semantic_result = semantic_result
        self.last_hybrid_call = None
        self.last_semantic_call = None

    async def query_hybrid(self, query_text, n_results, where=None):
        self.last_hybrid_call = {
            "query_text": query_text,
            "n_results": n_results,
            "where": where,
        }
        return self.hybrid_result

    async def query(self, query_texts, n_results, where=None, include=None):
        self.last_semantic_call = {
            "query_texts": query_texts,
            "n_results": n_results,
            "where": where,
            "include": include,
        }
        return self.semantic_result


def _meta(chunk_id="c1", source="doc.md"):
    return {
        "dept_id": "eng",
        "user_id": "u@example.com",
        "file_for_user": False,
        "chunk_id": chunk_id,
        "file_id": "f1",
        "source": source,
        "ext": "md",
        "tags": "",
        "size_kb": 1,
        "upload_at": "2026-01-01",
        "uploaded_at_ts": 1704067200,
        "page": 0,
    }


def test_build_where_validates_required_ids():
    with pytest.raises(ValueError, match="organization ID"):
        rq.build_where({}, "", "user@example.com")

    with pytest.raises(ValueError, match="user ID"):
        rq.build_where({}, "eng", "")


def test_build_where_with_multi_ext_filters():
    where = rq.build_where(
        {"filters": [{"exts": ["pdf", "docx"]}]},
        dept_id="eng",
        user_id="user@example.com",
    )
    assert "$and" in where
    clauses = where["$and"]
    assert {"dept_id": "eng"} in clauses
    assert {"$or": [{"file_for_user": False}, {"user_id": "user@example.com"}]} in clauses
    assert {"$or": [{"ext": "pdf"}, {"ext": "docx"}]} in clauses


def test_retrieve_hybrid_normalizes_rrf_scores(monkeypatch):
    monkeypatch.setattr(rq, "coverage_ok", lambda *args, **kwargs: True)
    monkeypatch.setattr(rq.Config, "RRF_K", 60, raising=False)
    monkeypatch.setattr(rq.Config, "CANDIDATES", 2, raising=False)
    monkeypatch.setattr(rq.Config, "SHOW_SCORES", False, raising=False)

    rrf_max = 2 / (rq.Config.RRF_K + 1)
    vector_db = _FakeVectorDB(
        hybrid_result={
            "documents": [["doc one", "doc two"]],
            "metadatas": [[_meta("c1", "a.md"), _meta("c2", "b.md")]],
            "distances": [[rrf_max * 0.5, rrf_max * 2.0]],
        }
    )

    contexts, err = asyncio.run(
        rq.retrieve(
            vector_db=vector_db,
            query="q",
            dept_id="eng",
            user_id="u@example.com",
            top_k=2,
            where={"dept_id": "eng"},
            use_hybrid=True,
            use_reranker=False,
        )
    )

    assert err is None
    assert len(contexts) == 2
    # Sorted descending by hybrid score; second item should be clipped to 1.0.
    assert contexts[0]["hybrid"] == pytest.approx(1.0)
    assert contexts[1]["hybrid"] == pytest.approx(0.5)
    assert all(c["bm25"] == 0.0 for c in contexts)
    assert all(c["sem_sim"] == 0.0 for c in contexts)


def test_retrieve_hybrid_coverage_gate_blocks_when_not_reranking(monkeypatch):
    monkeypatch.setattr(rq, "coverage_ok", lambda *args, **kwargs: False)
    monkeypatch.setattr(rq.Config, "RRF_K", 60, raising=False)
    monkeypatch.setattr(rq.Config, "CANDIDATES", 1, raising=False)

    vector_db = _FakeVectorDB(
        hybrid_result={
            "documents": [["doc one"]],
            "metadatas": [[_meta("c1", "a.md")]],
            "distances": [[2 / 61]],
        }
    )

    contexts, err = asyncio.run(
        rq.retrieve(
            vector_db=vector_db,
            query="q",
            dept_id="eng",
            user_id="u@example.com",
            top_k=1,
            where=None,
            use_hybrid=True,
            use_reranker=False,
        )
    )

    assert contexts == []
    assert "hybrid coverage check" in err


def test_retrieve_semantic_confidence_gate_blocks_low_similarity(monkeypatch):
    monkeypatch.setattr(rq.Config, "CANDIDATES", 1, raising=False)
    monkeypatch.setattr(rq.Config, "MIN_SEM_SIM", 0.4, raising=False)
    monkeypatch.setattr(rq.Config, "RERANKER_THRESHOLD_RELAXATION", 0.75, raising=False)
    monkeypatch.setattr(rq.Config, "SHOW_SCORES", False, raising=False)

    vector_db = _FakeVectorDB(
        semantic_result={
            "documents": [["weak doc"]],
            "metadatas": [[_meta("c1", "a.md")]],
            # similarity = 1 - distance = 0.1 -> below MIN_SEM_SIM
            "distances": [[0.9]],
        }
    )

    contexts, err = asyncio.run(
        rq.retrieve(
            vector_db=vector_db,
            query="q",
            dept_id="eng",
            user_id="u@example.com",
            top_k=1,
            where=None,
            use_hybrid=False,
            use_reranker=False,
        )
    )

    assert contexts == []
    assert "semantic confidence threshold" in err


def test_retrieve_semantic_with_local_reranker_uses_rerank_scores(monkeypatch):
    monkeypatch.setattr(rq.Config, "CANDIDATES", 2, raising=False)
    monkeypatch.setattr(rq.Config, "MIN_SEM_SIM", 0.6, raising=False)
    monkeypatch.setattr(rq.Config, "RERANKER_THRESHOLD_RELAXATION", 0.5, raising=False)
    monkeypatch.setattr(rq.Config, "MIN_RERANK", 0.1, raising=False)
    monkeypatch.setattr(rq.Config, "AVG_RERANK", 0.05, raising=False)
    monkeypatch.setattr(rq.Config, "RERANKER_PROVIDER", "local", raising=False)
    monkeypatch.setattr(rq.Config, "SHOW_SCORES", False, raising=False)
    monkeypatch.setattr(rq, "coverage_ok", lambda *args, **kwargs: True)

    class _FakeReranker:
        def predict(self, _inputs):
            # Inputs are already sem-sim sorted (c2 first, then c1).
            # Give higher score to second input so reranker should flip order.
            return [0.2, 0.9]

    monkeypatch.setattr(rq, "get_reranker", lambda: _FakeReranker())

    vector_db = _FakeVectorDB(
        semantic_result={
            "documents": [["doc one", "doc two"]],
            "metadatas": [[_meta("c1", "a.md"), _meta("c2", "b.md")]],
            # similarities = [0.35, 0.4] (above relaxed threshold 0.3)
            "distances": [[0.65, 0.6]],
        }
    )

    contexts, err = asyncio.run(
        rq.retrieve(
            vector_db=vector_db,
            query="q",
            dept_id="eng",
            user_id="u@example.com",
            top_k=2,
            where=None,
            use_hybrid=False,
            use_reranker=True,
        )
    )

    assert err is None
    assert len(contexts) == 2
    assert contexts[0]["chunk_id"] == "c1"
    assert contexts[0]["rerank"] == pytest.approx(0.9)
    assert contexts[1]["rerank"] == pytest.approx(0.2)


def test_retrieve_returns_explicit_error_for_empty_query():
    contexts, err = asyncio.run(rq.retrieve(vector_db=_FakeVectorDB(), query="", dept_id="d", user_id="u"))
    assert contexts == []
    assert err == "Empty query"


def test_retrieve_returns_explicit_error_for_missing_vector_db():
    contexts, err = asyncio.run(rq.retrieve(vector_db=None, query="hello", dept_id="d", user_id="u"))
    assert contexts == []
    assert err == "No vector database provided"
