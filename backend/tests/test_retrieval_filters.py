import importlib
import sys
import types

import pytest


def _load_build_where():
    """
    Import retrieval_qdrant in a lightweight way for filter-logic testing.
    """
    if "sentence_transformers" not in sys.modules:
        stub_st = types.ModuleType("sentence_transformers")

        class _DummyCrossEncoder:  # pragma: no cover - only for import shim
            def __init__(self, *args, **kwargs):
                pass

        stub_st.CrossEncoder = _DummyCrossEncoder
        sys.modules["sentence_transformers"] = stub_st

    if "src.services.vector_db_qdrant" not in sys.modules:
        stub_vdb = types.ModuleType("src.services.vector_db_qdrant")

        class _DummyQdrantVectorDB:  # pragma: no cover - only for import shim
            pass

        stub_vdb.QdrantVectorDB = _DummyQdrantVectorDB
        sys.modules["src.services.vector_db_qdrant"] = stub_vdb

    module = importlib.import_module("src.services.retrieval_qdrant")
    return module.build_where


def test_build_where_requires_dept_id():
    build_where = _load_build_where()

    with pytest.raises(ValueError) as exc_info:
        build_where(payload={}, dept_id="", user_id="u1")

    assert "organization id" in str(exc_info.value).lower()


def test_build_where_requires_user_id():
    build_where = _load_build_where()

    with pytest.raises(ValueError) as exc_info:
        build_where(payload={}, dept_id="D1", user_id="")

    assert "user id" in str(exc_info.value).lower()


def test_build_where_default_scope_contains_dept_and_user_visibility():
    build_where = _load_build_where()

    where = build_where(payload={}, dept_id="D1", user_id="u1")
    assert where == {
        "$and": [
            {"dept_id": "D1"},
            {"$or": [{"file_for_user": False}, {"user_id": "u1"}]},
        ]
    }


def test_build_where_single_extension():
    build_where = _load_build_where()

    where = build_where(
        payload={"filters": [{"exts": ["pdf"]}]},
        dept_id="D1",
        user_id="u1",
    )
    assert where == {
        "$and": [
            {"ext": "pdf"},
            {"dept_id": "D1"},
            {"$or": [{"file_for_user": False}, {"user_id": "u1"}]},
        ]
    }


def test_build_where_multiple_extensions():
    build_where = _load_build_where()

    where = build_where(
        payload={"filters": [{"exts": ["pdf", "docx"]}]},
        dept_id="D1",
        user_id="u1",
    )
    assert where == {
        "$and": [
            {"$or": [{"ext": "pdf"}, {"ext": "docx"}]},
            {"dept_id": "D1"},
            {"$or": [{"file_for_user": False}, {"user_id": "u1"}]},
        ]
    }
