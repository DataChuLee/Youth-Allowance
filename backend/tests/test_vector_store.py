from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.core.errors import IndexMissingError
from app.rag import vector_store


def test_assert_index_exists_rejects_missing_directory(tmp_path):
    with pytest.raises(IndexMissingError):
        vector_store.assert_index_exists(tmp_path / "missing")


def test_assert_index_exists_rejects_empty_directory(tmp_path):
    empty = tmp_path / "chroma"
    empty.mkdir()

    with pytest.raises(IndexMissingError):
        vector_store.assert_index_exists(empty)


def test_create_vector_store_does_not_auto_create_collection(monkeypatch, tmp_path):
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    (chroma_dir / "marker").write_text("exists", encoding="utf-8")
    captured = {}
    settings = SimpleNamespace(
        chroma_dir=chroma_dir,
        openai_api_key="test-key",
        openai_embedding_model="test-embedding-model",
    )

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured["embedding_kwargs"] = kwargs

    class FakeChroma:
        def __init__(self, **kwargs):
            captured["chroma_kwargs"] = kwargs

    monkeypatch.setattr(vector_store, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(vector_store, "Chroma", FakeChroma)

    vector_store.create_vector_store(settings)

    assert captured["embedding_kwargs"] == {
        "api_key": "test-key",
        "model": "test-embedding-model",
    }
    assert captured["chroma_kwargs"]["collection_name"] == "youth_allowance_booklet"
    assert captured["chroma_kwargs"]["persist_directory"] == str(chroma_dir)
    assert captured["chroma_kwargs"]["create_collection_if_not_exists"] is False


def test_retrieve_pdf_documents_uses_configured_top_k(monkeypatch, tmp_path):
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    (chroma_dir / "marker").write_text("exists", encoding="utf-8")
    document = Document(
        page_content="청년수당 카드 사용처 기준 안내",
        metadata={"page": 3, "chunk_id": "pdf-page-3-chunk-0"},
    )
    settings = SimpleNamespace(
        chroma_dir=chroma_dir,
        openai_api_key="test-key",
        openai_embedding_model="test-embedding-model",
        retrieval_top_k=7,
    )
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            pass

    class FakeChroma:
        def __init__(self, **kwargs):
            pass

        def similarity_search_with_relevance_scores(self, question, k):
            captured["question"] = question
            captured["k"] = k
            return [(document, 0.87)]

    monkeypatch.setattr(vector_store, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(vector_store, "Chroma", FakeChroma)

    results = vector_store.retrieve_pdf_documents("카드 사용처는?", settings)

    assert captured == {"question": "카드 사용처는?", "k": 7}
    assert results[0].document == document
    assert results[0].score == 0.87
