from types import SimpleNamespace
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.core.errors import IndexMissingError
from app.rag import retrieval
from app.rag.retrieval import (
    HybridRRFRetriever,
    RetrievalUnavailableError,
    create_ensemble_retriever,
    retrieve_pdf_documents_with_ensemble,
    rrf_merge_documents,
)


def make_document(chunk_id: str, text: str = "content") -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "page": 1, "title": "청년수당 참여자 안내책자"},
    )


def chunk_ids(documents: list[Document]) -> list[str]:
    return [str(document.metadata["chunk_id"]) for document in documents]


class FakeRetriever:
    def __init__(
        self,
        documents: list[Document] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.documents = documents or []
        self.error = error
        self.k = 0

    def invoke(self, _query: str) -> list[Document]:
        if self.error is not None:
            raise self.error
        return self.documents


def test_rrf_merge_documents_uses_alpha_to_shift_bm25_vector_weight() -> None:
    bm25_doc = make_document("bm25-only")
    vector_doc = make_document("vector-only")

    bm25_weighted = rrf_merge_documents(
        [bm25_doc],
        [vector_doc],
        alpha=0.8,
        rrf_k=60,
        top_n=2,
    )
    vector_weighted = rrf_merge_documents(
        [bm25_doc],
        [vector_doc],
        alpha=0.2,
        rrf_k=60,
        top_n=2,
    )

    assert chunk_ids(bm25_weighted) == ["bm25-only", "vector-only"]
    assert chunk_ids(vector_weighted) == ["vector-only", "bm25-only"]


def test_rrf_merge_documents_deduplicates_and_boosts_overlapping_chunks() -> None:
    shared_bm25 = make_document("shared", "bm25 copy")
    shared_vector = make_document("shared", "vector copy")

    merged = rrf_merge_documents(
        [shared_bm25, make_document("bm25-tail")],
        [make_document("vector-head"), shared_vector],
        alpha=0.5,
        rrf_k=60,
        top_n=3,
    )

    assert chunk_ids(merged) == ["shared", "vector-head", "bm25-tail"]
    assert merged[0].page_content == "bm25 copy"


def test_hybrid_retriever_uses_bm25_when_vector_search_fails() -> None:
    bm25_document = make_document("bm25-only")
    retriever = HybridRRFRetriever(
        bm25_retriever=FakeRetriever([bm25_document]),
        vector_retriever=FakeRetriever(error=RuntimeError("vector unavailable")),
        top_k=5,
    )

    assert chunk_ids(retriever.invoke("청년수당 신청 자격")) == ["bm25-only"]


def test_hybrid_retriever_returns_empty_when_searches_succeed_without_results() -> None:
    retriever = HybridRRFRetriever(
        bm25_retriever=FakeRetriever(),
        vector_retriever=FakeRetriever(),
        top_k=5,
    )

    assert retriever.invoke("검색 결과 없는 질문") == []


def test_hybrid_retriever_raises_when_all_search_backends_fail() -> None:
    retriever = HybridRRFRetriever(
        bm25_retriever=FakeRetriever(error=RuntimeError("bm25 unavailable")),
        vector_retriever=FakeRetriever(error=RuntimeError("vector unavailable")),
        top_k=5,
    )

    with pytest.raises(RetrievalUnavailableError):
        retriever.invoke("검색 시스템 장애")


def test_create_ensemble_retriever_loads_faiss_index_and_builds_bm25_from_docstore(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bm25_doc = make_document("bm25-id", "청년수당 콜센터는 1566-3344입니다.")
    vector_doc = make_document("vector-id", "문의처 안내")
    captured: dict[str, object] = {}

    class FakeRetriever:
        def __init__(self, docs_by_query: dict[str, list[Document]]) -> None:
            self.docs_by_query = docs_by_query
            self.queries: list[str] = []

        def invoke(self, query: str) -> list[Document]:
            self.queries.append(query)
            return self.docs_by_query.get(query, [])

    bm25_retriever = FakeRetriever({"1566-3344": [bm25_doc]})
    vector_retriever = FakeRetriever({"1566-3344": [vector_doc]})

    class FakeEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            captured["embedding_kwargs"] = kwargs

    class FakeDocstore:
        _dict = {
            "doc-1": bm25_doc,
            "doc-2": vector_doc,
        }

    class FakeFAISSStore:
        docstore = FakeDocstore()

        @classmethod
        def load_local(cls, index_dir: str, embedding: object, **kwargs: object):
            captured["faiss_index_dir"] = index_dir
            captured["faiss_embedding"] = embedding
            captured["faiss_kwargs"] = kwargs
            return cls()

        def as_retriever(self, **kwargs: object) -> FakeRetriever:
            captured["vector_kwargs"] = kwargs
            return vector_retriever

    class FakeBM25Retriever:
        @classmethod
        def from_documents(cls, documents: list[Document], **kwargs: object) -> FakeRetriever:
            captured["bm25_documents"] = documents
            captured["bm25_kwargs"] = kwargs
            return bm25_retriever

    index_dir = tmp_path / "faiss_rag_final"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"trusted-test-index")
    (index_dir / "index.pkl").write_bytes(b"trusted-test-docstore")
    settings = SimpleNamespace(
        openai_api_key="test-key",
        openai_embedding_model="test-embedding-model",
        faiss_index_dir=index_dir,
        retrieval_top_k=2,
        retrieval_candidate_k=20,
        retrieval_rrf_k=60,
    )

    monkeypatch.setattr(retrieval, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(retrieval, "FAISS", FakeFAISSStore)
    monkeypatch.setattr(retrieval, "BM25Retriever", FakeBM25Retriever)
    monkeypatch.setattr(retrieval, "TRUSTED_FAISS_INDEX_ROOT", tmp_path)

    hybrid_retriever = create_ensemble_retriever(settings)
    results = hybrid_retriever.invoke("1566-3344")

    assert [document.metadata["chunk_id"] for document in results] == ["bm25-id", "vector-id"]
    assert captured["bm25_kwargs"] == {"preprocess_func": retrieval.tokenize_for_bm25}
    assert captured["embedding_kwargs"] == {
        "api_key": "test-key",
        "model": "test-embedding-model",
    }
    assert Path(str(captured["faiss_index_dir"])) == index_dir.resolve()
    assert captured["faiss_kwargs"] == {"allow_dangerous_deserialization": True}
    assert captured["bm25_documents"] == [bm25_doc, vector_doc]
    assert bm25_retriever.queries == ["1566-3344"]
    assert vector_retriever.queries == ["1566-3344"]
    assert captured["vector_kwargs"] == {
        "search_kwargs": {"k": 20},
    }


def test_create_ensemble_retriever_reports_missing_when_faiss_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        faiss_index_dir=tmp_path / "missing-index",
        retrieval_top_k=5,
        retrieval_candidate_k=20,
        retrieval_rrf_k=60,
    )
    monkeypatch.setattr(retrieval, "TRUSTED_FAISS_INDEX_ROOT", tmp_path)

    with pytest.raises(IndexMissingError):
        create_ensemble_retriever(settings)


def test_create_ensemble_retriever_reports_unavailable_for_corrupt_faiss(
    monkeypatch,
    tmp_path: Path,
) -> None:
    index_dir = tmp_path / "faiss"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"corrupt")
    (index_dir / "index.pkl").write_bytes(b"corrupt")
    settings = SimpleNamespace(
        openai_api_key="test-key",
        openai_embedding_model="test-embedding-model",
        faiss_index_dir=index_dir,
        retrieval_top_k=5,
        retrieval_candidate_k=20,
        retrieval_rrf_k=60,
    )
    monkeypatch.setattr(retrieval, "TRUSTED_FAISS_INDEX_ROOT", tmp_path)

    monkeypatch.setattr(
        retrieval.FAISS,
        "load_local",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("corrupt index")),
    )

    with pytest.raises(RetrievalUnavailableError):
        create_ensemble_retriever(settings)


def test_create_ensemble_retriever_never_loads_faiss_outside_trusted_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "managed"
    trusted_root.mkdir()
    outside_index = tmp_path / "outside"
    outside_index.mkdir()
    (outside_index / "index.faiss").write_bytes(b"untrusted")
    (outside_index / "index.pkl").write_bytes(b"untrusted")
    load_calls = 0

    def fail_if_loaded(*_args, **_kwargs):
        nonlocal load_calls
        load_calls += 1
        raise AssertionError("untrusted FAISS must not be loaded")

    settings = SimpleNamespace(
        openai_api_key="test-key",
        openai_embedding_model="test-embedding-model",
        faiss_index_dir=outside_index,
        retrieval_top_k=5,
        retrieval_candidate_k=20,
        retrieval_rrf_k=60,
    )
    monkeypatch.setattr(retrieval, "TRUSTED_FAISS_INDEX_ROOT", trusted_root)
    monkeypatch.setattr(retrieval.FAISS, "load_local", fail_if_loaded)

    with pytest.raises(RetrievalUnavailableError):
        create_ensemble_retriever(settings)

    assert load_calls == 0


def test_faiss_expected_paths_must_be_regular_files(monkeypatch, tmp_path: Path) -> None:
    index_dir = tmp_path / "faiss"
    index_dir.mkdir()
    (index_dir / "index.faiss").mkdir()
    (index_dir / "index.pkl").write_bytes(b"docstore")
    monkeypatch.setattr(retrieval, "TRUSTED_FAISS_INDEX_ROOT", tmp_path)

    with pytest.raises(RetrievalUnavailableError):
        retrieval.assert_faiss_index_exists(index_dir)


def test_retrieve_pdf_documents_with_ensemble_returns_rank_scored_documents(monkeypatch) -> None:
    top = make_document("top")
    tail = make_document("tail")

    class FakeRetriever:
        def invoke(self, question: str) -> list[Document]:
            assert question == "청년수당 신청 자격"
            return [top, tail]

    settings = SimpleNamespace(retrieval_top_k=2)

    monkeypatch.setattr(retrieval, "create_ensemble_retriever", lambda _: FakeRetriever())

    results = retrieve_pdf_documents_with_ensemble("청년수당 신청 자격", settings)

    assert [item.document.metadata["chunk_id"] for item in results] == ["top", "tail"]
    assert [item.score for item in results] == [1.0, 0.99]


def test_tokenize_for_bm25_splits_korean_into_bigrams_and_keeps_ascii_words() -> None:
    assert retrieval.tokenize_for_bm25("노래방에서 ATM 1566-3344") == [
        "노래", "래방", "방에", "에서", "atm", "1566", "3344",
    ]
    assert retrieval.tokenize_for_bm25("<td>카지노,</td> 및") == ["td", "카지", "지노", "td", "및"]


def test_bm25_with_korean_bigrams_matches_question_with_particle() -> None:
    from langchain_community.retrievers import BM25Retriever

    documents = [
        Document(page_content="| 24 | 노래방 | 25 | 비디오방 |", metadata={"chunk_id": "table"}),
        Document(page_content="자기성장기록서는 매월 제출합니다.", metadata={"chunk_id": "report"}),
        Document(page_content="월세와 관리비는 계좌이체로 낼 수 있습니다.", metadata={"chunk_id": "rent"}),
    ]
    bm25 = BM25Retriever.from_documents(documents, preprocess_func=retrieval.tokenize_for_bm25, k=1)

    assert bm25.invoke("노래방에서 쓴 돈")[0].metadata["chunk_id"] == "table"
