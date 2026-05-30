from langchain_core.documents import Document

from app.core.errors import IndexMissingError
from app.graph.state import GraphState, RetrievedDocument
from app.graph.workflow import (
    assess_evidence,
    fallback_no_answer,
    generate_answer_from_documents,
    run_chat_graph,
)


def test_assess_evidence_rejects_no_documents() -> None:
    state = GraphState(question="없는 내용인가요?", retrieved_documents=[])

    result = assess_evidence(state)

    assert result.evidence.is_sufficient is False
    assert result.status == "insufficient_pdf_evidence"


def test_assess_evidence_accepts_high_score_document() -> None:
    document = Document(
        page_content="청년수당 카드 사용처 기준 안내",
        metadata={"page": 3, "chunk_id": "pdf-page-3-chunk-0", "title": "청년수당 참여자 안내책자"},
    )
    state = GraphState(
        question="카드 사용처는?",
        retrieved_documents=[RetrievedDocument(document=document, score=0.9)],
    )

    result = assess_evidence(state)

    assert result.evidence.is_sufficient is True
    assert result.evidence.source_chunk_ids == ["pdf-page-3-chunk-0"]


def test_assess_evidence_rejects_low_score_document() -> None:
    document = Document(
        page_content="청년수당 카드 사용처 기준 안내",
        metadata={"page": 3, "chunk_id": "pdf-page-3-chunk-0", "title": "청년수당 참여자 안내책자"},
    )
    state = GraphState(
        question="카드 사용처는?",
        retrieved_documents=[RetrievedDocument(document=document, score=0.1)],
        answer="old answer",
        sources=[],
    )

    result = assess_evidence(state)

    assert result.evidence.is_sufficient is False
    assert result.status == "insufficient_pdf_evidence"
    assert result.answer == ""
    assert result.sources == []


def test_generate_answer_from_documents_uses_matching_sources() -> None:
    document = Document(
        page_content="청년수당 카드 사용처 기준 안내",
        metadata={"page": 3, "chunk_id": "pdf-page-3-chunk-0", "title": "청년수당 참여자 안내책자"},
    )
    state = GraphState(
        question="카드 사용처는?",
        retrieved_documents=[RetrievedDocument(document=document, score=0.9)],
    )
    state = assess_evidence(state)

    result = generate_answer_from_documents(state)

    assert result.status == "answered_from_pdf"
    assert result.needs_external_search is False
    assert result.sources
    assert result.sources[0].chunk_id == "pdf-page-3-chunk-0"


def test_generate_answer_from_documents_falls_back_without_matching_sources() -> None:
    document = Document(
        page_content="청년수당 카드 사용처 기준 안내",
        metadata={"page": 3, "chunk_id": "pdf-page-3-chunk-0", "title": "청년수당 참여자 안내책자"},
    )
    state = GraphState(
        question="카드 사용처는?",
        retrieved_documents=[RetrievedDocument(document=document, score=0.9)],
    )
    state.evidence.source_chunk_ids = ["missing-chunk"]

    result = generate_answer_from_documents(state)

    assert result.status == "insufficient_pdf_evidence"
    assert result.needs_external_search is True
    assert result.sources == []


def test_fallback_returns_external_search_signal() -> None:
    state = GraphState(question="최신 변경 사항은?")

    result = fallback_no_answer(state)

    assert result.answer.startswith("안내책자에서 해당 내용을 확인하지 못했습니다.")
    assert result.needs_external_search is True
    assert result.status == "insufficient_pdf_evidence"


def test_run_chat_graph_answers_from_retrieved_pdf_documents() -> None:
    document = Document(
        page_content="청년수당 카드는 사업 목적에 맞는 진로탐색 및 구직활동 비용에 사용할 수 있습니다.",
        metadata={"page": 12, "chunk_id": "pdf-page-12-chunk-0", "title": "청년수당 참여자 안내책자"},
    )

    def fake_retrieve(question: str) -> list[RetrievedDocument]:
        assert question == "청년수당 카드는 어디에 쓸 수 있어?"
        return [RetrievedDocument(document=document, score=0.95)]

    def fake_generate(question: str, documents: list[Document]) -> str:
        assert question == "청년수당 카드는 어디에 쓸 수 있어?"
        assert documents == [document]
        return "청년수당 카드는 진로탐색 및 구직활동 비용에 사용할 수 있습니다."

    response = run_chat_graph(
        "청년수당 카드는 어디에 쓸 수 있어?",
        retrieve_documents=fake_retrieve,
        generate_answer=fake_generate,
    )

    assert response.status == "answered_from_pdf"
    assert response.needs_external_search is False
    assert response.answer == "청년수당 카드는 진로탐색 및 구직활동 비용에 사용할 수 있습니다."
    assert response.sources[0].chunk_id == "pdf-page-12-chunk-0"


def test_run_chat_graph_falls_back_when_index_is_missing() -> None:
    def fake_retrieve(question: str) -> list[RetrievedDocument]:
        raise IndexMissingError("index missing")

    response = run_chat_graph("청년수당 신청 조건은?", retrieve_documents=fake_retrieve)

    assert response.status == "insufficient_pdf_evidence"
    assert response.needs_external_search is True
    assert response.sources == []


def test_run_chat_graph_falls_back_when_pdf_evidence_is_weak() -> None:
    document = Document(
        page_content="전혀 다른 안내입니다.",
        metadata={"page": 2, "chunk_id": "pdf-page-2-chunk-0", "title": "청년수당 참여자 안내책자"},
    )

    def fake_retrieve(question: str) -> list[RetrievedDocument]:
        return [RetrievedDocument(document=document, score=0.01)]

    response = run_chat_graph("청년수당 카드 사용처는?", retrieve_documents=fake_retrieve)

    assert response.status == "insufficient_pdf_evidence"
    assert response.needs_external_search is True
    assert response.sources == []
