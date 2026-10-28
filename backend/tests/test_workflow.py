from langchain_core.documents import Document

from app.core.errors import IndexMissingError
from app.graph.state import GraphState, RetrievedDocument
from app.graph.workflow import (
    assess_evidence,
    fallback_no_answer,
    run_chat_graph,
)
from app.rag.retrieval import RetrievalUnavailableError


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


def test_run_chat_graph_falls_back_when_retrieval_system_is_unavailable() -> None:
    def fail_retrieval(_: str) -> list[RetrievedDocument]:
        raise RetrievalUnavailableError("PDF 검색 backend를 사용할 수 없습니다.")

    response = run_chat_graph(
        "검색 시스템 장애 질문",
        retrieve_documents=fail_retrieval,
    )

    assert response.status == "insufficient_pdf_evidence"
    assert response.needs_external_search is True
    assert response.sources == []
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


def test_run_chat_graph_passes_pdf_context_to_policy_generator() -> None:
    pdf_document = Document(
        page_content="청년수당은 생활비와 교육비에 사용할 수 있고 온라인 카드 결제가 가능합니다.",
        metadata={"page": 12, "chunk_id": "pdf-page-12-chunk-0", "title": "청년수당 안내책자"},
    )
    captured: dict[str, object] = {}

    def fake_retrieve_pdf(question: str) -> list[RetrievedDocument]:
        return [RetrievedDocument(document=pdf_document, score=0.95)]

    def fake_generate_policy_answer(question: str, documents: list[Document]) -> str:
        captured["question"] = question
        captured["documents"] = documents
        return "PDF 근거를 사용한 답변"

    response = run_chat_graph(
        "배민으로 청년수당 사용해도 돼?",
        retrieve_documents=fake_retrieve_pdf,
        generate_policy_answer=fake_generate_policy_answer,
    )

    assert captured["question"] == "배민으로 청년수당 사용해도 돼?"
    assert captured["documents"] == [pdf_document]
    assert response.status == "answered_from_pdf"
    assert response.needs_external_search is False
    assert response.answer == "PDF 근거를 사용한 답변"
    assert [source.type for source in response.sources] == ["pdf"]


def test_run_chat_graph_uses_request_history_without_persisting_it() -> None:
    document = Document(
        page_content="청년수당 사용처 안내",
        metadata={"page": 12, "chunk_id": "pdf-page-12-chunk-0", "title": "안내책자"},
    )
    generated_questions: list[str] = []

    def retrieve(_: str) -> list[RetrievedDocument]:
        return [RetrievedDocument(document=document, score=0.9)]

    def generate(question: str, _: list[Document]) -> str:
        generated_questions.append(question)
        return "답변"

    run_chat_graph(
        "그건 어디에 써?",
        thread_id="same-tab",
        history=[
            {"role": "user", "content": "청년수당 카드가 뭐야?"},
            {"role": "assistant", "content": "청년수당 전용 카드입니다."},
        ],
        retrieve_documents=retrieve,
        generate_answer=generate,
    )
    run_chat_graph(
        "새 질문",
        thread_id="same-tab",
        retrieve_documents=retrieve,
        generate_answer=generate,
    )

    assert "청년수당 카드가 뭐야?" in generated_questions[0]
    assert "[이전 대화]" not in generated_questions[1]
