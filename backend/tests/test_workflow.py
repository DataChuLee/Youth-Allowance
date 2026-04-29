from langchain_core.documents import Document

from app.graph.workflow import assess_evidence, fallback_no_answer
from app.graph.state import GraphState, RetrievedDocument


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


def test_fallback_returns_external_search_signal() -> None:
    state = GraphState(question="최신 변경 사항은?")

    result = fallback_no_answer(state)

    assert result.answer.startswith("안내책자에서 해당 내용을 확인하지 못했습니다")
    assert result.needs_external_search is True
    assert result.status == "insufficient_pdf_evidence"
