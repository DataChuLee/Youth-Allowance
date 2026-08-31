from langchain_core.documents import Document

from app.core.errors import IndexMissingError
from app.graph.state import GraphPolicyResult, GraphState, RetrievedDocument
from app.graph.workflow import (
    assess_evidence,
    fallback_no_answer,
    generate_answer_from_documents,
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


def test_run_chat_graph_passes_graph_and_pdf_context_to_fixed_rag_generator() -> None:
    pdf_document = Document(
        page_content="청년수당은 생활비와 교육비에 사용할 수 있고 온라인 카드 결제가 가능합니다.",
        metadata={"page": 12, "chunk_id": "pdf-page-12-chunk-0", "title": "청년수당 안내책자"},
    )
    graph_policy = GraphPolicyResult(
        rule_id="spending.living.allowed",
        rule_name="생활비 사용 가능",
        decision="allowed",
        page=12,
        chunk_id="ocr-page-12-block-3",
        evidence="생활비 | 식비, 통신비, 교통비 등",
        score=0.91,
    )
    captured: dict[str, object] = {}

    def fake_retrieve_pdf(question: str) -> list[RetrievedDocument]:
        return [RetrievedDocument(document=pdf_document, score=0.95)]

    def fake_retrieve_graph(question: str) -> list[GraphPolicyResult]:
        return [graph_policy]

    def fake_generate_policy_answer(
        question: str,
        documents: list[Document],
        graph_policies: list[GraphPolicyResult],
    ) -> str:
        captured["question"] = question
        captured["documents"] = documents
        captured["graph_policies"] = graph_policies
        return "그래프와 PDF 근거를 함께 사용한 답변"

    response = run_chat_graph(
        "배민으로 청년수당 사용해도 돼?",
        retrieve_documents=fake_retrieve_pdf,
        retrieve_graph_policies=fake_retrieve_graph,
        generate_policy_answer=fake_generate_policy_answer,
    )

    assert captured["question"] == "배민으로 청년수당 사용해도 돼?"
    assert captured["documents"] == [pdf_document]
    assert captured["graph_policies"] == [graph_policy]
    assert response.status == "answered_from_pdf"
    assert response.needs_external_search is False
    assert response.answer == "그래프와 PDF 근거를 함께 사용한 답변"
    assert [source.type for source in response.sources] == ["graph", "pdf"]
    assert response.sources[0].chunk_id == "ocr-page-12-block-3"


def test_run_chat_graph_continues_with_pdf_when_graph_retrieval_fails() -> None:
    pdf_document = Document(
        page_content="월세는 계좌이체 가능하지만 증빙서류가 필요합니다.",
        metadata={"page": 13, "chunk_id": "pdf-page-13-chunk-0", "title": "청년수당 안내책자"},
    )

    def fake_retrieve_pdf(question: str) -> list[RetrievedDocument]:
        return [RetrievedDocument(document=pdf_document, score=0.9)]

    def fail_retrieve_graph(question: str) -> list[GraphPolicyResult]:
        raise RuntimeError("neo4j unavailable")

    def fake_generate_policy_answer(
        question: str,
        documents: list[Document],
        graph_policies: list[GraphPolicyResult],
    ) -> str:
        assert documents == [pdf_document]
        assert graph_policies == []
        return "PDF 근거만 사용한 답변"

    response = run_chat_graph(
        "월세 계좌이체 가능해?",
        retrieve_documents=fake_retrieve_pdf,
        retrieve_graph_policies=fail_retrieve_graph,
        generate_policy_answer=fake_generate_policy_answer,
    )

    assert response.status == "answered_from_pdf"
    assert response.answer == "PDF 근거만 사용한 답변"
    assert [source.type for source in response.sources] == ["pdf"]


def test_run_chat_graph_filters_non_direct_food_delivery_graph_sources() -> None:
    policies = [
        GraphPolicyResult(
            rule_id="spending.clean_card.blocked_categories",
            rule_name="클린카드 사용 불가 업종 제한",
            decision="blocked",
            page=12,
            chunk_id="ocr-page-12-block-4",
            evidence="호텔, 주점, 복권 판매, 피부미용, 마사지 등 48개 업종",
            score=1.0,
        ),
        GraphPolicyResult(
            rule_id="card.simple_payment.blocked",
            rule_name="간편결제 사용 불가",
            decision="blocked",
            page=12,
            chunk_id="ocr-page-12-block-1",
            evidence="카카오페이, 네이버페이 등 간편 결제 불가",
            score=0.8,
        ),
        GraphPolicyResult(
            rule_id="spending.living.allowed",
            rule_name="생활비 사용 가능",
            decision="allowed",
            page=12,
            chunk_id="ocr-page-12-block-3",
            evidence="생활비 | 식비, 통신비, 교통비 등",
            score=0.7,
        ),
    ]
    captured: dict[str, object] = {}

    def fake_retrieve_pdf(question: str) -> list[RetrievedDocument]:
        return []

    def fake_retrieve_graph(question: str) -> list[GraphPolicyResult]:
        return policies

    def fake_generate_policy_answer(
        question: str,
        documents: list[Document],
        graph_policies: list[GraphPolicyResult],
    ) -> str:
        captured["graph_policies"] = graph_policies
        return "배민은 식비 목적의 전용 체크카드 직접 결제라면 가능하고 간편결제는 불가합니다."

    response = run_chat_graph(
        "배민에서 청년수당을 사용해도 돼?",
        retrieve_documents=fake_retrieve_pdf,
        retrieve_graph_policies=fake_retrieve_graph,
        generate_policy_answer=fake_generate_policy_answer,
    )

    assert [policy.rule_id for policy in captured["graph_policies"]] == [
        "card.simple_payment.blocked",
        "spending.living.allowed",
    ]
    assert [source.chunk_id for source in response.sources] == [
        "ocr-page-12-block-1",
        "ocr-page-12-block-3",
    ]
    assert response.status == "answered_from_pdf"


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
