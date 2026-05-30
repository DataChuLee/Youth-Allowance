from langchain_core.documents import Document

from app.graph.state import EvidenceDecision, IntentDecision, RetrievedDocument
from app.graph.workflow import run_chat_graph
from app.rag.evidence import grade_evidence
from app.rag.intent import normalize_intent_decision
from app.rag.query_planner import QueryPlan, build_search_queries
from app.rag.retrieval import retrieve_with_queries


def make_retrieved_document(
    chunk_id: str,
    content: str,
    *,
    page: int = 1,
    score: float = 1.0,
) -> RetrievedDocument:
    return RetrievedDocument(
        document=Document(
            page_content=content,
            metadata={
                "page": page,
                "chunk_id": chunk_id,
                "title": "청년수당 참여자 안내책자",
            },
        ),
        score=score,
    )


def test_build_search_queries_includes_original_and_deduplicates_planned_queries() -> None:
    def fake_plan(question: str) -> QueryPlan:
        assert question == "청년수당으로 배민 결제 가능해?"
        return QueryPlan(
            rewritten_query="청년수당 배달앱 식비 온라인 카드 결제 가능 여부",
            queries=[
                "청년수당 카드 사용 가능 항목 식비 생활비",
                "청년수당 배달앱 식비 온라인 카드 결제 가능 여부",
                "청년수당 간편결제 포인트 충전 상품권 기프티콘 사용 불가",
            ],
        )

    queries = build_search_queries(
        "청년수당으로 배민 결제 가능해?",
        plan_query=fake_plan,
        max_queries=4,
    )

    assert queries == [
        "청년수당으로 배민 결제 가능해?",
        "청년수당 배달앱 식비 온라인 카드 결제 가능 여부",
        "청년수당 카드 사용 가능 항목 식비 생활비",
        "청년수당 간편결제 포인트 충전 상품권 기프티콘 사용 불가",
    ]


def test_retrieve_with_queries_interleaves_and_deduplicates_ensemble_results() -> None:
    documents_by_query = {
        "q1": [
            make_retrieved_document("chunk-a", "첫 번째 쿼리 상위 문서"),
            make_retrieved_document("chunk-b", "첫 번째 쿼리 두 번째 문서"),
        ],
        "q2": [
            make_retrieved_document("chunk-c", "두 번째 쿼리 상위 문서"),
            make_retrieved_document("chunk-a", "중복 문서"),
        ],
    }

    def fake_retrieve(query: str) -> list[RetrievedDocument]:
        return documents_by_query[query]

    results = retrieve_with_queries(
        "일반 질문",
        ["q1", "q2"],
        retrieve_documents=fake_retrieve,
        max_docs=3,
        per_query_k=2,
    )

    assert [item.document.metadata["chunk_id"] for item in results] == [
        "chunk-a",
        "chunk-c",
        "chunk-b",
    ]
    assert [item.document.metadata["retrieved_by_query"] for item in results] == [
        "q1",
        "q2",
        "q1",
    ]


def test_grade_evidence_keeps_llm_insufficient_decision_even_with_source_ids() -> None:
    document = make_retrieved_document(
        "chunk-a",
        "질문과 직접 관련이 없어 충분하지 않은 문서",
    ).document

    def fake_grade(question: str, documents: list[Document]) -> EvidenceDecision:
        return EvidenceDecision(
            is_sufficient=False,
            support_level="direct",
            reason="근거가 충분하지 않습니다.",
            source_chunk_ids=["chunk-a"],
        )

    result = grade_evidence("청년수당 신청 자격은?", [document], grade_with_llm=fake_grade)

    assert result.is_sufficient is False
    assert result.support_level == "direct"
    assert result.source_chunk_ids == ["chunk-a"]


def test_normalize_intent_decision_routes_invalid_intent_to_rag() -> None:
    result = normalize_intent_decision({"intent": "unknown", "reason": "bad"})

    assert result.intent == "rag"
    assert result.reason == "invalid classifier output"


def test_run_chat_graph_returns_general_answer_without_retrieval() -> None:
    def fail_retrieve(_: str) -> list[RetrievedDocument]:
        raise AssertionError("general_answer must not retrieve")

    response = run_chat_graph(
        "안녕?",
        classify_intent=lambda question: IntentDecision(
            intent="general_answer",
            reason="greeting",
        ),
        retrieve_documents=fail_retrieve,
    )

    assert response.intent == "general_answer"
    assert response.status == "general_answer"
    assert response.needs_external_search is False
    assert response.sources == []
    assert "청년수당" in response.answer


def test_run_chat_graph_routes_classifier_failure_to_rag() -> None:
    document = make_retrieved_document(
        "chunk-answer",
        "청년수당 카드는 구직활동 비용에 사용할 수 있습니다.",
    )

    def fail_classify(_: str) -> IntentDecision:
        raise RuntimeError("classifier failed")

    def fake_retrieve(query: str) -> list[RetrievedDocument]:
        assert query == "청년수당 카드 사용처는?"
        return [document]

    def fake_generate(question: str, documents: list[Document]) -> str:
        return "청년수당 카드는 구직활동 비용에 사용할 수 있습니다."

    response = run_chat_graph(
        "청년수당 카드 사용처는?",
        classify_intent=fail_classify,
        retrieve_documents=fake_retrieve,
        generate_answer=fake_generate,
    )

    assert response.intent == "rag"
    assert response.status == "answered_from_pdf"


def test_run_chat_graph_returns_policy_blocked_answer_with_blocked_status() -> None:
    blocked_document = make_retrieved_document(
        "chunk-blocked",
        "카카오페이, 네이버페이 등 간편결제 불가",
        page=7,
    )

    def fake_retrieve(query: str) -> list[RetrievedDocument]:
        assert query == "청년수당 간편결제 불가"
        return [blocked_document]

    def fake_grade(
        question: str,
        documents: list[Document],
    ) -> EvidenceDecision:
        assert question == "청년수당으로 카카오페이 결제해도 돼?"
        assert [document.metadata["chunk_id"] for document in documents] == [
            "chunk-blocked"
        ]
        return EvidenceDecision(
            is_sufficient=True,
            support_level="direct",
            reason="간편결제 불가 근거가 확인됩니다.",
            source_chunk_ids=["chunk-blocked"],
        )

    response = run_chat_graph(
        "청년수당으로 카카오페이 결제해도 돼?",
        classify_intent=lambda question: IntentDecision(
            intent="rag",
            reason="policy question",
        ),
        plan_search_queries=lambda question: ["청년수당 간편결제 불가"],
        retrieve_documents=fake_retrieve,
        grade_evidence=fake_grade,
    )

    assert response.intent == "rag"
    assert response.status == "blocked_by_policy"
    assert response.needs_external_search is False
    assert "간편결제" in response.answer
    assert response.sources[0].chunk_id == "chunk-blocked"
