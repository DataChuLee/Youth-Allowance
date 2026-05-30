from collections.abc import Callable
from typing import Literal

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from app.api.schemas import ChatResponse
from app.core.config import Settings, get_settings
from app.core.errors import IndexMissingError
from app.graph.state import EvidenceDecision, GraphState, IntentDecision, RetrievedDocument
from app.rag.evidence import GradeEvidence, grade_evidence as grade_pdf_evidence
from app.rag.generation import (
    build_blocked_answer,
    build_fallback_answer,
    build_food_delivery_answer,
    build_general_answer,
    build_study_device_answer,
    generate_pdf_answer,
)
from app.rag.intent import ClassifyIntent, create_intent_classifier, normalize_intent_decision
from app.rag.policy import (
    is_food_delivery_question,
    is_study_device_question,
    resolve_policy_decision,
)
from app.rag.query_planner import build_search_queries
from app.rag.retrieval import (
    RetrieveDocuments,
    retrieve_pdf_documents_with_ensemble,
    retrieve_with_queries,
)
from app.rag.sources import document_to_source

DEFAULT_MIN_SIMILARITY_SCORE = 0.2

PlanSearchQueries = Callable[[str], list[str]]
GenerateAnswer = Callable[[str, list[Document]], str]
IntentRouteName = Literal["general_answer", "rag"]
RouteName = Literal["blocked", "insufficient", "answerable"]


def documents_from_state(state: GraphState) -> list[Document]:
    return [item.document for item in state.retrieved_documents]


def assess_evidence(
    state: GraphState,
    min_similarity_score: float = DEFAULT_MIN_SIMILARITY_SCORE,
) -> GraphState:
    if not state.retrieved_documents:
        state.evidence = EvidenceDecision(
            is_sufficient=False,
            support_level="insufficient",
            reason="검색된 PDF 청크가 없습니다.",
            source_chunk_ids=[],
        )
        state.status = "insufficient_pdf_evidence"
        state.needs_external_search = True
        state.answer = ""
        state.sources = []
        return state

    best = max(state.retrieved_documents, key=lambda item: item.score)
    if best.score < min_similarity_score:
        state.evidence = EvidenceDecision(
            is_sufficient=False,
            support_level="insufficient",
            reason="검색 점수가 최소 기준보다 낮습니다.",
            source_chunk_ids=[],
        )
        state.status = "insufficient_pdf_evidence"
        state.needs_external_search = True
        state.answer = ""
        state.sources = []
        return state

    chunk_id = str(best.document.metadata["chunk_id"])
    state.evidence = EvidenceDecision(
        is_sufficient=True,
        support_level="direct",
        reason="상위 검색 청크가 최소 검색 점수 기준을 통과했습니다.",
        source_chunk_ids=[chunk_id],
    )
    state.status = "answered_from_pdf"
    state.needs_external_search = False
    return state


def select_evidence_documents(state: GraphState) -> list[Document]:
    source_chunk_ids = state.policy.source_chunk_ids or state.evidence.source_chunk_ids
    allowed = set(source_chunk_ids)
    return [
        item.document
        for item in state.retrieved_documents
        if str(item.document.metadata.get("chunk_id")) in allowed
    ]


def generate_answer_from_documents(
    state: GraphState,
    generate_answer: GenerateAnswer | None = None,
) -> GraphState:
    documents = select_evidence_documents(state)
    if not documents:
        return fallback_no_answer(state)

    if generate_answer is None:
        context = "\n\n".join(document.page_content for document in documents)
        state.answer = f"안내책자 근거로 확인한 내용입니다.\n\n{context}"
    else:
        state.answer = generate_answer(state.question, documents)
    state.sources = [document_to_source(document) for document in documents]
    state.status = "answered_from_pdf"
    state.needs_external_search = False
    return state


def fallback_no_answer(state: GraphState) -> GraphState:
    state.answer = build_fallback_answer()
    state.sources = []
    state.status = "insufficient_pdf_evidence"
    state.needs_external_search = True
    return state


def classify_intent_node(classify_intent: ClassifyIntent) -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        try:
            decision = normalize_intent_decision(classify_intent(state.question))
        except Exception:
            decision = IntentDecision(intent="rag", reason="classifier failed")
        state.intent = decision.intent
        return state

    return node


def route_intent(state: GraphState) -> IntentRouteName:
    return state.intent


def general_answer_node() -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        state.answer = build_general_answer(state.question)
        state.sources = []
        state.status = "general_answer"
        state.needs_external_search = False
        state.intent = "general_answer"
        return state

    return node


def plan_queries_node(plan_search_queries: PlanSearchQueries) -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        try:
            state.search_queries = plan_search_queries(state.question)
        except Exception:
            # Query planner는 검색 품질 개선 단계이므로 실패해도 원 질문 검색으로 계속 진행한다.
            state.search_queries = [state.question]
        return state

    return node


def retrieve_pdf_node(
    retrieve_documents: RetrieveDocuments,
    settings: Settings,
    use_query_expansion: bool,
) -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        try:
            if use_query_expansion:
                search_queries = state.search_queries or [state.question]
                state.retrieved_documents = retrieve_with_queries(
                    state.question,
                    search_queries,
                    retrieve_documents=retrieve_documents,
                    max_docs=max(settings.retrieval_top_k, 8),
                    per_query_k=settings.retrieval_top_k,
                )
            else:
                state.retrieved_documents = retrieve_documents(state.question)
        except IndexMissingError:
            state.retrieved_documents = []
            state.evidence = EvidenceDecision(
                is_sufficient=False,
                support_level="insufficient",
                reason="PDF 인덱스가 없습니다. 먼저 indexing 명령을 실행해야 합니다.",
                source_chunk_ids=[],
            )
        return state

    return node


def grade_evidence_node(
    grade_evidence: GradeEvidence | None,
    settings: Settings,
) -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        if state.evidence.reason and not state.retrieved_documents:
            return state
        if grade_evidence is None:
            return assess_evidence(state, settings.min_similarity_score)
        state.evidence = EvidenceDecision.model_validate(
            grade_evidence(state.question, documents_from_state(state))
        )
        return state

    return node


def resolve_policy_node() -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        state.policy = resolve_policy_decision(
            state.question,
            documents_from_state(state),
            state.evidence,
        )
        return state

    return node


def route_policy(state: GraphState) -> RouteName:
    if state.policy.decision == "blocked":
        return "blocked"
    if state.policy.decision == "insufficient":
        return "insufficient"
    return "answerable"


def generate_policy_blocked_answer_node() -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        documents = select_evidence_documents(state)
        if not documents:
            return fallback_no_answer(state)
        state.answer = build_blocked_answer(state.policy)
        state.sources = [document_to_source(document) for document in documents]
        state.status = "blocked_by_policy"
        state.needs_external_search = False
        return state

    return node


def generate_answer_node(generate_answer: GenerateAnswer) -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        documents = select_evidence_documents(state)
        if not documents:
            return fallback_no_answer(state)

        if is_food_delivery_question(state.question):
            state.answer = build_food_delivery_answer()
        elif is_study_device_question(state.question):
            state.answer = build_study_device_answer()
        else:
            state.answer = generate_answer(state.question, documents)

        state.sources = [document_to_source(document) for document in documents]
        state.status = "answered_from_pdf"
        state.needs_external_search = False
        return state

    return node


def fallback_no_answer_node() -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        return fallback_no_answer(state)

    return node


def build_chat_graph(
    classify_intent: ClassifyIntent,
    plan_search_queries: PlanSearchQueries,
    retrieve_documents: RetrieveDocuments,
    grade_evidence: GradeEvidence | None,
    generate_answer: GenerateAnswer,
    settings: Settings,
    use_query_expansion: bool = True,
):
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent_node(classify_intent))
    graph.add_node("general_answer", general_answer_node())
    graph.add_node("plan_queries", plan_queries_node(plan_search_queries))
    graph.add_node(
        "retrieve_pdf",
        retrieve_pdf_node(retrieve_documents, settings, use_query_expansion),
    )
    graph.add_node("grade_evidence", grade_evidence_node(grade_evidence, settings))
    graph.add_node("resolve_policy", resolve_policy_node())
    graph.add_node("generate_blocked_answer", generate_policy_blocked_answer_node())
    graph.add_node("generate_answer", generate_answer_node(generate_answer))
    graph.add_node("fallback_no_answer", fallback_no_answer_node())

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "general_answer": "general_answer",
            "rag": "plan_queries",
        },
    )
    graph.add_edge("general_answer", END)
    graph.add_edge("plan_queries", "retrieve_pdf")
    graph.add_edge("retrieve_pdf", "grade_evidence")
    graph.add_edge("grade_evidence", "resolve_policy")
    graph.add_conditional_edges(
        "resolve_policy",
        route_policy,
        {
            "blocked": "generate_blocked_answer",
            "insufficient": "fallback_no_answer",
            "answerable": "generate_answer",
        },
    )
    graph.add_edge("generate_blocked_answer", END)
    graph.add_edge("generate_answer", END)
    graph.add_edge("fallback_no_answer", END)
    return graph.compile()


def run_chat_graph(
    question: str,
    classify_intent: ClassifyIntent | None = None,
    plan_search_queries: PlanSearchQueries | None = None,
    retrieve_documents: RetrieveDocuments | None = None,
    grade_evidence: GradeEvidence | None = None,
    generate_answer: GenerateAnswer | None = None,
    settings: Settings | None = None,
) -> ChatResponse:
    resolved_settings = settings or get_settings()
    if classify_intent is not None:
        resolved_classify_intent = classify_intent
    elif resolved_settings.app_env == "test" or retrieve_documents is not None:
        resolved_classify_intent = lambda query: IntentDecision(intent="rag", reason="test default")
    else:
        resolved_classify_intent = create_intent_classifier(resolved_settings)

    if plan_search_queries is not None:
        resolved_plan_search_queries = plan_search_queries
    elif retrieve_documents is not None or resolved_settings.app_env == "test":
        resolved_plan_search_queries = lambda query: [query]
    else:
        resolved_plan_search_queries = lambda query: build_search_queries(
            query,
            settings=resolved_settings,
        )
    resolved_retrieve_documents = retrieve_documents or (
        lambda query: retrieve_pdf_documents_with_ensemble(query, resolved_settings)
    )
    if grade_evidence is not None:
        resolved_grade_evidence = grade_evidence
    elif retrieve_documents is not None or resolved_settings.app_env == "test":
        resolved_grade_evidence = None
    else:
        resolved_grade_evidence = lambda query, documents: grade_pdf_evidence(
            query,
            documents,
            settings=resolved_settings,
        )
    resolved_generate_answer = generate_answer or (
        lambda query, documents: generate_pdf_answer(query, documents, resolved_settings)
    )

    graph = build_chat_graph(
        classify_intent=resolved_classify_intent,
        plan_search_queries=resolved_plan_search_queries,
        retrieve_documents=resolved_retrieve_documents,
        grade_evidence=resolved_grade_evidence,
        generate_answer=resolved_generate_answer,
        settings=resolved_settings,
        use_query_expansion=not (retrieve_documents is not None and plan_search_queries is None),
    )
    result = graph.invoke(GraphState(question=question))
    state = GraphState.model_validate(result)
    return ChatResponse(
        answer=state.answer,
        sources=state.sources,
        status=state.status,
        needs_external_search=state.needs_external_search,
        intent=state.intent,
    )


def retrieved_document_from_score(document: Document, score: float) -> RetrievedDocument:
    return RetrievedDocument(document=document, score=score)
