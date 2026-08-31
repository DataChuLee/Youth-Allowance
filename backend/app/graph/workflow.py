from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from app.api.schemas import ChatResponse, Source
from app.core.cache import get_query_cache
from app.core.config import Settings, get_settings
from app.core.errors import IndexMissingError
from app.graph.state import EvidenceDecision, GraphPolicyResult, GraphState, IntentDecision, RetrievedDocument
from app.rag.evidence import GradeEvidence, grade_evidence as grade_pdf_evidence
from app.rag.generation import (
    build_blocked_answer,
    build_fallback_answer,
    build_general_answer,
    generate_pdf_answer,
    generate_policy_context_answer,
    select_graph_policies_for_question,
    stream_pdf_answer,
    stream_policy_context_answer,
)
from app.rag.intent import ClassifyIntent, create_intent_classifier, normalize_intent_decision
from app.rag.policy import build_policy_evidence_queries, resolve_policy_decision
from app.rag.query_planner import build_search_queries
from app.rag.retrieval import (
    RetrieveDocuments,
    RetrievalUnavailableError,
    retrieve_pdf_documents_with_ensemble,
    retrieve_with_queries,
)
from app.rag.sources import document_to_source, graph_policy_to_source

DEFAULT_MIN_SIMILARITY_SCORE = 0.2


def _augment_question_with_history(question: str, history: list[dict]) -> str:
    """이전 대화를 질문 앞에 붙여 멀티턴 컨텍스트를 제공한다."""
    if not history:
        return question
    recent = history[-4:]  # 최대 2턴(4메시지)
    lines = "\n".join(
        f"{'사용자' if m.get('role') == 'user' else '챗봇'}: {m.get('content', '')}"
        for m in recent
    )
    return f"[이전 대화]\n{lines}\n\n[현재 질문]\n{question}"


def _score_lookup(state: GraphState) -> dict[str, float]:
    """retrieved_documents에서 chunk_id → score 맵을 생성한다."""
    return {
        str(item.document.metadata.get("chunk_id", "")): item.score
        for item in state.retrieved_documents
        if item.document.metadata.get("chunk_id")
    }


PlanSearchQueries = Callable[[str], list[str]]
GenerateAnswer = Callable[[str, list[Document]], str]
RetrieveGraphPolicies = Callable[[str], list[GraphPolicyResult]]
GeneratePolicyAnswer = Callable[[str, list[Document], list[GraphPolicyResult]], str]
IntentRouteName = Literal["general_answer", "rag"]
RouteName = Literal["blocked", "insufficient", "answerable"]
AnswerStatus = Literal[
    "general_answer",
    "answered_from_pdf",
    "insufficient_pdf_evidence",
    "blocked_by_policy",
]


@dataclass(frozen=True)
class ChatGraphDependencies:
    classify_intent: ClassifyIntent
    plan_search_queries: PlanSearchQueries
    retrieve_documents: RetrieveDocuments
    retrieve_graph_policies: RetrieveGraphPolicies
    grade_evidence: GradeEvidence | None
    generate_answer: GenerateAnswer
    generate_policy_answer: GeneratePolicyAnswer
    settings: Settings
    use_query_expansion: bool
    use_policy_context_generation: bool


@dataclass(frozen=True)
class PreparedChatStream:
    chunks: Iterator[str]
    sources: list[Source]
    status: AnswerStatus
    needs_external_search: bool
    intent: Literal["general_answer", "rag"]


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


def select_context_documents(state: GraphState) -> list[Document]:
    documents = select_evidence_documents(state)
    if documents:
        return documents
    if state.graph_policy_results:
        return documents_from_state(state)
    return []


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
    scores = _score_lookup(state)
    state.sources = [
        document_to_source(doc, scores.get(str(doc.metadata.get("chunk_id", "")), 0.0))
        for doc in documents
    ]
    state.status = "answered_from_pdf"
    state.needs_external_search = False
    return state


def fallback_no_answer(state: GraphState) -> GraphState:
    state.answer = build_fallback_answer()
    state.sources = []
    state.status = "insufficient_pdf_evidence"
    state.needs_external_search = True
    state.messages = state.messages + [
        {"role": "user", "content": state.question},
        {"role": "assistant", "content": state.answer},
    ]
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
        state.messages = state.messages + [
            {"role": "user", "content": state.question},
            {"role": "assistant", "content": state.answer},
        ]
        return state

    return node


def plan_queries_node(plan_search_queries: PlanSearchQueries) -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        policy_queries = build_policy_evidence_queries(state.question)
        if policy_queries:
            # 명시적 제한 질문은 LLM query planner까지 호출하지 않는다. 원문 제한 조항을
            # 바로 검색해 근거가 있을 때만 차단하므로 지연과 토큰 사용을 함께 줄인다.
            # 이 경로에서는 사용자 표현보다 규정 용어의 lexical match를 우선한다.
            state.search_queries = policy_queries
            return state
        try:
            state.search_queries = plan_search_queries(state.question)
        except Exception:
            # Query planner는 검색 품질 개선 단계이므로 실패해도 원 질문 검색으로 계속 진행한다.
            state.search_queries = [state.question]
        # 정책 제한 신호는 LLM planner 결과와 무관하게 원문 규정 표현으로 한 번 더 검색한다.
        state.search_queries = list(dict.fromkeys(state.search_queries))
        return state

    return node


def retrieve_pdf_node(
    retrieve_documents: RetrieveDocuments,
    settings: Settings,
    use_query_expansion: bool,
) -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        cache = get_query_cache()
        cache_key = state.question.strip()

        cached = cache.get(cache_key)
        if cached is not None:
            state.retrieved_documents = cached
            return state

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
        except (IndexMissingError, RetrievalUnavailableError) as exc:
            state.retrieved_documents = []
            state.evidence = EvidenceDecision(
                is_sufficient=False,
                support_level="insufficient",
                reason=str(exc),
                source_chunk_ids=[],
            )
            return state

        cache.set(cache_key, state.retrieved_documents)
        return state

    return node


def retrieve_graph_node(
    retrieve_graph_policies: RetrieveGraphPolicies,
) -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        try:
            state.graph_policy_results = retrieve_graph_policies(state.question)
        except Exception:
            # Graph RAG는 보조 정책 근거 레이어입니다. Neo4j 장애가 있어도 PDF RAG 답변 경로는 유지합니다.
            state.graph_policy_results = []
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


def resolve_policy_node(*, final: bool) -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        state.policy = resolve_policy_decision(
            state.question,
            documents_from_state(state),
            state.evidence if final else None,
        )
        return state

    return node


def route_policy(state: GraphState) -> RouteName:
    if state.policy.decision == "blocked":
        return "blocked"
    if state.graph_policy_results:
        return "answerable"
    if state.policy.decision == "insufficient":
        return "insufficient"
    return "answerable"


def generate_policy_blocked_answer_node() -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        documents = select_evidence_documents(state)
        if not documents:
            return fallback_no_answer(state)
        state.answer = build_blocked_answer(state.policy)
        scores = _score_lookup(state)
        state.sources = [
            document_to_source(doc, scores.get(str(doc.metadata.get("chunk_id", "")), 0.0))
            for doc in documents
        ]
        state.status = "blocked_by_policy"
        state.needs_external_search = False
        state.messages = state.messages + [
            {"role": "user", "content": state.question},
            {"role": "assistant", "content": state.answer},
        ]
        return state

    return node


def generate_answer_node(
    generate_answer: GenerateAnswer,
    generate_policy_answer: GeneratePolicyAnswer,
    use_policy_context_generation: bool,
) -> Callable[[GraphState], GraphState]:
    def node(state: GraphState) -> GraphState:
        documents = select_context_documents(state)
        if not documents and not state.graph_policy_results:
            return fallback_no_answer(state)

        graph_policies = select_graph_policies_for_question(
            state.question,
            state.graph_policy_results,
        )

        # 이전 대화가 있으면 질문에 컨텍스트를 주입한다 (멀티턴 지원)
        question_with_history = _augment_question_with_history(state.question, state.messages)

        if use_policy_context_generation or graph_policies:
            state.answer = generate_policy_answer(
                question_with_history,
                documents,
                graph_policies,
            )
        else:
            state.answer = generate_answer(question_with_history, documents)

        scores = _score_lookup(state)
        state.sources = [
            graph_policy_to_source(policy)
            for policy in graph_policies
        ] + [
            document_to_source(doc, scores.get(str(doc.metadata.get("chunk_id", "")), 0.0))
            for doc in documents
        ]
        state.status = "answered_from_pdf"
        state.needs_external_search = False
        state.messages = state.messages + [
            {"role": "user", "content": state.question},
            {"role": "assistant", "content": state.answer},
        ]
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
    retrieve_graph_policies: RetrieveGraphPolicies,
    grade_evidence: GradeEvidence | None,
    generate_answer: GenerateAnswer,
    generate_policy_answer: GeneratePolicyAnswer,
    settings: Settings,
    use_query_expansion: bool = True,
    use_policy_context_generation: bool = True,
):
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent_node(classify_intent))
    graph.add_node("general_answer", general_answer_node())
    graph.add_node("plan_queries", plan_queries_node(plan_search_queries))
    graph.add_node(
        "retrieve_pdf",
        retrieve_pdf_node(retrieve_documents, settings, use_query_expansion),
    )
    graph.add_node("retrieve_graph", retrieve_graph_node(retrieve_graph_policies))
    graph.add_node("resolve_policy_precheck", resolve_policy_node(final=False))
    graph.add_node("grade_evidence", grade_evidence_node(grade_evidence, settings))
    graph.add_node("resolve_policy", resolve_policy_node(final=True))
    graph.add_node("generate_blocked_answer", generate_policy_blocked_answer_node())
    graph.add_node(
        "generate_answer",
        generate_answer_node(
            generate_answer,
            generate_policy_answer,
            use_policy_context_generation,
        ),
    )
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
    graph.add_edge("retrieve_pdf", "retrieve_graph")
    graph.add_edge("retrieve_graph", "resolve_policy_precheck")
    graph.add_conditional_edges(
        "resolve_policy_precheck",
        route_policy,
        {
            "blocked": "generate_blocked_answer",
            "insufficient": "grade_evidence",
            "answerable": "grade_evidence",
        },
    )
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


def resolve_chat_graph_dependencies(
    question: str,
    classify_intent: ClassifyIntent | None = None,
    plan_search_queries: PlanSearchQueries | None = None,
    retrieve_documents: RetrieveDocuments | None = None,
    retrieve_graph_policies: RetrieveGraphPolicies | None = None,
    grade_evidence: GradeEvidence | None = None,
    generate_answer: GenerateAnswer | None = None,
    generate_policy_answer: GeneratePolicyAnswer | None = None,
    settings: Settings | None = None,
) -> ChatGraphDependencies:
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
    if retrieve_graph_policies is not None:
        resolved_retrieve_graph_policies = retrieve_graph_policies
    else:
        resolved_retrieve_graph_policies = lambda query: []
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
    resolved_generate_policy_answer = generate_policy_answer or (
        lambda query, documents, graph_policies: generate_policy_context_answer(
            query,
            documents,
            graph_policies,
            resolved_settings,
        )
    )
    use_policy_context_generation = generate_policy_answer is not None or (
        retrieve_documents is None and resolved_settings.app_env != "test"
    )

    return ChatGraphDependencies(
        classify_intent=resolved_classify_intent,
        plan_search_queries=resolved_plan_search_queries,
        retrieve_documents=resolved_retrieve_documents,
        retrieve_graph_policies=resolved_retrieve_graph_policies,
        grade_evidence=resolved_grade_evidence,
        generate_answer=resolved_generate_answer,
        generate_policy_answer=resolved_generate_policy_answer,
        settings=resolved_settings,
        use_query_expansion=not (retrieve_documents is not None and plan_search_queries is None),
        use_policy_context_generation=use_policy_context_generation,
    )


def run_chat_graph(
    question: str,
    classify_intent: ClassifyIntent | None = None,
    plan_search_queries: PlanSearchQueries | None = None,
    retrieve_documents: RetrieveDocuments | None = None,
    retrieve_graph_policies: RetrieveGraphPolicies | None = None,
    grade_evidence: GradeEvidence | None = None,
    generate_answer: GenerateAnswer | None = None,
    generate_policy_answer: GeneratePolicyAnswer | None = None,
    settings: Settings | None = None,
    thread_id: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> ChatResponse:
    dependencies = resolve_chat_graph_dependencies(
        question=question,
        classify_intent=classify_intent,
        plan_search_queries=plan_search_queries,
        retrieve_documents=retrieve_documents,
        retrieve_graph_policies=retrieve_graph_policies,
        grade_evidence=grade_evidence,
        generate_answer=generate_answer,
        generate_policy_answer=generate_policy_answer,
        settings=settings,
    )
    graph = build_chat_graph(
        classify_intent=dependencies.classify_intent,
        plan_search_queries=dependencies.plan_search_queries,
        retrieve_documents=dependencies.retrieve_documents,
        retrieve_graph_policies=dependencies.retrieve_graph_policies,
        grade_evidence=dependencies.grade_evidence,
        generate_answer=dependencies.generate_answer,
        generate_policy_answer=dependencies.generate_policy_answer,
        settings=dependencies.settings,
        use_query_expansion=dependencies.use_query_expansion,
        use_policy_context_generation=dependencies.use_policy_context_generation,
    )
    # thread_id는 요청 상관관계 식별자로만 받고 서버 상태 저장에는 사용하지 않는다.
    _ = thread_id
    recent_history = list(history or [])[-4:]
    result = graph.invoke(GraphState(question=question, messages=recent_history))
    state = GraphState.model_validate(result)
    return ChatResponse(
        answer=state.answer,
        sources=state.sources,
        status=state.status,
        needs_external_search=state.needs_external_search,
        intent=state.intent,
    )


def prepare_chat_stream(
    question: str,
    classify_intent: ClassifyIntent | None = None,
    plan_search_queries: PlanSearchQueries | None = None,
    retrieve_documents: RetrieveDocuments | None = None,
    retrieve_graph_policies: RetrieveGraphPolicies | None = None,
    grade_evidence: GradeEvidence | None = None,
    generate_answer: GenerateAnswer | None = None,
    generate_policy_answer: GeneratePolicyAnswer | None = None,
    settings: Settings | None = None,
    thread_id: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> PreparedChatStream:
    dependencies = resolve_chat_graph_dependencies(
        question=question,
        classify_intent=classify_intent,
        plan_search_queries=plan_search_queries,
        retrieve_documents=retrieve_documents,
        retrieve_graph_policies=retrieve_graph_policies,
        grade_evidence=grade_evidence,
        generate_answer=generate_answer,
        generate_policy_answer=generate_policy_answer,
        settings=settings,
    )
    captured: dict[str, object] = {}

    def capture_pdf_answer(query: str, documents: list[Document]) -> str:
        captured["kind"] = "pdf"
        captured["question"] = query
        captured["documents"] = documents
        return ""

    def capture_policy_answer(
        query: str,
        documents: list[Document],
        graph_policies: list[GraphPolicyResult],
    ) -> str:
        captured["kind"] = "policy"
        captured["question"] = query
        captured["documents"] = documents
        captured["graph_policies"] = graph_policies
        return ""

    graph = build_chat_graph(
        classify_intent=dependencies.classify_intent,
        plan_search_queries=dependencies.plan_search_queries,
        retrieve_documents=dependencies.retrieve_documents,
        retrieve_graph_policies=dependencies.retrieve_graph_policies,
        grade_evidence=dependencies.grade_evidence,
        generate_answer=capture_pdf_answer,
        generate_policy_answer=capture_policy_answer,
        settings=dependencies.settings,
        use_query_expansion=dependencies.use_query_expansion,
        use_policy_context_generation=dependencies.use_policy_context_generation,
    )
    _ = thread_id
    recent_history = list(history or [])[-4:]
    result = graph.invoke(GraphState(question=question, messages=recent_history))
    state = GraphState.model_validate(result)

    if captured.get("kind") == "policy":
        chunks = stream_policy_context_answer(
            str(captured["question"]),
            captured["documents"],  # type: ignore[arg-type]
            captured["graph_policies"],  # type: ignore[arg-type]
            dependencies.settings,
        )
    elif captured.get("kind") == "pdf":
        chunks = stream_pdf_answer(
            str(captured["question"]),
            captured["documents"],  # type: ignore[arg-type]
            dependencies.settings,
        )
    else:
        chunks = iter([state.answer]) if state.answer else iter(())

    return PreparedChatStream(
        chunks=chunks,
        sources=state.sources,
        status=state.status,
        needs_external_search=state.needs_external_search,
        intent=state.intent,
    )


def retrieved_document_from_score(document: Document, score: float) -> RetrievedDocument:
    return RetrievedDocument(document=document, score=score)
