from langchain_core.documents import Document

from app.graph.state import EvidenceDecision, GraphState, RetrievedDocument
from app.rag.sources import document_to_source

DEFAULT_MIN_SIMILARITY_SCORE = 0.2


def assess_evidence(
    state: GraphState,
    min_similarity_score: float = DEFAULT_MIN_SIMILARITY_SCORE,
) -> GraphState:
    if not state.retrieved_documents:
        state.evidence = EvidenceDecision(
            is_sufficient=False,
            reason="검색된 PDF 청크가 없습니다.",
            source_chunk_ids=[],
        )
        state.status = "insufficient_pdf_evidence"
        state.needs_external_search = True
        return state

    best = max(state.retrieved_documents, key=lambda item: item.score)
    if best.score < min_similarity_score:
        state.evidence = EvidenceDecision(
            is_sufficient=False,
            reason="검색 점수가 최소 기준보다 낮습니다.",
            source_chunk_ids=[],
        )
        state.status = "insufficient_pdf_evidence"
        state.needs_external_search = True
        return state

    chunk_id = str(best.document.metadata["chunk_id"])
    state.evidence = EvidenceDecision(
        is_sufficient=True,
        reason="상위 검색 청크가 최소 검색 점수 기준을 통과했습니다.",
        source_chunk_ids=[chunk_id],
    )
    state.status = "answered_from_pdf"
    state.needs_external_search = False
    return state


def generate_answer_from_documents(state: GraphState) -> GraphState:
    allowed = set(state.evidence.source_chunk_ids)
    documents = [
        item.document for item in state.retrieved_documents
        if str(item.document.metadata.get("chunk_id")) in allowed
    ]
    context = "\n\n".join(document.page_content for document in documents)
    state.answer = f"안내책자 기준으로 확인한 내용입니다.\n\n{context}"
    state.sources = [document_to_source(document) for document in documents]
    state.status = "answered_from_pdf"
    state.needs_external_search = False
    return state


def fallback_no_answer(state: GraphState) -> GraphState:
    state.answer = "안내책자에서 해당 내용을 확인하지 못했습니다. 최신 공식 안내 확인이 필요할 수 있습니다."
    state.sources = []
    state.status = "insufficient_pdf_evidence"
    state.needs_external_search = True
    return state


def retrieved_document_from_score(document: Document, score: float) -> RetrievedDocument:
    return RetrievedDocument(document=document, score=score)
