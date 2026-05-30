from collections.abc import Callable

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever

from app.core.config import Settings, get_settings
from app.core.errors import IndexMissingError
from app.graph.state import RetrievedDocument
from app.rag.policy import (
    is_food_delivery_question,
    is_study_device_question,
    score_food_delivery_document,
    score_study_device_document,
)
from app.rag.vector_store import create_vector_store

BM25_WEIGHT = 0.6
VECTOR_WEIGHT = 0.4

RetrieveDocuments = Callable[[str], list[RetrievedDocument]]


def retrieved_document_from_rank(document: Document, rank: int) -> RetrievedDocument:
    return RetrievedDocument(document=document, score=max(0.0, 1.0 - (rank * 0.01)))


def clone_with_retrieved_query(
    retrieved_document: RetrievedDocument,
    query: str,
) -> RetrievedDocument:
    document = retrieved_document.document
    return RetrievedDocument(
        document=Document(
            page_content=document.page_content,
            metadata={**document.metadata, "retrieved_by_query": query},
        ),
        score=retrieved_document.score,
    )


def normalize_retrieved_documents(
    documents: list[RetrievedDocument] | list[Document],
) -> list[RetrievedDocument]:
    normalized: list[RetrievedDocument] = []
    for rank, item in enumerate(documents):
        if isinstance(item, RetrievedDocument):
            normalized.append(item)
        else:
            normalized.append(retrieved_document_from_rank(item, rank))
    return normalized


def retrieve_with_queries(
    question: str,
    search_queries: list[str],
    retrieve_documents: RetrieveDocuments,
    max_docs: int = 8,
    per_query_k: int = 5,
) -> list[RetrievedDocument]:
    documents_by_query: list[list[RetrievedDocument]] = []
    for query in search_queries:
        retrieved_documents = normalize_retrieved_documents(retrieve_documents(query))
        documents_by_query.append(
            [
                clone_with_retrieved_query(item, query)
                for item in retrieved_documents[:per_query_k]
            ]
        )

    interleaved: list[RetrievedDocument] = []
    max_len = max((len(documents) for documents in documents_by_query), default=0)
    for rank in range(max_len):
        for documents in documents_by_query:
            if rank < len(documents):
                interleaved.append(documents[rank])

    deduped: list[RetrievedDocument] = []
    seen: set[str] = set()
    for item in interleaved:
        key = str(item.document.metadata.get("chunk_id") or item.document.page_content)
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)

    if is_food_delivery_question(question):
        deduped.sort(
            key=lambda item: score_food_delivery_document(item.document),
            reverse=True,
        )
    elif is_study_device_question(question):
        deduped.sort(
            key=lambda item: score_study_device_document(item.document),
            reverse=True,
        )

    return deduped[:max_docs]


def load_documents_from_vector_store(vector_store: Chroma) -> list[Document]:
    raw_collection = vector_store.get(include=["documents", "metadatas"])
    contents = raw_collection.get("documents") or []
    metadatas = raw_collection.get("metadatas") or []
    documents = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(contents, metadatas, strict=False)
        if content
    ]
    if not documents:
        raise IndexMissingError("Chroma 저장소에 BM25 retriever를 만들 문서가 없습니다.")
    return documents


def create_ensemble_retriever(settings: Settings | None = None) -> BaseRetriever:
    resolved_settings = settings or get_settings()
    vector_store = create_vector_store(resolved_settings)
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": resolved_settings.retrieval_top_k},
    )
    bm25_retriever = BM25Retriever.from_documents(load_documents_from_vector_store(vector_store))
    bm25_retriever.k = resolved_settings.retrieval_top_k

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[BM25_WEIGHT, VECTOR_WEIGHT],
    )


def retrieve_pdf_documents_with_ensemble(
    question: str,
    settings: Settings | None = None,
) -> list[RetrievedDocument]:
    resolved_settings = settings or get_settings()
    retriever = create_ensemble_retriever(resolved_settings)
    documents = retriever.invoke(question)[: resolved_settings.retrieval_top_k]
    return normalize_retrieved_documents(documents)
