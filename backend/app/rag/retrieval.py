from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.core.config import BACKEND_DIR, PROJECT_ROOT, Settings, get_settings
from app.core.errors import IndexMissingError
from app.graph.state import RetrievedDocument
from app.rag.policy import (
    is_food_delivery_question,
    is_study_device_question,
    score_food_delivery_document,
    score_study_device_document,
)
from app.rag.query_preprocessing import (
    classify_retrieval_intent,
    expand_synonyms,
    normalize_for_retrieval,
)

DEFAULT_RETRIEVAL_CANDIDATE_K = 20
DEFAULT_RETRIEVAL_RRF_K = 60
TRUSTED_FAISS_INDEX_ROOT = BACKEND_DIR.resolve()
FAISS_INDEX_FILENAMES = ("index.faiss", "index.pkl")

RetrieveDocuments = Callable[[str], list[RetrievedDocument]]


class RetrievalUnavailableError(RuntimeError):
    """모든 PDF 검색 backend를 사용할 수 없을 때 발생한다."""


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


def document_key(document: Document, fallback: str) -> str:
    return str(document.metadata.get("chunk_id") or document.page_content or fallback)


def rrf_merge_documents(
    bm25_docs: list[Document],
    vector_docs: list[Document],
    *,
    alpha: float = 0.5,
    rrf_k: int = DEFAULT_RETRIEVAL_RRF_K,
    top_n: int = DEFAULT_RETRIEVAL_CANDIDATE_K,
) -> list[Document]:
    scores: dict[str, float] = {}
    documents_by_key: dict[str, Document] = {}

    for rank, document in enumerate(bm25_docs):
        key = document_key(document, f"bm25-{rank}")
        scores[key] = scores.get(key, 0.0) + alpha * (1 / (rrf_k + rank + 1))
        documents_by_key.setdefault(key, document)

    for rank, document in enumerate(vector_docs):
        key = document_key(document, f"vector-{rank}")
        scores[key] = scores.get(key, 0.0) + (1 - alpha) * (1 / (rrf_k + rank + 1))
        documents_by_key.setdefault(key, document)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [documents_by_key[key] for key in ranked_keys[:top_n]]


def retrieve_bm25_expanded(
    question: str,
    bm25_retriever,
    *,
    candidate_k: int = DEFAULT_RETRIEVAL_CANDIDATE_K,
) -> list[Document]:
    bm25_retriever.k = candidate_k
    deduped: dict[str, Document] = {}
    for query in expand_synonyms(question):
        for rank, document in enumerate(bm25_retriever.invoke(query)):
            key = document_key(document, f"bm25-{query}-{rank}")
            deduped.setdefault(key, document)
    return list(deduped.values())


@dataclass(frozen=True)
class HybridRRFRetriever:
    bm25_retriever: object | None
    vector_retriever: object | None
    top_k: int
    candidate_k: int = DEFAULT_RETRIEVAL_CANDIDATE_K
    rrf_k: int = DEFAULT_RETRIEVAL_RRF_K

    def invoke(self, question: str) -> list[Document]:
        normalized = normalize_for_retrieval(question)
        intent = classify_retrieval_intent(normalized)
        bm25_docs: list[Document] = []
        vector_docs: list[Document] = []
        failures = 0
        available_backends = 0

        if self.bm25_retriever is not None:
            available_backends += 1
            try:
                bm25_docs = retrieve_bm25_expanded(
                    normalized,
                    self.bm25_retriever,
                    candidate_k=self.candidate_k,
                )
            except Exception:
                failures += 1

        if self.vector_retriever is not None:
            available_backends += 1
            try:
                vector_docs = self.vector_retriever.invoke(normalized)
            except Exception:
                failures += 1

        if available_backends == 0 or failures == available_backends:
            raise RetrievalUnavailableError("PDF 검색 backend를 사용할 수 없습니다.")

        return rrf_merge_documents(
            bm25_docs,
            vector_docs,
            alpha=intent.alpha,
            rrf_k=self.rrf_k,
            top_n=max(self.top_k, self.candidate_k),
        )[: self.top_k]


def retrieve_with_queries(
    question: str,
    search_queries: list[str],
    retrieve_documents: RetrieveDocuments,
    max_docs: int = 8,
    per_query_k: int = 5,
) -> list[RetrievedDocument]:
    documents_by_query: list[list[RetrievedDocument]] = []
    failed_queries = 0
    for query in search_queries:
        try:
            retrieved_documents = normalize_retrieved_documents(retrieve_documents(query))
        except RetrievalUnavailableError:
            failed_queries += 1
            continue
        documents_by_query.append(
            [
                clone_with_retrieved_query(item, query)
                for item in retrieved_documents[:per_query_k]
            ]
        )

    if failed_queries and not documents_by_query:
        raise RetrievalUnavailableError("모든 확장 질의의 PDF 검색이 실패했습니다.")

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


def _is_path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_trusted_faiss_index_dir(index_dir: Path) -> Path:
    """배포자가 관리하는 backend 루트 내부의 고정 인덱스만 허용한다."""
    candidate = Path(index_dir)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved_index_dir = candidate.resolve()
    trusted_root = TRUSTED_FAISS_INDEX_ROOT.resolve()
    if not _is_path_within(resolved_index_dir, trusted_root):
        raise RetrievalUnavailableError(
            "FAISS 인덱스 경로가 애플리케이션 관리 범위를 벗어났습니다."
        )
    return resolved_index_dir


def assert_faiss_index_exists(index_dir: Path) -> Path:
    resolved_index_dir = resolve_trusted_faiss_index_dir(index_dir)
    if not resolved_index_dir.exists():
        raise IndexMissingError("FAISS 인덱스가 없습니다. 먼저 FAISS indexing을 실행하세요.")
    if not resolved_index_dir.is_dir():
        raise RetrievalUnavailableError("FAISS 인덱스 경로가 디렉터리가 아닙니다.")

    for filename in FAISS_INDEX_FILENAMES:
        index_file = resolved_index_dir / filename
        if not index_file.exists():
            raise IndexMissingError("FAISS 인덱스가 없습니다. 먼저 FAISS indexing을 실행하세요.")
        # symlink가 신뢰 루트 밖 파일을 가리키는 경우도 역직렬화 전에 차단한다.
        if (
            index_file.is_symlink()
            or not index_file.is_file()
            or not _is_path_within(
                index_file.resolve(),
                TRUSTED_FAISS_INDEX_ROOT.resolve(),
            )
        ):
            raise RetrievalUnavailableError("FAISS 인덱스 파일이 신뢰 경계를 벗어났습니다.")
    return resolved_index_dir


def load_trusted_faiss_store(index_dir: Path, embeddings):
    """검증된 로컬 인덱스에 한해서만 LangChain의 pickle 기반 포맷을 연다."""
    trusted_index_dir = assert_faiss_index_exists(index_dir)
    return FAISS.load_local(
        str(trusted_index_dir),
        embeddings,
        # 현행 포맷은 docstore와 ID 매핑을 index.pkl로 저장한다.
        # 신뢰 경계를 통과한 애플리케이션 생성 인덱스에만 pickle 로드를 허용한다.
        allow_dangerous_deserialization=True,
    )


def load_documents_from_faiss_store(vector_store) -> list[Document]:
    docstore_items = getattr(vector_store.docstore, "_dict", {})
    documents = [
        document
        for document in docstore_items.values()
        if isinstance(document, Document) and document.page_content
    ]
    if not documents:
        raise IndexMissingError("FAISS 인덱스에 BM25 retriever를 만들 문서가 없습니다.")
    return documents


def create_ensemble_retriever(settings: Settings | None = None) -> HybridRRFRetriever:
    resolved_settings = settings or get_settings()
    faiss_index_dir = Path(resolved_settings.faiss_index_dir)
    vector_retriever = None
    bm25_documents: list[Document] = []
    faiss_exists = True
    trusted_faiss_index_dir: Path | None = None

    try:
        trusted_faiss_index_dir = assert_faiss_index_exists(faiss_index_dir)
    except IndexMissingError:
        faiss_exists = False
    except RetrievalUnavailableError:
        # 신뢰 경계를 벗어나거나 예상 파일이 아니면 위험한 역직렬화를 시도하지 않는다.
        vector_retriever = None

    if trusted_faiss_index_dir is not None:
        try:
            embeddings = OpenAIEmbeddings(
                api_key=resolved_settings.openai_api_key,
                model=resolved_settings.openai_embedding_model,
            )
            vector_store = load_trusted_faiss_store(
                trusted_faiss_index_dir,
                embeddings,
            )
            vector_retriever = vector_store.as_retriever(
                search_kwargs={"k": resolved_settings.retrieval_candidate_k},
            )
            try:
                bm25_documents = load_documents_from_faiss_store(vector_store)
            except IndexMissingError:
                bm25_documents = []
        except Exception:
            # 손상되거나 호환되지 않는 FAISS 파일은 사용하지 않는다.
            vector_retriever = None

    bm25_retriever = None
    if bm25_documents:
        bm25_retriever = BM25Retriever.from_documents(bm25_documents)
        bm25_retriever.k = resolved_settings.retrieval_candidate_k

    if vector_retriever is None and bm25_retriever is None:
        if not faiss_exists:
            raise IndexMissingError(
                "FAISS 인덱스가 없습니다. 먼저 python -m app.indexing.index_pdf를 실행하세요."
            )
        raise RetrievalUnavailableError("FAISS 인덱스를 읽을 수 없습니다.")

    return HybridRRFRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        top_k=resolved_settings.retrieval_top_k,
        candidate_k=resolved_settings.retrieval_candidate_k,
        rrf_k=resolved_settings.retrieval_rrf_k,
    )


def retrieve_pdf_documents_with_ensemble(
    question: str,
    settings: Settings | None = None,
) -> list[RetrievedDocument]:
    resolved_settings = settings or get_settings()
    retriever = create_ensemble_retriever(resolved_settings)
    documents = retriever.invoke(question)[: resolved_settings.retrieval_top_k]
    return normalize_retrieved_documents(documents)
