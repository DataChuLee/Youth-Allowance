from langchain_core.documents import Document

from app.graph.state import RetrievedDocument
from app.rag.retrieval import expand_to_parent, retrieve_with_queries


def child(chunk_id: str, parent_id: str | None, score: float = 1.0) -> RetrievedDocument:
    metadata = {"chunk_id": chunk_id, "page": 12}
    if parent_id:
        metadata |= {"parent_id": parent_id, "parent_text": f"부모 {parent_id}"}
    return RetrievedDocument(document=Document(page_content=f"자식 {chunk_id}", metadata=metadata), score=score)


def test_chunk_without_parent_is_unchanged() -> None:
    item = child("sem_0001", None)
    assert expand_to_parent(item) is item


def test_siblings_collapse_to_one_parent_before_max_docs_cap() -> None:
    hits = [child("st_1", "p12_s1"), child("st_2", "p12_s1"), child("st_3", "p12_s1"), child("st_4", "p14_s1")]
    result = retrieve_with_queries("질문", ["질문"], retrieve_documents=lambda _: hits, max_docs=2, per_query_k=4)
    assert [item.document.metadata["chunk_id"] for item in result] == ["p12_s1", "p14_s1"]
    assert result[0].document.page_content == "부모 p12_s1"
    assert "parent_text" not in result[0].document.metadata
