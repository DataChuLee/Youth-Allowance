from langchain_core.documents import Document

from app.rag.sources import document_to_source


def test_document_to_source_preserves_required_metadata() -> None:
    document = Document(
        page_content="카드 사용 관련 안내 문단입니다.",
        metadata={
            "title": "청년수당 참여자 안내책자",
            "page": 12,
            "chunk_id": "pdf-page-12-chunk-2",
        },
    )

    source = document_to_source(document)

    assert source.type == "pdf"
    assert source.title == "청년수당 참여자 안내책자"
    assert source.page == 12
    assert source.chunk_id == "pdf-page-12-chunk-2"
    assert source.excerpt == "카드 사용 관련 안내 문단입니다."
