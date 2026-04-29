from langchain_core.documents import Document

from app.api.schemas import Source

MAX_EXCERPT_LENGTH = 280


def compact_text(text: str) -> str:
    return " ".join(text.split())


def document_to_source(document: Document) -> Source:
    text = compact_text(document.page_content)
    excerpt = text[:MAX_EXCERPT_LENGTH]
    return Source(
        type="pdf",
        title=str(document.metadata.get("title", "청년수당 참여자 안내책자")),
        page=int(document.metadata["page"]),
        excerpt=excerpt,
        chunk_id=str(document.metadata["chunk_id"]),
    )
