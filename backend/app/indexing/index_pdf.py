from pathlib import Path
import sys
import base64
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import Settings, get_settings
from app.core.errors import IndexingError
from app.indexing.quality import (
    IndexingStats,
    format_indexing_stats,
    validate_indexing_stats,
)

load_dotenv()

BOOKLET_TITLE = "청년수당 참여자 안내책자"
OCR_MODEL = "gpt-4o-mini"


def build_chunk_id(page: int, index: int) -> str:
    return f"pdf-page-{page}-chunk-{index}"


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    if not pdf_path.exists():
        raise IndexingError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    if pages and any(page.page_content.strip() for page in pages):
        return pages
    return load_pdf_pages_with_ocr(pdf_path)


def load_pdf_pages_with_ocr(pdf_path: Path, settings: Settings | None = None) -> list[Document]:
    settings = settings or get_settings()
    try:
        import fitz
    except ImportError as exc:
        raise IndexingError(
            "PDF 텍스트를 추출하지 못했습니다. 이미지 기반 PDF OCR을 위해 pymupdf 설치가 필요합니다."
        ) from exc

    client = OpenAI(api_key=settings.openai_api_key)
    pdf = fitz.open(str(pdf_path))
    pages: list[Document] = []

    try:
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image_bytes = pixmap.tobytes("png")
            text = extract_text_from_page_image(client, image_bytes)
            pages.append(
                Document(
                    page_content=text,
                    metadata={"page": page_index, "source": str(pdf_path)},
                )
            )
    finally:
        pdf.close()

    return pages


def extract_text_from_page_image(client: OpenAI, image_bytes: bytes) -> str:
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    response = client.chat.completions.create(
        model=OCR_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "이미지에 보이는 한국어 문서를 OCR로 전사하세요. "
                    "문서에 없는 내용을 추가하지 말고, 읽을 수 없는 부분은 생략하세요."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "이 페이지의 모든 읽을 수 있는 텍스트를 줄바꿈을 유지해 전사하세요.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )
    return response.choices[0].message.content or ""


def split_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks: list[Document] = []
    per_page_counts: dict[int, int] = {}

    for page in pages:
        page_number = int(page.metadata.get("page", 0)) + 1
        page_chunks = splitter.split_documents([page])

        for page_chunk in page_chunks:
            index = per_page_counts.get(page_number, 0)
            per_page_counts[page_number] = index + 1
            chunk_id = build_chunk_id(page_number, index)
            page_chunk.metadata.update(
                {
                    "source": str(page.metadata.get("source", "")),
                    "title": BOOKLET_TITLE,
                    "page": page_number,
                    "chunk_id": chunk_id,
                }
            )
            chunks.append(page_chunk)

    return chunks


def collect_stats(pages: list[Document], chunks: list[Document]) -> IndexingStats:
    total_pages = len(pages)
    empty_pages = sum(1 for page in pages if not page.page_content.strip())
    total_characters = sum(len(page.page_content.strip()) for page in pages)
    sample_chunks = [
        f"page={chunk.metadata['page']} chunk_id={chunk.metadata['chunk_id']}"
        for chunk in chunks[:3]
    ]
    return IndexingStats(
        total_pages=total_pages,
        extracted_pages=total_pages - empty_pages,
        empty_pages=empty_pages,
        total_characters=total_characters,
        chunk_count=len(chunks),
        sample_chunks=sample_chunks,
    )


def index_pdf() -> IndexingStats:
    settings = get_settings()
    pages = load_pdf_pages(settings.pdf_path)
    chunks = split_pages(pages)
    stats = collect_stats(pages, chunks)
    validate_indexing_stats(stats)

    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )
    Chroma.from_documents(
        documents=chunks,
        ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks],
        embedding=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_name="youth_allowance_booklet",
    )
    return stats


def main() -> None:
    stats = index_pdf()
    print(format_indexing_stats(stats))


if __name__ == "__main__":
    main()
