import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import PROJECT_ROOT, get_settings
from app.core.errors import IndexingError
from app.indexing.quality import (
    IndexingStats,
    format_indexing_stats,
    validate_indexing_stats,
)
from app.rag.retrieval import resolve_trusted_faiss_index_dir

load_dotenv()

BOOKLET_TITLE = "청년수당 참여자 안내책자"
# 기준선(baseline) 청킹 설정. 바꾸면 manifest.json으로 실험 결과와 짝지을 수 있다.
CHUNKER_CONFIG = {
    "type": "semantic",
    "scope": "page",
    "breakpoint_threshold_type": "percentile",
    "breakpoint_threshold_amount": 85,
}
MANIFEST_FILENAME = "manifest.json"


def build_chunk_id(index: int) -> str:
    return f"sem_{index:04d}"


def load_ocr_pages(jsonl_path: Path) -> list[Document]:
    """Upstage OCR JSONL(한 줄에 한 페이지)을 페이지 문서로 읽는다."""
    if not jsonl_path.is_file():
        raise IndexingError(f"OCR JSONL 파일을 찾을 수 없습니다: {jsonl_path}")

    pages: list[Document] = []
    with jsonl_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            page = int(row.get("metadata", {}).get("page", len(pages) + 1))
            pages.append(Document(page_content=row["page_content"], metadata={"page": page}))
    return pages


def assign_chunk_metadata(chunks: list[Document], source: str) -> list[Document]:
    # 빈 청크를 먼저 버린 뒤 번호를 매겨야 sem_ ID가 문서 순서대로 연속된다.
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    for index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {"chunk_id": build_chunk_id(index), "title": BOOKLET_TITLE, "source": source}
        )
    return chunks


def split_pages_semantic(pages: list[Document], embeddings, source: str) -> list[Document]:
    # 페이지 안에서 의미가 크게 바뀌는 지점(문장 간 임베딩 거리 상위 15%)에서 자른다.
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=CHUNKER_CONFIG["breakpoint_threshold_type"],
        breakpoint_threshold_amount=CHUNKER_CONFIG["breakpoint_threshold_amount"],
    )
    chunks: list[Document] = []
    for page in pages:
        chunks.extend(
            splitter.create_documents([page.page_content], metadatas=[dict(page.metadata)])
        )
    return assign_chunk_metadata(chunks, source)


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def current_git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    chunks: list[Document], source_sha256: str, embedding_model: str
) -> dict:
    """실험 결과를 어떤 인덱스로 측정했는지 추적하기 위한 메타데이터. 원문은 넣지 않는다."""
    chunk_digest = sha256_text(
        "\n".join(f"{c.metadata['chunk_id']}\t{c.page_content}" for c in chunks)
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "source_sha256": source_sha256,
        "embedding_model": embedding_model,
        "chunker": CHUNKER_CONFIG,
        "chunk_count": len(chunks),
        "chunks_sha256": chunk_digest,
    }


def index_pdf() -> IndexingStats:
    settings = get_settings()
    # 상대 경로는 실행 위치와 무관하게 프로젝트 루트 기준으로 해석한다(서비스 로드 규칙과 동일).
    jsonl_path = PROJECT_ROOT / settings.ocr_jsonl_path
    index_dir = resolve_trusted_faiss_index_dir(settings.faiss_index_dir)
    pages = load_ocr_pages(jsonl_path)
    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )
    chunks = split_pages_semantic(pages, embeddings, jsonl_path.name)
    stats = collect_stats(pages, chunks)
    validate_indexing_stats(stats)

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
        ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks],
    )
    vector_store.save_local(str(index_dir))
    manifest = build_manifest(
        chunks,
        hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
        settings.openai_embedding_model,
    )
    (index_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> None:
    stats = index_pdf()
    print(format_indexing_stats(stats))


if __name__ == "__main__":
    main()
