from dataclasses import dataclass

from app.core.errors import IndexingError


@dataclass(frozen=True)
class IndexingStats:
    total_pages: int
    extracted_pages: int
    empty_pages: int
    total_characters: int
    chunk_count: int
    sample_chunks: list[str]


def validate_indexing_stats(stats: IndexingStats) -> None:
    if stats.total_pages <= 0:
        raise IndexingError("PDF 페이지를 찾을 수 없습니다.")
    if stats.chunk_count <= 0:
        raise IndexingError("검색 청크가 생성되지 않았습니다.")

    empty_ratio = stats.empty_pages / stats.total_pages
    if stats.total_characters < 500 or empty_ratio > 0.5:
        raise IndexingError(
            "텍스트 추출 품질이 낮습니다. 이미지 기반 PDF일 수 있으므로 OCR 전처리가 필요합니다."
        )


def format_indexing_stats(stats: IndexingStats) -> str:
    samples = "\n".join(f"- {sample}" for sample in stats.sample_chunks)
    return (
        "Indexing quality report\n"
        f"total_pages={stats.total_pages}\n"
        f"extracted_pages={stats.extracted_pages}\n"
        f"empty_pages={stats.empty_pages}\n"
        f"total_characters={stats.total_characters}\n"
        f"chunk_count={stats.chunk_count}\n"
        f"sample_chunks:\n{samples}"
    )
