import pytest

from app.core.errors import IndexingError
from app.indexing.quality import IndexingStats, validate_indexing_stats


def test_validate_indexing_stats_accepts_text_rich_pdf() -> None:
    stats = IndexingStats(
        total_pages=10,
        extracted_pages=10,
        empty_pages=0,
        total_characters=25_000,
        chunk_count=80,
        sample_chunks=["page=1 chunk=pdf-page-1-chunk-0 text=hello"],
    )

    validate_indexing_stats(stats)


def test_validate_indexing_stats_rejects_low_text_pdf() -> None:
    stats = IndexingStats(
        total_pages=20,
        extracted_pages=1,
        empty_pages=19,
        total_characters=50,
        chunk_count=1,
        sample_chunks=[],
    )

    with pytest.raises(IndexingError, match="텍스트 추출 품질이 낮습니다"):
        validate_indexing_stats(stats)
