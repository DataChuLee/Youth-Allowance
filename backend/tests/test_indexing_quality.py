from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.core.errors import IndexingError
from app.indexing import index_pdf as indexing_module
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


def test_load_pdf_pages_uses_ocr_when_pypdf_extracts_no_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "booklet.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    captured = {}

    class FakeLoader:
        def __init__(self, path: str):
            captured["loader_path"] = path

        def load(self) -> list[Document]:
            return [
                Document(page_content="", metadata={"page": 0, "source": str(pdf_path)}),
                Document(page_content="   ", metadata={"page": 1, "source": str(pdf_path)}),
            ]

    def fake_ocr(path: Path) -> list[Document]:
        captured["ocr_path"] = path
        return [
            Document(
                page_content="OCR로 추출한 청년수당 안내 텍스트",
                metadata={"page": 0, "source": str(path)},
            )
        ]

    monkeypatch.setattr(indexing_module, "PyPDFLoader", FakeLoader)
    monkeypatch.setattr(indexing_module, "load_pdf_pages_with_ocr", fake_ocr)

    pages = indexing_module.load_pdf_pages(pdf_path)

    assert captured["loader_path"] == str(pdf_path)
    assert captured["ocr_path"] == pdf_path
    assert pages[0].page_content == "OCR로 추출한 청년수당 안내 텍스트"


def test_index_pdf_uses_deterministic_chroma_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = {}
    settings = SimpleNamespace(
        pdf_path=tmp_path / "booklet.pdf",
        chroma_dir=tmp_path / "chroma",
        openai_api_key="test-key",
        openai_embedding_model="test-embedding-model",
    )
    monkeypatch.setattr(indexing_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        indexing_module,
        "load_pdf_pages",
        lambda _: [
            Document(
                page_content="충분한 텍스트" * 100,
                metadata={"page": 0, "source": "booklet.pdf"},
            )
        ],
    )

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured["embedding_kwargs"] = kwargs

    class FakeChroma:
        @staticmethod
        def from_documents(**kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(indexing_module, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(indexing_module, "Chroma", FakeChroma)

    indexing_module.index_pdf()

    assert captured["ids"]
    assert captured["ids"] == [
        document.metadata["chunk_id"] for document in captured["documents"]
    ]
