import json
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
        sample_chunks=["page=1 chunk_id=sem_0000"],
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


def test_collect_stats_does_not_include_source_content() -> None:
    sensitive_text = "주민등록번호와 계좌번호가 포함될 수 있는 원문"
    page = Document(page_content=sensitive_text, metadata={"page": 1})
    chunk = Document(
        page_content=sensitive_text,
        metadata={"page": 1, "chunk_id": "sem_0000"},
    )

    stats = indexing_module.collect_stats([page], [chunk])

    assert stats.sample_chunks == ["page=1 chunk_id=sem_0000"]
    assert sensitive_text not in "\n".join(stats.sample_chunks)


def write_ocr_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n\n",
        encoding="utf-8",
    )
    return path


def test_load_ocr_pages_reads_page_numbers_from_metadata(tmp_path: Path) -> None:
    jsonl_path = write_ocr_jsonl(
        tmp_path / "ocr.jsonl",
        [
            {"page_content": "1쪽 내용", "metadata": {"page": 1, "coordinates": []}},
            {"page_content": "2쪽 내용", "metadata": {"page": 2}},
        ],
    )

    pages = indexing_module.load_ocr_pages(jsonl_path)

    assert [page.metadata for page in pages] == [{"page": 1}, {"page": 2}]
    assert pages[1].page_content == "2쪽 내용"


def test_load_ocr_pages_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IndexingError, match="OCR JSONL"):
        indexing_module.load_ocr_pages(tmp_path / "missing.jsonl")


def test_assign_chunk_metadata_drops_empty_chunks_before_numbering() -> None:
    chunks = [
        Document(page_content="첫 청크", metadata={"page": 1}),
        Document(page_content="  ", metadata={"page": 1}),
        Document(page_content="둘째 청크", metadata={"page": 2}),
    ]

    result = indexing_module.assign_chunk_metadata(chunks, "ocr.jsonl")

    assert [chunk.metadata["chunk_id"] for chunk in result] == ["sem_0000", "sem_0001"]
    assert [chunk.metadata["page"] for chunk in result] == [1, 2]
    assert result[0].metadata["title"] == "청년수당 참여자 안내책자"


def test_index_pdf_saves_faiss_with_chunk_ids_and_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jsonl_path = write_ocr_jsonl(
        tmp_path / "ocr.jsonl",
        [
            {"page_content": "충분한 텍스트" * 50, "metadata": {"page": 1}},
            {"page_content": "청년수당 안내" * 50, "metadata": {"page": 2}},
        ],
    )
    index_dir = tmp_path / "faiss"
    index_dir.mkdir()
    settings = SimpleNamespace(
        ocr_jsonl_path=jsonl_path,
        faiss_index_dir=index_dir,
        openai_api_key="test-key",
        openai_embedding_model="test-embedding-model",
    )
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured["embedding_kwargs"] = kwargs

    class FakeChunker:
        def __init__(self, _embeddings, **kwargs):
            captured["chunker_kwargs"] = kwargs

        def create_documents(self, texts, metadatas):
            return [Document(page_content=texts[0], metadata=metadatas[0])]

    class FakeFAISS:
        @classmethod
        def from_documents(cls, documents, _embeddings, ids):
            captured["documents"] = documents
            captured["ids"] = ids
            return cls()

        def save_local(self, path: str) -> None:
            captured["saved_to"] = path

    monkeypatch.setattr(indexing_module, "get_settings", lambda: settings)
    monkeypatch.setattr("app.rag.retrieval.TRUSTED_FAISS_INDEX_ROOT", tmp_path)
    monkeypatch.setattr(indexing_module, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(indexing_module, "SemanticChunker", FakeChunker)
    monkeypatch.setattr(indexing_module, "FAISS", FakeFAISS)

    indexing_module.index_pdf()

    assert captured["ids"] == ["sem_0000", "sem_0001"]
    assert captured["ids"] == [doc.metadata["chunk_id"] for doc in captured["documents"]]
    assert captured["chunker_kwargs"] == {
        "breakpoint_threshold_type": "percentile",
        "breakpoint_threshold_amount": 85,
    }
    assert captured["saved_to"] == str(index_dir.resolve())
    manifest_text = (index_dir / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["chunk_count"] == 2
    assert manifest["embedding_model"] == "test-embedding-model"
    assert manifest["chunker"]["breakpoint_threshold_amount"] == 85
    assert "충분한 텍스트" not in manifest_text
