from langchain_core.documents import Document

from benchmarks.build_table_split_index import split_table_blocks


def test_split_table_blocks_moves_tables_into_chunks_with_preceding_context() -> None:
    chunk = Document(
        page_content=(
            "사용 불가 항목 클린카드 사용 불가 업종 48개 | 1 | 특별호텔 |\n"
            "| --- | --- |\n"
            "| 33 | 카지노 |\n"
            " # 자산 운영 지출 불가"
        ),
        metadata={"chunk_id": "sem_0032", "page": 12},
    )

    prose, table = split_table_blocks(chunk)

    assert prose.metadata["chunk_id"] == "sem_0032"
    assert "카지노" not in prose.page_content and "자산 운영" in prose.page_content
    assert table.metadata == {"chunk_id": "sem_0032_t1", "page": 12}
    assert table.page_content == (
        "사용 불가 항목 클린카드 사용 불가 업종 48개\n| 1 | 특별호텔 |\n| --- | --- |\n| 33 | 카지노 |"
    )


def test_split_table_blocks_keeps_chunks_without_tables() -> None:
    chunk = Document(page_content="표 없는 문단 | 세로선만 있음", metadata={"chunk_id": "sem_0001"})

    assert split_table_blocks(chunk) == [chunk]


def test_split_table_blocks_uses_only_text_after_previous_table_as_context() -> None:
    chunk = Document(
        page_content="사용 예 | 주거비 | 월세 |\n| --- | --- |\n 사용 불가 업종 | 1 | 카지노 |\n| --- | --- |",
        metadata={"chunk_id": "sem_0032"},
    )

    _, first, second = split_table_blocks(chunk)

    assert first.page_content.startswith("사용 예\n")
    assert second.page_content.startswith("사용 불가 업종\n")
