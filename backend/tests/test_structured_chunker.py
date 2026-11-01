from langchain_core.documents import Document

from app.indexing.structured_chunker import chunk_pages, clean_line, numbered_items, table_pieces

PAGE_12 = """청년수당
사용법 # 2.카드
안내 # 사용 가능 항목 # 청년수당은 주거비, 생활비 등에서 자유롭게 사용 가능합니다. 사용 예 | 생활비 | 식비, 통신비(소액결제 제외) |
| --- | --- |
| 기타 | 문화 관람비 |
 사용 불가 항목 클린카드 사용 불가 업종 | 1 | 특별호텔 | 3 | 귀금속, 금, 은, 보석 |
| --- | --- | --- | --- |
| 2 | 1급호텔 |  |  |
 # 자산 운영·축적 및 투자를 위한 지출 예금, 적금, 국민연금 납부 12 SEOUL
M! 100S
서울광역
청년센터"""


def chunks_of(text: str, page: int = 12) -> list[str]:
    return [chunk.page_content for chunk in chunk_pages([Document(page_content=text, metadata={"page": page})])]


def test_logo_footer_lines_are_removed_but_content_numbers_stay() -> None:
    assert clean_line("서울광역") == "" and clean_line("청년센터 13") == ""
    assert clean_line("투자와 관련 있는 지출 등 12 SEOUL") == "투자와 관련 있는 지출 등"
    assert clean_line("주 30") == "주 30"


def test_allowed_items_and_blocked_merchants_land_in_separate_chunks() -> None:
    chunks = chunks_of(PAGE_12)
    telecom = next(chunk for chunk in chunks if "통신비(소액결제 제외)" in chunk)
    jewelry = next(chunk for chunk in chunks if "귀금속, 금, 은, 보석" in chunk)
    assert telecom != jewelry
    assert "특별호텔" not in jewelry and "국민연금" not in telecom
    # 업종 항목에는 무엇의 목록인지 알려 주는 표 앞 문장이 붙는다.
    assert "사용 불가 업종" in jewelry
    assert all(chunk.startswith("[12쪽 청년수당 사용법 2.카드]") for chunk in chunks)
    assert not any("SEOUL" in chunk for chunk in chunks)


def test_numbered_multi_column_list_is_unpacked_in_number_order() -> None:
    rows = ["| 1 | 특별호텔 | 3 | 귀금속 |", "| 2 | 1급호텔 |  |  |"]
    assert numbered_items(rows) == ["특별호텔", "1급호텔", "귀금속"]
    assert numbered_items(["| 주거비 | 월세 | 생활비 | 식비 |"]) is None


def test_faq_answer_row_stays_with_its_question() -> None:
    pieces = table_pieces(["| Q5 | 개인 돈 입금 돼요? |", "| A5 | 가능합니다. |", "| Q6 | 이월 돼요? |"])
    assert pieces == ["| Q5 | 개인 돈 입금 돼요? |\n| A5 | 가능합니다. |", "| Q6 | 이월 돼요? |"]


def test_calendar_table_is_kept_whole() -> None:
    row = "| 1 | 2 | 3 | 4 | 5 | 6 | 7 |"
    assert table_pieces([row, row]) == [f"{row}\n{row}"]


def test_chunking_is_deterministic() -> None:
    assert chunks_of(PAGE_12) == chunks_of(PAGE_12)


def test_children_carry_parent_section_without_search_context() -> None:
    chunks = chunk_pages([Document(page_content=PAGE_12, metadata={"page": 12})])
    telecom = next(chunk for chunk in chunks if "통신비(소액결제 제외)" in chunk.page_content)
    hotel = next(chunk for chunk in chunks if chunk.page_content.endswith("특별호텔"))
    jewelry = next(chunk for chunk in chunks if chunk.page_content.endswith("귀금속, 금, 은, 보석"))
    # 업종 항목은 같은 부모(업종 목록 전체)를 공유하고, 사용 가능 표와는 다른 부모다.
    assert hotel.metadata["parent_id"] == jewelry.metadata["parent_id"] != telecom.metadata["parent_id"]
    assert "특별호텔, 1급호텔, 귀금속, 금, 은, 보석" in jewelry.metadata["parent_text"]
    assert "| 기타 | 문화 관람비 |" in telecom.metadata["parent_text"]
