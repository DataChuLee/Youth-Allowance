"""OCR 페이지를 문서 구조(제목·표·FAQ) 기준으로 자르는 결정적 청커 (실험 8).

기준선 SemanticChunker의 문제(실험 8 진단):
- 로고·쪽 꼬리말("SOUL 청년센터")이 독립 청크가 되어 102개 중 38개가 100자 미만이었다.
- 12쪽 한 청크(1,361자)에 사용 가능 표, 사용 불가 업종 48개, 자산 축적 금지가 섞여 있었다.
- 문장 임베딩 거리로 자르므로 다시 돌리면 경계가 2~4개 달라졌다.

여기서는 임베딩 없이 규칙으로만 자른다. 같은 입력이면 항상 같은 청크가 나온다.
표 셀 안의 원문은 고치지 않는다. 평가 인용문(셀 안의 문구)이 그대로 남아야 청킹 간 비교가 된다.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

CHUNKER_CONFIG = {
    "type": "structured",
    "version": 2,
    "max_chars": 500,
    "min_text_chars": 10,
    "table_context_chars": 150,
    "parent_max_chars": 1200,
}
MAX_CHUNK_CHARS = CHUNKER_CONFIG["max_chars"]
# 이보다 짧은 본문 조각("6 월", "질문")은 쪽 제목이나 다음 표의 맥락으로 이미 들어가므로 버린다.
MIN_TEXT_CHARS = CHUNKER_CONFIG["min_text_chars"]
TABLE_CONTEXT_CHARS = CHUNKER_CONFIG["table_context_chars"]
# 답변 모델에 넘기는 부모 절의 최대 길이. 기준선 최대 청크(1,750자)보다 작게 둔다.
PARENT_MAX_CHARS = CHUNKER_CONFIG["parent_max_chars"]
# 쪽 머리의 장·절 제목은 OCR에서 짧은 줄 여러 개로 쪼개져 나온다(예: "3.현금", "사용", "(계좌이체)").
TITLE_LINE_MAX_CHARS = 12
# 7칸 이상 표는 달력이다. 행 단위로 자르면 날짜만 남으므로 통째로 둔다.
CALENDAR_MIN_COLUMNS = 7

_IMAGE = "![image](/image/placeholder)"
_NOISE_TOKEN = r"(?:SEOULM|SEOUL|SE♡UL|M!|SOUL|서울광역\\?|청년센터|100S|TOOS|\d{1,3})"
_NOISE_LINE = re.compile(rf"^(?:\s*{_NOISE_TOKEN})+\s*$")
_TRAILING_NOISE = re.compile(rf"(?:\s+{_NOISE_TOKEN})+\s*$")
_TABLE_SEPARATOR = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_SENTENCE_END = re.compile(r"(?<=[다요]\.)\s+|\n")
# FAQ 문항(Q1, Q2…)마다 새 조각을 시작한다. 답(A1)은 같은 조각에 남긴다.
_FAQ_START = re.compile(r"\s(?=Q\d+\s)")


def clean_line(line: str) -> str:
    """로고·쪽 번호 같은 꼬리말 잡음을 지운다. 내용 줄 끝에 붙은 잡음도 뗀다."""
    line = line.replace(_IMAGE, "").rstrip()
    if _NOISE_LINE.match(line):
        return ""
    trailing = _TRAILING_NOISE.search(line)
    # 숫자만 남는 꼬리("주 30")는 내용일 수 있으므로, 로고 글자가 섞인 경우에만 지운다.
    if trailing and re.search(r"[A-Za-z가-힣!]", trailing.group()):
        line = line[: trailing.start()]
    return line


def split_title(lines: list[str]) -> tuple[str, list[str]]:
    """쪽 머리의 짧은 줄들을 절 제목으로 떼어 낸다."""
    title: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("|") or len(line.replace("#", "").strip()) > TITLE_LINE_MAX_CHARS:
            break
        if line:
            title.append(line.replace("#", "").strip())
        index += 1
    return " ".join(" ".join(title).split()), lines[index:]


def split_blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    """줄을 ("text", 줄들) / ("table", 표 줄들) 블록으로 나눈다."""
    blocks: list[tuple[str, list[str]]] = []
    text: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        is_table_start = (
            "|" in line and index + 1 < len(lines) and _TABLE_SEPARATOR.match(lines[index + 1].strip())
        )
        if not is_table_start:
            text.append(line)
            index += 1
            continue
        # OCR은 표 머리행을 앞 문장과 같은 줄에 붙여 내기도 한다.
        lead, header = line.split("|", 1)
        if lead.strip():
            text.append(lead)
        if text:
            blocks.append(("text", text))
            text = []
        table = ["|" + header]
        index += 2
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            table.append(lines[index].strip())
            index += 1
        blocks.append(("table", table))
    if text:
        blocks.append(("text", text))
    return blocks


def row_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def numbered_items(rows: list[str]) -> list[str] | None:
    """'번호 | 이름 | 번호 | 이름' 다단 목록(12쪽 사용 불가 업종 48개)이면 이름만 번호순으로 돌려준다."""
    items: list[tuple[int, str]] = []
    for row in rows:
        cells = row_cells(row)
        if len(cells) < 4 or len(cells) % 2:
            return None
        for number, name in zip(cells[::2], cells[1::2]):
            if not number and not name:
                continue
            if not number.isdigit() or not name:
                return None
            items.append((int(number), name))
    return [name for _, name in sorted(items)] if items else None


def table_pieces(rows: list[str]) -> list[str]:
    """표를 검색 단위로 나눈다. 행 하나가 항목 하나인 표는 행마다 따로 둔다."""
    rows = [row for row in rows if any(row_cells(row)) and not _TABLE_SEPARATOR.match(row)]
    if not rows:
        return []
    items = numbered_items(rows)
    if items is not None:
        return items
    columns = max(len(row_cells(row)) for row in rows)
    # 한 칸짜리 표는 OCR이 표 모양을 잘못 읽은 경우(34쪽 센터 목록)라 행이 독립된 의미를 갖지 않는다.
    if columns >= CALENDAR_MIN_COLUMNS or columns == 1:
        return ["\n".join(rows)]
    pieces: list[str] = []
    for row in rows:
        # FAQ 표는 질문 행(Q5)과 답 행(A5)이 나뉘어 있다. 답은 질문과 같은 조각에 둔다.
        if pieces and re.match(r"A\d+$", row_cells(row)[0]):
            pieces[-1] += "\n" + row
        else:
            pieces.append(row)
    return pieces


def table_context(text: str) -> str:
    """표 바로 앞 절(마지막 # 제목 표시 이후). 표가 무엇의 목록인지 알려 준다.

    길면 앞부분(절 주제)과 끝부분(표 바로 앞 문장)만 남긴다. 맥락이 길면 행마다 같은 문장이 반복돼
    행끼리 임베딩이 비슷해진다.
    """
    section = " ".join(re.split(r"(?:^|\s)#\s", text)[-1].split())
    half = TABLE_CONTEXT_CHARS // 2
    return section if len(section) <= TABLE_CONTEXT_CHARS else f"{section[:half]} … {section[-half:]}"


def pack(pieces: list[str], max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """짧은 조각은 이어 붙이고, 긴 조각은 문장 끝에서 나눠 max_chars 안에 맞춘다."""
    units: list[str] = []
    for piece in pieces:
        units.extend(part.strip() for part in (_SENTENCE_END.split(piece) if len(piece) > max_chars else [piece]))
    chunks: list[str] = []
    for unit in filter(None, units):
        if chunks and len(chunks[-1]) + 1 + len(unit) <= max_chars:
            chunks[-1] += "\n" + unit
        else:
            chunks.append(unit)
    return chunks


def text_pieces(lines: list[str]) -> list[str]:
    """본문을 OCR 제목 표시(#)와 FAQ 문항 경계에서 나눈 뒤 길이에 맞춰 묶는다."""
    text = "\n".join(lines)
    pieces = [piece for part in re.split(r"(?:^|\s)#\s", text) for piece in _FAQ_START.split(part)]
    return pack([piece.strip() for piece in pieces if piece.strip()])


def block_text(kind: str, block: list[str]) -> str:
    """부모 절에 넣을 원문. 검색용 맥락 문장은 붙이지 않는다(실험 8: 행에 붙인 '…불가' 맥락을 모델이 오독)."""
    if kind == "text":
        return " ".join(" ".join(block).replace(" # ", " ").split()).lstrip("# ")
    rows = [row for row in block if any(row_cells(row)) and not _TABLE_SEPARATOR.match(row)]
    items = numbered_items(rows)
    return ", ".join(items) if items is not None else "\n".join(rows)


def group_sections(blocks: list[tuple[str, list[str]]]) -> list[list[tuple[str, list[str]]]]:
    """본문 블록과 그 뒤에 이어지는 표들을 한 절로 묶는다. 표는 바로 앞 문장이 제목 역할을 한다."""
    sections: list[list[tuple[str, list[str]]]] = []
    for kind, block in blocks:
        if kind == "text" or not sections:
            sections.append([])
        sections[-1].append((kind, block))
    return sections


def block_chunks(kind: str, block: list[str], previous_text: str) -> list[str]:
    if kind == "text":
        return [piece for piece in text_pieces(block) if len(piece) >= MIN_TEXT_CHARS]
    # FAQ 표(Q5 | …)는 질문이 곧 맥락이라 앞 답변 문장을 붙이지 않는다.
    is_faq = re.match(r"Q\d+$", row_cells(block[0])[0]) is not None
    context = "" if is_faq else table_context(previous_text)
    return [f"{context}\n{piece}" if context else piece for piece in table_pieces(block)]


def chunk_page(page: Document) -> list[Document]:
    """작은 청크(검색용)마다 부모 절(답변용) 원문을 붙인다.

    부모 절은 '본문 + 이어지는 표'이고 PARENT_MAX_CHARS를 넘으면 블록 하나로 줄인다.
    블록 하나도 넘으면 부모 없이 작은 청크 그대로 쓴다(예전 12쪽 1,361자처럼 가능·불가가 섞이지 않게).
    """
    lines = [clean_line(line) for line in page.page_content.split("\n")]
    title, body = split_title([line for line in lines if line.strip()])
    page_number = page.metadata["page"]
    header = f"[{page_number}쪽 {title}]".replace(" ]", "]")
    chunks: list[Document] = []
    for section_index, section in enumerate(group_sections(split_blocks(body))):
        section_text = "\n".join(block_text(kind, block) for kind, block in section)
        previous_text = ""
        for block_index, (kind, block) in enumerate(section):
            text = block_text(kind, block)
            if len(section_text) <= PARENT_MAX_CHARS:
                parent = (f"p{page_number:02d}_s{section_index}", section_text)
            elif len(text) <= PARENT_MAX_CHARS:
                parent = (f"p{page_number:02d}_s{section_index}_b{block_index}", text)
            else:
                parent = None
            for body_text in block_chunks(kind, block, previous_text):
                metadata = {"page": page_number, "section": title}
                if parent:
                    metadata |= {"parent_id": parent[0], "parent_text": f"{header}\n{parent[1]}"}
                chunks.append(Document(page_content=f"{header}\n{body_text}", metadata=metadata))
            previous_text = "\n".join(block) if kind == "text" else ""
    return chunks


def chunk_pages(pages: list[Document]) -> list[Document]:
    return [chunk for page in pages for chunk in chunk_page(page)]
